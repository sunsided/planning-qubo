//! Time-indexed scheduling: jobs on a shared resource over discrete time slots.
//!
//! Time is split into the slots `0..horizon`. Each [`Job`] runs for `duration` consecutive
//! slots, starts no earlier than its `release` and ends no later than its `deadline` (and
//! the horizon). The encoding has one variable `job{j}.start{t}` per job `j` and allowed
//! start `t`, which is 1 when the job starts at `t`. The objective is the weighted
//! completion time `Σ weightⱼ · (startⱼ + durationⱼ)`. The constraints are:
//!
//! - `job{j}.exactly_one`: job `j` starts exactly once;
//! - `slot{τ}.capacity`: at most `capacity` jobs run in slot `τ`. Capacity 1 uses an
//!   at-most-one penalty, other capacities a linear constraint with slack bits. A slot that
//!   fewer jobs can cover than the capacity has no constraint;
//! - `job{a}.before.job{b}`: for a precedence `(a, b)`, job `b` starts after job `a` ends.
//!   Each start pair that breaks it adds `P·x·y`.
//!
//! Each violation costs at least the penalty weight `P`. Use [`penalty::safe`] for a weight
//! that makes each ground state a feasible optimum.
//!
//! [`penalty::safe`]: crate::penalty::safe
//!
//! ```
//! use planning_qubo::scheduling::{self, Job};
//! use planning_qubo::{Exhaustive, Sampler, Unlimited, penalty};
//!
//! // Two jobs on one machine, the long one with a small weight.
//! let jobs = [Job::new(3, 1), Job::new(1, 5)];
//! let (_, encoding) = penalty::safe(|p| scheduling::encode(&jobs, 4, 1, &[], p))?;
//!
//! let mut sampler = Exhaustive::new();
//! let run = sampler.sample(encoding.problem().qubo(), Unlimited)?;
//! let decoded = encoding.decode(run.sample.bits);
//!
//! // The short, heavy job goes first: 5·1 + 1·4 = 9.
//! assert_eq!(decoded.starts, [Some(1), Some(0)]);
//! assert_eq!(decoded.cost, 9);
//! assert!(decoded.is_feasible());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use crate::csp::one_hot;
use crate::{Builder, Coefficient, Encoded, Problem, Var};

/// A job: its duration, its time window and its weight in the objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Job<T> {
    /// The number of consecutive slots that the job runs. At least 1.
    pub duration: usize,
    /// The earliest start slot.
    pub release: usize,
    /// The slot by which the job must end: `start + duration ≤ deadline`.
    pub deadline: usize,
    /// The weight of the job's completion time in the objective.
    pub weight: T,
}

impl<T> Job<T> {
    /// A job of `duration` slots and `weight`, released at 0, with no deadline but the horizon.
    #[must_use]
    pub fn new(duration: usize, weight: T) -> Self {
        Self {
            duration,
            release: 0,
            deadline: usize::MAX,
            weight,
        }
    }
}

/// The error when a scheduling instance cannot be encoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ScheduleError {
    /// A job has the duration 0.
    ZeroDuration {
        /// The job.
        job: usize,
    },
    /// A precedence names a job that does not exist.
    UnknownJob {
        /// The index of the precedence.
        precedence: usize,
    },
    /// The horizon exceeds `u32::MAX`, so completion times do not fit the coefficients.
    HorizonTooLong,
}

impl fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDuration { job } => write!(f, "job {job} has the duration 0"),
            Self::UnknownJob { precedence } => {
                write!(f, "precedence {precedence} names a job that does not exist")
            }
            Self::HorizonTooLong => f.write_str("the horizon exceeds u32::MAX slots"),
        }
    }
}

impl core::error::Error for ScheduleError {}

/// The start variables of each job with their start slots, in slot order.
type Starts = Vec<Vec<(usize, Var)>>;

/// An encoded scheduling instance: the QUBO problem and the data to decode samples.
#[derive(Debug, Clone, PartialEq)]
pub struct Encoding<T> {
    /// The QUBO problem with its constraints.
    problem: Problem<T>,
    /// The start variables of each job, with their start slots, in slot order.
    starts: Starts,
    /// The completion cost `weight · (start + duration)` of each start variable, as `starts`.
    costs: Vec<Vec<T>>,
}

/// A sample, decoded into a schedule.
#[derive(Debug, Clone, PartialEq)]
pub struct Decoded<T> {
    /// The start slot of each job. `None` if the sample gives the job no start or several.
    pub starts: Vec<Option<usize>>,
    /// The weighted completion time of all chosen starts, also for an infeasible sample.
    pub cost: T,
    /// The names of the violated constraints, in the order of [`Problem::constraints`].
    pub violations: Vec<String>,
}

impl<T> Decoded<T> {
    /// Returns `true` if the sample violates no constraint.
    #[must_use]
    pub fn is_feasible(&self) -> bool {
        self.violations.is_empty()
    }
}

/// Encodes `jobs` on a resource that runs `capacity` jobs at once, over `horizon` slots.
///
/// `precedences` holds pairs `(a, b)`: job `b` starts after job `a` ends. `penalty` is the
/// weight `P` of each constraint. A job whose window holds no start gets no variables, so
/// its `exactly_one` constraint is always violated.
///
/// # Errors
///
/// Returns [`ScheduleError`] for a zero duration, a precedence with an unknown job, or a
/// horizon above `u32::MAX`.
pub fn encode<T: Coefficient>(
    jobs: &[Job<T>],
    horizon: usize,
    capacity: u32,
    precedences: &[(usize, usize)],
    penalty: T,
) -> Result<Encoding<T>, ScheduleError> {
    validate(jobs, horizon, precedences)?;

    let mut builder = Builder::new();
    let (starts, costs) = add_starts(&mut builder, jobs, horizon);

    for (j, row) in starts.iter().enumerate() {
        let vars: Vec<Var> = row.iter().map(|&(_, v)| v).collect();
        builder.exactly_one(&format!("job{j}.exactly_one"), &vars, penalty);
    }

    add_capacity_constraints(&mut builder, jobs, &starts, horizon, capacity, penalty);
    add_precedences(&mut builder, jobs, &starts, precedences, penalty);

    Ok(Encoding {
        problem: builder.build(),
        starts,
        costs,
    })
}

/// Checks the durations, the precedence indices and the horizon.
fn validate<T>(
    jobs: &[Job<T>],
    horizon: usize,
    precedences: &[(usize, usize)],
) -> Result<(), ScheduleError> {
    if let Some(job) = jobs.iter().position(|job| job.duration == 0) {
        return Err(ScheduleError::ZeroDuration { job });
    }

    if let Some(precedence) = precedences
        .iter()
        .position(|&(a, b)| a >= jobs.len() || b >= jobs.len())
    {
        return Err(ScheduleError::UnknownJob { precedence });
    }

    if u32::try_from(horizon).is_err() {
        return Err(ScheduleError::HorizonTooLong);
    }

    Ok(())
}

/// Creates the start variables of each job with their completion costs.
fn add_starts<T: Coefficient>(
    builder: &mut Builder<T>,
    jobs: &[Job<T>],
    horizon: usize,
) -> (Starts, Vec<Vec<T>>) {
    let mut starts = Vec::with_capacity(jobs.len());
    let mut costs = Vec::with_capacity(jobs.len());

    for (j, job) in jobs.iter().enumerate() {
        let mut row = Vec::new();
        let mut row_costs = Vec::new();

        for t in start_window(job, horizon) {
            let var = builder.variable(&format!("job{j}.start{t}"));
            let cost = job.weight.times(slot_count(t + job.duration));
            builder.add_linear(var, cost);
            row.push((t, var));
            row_costs.push(cost);
        }

        starts.push(row);
        costs.push(row_costs);
    }

    (starts, costs)
}

/// The allowed start slots of `job`: from its release until it ends by its deadline and the horizon.
fn start_window<T>(job: &Job<T>, horizon: usize) -> core::ops::Range<usize> {
    let end = job.deadline.min(horizon);
    let last = end.saturating_sub(job.duration - 1);

    job.release..last.max(job.release)
}

/// A slot count as a coefficient. The horizon is validated to fit `u32`.
fn slot_count<T: Coefficient>(slots: usize) -> T {
    T::from_u32(u32::try_from(slots).expect("slots stay within the validated horizon"))
}

/// Adds `slot{τ}.capacity` for each slot that more jobs can cover than the capacity.
fn add_capacity_constraints<T: Coefficient>(
    builder: &mut Builder<T>,
    jobs: &[Job<T>],
    starts: &[Vec<(usize, Var)>],
    horizon: usize,
    capacity: u32,
    penalty: T,
) {
    for slot in 0..horizon {
        let running: Vec<Var> = jobs
            .iter()
            .zip(starts)
            .flat_map(|(job, row)| {
                row.iter()
                    .filter(move |&&(t, _)| t <= slot && slot < t + job.duration)
                    .map(|&(_, v)| v)
            })
            .collect();

        if usize::try_from(capacity).is_ok_and(|c| running.len() <= c) {
            continue;
        }

        let name = format!("slot{slot}.capacity");

        if capacity == 1 {
            builder.at_most_one(&name, &running, penalty);
        } else {
            let terms: Vec<(Var, T)> = running.iter().map(|&v| (v, T::ONE)).collect();
            builder.linear_equality(&name, &terms, T::from_u32(capacity), capacity, penalty);
        }
    }
}

/// Adds `job{a}.before.job{b}`: each start pair where `b` starts before `a` ends.
fn add_precedences<T: Coefficient>(
    builder: &mut Builder<T>,
    jobs: &[Job<T>],
    starts: &[Vec<(usize, Var)>],
    precedences: &[(usize, usize)],
    penalty: T,
) {
    for &(a, b) in precedences {
        let name = format!("job{a}.before.job{b}");
        let duration = jobs[a].duration;

        for &(t, x) in &starts[a] {
            for &(_, y) in starts[b].iter().filter(|&&(s, _)| s < t + duration) {
                builder.at_most_one(&name, &[x, y], penalty);
            }
        }
    }
}

impl<T> Encoded<T> for Encoding<T> {
    fn problem(&self) -> &Problem<T> {
        &self.problem
    }
}

impl<T: Coefficient> Encoding<T> {
    /// The QUBO problem with its constraints.
    #[must_use]
    pub fn problem(&self) -> &Problem<T> {
        &self.problem
    }

    /// The variable that is 1 when job `job` starts at slot `start`, if that start is allowed.
    ///
    /// # Panics
    ///
    /// Panics if `job` is out of range.
    #[must_use]
    pub fn variable(&self, job: usize, start: usize) -> Option<Var> {
        self.starts[job]
            .iter()
            .find(|&&(t, _)| t == start)
            .map(|&(_, v)| v)
    }

    /// Decodes the assignment `bits` of all QUBO variables, slack bits included.
    ///
    /// # Panics
    ///
    /// Panics if `bits.len()` is not the number of QUBO variables.
    #[must_use]
    pub fn decode(&self, bits: &[bool]) -> Decoded<T> {
        let violations = self.problem.violation_names(bits);
        let starts = self
            .starts
            .iter()
            .map(|row| {
                let vars: Vec<Var> = row.iter().map(|&(_, v)| v).collect();
                one_hot(&vars, bits).map(|k| row[k].0)
            })
            .collect();

        Decoded {
            starts,
            cost: self.cost(bits),
            violations,
        }
    }

    /// The weighted completion time of the chosen starts.
    fn cost(&self, bits: &[bool]) -> T {
        self.starts
            .iter()
            .flatten()
            .zip(self.costs.iter().flatten())
            .filter(|&(&(_, v), _)| bits[v.index()])
            .fold(T::ZERO, |sum, (_, &c)| sum.plus(c))
    }
}
