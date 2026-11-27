//! The simulated annealing sampler.

use alloc::vec::Vec;
use core::fmt;

use rand_core::Rng;
use search_core::Budget;

use super::adjacency::Adjacency;
use super::metropolis::{self, beta_range, geometric};
use crate::{Coefficient, Qubo, Run, Sample, SampleError, Sampler, Status};

/// Simulated annealing with single-bit Metropolis flips and a geometric schedule.
///
/// Each [`Sampler::sample`] call starts from a uniform random assignment and runs the
/// sweeps of its [`Schedule`]. A sweep visits each variable once, in index order, and
/// tries to flip it with the Metropolis rule. The inverse temperature `β` grows
/// geometrically from a hot to a cold value over the sweeps.
///
/// The unit of work is one flip attempt ([`AnnealingStats::flips`]). A run that the budget
/// stops resumes at the same variable, sweep and RNG state, so a run in slices gives the
/// same result as one run without a budget. The sampler returns the lowest-energy
/// assignment that it visits, not the last one. Energies are kept in `T`, so they are
/// exact for [`i64`]; only the acceptance test uses [`f64`].
///
/// The RNG is the only randomness, so a seeded RNG gives reproducible samples. The
/// sampler keeps its workspace between calls: a warm call allocates nothing.
#[derive(Debug, Clone)]
pub struct SimulatedAnnealing<R, T> {
    /// The random source for the initial state and the acceptance tests.
    rng: R,
    /// The number of sweeps and the `β` range.
    schedule: Schedule,
    /// The model copy of the current run.
    adjacency: Adjacency<T>,
    /// The current assignment. Same length as the model.
    current: Vec<bool>,
    /// The energy of `current`.
    energy: T,
    /// The lowest-energy assignment that the run has visited.
    best: Vec<bool>,
    /// The energy of `best`.
    best_energy: T,
    /// The hot and cold `β` of the current run.
    betas: (f64, f64),
    /// The index of the current sweep. The run is finished when it reaches the sweep count.
    sweep: u32,
    /// The next variable to visit in the current sweep.
    position: usize,
    /// `false` until the first [`Sampler::sample`] call.
    started: bool,
}

/// The annealing schedule: the number of sweeps and the range of the inverse temperature.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Schedule {
    /// The number of sweeps per run. At least 1.
    sweeps: u32,
    /// The hot and cold `β`, or `None` to derive them from the model.
    betas: Option<(f64, f64)>,
}

impl Schedule {
    /// A schedule with `sweeps` sweeps and a `β` range from the model. This is the default.
    ///
    /// - hot: the largest possible flip delta is accepted with probability 1/2;
    /// - cold: the smallest nonzero coefficient magnitude is accepted with probability 1/100.
    ///
    /// A `sweeps` value of 0 counts as 1.
    #[must_use]
    pub fn from_model(sweeps: u32) -> Self {
        Self {
            sweeps: sweeps.max(1),
            betas: None,
        }
    }

    /// A schedule with `sweeps` sweeps and `β` from `hot` to `cold`, geometrically.
    ///
    /// The units of `β` are inverse energy units of the model. A `sweeps` value of 0 counts
    /// as 1; a single sweep runs at `cold`.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidSchedule`] if `hot` or `cold` is not finite and positive.
    pub fn geometric(sweeps: u32, hot: f64, cold: f64) -> Result<Self, InvalidSchedule> {
        if !is_valid_beta(hot) || !is_valid_beta(cold) {
            return Err(InvalidSchedule);
        }

        Ok(Self {
            sweeps: sweeps.max(1),
            betas: Some((hot, cold)),
        })
    }

    /// The number of sweeps per run.
    #[must_use]
    pub fn sweeps(&self) -> u32 {
        self.sweeps
    }

    /// The hot and cold `β` for the model in `adjacency`: the given range, or the derived one.
    pub(super) fn betas_for<T: Coefficient>(&self, adjacency: &Adjacency<T>) -> (f64, f64) {
        self.betas.unwrap_or_else(|| beta_range(adjacency))
    }
}

/// Returns `true` if `beta` is finite and positive.
fn is_valid_beta(beta: f64) -> bool {
    beta.is_finite() && beta > 0.0
}

/// The error when a [`Schedule`] has a `β` that is not finite and positive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidSchedule;

impl fmt::Display for InvalidSchedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the annealing betas must be finite and positive")
    }
}

impl core::error::Error for InvalidSchedule {}

/// The work counters of one [`SimulatedAnnealing`] call.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AnnealingStats {
    /// The number of sweeps that the call completed.
    pub sweeps: u64,
    /// The number of flip attempts (Metropolis steps) that the call made.
    pub flips: u64,
    /// The number of flip attempts that the Metropolis rule accepted.
    pub accepted: u64,
}

impl<R: Rng, T: Coefficient> SimulatedAnnealing<R, T> {
    /// Creates a sampler that draws from `rng` and follows `schedule`.
    #[must_use]
    pub fn new(rng: R, schedule: Schedule) -> Self {
        Self {
            rng,
            schedule,
            adjacency: Adjacency::new(),
            current: Vec::new(),
            energy: T::ZERO,
            best: Vec::new(),
            best_energy: T::ZERO,
            betas: (1.0, 1.0),
            sweep: 0,
            position: 0,
            started: false,
        }
    }

    /// Resets the workspace for a new run on `model`, from a random assignment.
    fn start(&mut self, model: &Qubo<T>) {
        self.adjacency.rebuild(model);
        self.randomize_current();
        self.energy = self.adjacency.energy(&self.current);
        self.best.clear();
        self.best.extend_from_slice(&self.current);
        self.best_energy = self.energy;
        self.betas = self.schedule.betas_for(&self.adjacency);
        self.sweep = 0;
        self.position = 0;
        self.started = true;
    }

    /// Sets the current assignment to uniform random bits. Reuses the buffer.
    fn randomize_current(&mut self) {
        metropolis::random_bits(&mut self.rng, &mut self.current, self.adjacency.len());
    }

    /// Runs the remaining sweeps until the run finishes or `budget` is exhausted.
    fn run<B: Budget<AnnealingStats>>(
        &mut self,
        mut budget: B,
    ) -> Run<Sample<'_, T>, AnnealingStats> {
        let mut stats = AnnealingStats::default();

        while self.sweep < self.schedule.sweeps {
            if !self.continue_sweep(&mut budget, &mut stats) {
                return self.result(Status::OutOfBudget, stats);
            }

            self.position = 0;
            self.sweep += 1;
            stats.sweeps += 1;
        }

        self.result(Status::Finished, stats)
    }

    /// Runs the current sweep from `position` to its end.
    ///
    /// Returns `false` if `budget` is exhausted before the sweep ends.
    fn continue_sweep<B: Budget<AnnealingStats>>(
        &mut self,
        budget: &mut B,
        stats: &mut AnnealingStats,
    ) -> bool {
        let (hot, cold) = self.betas;
        let beta = geometric(hot, cold, self.sweep, self.schedule.sweeps);

        while self.position < self.current.len() {
            if budget.exhausted(stats) {
                return false;
            }

            stats.accepted += u64::from(self.metropolis_step(self.position, beta));
            stats.flips += 1;
            self.position += 1;
        }

        true
    }

    /// Tries to flip variable `i` at `beta`, and keeps a new best. Returns `true` on a flip.
    fn metropolis_step(&mut self, i: usize, beta: f64) -> bool {
        let delta = self.adjacency.flip_delta(i, &self.current);

        if !self.accepts(delta.to_f64(), beta) {
            return false;
        }

        self.current[i] = !self.current[i];
        self.energy = self.energy.plus(delta);

        if self.energy < self.best_energy {
            self.best_energy = self.energy;
            self.best.copy_from_slice(&self.current);
        }

        true
    }

    /// The Metropolis rule: accept a decrease, and an increase with probability `exp(−β·Δ)`.
    fn accepts(&mut self, delta: f64, beta: f64) -> bool {
        metropolis::accepts(&mut self.rng, delta, beta)
    }

    /// Borrows the best assignment with its exact energy.
    fn result(&self, status: Status, work: AnnealingStats) -> Run<Sample<'_, T>, AnnealingStats> {
        Run {
            sample: Sample {
                bits: &self.best,
                energy: self.adjacency.energy(&self.best),
            },
            status,
            stats: work,
        }
    }
}

impl<R: Rng, T: Coefficient> Sampler<Qubo<T>> for SimulatedAnnealing<R, T> {
    type Stats = AnnealingStats;
    type Sample<'a>
        = Sample<'a, T>
    where
        Self: 'a;
    type Error = SampleError;

    fn sample<B: Budget<AnnealingStats>>(
        &mut self,
        model: &Qubo<T>,
        budget: B,
    ) -> Result<Run<Sample<'_, T>, AnnealingStats>, SampleError> {
        self.start(model);

        Ok(self.run(budget))
    }

    fn resume<B: Budget<AnnealingStats>>(
        &mut self,
        budget: B,
    ) -> Result<Run<Sample<'_, T>, AnnealingStats>, SampleError> {
        if !self.started {
            return Err(SampleError::NothingToResume);
        }

        Ok(self.run(budget))
    }
}

/// Unit tests for the schedule.
#[cfg(test)]
mod tests {
    use super::{InvalidSchedule, Schedule};

    /// A geometric schedule needs finite, positive betas.
    #[test]
    fn geometric_schedule_validates_betas() {
        assert!(Schedule::geometric(10, 0.1, 5.0).is_ok());
        assert_eq!(Schedule::geometric(10, 0.0, 5.0), Err(InvalidSchedule));
        assert_eq!(Schedule::geometric(10, 0.1, f64::NAN), Err(InvalidSchedule));
        assert_eq!(Schedule::from_model(0).sweeps(), 1);
    }
}
