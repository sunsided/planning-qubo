//! The exhaustive sampler: the exact baseline for small models.

use alloc::vec::Vec;

use search_core::Budget;

use super::adjacency::Adjacency;
use crate::{Coefficient, Qubo, Run, Sample, SampleError, Sampler, Status};

/// Tries every assignment and returns one with the lowest energy.
///
/// The enumeration is in Gray code order, so each step flips one bit and costs one flip
/// delta. Among equal energies, the first one in that order wins, so the result is
/// deterministic. The work is `2ⁿ` assignments, so models are limited to
/// [`Exhaustive::MAX_VARIABLES`] variables.
///
/// The unit of work is one evaluated assignment ([`ExhaustiveStats::evaluated`]). A run
/// that the budget stops resumes at the next assignment. For [`f64`], the comparison uses
/// a running energy sum, so near-ties can resolve differently than with exact sums.
#[derive(Debug, Clone)]
pub struct Exhaustive<T> {
    /// The model copy of the current run.
    adjacency: Adjacency<T>,
    /// The assignment at Gray code step `next − 1`. Same length as the model.
    current: Vec<bool>,
    /// The running energy of `current`.
    energy: T,
    /// The number of assignments that the run has evaluated: the next Gray code step.
    next: u64,
    /// The Gray code step of the best assignment so far.
    best_step: u64,
    /// The energy of the best assignment so far. Equal to the offset while `next == 0`.
    best_energy: T,
    /// The best assignment, written from `best_step` when a call returns.
    best: Vec<bool>,
    /// `false` until the first [`Sampler::sample`] call.
    started: bool,
}

/// The work counters of one [`Exhaustive`] call.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExhaustiveStats {
    /// The number of assignments that the call evaluated.
    pub evaluated: u64,
}

impl<T: Coefficient> Default for Exhaustive<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Coefficient> Exhaustive<T> {
    /// The largest number of variables that the sampler accepts.
    pub const MAX_VARIABLES: usize = 30;

    /// Creates a sampler with an empty workspace.
    #[must_use]
    pub fn new() -> Self {
        Self {
            adjacency: Adjacency::new(),
            current: Vec::new(),
            energy: T::ZERO,
            next: 0,
            best_step: 0,
            best_energy: T::ZERO,
            best: Vec::new(),
            started: false,
        }
    }

    /// Resets the workspace for a new run on `model`. Reuses the buffers.
    fn start(&mut self, model: &Qubo<T>) {
        self.adjacency.rebuild(model);
        self.current.clear();
        self.current.resize(model.num_variables(), false);
        self.energy = model.offset();
        self.next = 0;
        self.best_step = 0;
        self.best_energy = model.offset();
        self.started = true;
    }

    /// Evaluates assignments until all are done or `budget` is exhausted.
    fn run<B: Budget<ExhaustiveStats>>(
        &mut self,
        mut budget: B,
    ) -> Run<Sample<'_, T>, ExhaustiveStats> {
        let total = 1_u64 << self.adjacency.len();
        let mut stats = ExhaustiveStats::default();

        while self.next < total {
            if budget.exhausted(&stats) {
                return self.result(Status::OutOfBudget, stats);
            }

            self.evaluate_next();
            stats.evaluated += 1;
        }

        self.result(Status::Finished, stats)
    }

    /// Evaluates Gray code step `next`: flips one bit (none for step 0) and keeps a new best.
    fn evaluate_next(&mut self) {
        let step = self.next;

        if step > 0 {
            let i = bit_to_flip(step);
            self.energy = self
                .energy
                .plus(self.adjacency.flip_delta(i, &self.current));
            self.current[i] = !self.current[i];
        }

        if step == 0 || self.energy < self.best_energy {
            self.best_energy = self.energy;
            self.best_step = step;
        }

        self.next += 1;
    }

    /// Writes the best assignment into the workspace and borrows it.
    fn result(
        &mut self,
        status: Status,
        work: ExhaustiveStats,
    ) -> Run<Sample<'_, T>, ExhaustiveStats> {
        write_gray_code_bits(self.best_step, &mut self.best, self.adjacency.len());

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

impl<T: Coefficient> Sampler<Qubo<T>> for Exhaustive<T> {
    type Stats = ExhaustiveStats;
    type Sample<'a> = Sample<'a, T>;
    type Error = SampleError;

    fn sample<B: Budget<ExhaustiveStats>>(
        &mut self,
        model: &Qubo<T>,
        budget: B,
    ) -> Result<Run<Sample<'_, T>, ExhaustiveStats>, SampleError> {
        let variables = model.num_variables();

        if variables > Self::MAX_VARIABLES {
            return Err(SampleError::TooManyVariables { variables });
        }

        self.start(model);

        Ok(self.run(budget))
    }

    fn resume<B: Budget<ExhaustiveStats>>(
        &mut self,
        budget: B,
    ) -> Result<Run<Sample<'_, T>, ExhaustiveStats>, SampleError> {
        if !self.started {
            return Err(SampleError::NothingToResume);
        }

        Ok(self.run(budget))
    }
}

/// The bit that Gray code step `step` flips: the index of the lowest set bit of `step`.
fn bit_to_flip(step: u64) -> usize {
    usize::try_from(step.trailing_zeros()).expect("a u64 has at most 64 trailing zeros")
}

/// Writes the assignment after `step` Gray code steps into `bits`: the bits of `step ^ (step >> 1)`.
///
/// Reuses the capacity of `bits`, so a buffer with capacity `n` does not allocate.
fn write_gray_code_bits(step: u64, bits: &mut Vec<bool>, n: usize) {
    let code = step ^ (step >> 1);
    bits.clear();
    bits.extend((0..n).map(|i| code >> i & 1 == 1));
}

/// Unit tests for the exhaustive sampler.
#[cfg(test)]
mod tests {
    use super::{Exhaustive, ExhaustiveStats};
    use crate::{Qubo, SampleError, Sampler, Status, Unlimited};

    /// The sampler finds the ground state and reports its energy.
    #[test]
    fn finds_the_ground_state() {
        let mut qubo = Qubo::<i64>::new(2);
        qubo.add_linear(0, -2);
        qubo.add_linear(1, -3);
        qubo.add_quadratic(0, 1, 5);

        let mut sampler = Exhaustive::new();
        let run = sampler
            .sample(&qubo, Unlimited)
            .expect("two variables are allowed");

        assert_eq!(run.sample.bits, [false, true]);
        assert_eq!(run.sample.energy, -3);
        assert_eq!(run.status, Status::Finished);
        assert_eq!(run.stats, ExhaustiveStats { evaluated: 4 });
    }

    /// An empty model has one assignment with the offset as its energy.
    #[test]
    fn empty_model_returns_the_offset() {
        let mut qubo = Qubo::<f64>::new(0);
        qubo.add_offset(1.5);

        let mut sampler = Exhaustive::new();
        let run = sampler
            .sample(&qubo, Unlimited)
            .expect("zero variables are allowed");

        assert_eq!(run.sample.bits, [false; 0]);
        assert!((run.sample.energy - 1.5).abs() < f64::EPSILON);
    }

    /// Too many variables is an error, not a long wait.
    #[test]
    fn rejects_large_models() {
        let variables = Exhaustive::<i64>::MAX_VARIABLES + 1;
        let qubo = Qubo::<i64>::new(variables);

        assert_eq!(
            Exhaustive::new().sample(&qubo, Unlimited).err(),
            Some(SampleError::TooManyVariables { variables })
        );
    }

    /// Resume needs a run.
    #[test]
    fn resume_needs_a_run() {
        let mut sampler = Exhaustive::<i64>::new();

        assert_eq!(
            sampler.resume(Unlimited).err(),
            Some(SampleError::NothingToResume)
        );
    }
}
