//! The parallel tempering (replica exchange) sampler.

use alloc::vec::Vec;

use rand_core::Rng;
use search_core::Budget;

use super::adjacency::Adjacency;
use super::metropolis::{self, geometric};
use crate::{Coefficient, Qubo, Run, Sample, SampleError, Sampler, Schedule, Status};

/// Parallel tempering: replicas at fixed inverse temperatures that swap states.
///
/// Each [`Sampler::sample`] call starts every replica from a uniform random assignment.
/// The replicas sit on a geometric `β` ladder from the [`Schedule`]'s hot to its cold value;
/// the schedule's sweep count is the number of rounds. A round sweeps every replica once
/// with Metropolis flips, hottest first, and then proposes swaps between neighbors on the
/// ladder: the even pairs `(0, 1), (2, 3), …` in even rounds and the odd pairs in odd
/// rounds. A swap of replicas `k` and `k + 1` is accepted with probability
/// `min(1, exp((βₖ − βₖ₊₁)·(Eₖ − Eₖ₊₁)))`. The "replicas" are interleaved on one thread; use
/// the `parallel` feature to run several samplers at once.
///
/// The unit of work is one flip attempt ([`TemperingStats::flips`]); a swap phase is part of
/// the flip that ends its round. A run that the budget stops resumes at the same replica,
/// variable and RNG state, so a sliced run equals an unsliced one. The sampler returns the
/// lowest-energy assignment that any replica visits, with an exact energy for [`i64`]. A
/// warm call allocates nothing.
#[derive(Debug, Clone)]
pub struct ParallelTempering<R, T> {
    /// The random source for the initial states and all acceptance tests.
    rng: R,
    /// The number of rounds and the `β` range of the ladder.
    schedule: Schedule,
    /// The number of replicas. At least 1.
    replicas: usize,
    /// The model copy of the current run.
    adjacency: Adjacency<T>,
    /// The state of each replica slot, `replicas × n`, slot-major.
    states: Vec<bool>,
    /// The energy of each replica slot.
    energies: Vec<T>,
    /// The ladder position of each replica slot: slot `order[k]` runs at `betas[k]`.
    order: Vec<usize>,
    /// The inverse temperature of each ladder position, hottest first.
    betas: Vec<f64>,
    /// The lowest-energy assignment that the run has visited.
    best: Vec<bool>,
    /// The energy of `best`.
    best_energy: T,
    /// The index of the current round. The run is finished when it reaches the round count.
    round: u32,
    /// The ladder position that the current round sweeps next.
    rung: usize,
    /// The next variable of the replica at `rung`.
    position: usize,
    /// `false` until the first [`Sampler::sample`] call.
    started: bool,
}

/// The work counters of one [`ParallelTempering`] call.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TemperingStats {
    /// The number of rounds that the call completed.
    pub rounds: u64,
    /// The number of flip attempts (Metropolis steps) that the call made.
    pub flips: u64,
    /// The number of flip attempts that the Metropolis rule accepted.
    pub accepted: u64,
    /// The number of proposed replica swaps.
    pub swaps_proposed: u64,
    /// The number of accepted replica swaps.
    pub swaps_accepted: u64,
}

impl<R: Rng, T: Coefficient> ParallelTempering<R, T> {
    /// Creates a sampler with `replicas` replicas on the ladder of `schedule`.
    ///
    /// The schedule's sweep count is the number of rounds. A `replicas` value of 0 counts
    /// as 1; a single replica runs at the cold `β`, without swaps.
    #[must_use]
    pub fn new(rng: R, schedule: Schedule, replicas: usize) -> Self {
        Self {
            rng,
            schedule,
            replicas: replicas.max(1),
            adjacency: Adjacency::new(),
            states: Vec::new(),
            energies: Vec::new(),
            order: Vec::new(),
            betas: Vec::new(),
            best: Vec::new(),
            best_energy: T::ZERO,
            round: 0,
            rung: 0,
            position: 0,
            started: false,
        }
    }

    /// Resets the workspace for a new run on `model`, from random assignments.
    fn start(&mut self, model: &Qubo<T>) {
        self.adjacency.rebuild(model);
        self.randomize_states();
        self.fill_ladder();
        self.keep_best_replica();
        self.round = 0;
        self.rung = 0;
        self.position = 0;
        self.started = true;
    }

    /// Draws a random state for each replica and computes its energy. Reuses the buffers.
    fn randomize_states(&mut self) {
        let n = self.adjacency.len();
        metropolis::random_bits(&mut self.rng, &mut self.states, n * self.replicas);

        self.energies.clear();
        for slot in 0..self.replicas {
            let energy = self
                .adjacency
                .energy(&self.states[slot * n..(slot + 1) * n]);
            self.energies.push(energy);
        }
    }

    /// Puts slot `k` at ladder position `k`, and spaces the `β` values geometrically.
    fn fill_ladder(&mut self) {
        let (hot, cold) = self.schedule.betas_for(&self.adjacency);
        let rungs = u32::try_from(self.replicas).unwrap_or(u32::MAX);

        self.order.clear();
        self.order.extend(0..self.replicas);
        self.betas.clear();
        self.betas
            .extend((0..rungs).map(|k| geometric(hot, cold, k, rungs)));
    }

    /// Copies the lowest-energy replica into `best`.
    fn keep_best_replica(&mut self) {
        let n = self.adjacency.len();
        let slot = (1..self.replicas).fold(0, |low, k| {
            if self.energies[k] < self.energies[low] {
                k
            } else {
                low
            }
        });

        self.best.clear();
        self.best
            .extend_from_slice(&self.states[slot * n..(slot + 1) * n]);
        self.best_energy = self.energies[slot];
    }

    /// Runs the remaining rounds until the run finishes or `budget` is exhausted.
    fn run<B: Budget<TemperingStats>>(
        &mut self,
        mut budget: B,
    ) -> Run<Sample<'_, T>, TemperingStats> {
        let mut stats = TemperingStats::default();

        while self.round < self.schedule.sweeps() {
            if !self.continue_round(&mut budget, &mut stats) {
                return self.result(Status::OutOfBudget, stats);
            }

            self.exchange(&mut stats);
            self.rung = 0;
            self.round += 1;
            stats.rounds += 1;
        }

        self.result(Status::Finished, stats)
    }

    /// Sweeps the replicas from `rung` and `position` to the end of the round.
    ///
    /// Returns `false` if `budget` is exhausted before the round ends.
    fn continue_round<B: Budget<TemperingStats>>(
        &mut self,
        budget: &mut B,
        stats: &mut TemperingStats,
    ) -> bool {
        let n = self.adjacency.len();

        while self.rung < self.replicas {
            while self.position < n {
                if budget.exhausted(stats) {
                    return false;
                }

                stats.accepted += u64::from(self.metropolis_step());
                stats.flips += 1;
                self.position += 1;
            }

            self.position = 0;
            self.rung += 1;
        }

        true
    }

    /// Tries to flip variable `position` of the replica at `rung`, and keeps a new best.
    ///
    /// Returns `true` on a flip.
    fn metropolis_step(&mut self) -> bool {
        let n = self.adjacency.len();
        let (slot, i) = (self.order[self.rung], self.position);
        let state = &mut self.states[slot * n..(slot + 1) * n];
        let delta = self.adjacency.flip_delta(i, state);

        if !metropolis::accepts(&mut self.rng, delta.to_f64(), self.betas[self.rung]) {
            return false;
        }

        state[i] = !state[i];
        let energy = self.energies[slot].plus(delta);
        self.energies[slot] = energy;

        if energy < self.best_energy {
            self.best_energy = energy;
            self.best.copy_from_slice(state);
        }

        true
    }

    /// Proposes swaps between ladder neighbors, the even or odd pairs by round parity.
    fn exchange(&mut self, stats: &mut TemperingStats) {
        let first = usize::from(self.round % 2 == 1);

        for k in (first..self.replicas.saturating_sub(1)).step_by(2) {
            let (low, high) = (self.order[k], self.order[k + 1]);
            let gap = self.betas[k] - self.betas[k + 1];
            let difference = self.energies[low].to_f64() - self.energies[high].to_f64();

            stats.swaps_proposed += 1;

            if metropolis::accepts_exponent(&mut self.rng, gap * difference) {
                self.order.swap(k, k + 1);
                stats.swaps_accepted += 1;
            }
        }
    }

    /// Borrows the best assignment with its exact energy.
    fn result(&self, status: Status, work: TemperingStats) -> Run<Sample<'_, T>, TemperingStats> {
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

impl<R: Rng, T: Coefficient> Sampler<Qubo<T>> for ParallelTempering<R, T> {
    type Stats = TemperingStats;
    type Sample<'a>
        = Sample<'a, T>
    where
        Self: 'a;
    type Error = SampleError;

    fn sample<B: Budget<TemperingStats>>(
        &mut self,
        model: &Qubo<T>,
        budget: B,
    ) -> Result<Run<Sample<'_, T>, TemperingStats>, SampleError> {
        self.start(model);

        Ok(self.run(budget))
    }

    fn resume<B: Budget<TemperingStats>>(
        &mut self,
        budget: B,
    ) -> Result<Run<Sample<'_, T>, TemperingStats>, SampleError> {
        if !self.started {
            return Err(SampleError::NothingToResume);
        }

        Ok(self.run(budget))
    }
}

/// Unit tests for the replica energies and the ladder.
#[cfg(test)]
mod tests {
    use rand_core::SeedableRng;
    use rand_pcg::Pcg64Mcg;

    use super::ParallelTempering;
    use crate::{Qubo, Sampler, Schedule, Unlimited};

    /// A chain model with frustrated pairs.
    fn model() -> Qubo<i64> {
        let mut qubo = Qubo::new(6);

        for i in 0..6 {
            qubo.add_linear(i, -3);
        }

        for i in 0..5 {
            qubo.add_quadratic(i, i + 1, 4);
        }

        qubo
    }

    /// Each replica's kept energy matches its state, and the ladder runs hot to cold.
    #[test]
    fn replica_energies_stay_exact() {
        let qubo = model();
        let mut sampler =
            ParallelTempering::new(Pcg64Mcg::seed_from_u64(9), Schedule::from_model(30), 4);
        let run = sampler.sample(&qubo, Unlimited).expect("any model");

        assert_eq!(run.stats.rounds, 30);
        assert_eq!(run.stats.flips, 30 * 4 * 6);
        assert!(run.stats.swaps_proposed > 0);

        for slot in 0..4 {
            let state = &sampler.states[slot * 6..(slot + 1) * 6];
            assert_eq!(sampler.energies[slot], qubo.energy(state));
        }

        assert!(sampler.betas.windows(2).all(|w| w[0] < w[1]));
        let mut order = sampler.order.clone();
        order.sort_unstable();
        assert_eq!(order, [0, 1, 2, 3]);
    }
}
