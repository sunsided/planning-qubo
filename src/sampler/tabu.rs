//! The tabu search sampler.

use alloc::vec::Vec;

use rand_core::Rng;
use search_core::Budget;

use super::adjacency::Adjacency;
use super::metropolis;
use crate::{Coefficient, Qubo, Run, Sample, SampleError, Sampler, Status};

/// Tabu search with single-bit flips and a fixed tenure.
///
/// Each [`Sampler::sample`] call starts from a uniform random assignment. Each move flips
/// the variable with the lowest flip delta, also when that raises the energy. A flipped
/// variable is tabu for the next `tenure` moves, unless flipping it would give a new best
/// energy (the aspiration rule). Among equal deltas the lowest index wins, so the RNG is
/// used for the initial state only.
///
/// The tenure is clamped to the number of variables minus one, so each move has at least
/// one allowed variable. Typical values are a tenth to a quarter of the variable count.
///
/// The unit of work is one move ([`TabuStats::moves`]). The sampler keeps every flip delta
/// up to date, so a move costs `O(n + degree)`. A run that the budget stops resumes at the
/// same move. The sampler returns the lowest-energy assignment that it visits, with an
/// exact energy for [`i64`]. A warm call allocates nothing.
#[derive(Debug, Clone)]
pub struct TabuSearch<R, T> {
    /// The random source for the initial state.
    rng: R,
    /// The number of moves per run.
    moves: u64,
    /// The number of moves that a flipped variable stays tabu, before clamping.
    tenure: u64,
    /// The model copy of the current run.
    adjacency: Adjacency<T>,
    /// The current assignment. Same length as the model.
    current: Vec<bool>,
    /// The energy change when each variable of `current` flips.
    deltas: Vec<T>,
    /// The first move at which each variable is no longer tabu.
    tabu_until: Vec<u64>,
    /// The energy of `current`.
    energy: T,
    /// The lowest-energy assignment that the run has visited.
    best: Vec<bool>,
    /// The energy of `best`.
    best_energy: T,
    /// The index of the next move. The run is finished when it reaches `moves`.
    step: u64,
    /// `false` until the first [`Sampler::sample`] call.
    started: bool,
}

/// The work counters of one [`TabuSearch`] call.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TabuStats {
    /// The number of moves that the call made.
    pub moves: u64,
    /// The number of moves that set a new best energy.
    pub improvements: u64,
}

impl<R: Rng, T: Coefficient> TabuSearch<R, T> {
    /// Creates a sampler that runs `moves` moves per run with the given `tenure`.
    ///
    /// The initial state of each run is drawn from `rng`.
    #[must_use]
    pub fn new(rng: R, moves: u64, tenure: u64) -> Self {
        Self {
            rng,
            moves,
            tenure,
            adjacency: Adjacency::new(),
            current: Vec::new(),
            deltas: Vec::new(),
            tabu_until: Vec::new(),
            energy: T::ZERO,
            best: Vec::new(),
            best_energy: T::ZERO,
            step: 0,
            started: false,
        }
    }

    /// Resets the workspace for a new run on `model`, from a random assignment.
    fn start(&mut self, model: &Qubo<T>) {
        let n = model.num_variables();
        self.adjacency.rebuild(model);
        metropolis::random_bits(&mut self.rng, &mut self.current, n);
        self.energy = self.adjacency.energy(&self.current);

        self.deltas.clear();
        self.deltas
            .extend((0..n).map(|i| self.adjacency.flip_delta(i, &self.current)));
        self.tabu_until.clear();
        self.tabu_until.resize(n, 0);

        self.best.clear();
        self.best.extend_from_slice(&self.current);
        self.best_energy = self.energy;
        self.step = 0;
        self.started = true;
    }

    /// Makes the remaining moves until the run finishes or `budget` is exhausted.
    fn run<B: Budget<TabuStats>>(&mut self, mut budget: B) -> Run<Sample<'_, T>, TabuStats> {
        let mut stats = TabuStats::default();

        if self.current.is_empty() {
            return self.result(Status::Finished, stats);
        }

        while self.step < self.moves {
            if budget.exhausted(&stats) {
                return self.result(Status::OutOfBudget, stats);
            }

            stats.improvements += u64::from(self.make_move());
            stats.moves += 1;
        }

        self.result(Status::Finished, stats)
    }

    /// Flips the best allowed variable. Returns `true` if the move set a new best energy.
    fn make_move(&mut self) -> bool {
        let i = self.choose();
        self.flip(i);
        self.tabu_until[i] = self.step + 1 + self.clamped_tenure();
        self.step += 1;

        if self.energy < self.best_energy {
            self.best_energy = self.energy;
            self.best.copy_from_slice(&self.current);
            return true;
        }

        false
    }

    /// The tenure, at most the variable count minus one, so a move is always allowed.
    fn clamped_tenure(&self) -> u64 {
        let most = u64::try_from(self.current.len() - 1).unwrap_or(u64::MAX);

        self.tenure.min(most)
    }

    /// The variable with the lowest delta that is not tabu or that reaches a new best.
    fn choose(&self) -> usize {
        let allowed = |&(i, &delta): &(usize, &T)| {
            self.tabu_until[i] <= self.step || self.energy.plus(delta) < self.best_energy
        };

        self.deltas
            .iter()
            .enumerate()
            .filter(allowed)
            .fold(None, |best: Option<(usize, T)>, (i, &delta)| match best {
                Some((_, low)) if delta >= low => best,
                _ => Some((i, delta)),
            })
            .map(|(i, _)| i)
            .expect("the clamped tenure leaves at least one variable allowed")
    }

    /// Flips variable `i` and updates the energy and the deltas of `i` and its neighbors.
    ///
    /// Setting `xᵢ` adds `bᵢⱼ` to the local field of each neighbor `j`, clearing it
    /// removes `bᵢⱼ`. The delta of `j` is its field when `xⱼ = 0` and minus its field when
    /// `xⱼ = 1`.
    fn flip(&mut self, i: usize) {
        self.energy = self.energy.plus(self.deltas[i]);
        self.deltas[i] = T::ZERO.minus(self.deltas[i]);
        self.current[i] = !self.current[i];
        let set = self.current[i];

        for &(j, b) in self.adjacency.row(i) {
            let raises = set != self.current[j];
            let delta = &mut self.deltas[j];
            *delta = if raises {
                delta.plus(b)
            } else {
                delta.minus(b)
            };
        }
    }

    /// Borrows the best assignment with its exact energy.
    fn result(&self, status: Status, work: TabuStats) -> Run<Sample<'_, T>, TabuStats> {
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

impl<R: Rng, T: Coefficient> Sampler<Qubo<T>> for TabuSearch<R, T> {
    type Stats = TabuStats;
    type Sample<'a>
        = Sample<'a, T>
    where
        Self: 'a;
    type Error = SampleError;

    fn sample<B: Budget<TabuStats>>(
        &mut self,
        model: &Qubo<T>,
        budget: B,
    ) -> Result<Run<Sample<'_, T>, TabuStats>, SampleError> {
        self.start(model);

        Ok(self.run(budget))
    }

    fn resume<B: Budget<TabuStats>>(
        &mut self,
        budget: B,
    ) -> Result<Run<Sample<'_, T>, TabuStats>, SampleError> {
        if !self.started {
            return Err(SampleError::NothingToResume);
        }

        Ok(self.run(budget))
    }
}

/// Unit tests for the incremental deltas and the tabu rule.
#[cfg(test)]
mod tests {
    use rand_core::SeedableRng;
    use rand_pcg::Pcg64Mcg;

    use super::TabuSearch;
    use crate::{Exhaustive, Qubo, Sampler, Status, Unlimited};

    /// A frustrated model on four variables with all pair types.
    fn model() -> Qubo<i64> {
        let mut qubo = Qubo::new(4);
        qubo.add_offset(3);
        qubo.add_linear(0, -2);
        qubo.add_linear(1, 4);
        qubo.add_linear(3, -5);
        qubo.add_quadratic(0, 1, -3);
        qubo.add_quadratic(1, 2, 2);
        qubo.add_quadratic(2, 3, -4);
        qubo.add_quadratic(0, 3, 6);

        qubo
    }

    /// After every move the kept deltas equal freshly computed ones.
    #[test]
    fn incremental_deltas_match_the_model() {
        let qubo = model();
        let mut sampler = TabuSearch::new(Pcg64Mcg::seed_from_u64(3), 0, 1);
        let _ = sampler.sample(&qubo, Unlimited).expect("any model");

        for _ in 0..20 {
            let _ = sampler.make_move();

            for i in 0..4 {
                let fresh = sampler.adjacency.flip_delta(i, &sampler.current);
                assert_eq!(sampler.deltas[i], fresh);
            }

            assert_eq!(sampler.energy, qubo.energy(&sampler.current));
        }
    }

    /// The ground state of the small model is found, and an empty model finishes at once.
    #[test]
    fn finds_the_ground_state_and_handles_empty_models() {
        let ground = Exhaustive::new()
            .sample(&model(), Unlimited)
            .expect("four variables")
            .sample
            .energy;
        let mut sampler = TabuSearch::new(Pcg64Mcg::seed_from_u64(5), 50, 1);
        let run = sampler.sample(&model(), Unlimited).expect("any model");
        assert_eq!(run.sample.energy, ground);

        let mut empty = Qubo::<i64>::new(0);
        empty.add_offset(2);
        let run = sampler.sample(&empty, Unlimited).expect("any model");
        assert_eq!((run.sample.energy, run.status), (2, Status::Finished));
    }
}
