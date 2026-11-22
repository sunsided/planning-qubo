//! # Planning QUBO
//!
//! Planning, scheduling and constraint problems as QUBO and Ising models: encoders, decoders, penalty tuning, classical samplers and export for quantum annealers and QAOA.
//!
//! ## Scope
//!
//! - QUBO and Ising models with named binary variables and a penalty builder (exactly-one, at-most-one, linear constraints with slack bits, objective terms)
//! - Encoders for constraint satisfaction, time-indexed scheduling, assignment and allocation, time-expanded multi-agent path finding and routing
//! - Decoders that map samples back to plans and report violated constraints, plus penalty-weight tuning
//! - A sampler interface with classical baselines (exhaustive, simulated annealing, tabu search, parallel tempering)
//! - Export to annealer formats and adapters for quantum frameworks
//!
//! Repository: <https://github.com/sunsided/planning-qubo>

#![no_std]
#![forbid(unsafe_code)]

/// A source of low-energy samples for a QUBO model, classical or quantum.
pub trait Sampler<Model> {
    /// A sample: one value per binary variable of the model.
    type Sample;

    /// Draws a sample, aiming for low energy under `model`.
    fn sample(&mut self, model: &Model) -> Self::Sample;
}

/// Doctests for the code examples in README.md (not part of the API).
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
mod readme_doctests {}

/// Unit tests for the `Sampler` trait, through minimal samplers over a two-variable QUBO.
#[cfg(test)]
mod tests {
    use super::Sampler;

    /// An upper-triangular QUBO over two binary variables.
    struct Qubo {
        /// Coefficients: `q[i][i]` is linear, `q[0][1]` the pair term; `q[1][0]` is unused.
        q: [[i64; 2]; 2],
    }

    impl Qubo {
        /// The energy of the assignment `x`.
        fn energy(&self, x: [bool; 2]) -> i64 {
            let linear = i64::from(x[0]) * self.q[0][0] + i64::from(x[1]) * self.q[1][1];
            let pair = i64::from(x[0] && x[1]) * self.q[0][1];

            linear + pair
        }
    }

    /// Tries all four assignments and returns one with the lowest energy.
    struct Exhaustive;

    impl Sampler<Qubo> for Exhaustive {
        type Sample = [bool; 2];

        fn sample(&mut self, model: &Qubo) -> [bool; 2] {
            [[false, false], [true, false], [false, true], [true, true]]
                .into_iter()
                .min_by_key(|&x| model.energy(x))
                .expect("there are four assignments")
        }
    }

    /// A stand-in for a stochastic sampler: returns its scripted samples in turn.
    struct Scripted {
        /// The samples to hand out, in order, wrapping around.
        samples: [[bool; 2]; 3],
        /// Index of the next sample.
        next: usize,
    }

    impl Sampler<Qubo> for Scripted {
        type Sample = [bool; 2];

        fn sample(&mut self, _model: &Qubo) -> [bool; 2] {
            let sample = self.samples[self.next % self.samples.len()];
            self.next += 1;

            sample
        }
    }

    /// Draws `draws` samples and keeps the one with the lowest energy, the way a caller
    /// restarts a heuristic sampler.
    fn best_of<S: Sampler<Qubo, Sample = [bool; 2]>>(
        sampler: &mut S,
        model: &Qubo,
        draws: usize,
    ) -> Option<[bool; 2]> {
        (0..draws)
            .map(|_| sampler.sample(model))
            .min_by_key(|&x| model.energy(x))
    }

    /// The model rewards each variable on its own but penalizes setting both.
    const MODEL: Qubo = Qubo {
        q: [[-2, 5], [0, -3]],
    };

    /// The exhaustive sampler finds the ground state.
    #[test]
    fn exhaustive_finds_the_ground_state() {
        assert_eq!(Exhaustive.sample(&MODEL), [false, true]);
    }

    /// Repeated draws from a stateful sampler keep the lowest-energy sample seen.
    #[test]
    fn best_of_keeps_the_lowest_energy_draw() {
        let mut sampler = Scripted {
            samples: [[true, true], [true, false], [false, true]],
            next: 0,
        };

        assert_eq!(best_of(&mut sampler, &MODEL, 2), Some([true, false]));
        assert_eq!(best_of(&mut sampler, &MODEL, 3), Some([false, true]));
        assert_eq!(best_of(&mut sampler, &MODEL, 0), None);
    }
}
