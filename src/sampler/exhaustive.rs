//! The exhaustive sampler: the exact baseline for small models.

use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use super::adjacency::Adjacency;
use crate::{Coefficient, Qubo, Sample, Sampler};

/// Tries every assignment and returns one with the lowest energy.
///
/// The enumeration is in Gray code order, so each step flips one bit and costs one flip
/// delta. Among equal energies, the first one in that order wins, so the result is
/// deterministic. The work is `2ⁿ` flips, so models are limited to
/// [`Exhaustive::MAX_VARIABLES`] variables.
#[derive(Debug, Clone, Copy, Default)]
pub struct Exhaustive;

impl Exhaustive {
    /// The largest number of variables that the sampler accepts.
    pub const MAX_VARIABLES: usize = 30;
}

/// The error when a model has more than [`Exhaustive::MAX_VARIABLES`] variables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TooManyVariables {
    /// The number of variables of the model.
    pub variables: usize,
}

impl fmt::Display for TooManyVariables {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the model has {} variables; exhaustive sampling accepts at most {}",
            self.variables,
            Exhaustive::MAX_VARIABLES
        )
    }
}

impl core::error::Error for TooManyVariables {}

impl<T: Coefficient> Sampler<Qubo<T>> for Exhaustive {
    type Sample = Sample<T>;
    type Error = TooManyVariables;

    fn sample(&mut self, model: &Qubo<T>) -> Result<Sample<T>, TooManyVariables> {
        let variables = model.num_variables();

        if variables > Self::MAX_VARIABLES {
            return Err(TooManyVariables { variables });
        }

        let mut adjacency = Adjacency::default();
        adjacency.rebuild(model, |v| v);

        let bits = gray_code_minimum(&adjacency, model.offset());
        let energy = model.energy(&bits);

        Ok(Sample { bits, energy })
    }
}

/// Walks all assignments in Gray code order and returns the first one with the lowest energy.
///
/// Needs at most [`Exhaustive::MAX_VARIABLES`] variables, so the step count fits a `u64`.
fn gray_code_minimum<T: Coefficient>(adjacency: &Adjacency<T>, offset: T) -> Vec<bool> {
    let n = adjacency.len();
    let mut x = vec![false; n];
    let mut energy = offset;
    let mut best_energy = offset;
    let mut best_step = 0_u64;

    for step in 1..1_u64 << n {
        let i = bit_to_flip(step);
        energy = energy.plus(adjacency.flip_delta(i, &x));
        x[i] = !x[i];

        if energy < best_energy {
            best_energy = energy;
            best_step = step;
        }
    }

    gray_code_bits(best_step, n)
}

/// The bit that Gray code step `step` flips: the index of the lowest set bit of `step`.
fn bit_to_flip(step: u64) -> usize {
    usize::try_from(step.trailing_zeros()).expect("a u64 has at most 64 trailing zeros")
}

/// The assignment after `step` Gray code steps: the bits of `step ^ (step >> 1)`.
fn gray_code_bits(step: u64, n: usize) -> Vec<bool> {
    let code = step ^ (step >> 1);

    (0..n).map(|i| code >> i & 1 == 1).collect()
}

/// Unit tests for the exhaustive sampler.
#[cfg(test)]
mod tests {
    use super::{Exhaustive, TooManyVariables};
    use crate::{Qubo, Sampler};

    /// The sampler finds the ground state and reports its energy.
    #[test]
    fn finds_the_ground_state() {
        let mut qubo = Qubo::<i64>::new(2);
        qubo.add_linear(0, -2);
        qubo.add_linear(1, -3);
        qubo.add_quadratic(0, 1, 5);

        let sample = Exhaustive.sample(&qubo).expect("two variables are allowed");

        assert_eq!(sample.bits, [false, true]);
        assert_eq!(sample.energy, -3);
    }

    /// An empty model has one assignment with the offset as its energy.
    #[test]
    fn empty_model_returns_the_offset() {
        let mut qubo = Qubo::<f64>::new(0);
        qubo.add_offset(1.5);

        let sample = Exhaustive
            .sample(&qubo)
            .expect("zero variables are allowed");

        assert_eq!(sample.bits, [false; 0]);
        assert!((sample.energy - 1.5).abs() < f64::EPSILON);
    }

    /// Too many variables is an error, not a long wait.
    #[test]
    fn rejects_large_models() {
        let qubo = Qubo::<i64>::new(Exhaustive::MAX_VARIABLES + 1);

        assert_eq!(
            Exhaustive.sample(&qubo),
            Err(TooManyVariables {
                variables: Exhaustive::MAX_VARIABLES + 1
            })
        );
    }
}
