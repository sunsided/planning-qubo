//! The sampler interface and the classical samplers.

mod adjacency;
mod annealing;
mod exhaustive;

use alloc::vec::Vec;

pub use annealing::SimulatedAnnealing;
pub use exhaustive::{Exhaustive, TooManyVariables};

/// A source of low-energy samples for a model, classical or quantum.
///
/// A call draws one assignment of the model's binary variables. Lower energy is better. A
/// heuristic sampler need not return the optimum; callers draw again for more chances. A
/// sampler keeps its reusable buffers in `self`, so repeated calls allocate less.
pub trait Sampler<Model> {
    /// A sample: one value per binary variable of the model, usually with its energy.
    type Sample;

    /// The error when the sampler cannot handle the model.
    type Error;

    /// Draws a sample, aiming for low energy under `model`.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] if the sampler cannot handle the model, for example because
    /// it is too large.
    fn sample(&mut self, model: &Model) -> Result<Self::Sample, Self::Error>;
}

/// An assignment of a QUBO model's variables and its energy.
#[derive(Debug, Clone, PartialEq)]
pub struct Sample<T> {
    /// The value of each variable, by index.
    pub bits: Vec<bool>,
    /// The energy of `bits` under the model, as [`Qubo::energy`](crate::Qubo::energy) computes it.
    pub energy: T,
}
