//! The sampler interface and the classical samplers.
//!
//! A sampler owns its workspace: the model copy, the current state and the best state. A
//! call on a model of the same size as the previous call allocates nothing, and the result
//! borrows the best state from the workspace. Each thread uses its own sampler; models are
//! `Send + Sync` and shared.
//!
//! A run stops when it finishes or when its [`Budget`] is exhausted. A stopped run continues
//! with [`Sampler::resume`], so the work can spread over several time slices.
//!
//! ```mermaid
//! stateDiagram-v2
//!     [*] --> Running: sample
//!     Running --> OutOfBudget: budget exhausted
//!     OutOfBudget --> Running: resume
//!     OutOfBudget --> Running: sample (restart)
//!     Running --> Finished: all work done
//!     Finished --> Running: sample
//! ```

mod adjacency;
mod annealing;
mod exhaustive;

use core::fmt;

pub use annealing::{AnnealingStats, InvalidSchedule, Schedule, SimulatedAnnealing};
pub use exhaustive::{Exhaustive, ExhaustiveStats};
use search_core::{Budget, Status};

/// A source of low-energy samples for a model, classical or quantum.
///
/// [`Sampler::sample`] starts a run on a model; [`Sampler::resume`] continues a run that its
/// budget stopped. Lower energy is better. A heuristic sampler need not return the optimum;
/// callers draw again for more chances. The sampler checks the budget before each unit of
/// work, so a limit of `n` units stops after exactly `n` units. A model without variables
/// has no units of work, so its runs finish without a budget check.
///
/// The budget is taken by value, as in the other planning crates. Pass `&mut budget` for a
/// closure that must keep its state across calls; [`Unlimited`](search_core::Unlimited) runs
/// to the end.
pub trait Sampler<Model> {
    /// The work counters of one call, which the budget sees.
    type Stats;

    /// A sample: one value per binary variable of the model, usually with its energy. It
    /// borrows the sampler's workspace.
    type Sample<'a>
    where
        Self: 'a;

    /// The error when the sampler cannot handle a model or a call.
    type Error;

    /// Starts a new run on `model`, and runs until it finishes or `budget` is exhausted.
    ///
    /// The run replaces any earlier run. The returned sample is the best state so far.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] if the sampler cannot handle the model, for example because
    /// it is too large.
    fn sample<B: Budget<Self::Stats>>(
        &mut self,
        model: &Model,
        budget: B,
    ) -> Result<Run<Self::Sample<'_>, Self::Stats>, Self::Error>;

    /// Continues the current run until it finishes or `budget` is exhausted.
    ///
    /// The sampler keeps its own copy of the model, so the call needs no model. Resuming a
    /// finished run does no work and returns the same sample.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] if no run exists yet.
    fn resume<B: Budget<Self::Stats>>(
        &mut self,
        budget: B,
    ) -> Result<Run<Self::Sample<'_>, Self::Stats>, Self::Error>;
}

/// The result of one [`Sampler::sample`] or [`Sampler::resume`] call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Run<S, Stats> {
    /// The best state of the run so far.
    pub sample: S,
    /// Whether the run finished or the budget stopped it. With [`Status::Finished`] from
    /// [`Exhaustive`], the sample is a ground state; after [`Status::OutOfBudget`],
    /// [`Sampler::resume`] continues the run.
    pub status: Status,
    /// The work that this call did. Each call starts from zero.
    pub stats: Stats,
}

/// An assignment of a QUBO model's variables and its energy, borrowed from a sampler.
///
/// Use `bits.to_vec()` to keep the assignment beyond the next sampler call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample<'a, T> {
    /// The value of each variable, by index.
    pub bits: &'a [bool],
    /// The energy of `bits` under the model. Exact for [`i64`].
    pub energy: T,
}

/// The error when a sampler cannot handle a model or a call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SampleError {
    /// The model has more variables than the sampler accepts, for example
    /// [`Exhaustive::MAX_VARIABLES`].
    TooManyVariables {
        /// The number of variables of the model.
        variables: usize,
        /// The largest number of variables that the sampler accepts.
        max: usize,
    },
    /// [`Sampler::resume`] was called before any [`Sampler::sample`] call.
    NothingToResume,
}

impl fmt::Display for SampleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyVariables { variables, max } => write!(
                f,
                "the model has {variables} variables; the sampler accepts at most {max}"
            ),
            Self::NothingToResume => f.write_str("there is no run to resume"),
        }
    }
}

impl core::error::Error for SampleError {}
