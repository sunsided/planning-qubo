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
