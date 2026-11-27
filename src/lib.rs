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
//! Available now: [`Qubo`] and [`Ising`] models with exact conversion in both directions,
//! the penalty [`Builder`] with [`penalty`] weights chosen from the model (a safe bound, or
//! tuned against a sampler), the [`assignment`] encoder and decoder, and the [`Exhaustive`],
//! [`SimulatedAnnealing`], [`TabuSearch`] and [`ParallelTempering`] samplers. Samplers reuse their workspace, stop on a
//! [`Budget`] and [`resume`](Sampler::resume) later.
//!
//! ## Pipeline
//!
//! ```mermaid
//! graph LR
//!     problem[Assignment problem] -->|encode| builder[Builder]
//!     builder -->|build| encoded[Problem: QUBO + constraints]
//!     encoded -->|Sampler| sample[Sample]
//!     sample -->|decode| plan[Assignment + violations]
//!     encoded -.->|to_ising| ising[Ising]
//! ```
//!
//! ## Example
//!
//! ```
//! use planning_qubo::{Exhaustive, Sampler, Unlimited, assignment};
//!
//! // Two tasks, two agents, one task per agent. The penalty weight exceeds the sum of all costs.
//! let costs = [vec![4, 1], vec![2, 3]];
//! let encoding = assignment::encode(&costs, &[1, 1], 11)?;
//!
//! let mut sampler = Exhaustive::new();
//! let run = sampler.sample(encoding.problem().qubo(), Unlimited)?;
//! let decoded = encoding.decode(run.sample.bits);
//!
//! assert_eq!(decoded.agents, [Some(1), Some(0)]);
//! assert_eq!(decoded.cost, 3);
//! assert!(decoded.violations.is_empty());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! The crate is `no_std` and needs `alloc`. The `parallel` feature (which needs `std`) adds
//! `sample_parallel` and `resume_parallel`: several samplers on one model at once, one
//! scoped thread each.
//!
//! Repository: <https://github.com/sunsided/planning-qubo>

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod assignment;
mod builder;
mod coefficient;
mod ising;
#[cfg(feature = "parallel")]
mod parallel;
pub mod penalty;
mod qubo;
mod sampler;

pub use builder::{Builder, Constraint, ConstraintKind, Encoded, Problem, Var};
pub use coefficient::Coefficient;
pub use ising::{InexactScale, Ising};
#[cfg(feature = "parallel")]
pub use parallel::{RunResult, resume_parallel, sample_parallel};
pub use qubo::Qubo;
pub use sampler::{
    AnnealingStats, Exhaustive, ExhaustiveStats, InvalidSchedule, ParallelTempering, Run, Sample,
    SampleError, Sampler, Schedule, SimulatedAnnealing, TabuSearch, TabuStats, TemperingStats,
};
pub use search_core::{Budget, Status, Unlimited};

/// Doctests for the code examples in README.md (not part of the API).
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
mod readme_doctests {}
