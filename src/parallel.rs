//! Several samplers on one model at once, behind the `parallel` feature (needs `std`).
//!
//! Heuristic samplers get better with more independent draws. [`sample_parallel`] runs one
//! [`Sampler::sample`] call per sampler, each on its own scoped thread, and returns all runs
//! in sampler order; [`resume_parallel`] does the same for [`Sampler::resume`]. Each
//! sampler keeps its own workspace, so the threads share nothing but the model, and each run
//! equals the run of the same sampler on one thread.
//!
//! ```mermaid
//! flowchart LR
//!     model[model, shared] --> s1[sampler 1] & s2[sampler 2] & s3[sampler 3]
//!     s1 & s2 & s3 --> runs[runs in sampler order]
//! ```

extern crate std;

use alloc::vec::Vec;
use std::thread;

use search_core::Budget;

use crate::{Run, Sampler};

/// The result of one sampler call: the run, or the sampler's error.
pub type RunResult<'a, M, S> =
    Result<Run<<S as Sampler<M>>::Sample<'a>, <S as Sampler<M>>::Stats>, <S as Sampler<M>>::Error>;

/// Starts a run of each sampler on `model`, one thread per sampler, and waits for all.
///
/// Each sampler gets a clone of `budget`, so a limit applies to each run on its own. A
/// budget closure must be [`Clone`]: a closure that only reads copied values is, while
/// `&mut closure` is not. The runs come back in the order of `samplers`, with borrowed
/// samples; pick the lowest energy to keep the best one. Give stochastic samplers different
/// seeds, or they return the same sample.
///
/// # Panics
///
/// Panics if a sampler panics.
#[expect(
    clippy::needless_pass_by_value,
    reason = "budgets are taken by value across the planning crates; each thread gets a clone"
)]
pub fn sample_parallel<'s, M, S, B>(
    samplers: &'s mut [S],
    model: &M,
    budget: B,
) -> Vec<RunResult<'s, M, S>>
where
    M: Sync,
    S: Sampler<M> + Send,
    B: Budget<S::Stats> + Clone + Send,
    RunResult<'s, M, S>: Send,
{
    thread::scope(|scope| {
        let handles: Vec<_> = samplers
            .iter_mut()
            .map(|sampler| {
                let budget = budget.clone();
                scope.spawn(move || sampler.sample(model, budget))
            })
            .collect();

        join_all(handles)
    })
}

/// Resumes the run of each sampler, one thread per sampler, and waits for all.
///
/// The budget rules and the result order are those of [`sample_parallel`].
///
/// # Panics
///
/// Panics if a sampler panics.
#[expect(
    clippy::needless_pass_by_value,
    reason = "budgets are taken by value across the planning crates; each thread gets a clone"
)]
pub fn resume_parallel<'s, M, S, B>(samplers: &'s mut [S], budget: B) -> Vec<RunResult<'s, M, S>>
where
    S: Sampler<M> + Send,
    B: Budget<S::Stats> + Clone + Send,
    RunResult<'s, M, S>: Send,
{
    thread::scope(|scope| {
        let handles: Vec<_> = samplers
            .iter_mut()
            .map(|sampler| {
                let budget = budget.clone();
                scope.spawn(move || sampler.resume(budget))
            })
            .collect();

        join_all(handles)
    })
}

/// Joins the threads in order and passes a thread's panic on to the caller.
fn join_all<R>(handles: Vec<thread::ScopedJoinHandle<'_, R>>) -> Vec<R> {
    handles
        .into_iter()
        .map(|handle| {
            handle
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
        })
        .collect()
}
