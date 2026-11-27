//! Penalty weights chosen from the model instead of by the caller.
//!
//! Every encoder takes a penalty weight `P`, and each violated constraint costs at least
//! `P` when all constraint coefficients are integers (see [`Builder`](crate::Builder)).
//! Two ways to choose `P`:
//!
//! - [`safe`] computes a bound from the objective alone. With `P` above the objective's
//!   spread, every ground state is feasible if a feasible assignment exists. The bound is
//!   often large, which hurts samplers and annealers with a limited coefficient range.
//! - [`tune`] starts from the largest objective coefficient and doubles `P` until a sampler
//!   returns a feasible sample, up to the safe bound. The result is a smaller weight that
//!   works for that sampler and budget, but it carries no optimality guarantee.
//!
//! Both take the encoder as a closure from `P` to an encoding, so they work with every
//! encoder of the crate: an encoding at `P = 0` is the objective alone.
//!
//! ```mermaid
//! flowchart LR
//!     zero["encode(0)"] -->|objective| spread[spread bound]
//!     spread --> safe["safe: P = spread + 1"]
//!     start["P = largest objective coefficient"] --> sample{sample feasible?}
//!     sample -->|yes| done[keep P]
//!     sample -->|"no, P < bound"| double["P = min(2P, bound)"] --> sample
//!     sample -->|"no, P = bound"| give[return infeasible]
//! ```

use alloc::vec::Vec;
use core::fmt;

use search_core::Budget;

use crate::{Coefficient, Encoded, Qubo, Sample, Sampler};

/// An upper bound on `max E(x) − min E(x)` over all assignments: `Σᵢ |aᵢ| + Σ_{i<j} |bᵢⱼ|`.
///
/// The offset cancels out, so it is not part of the bound.
///
/// # Panics
///
/// For [`i64`], panics if the sum overflows.
#[must_use]
pub fn objective_spread<T: Coefficient>(qubo: &Qubo<T>) -> T {
    let linear = (0..qubo.num_variables())
        .map(|i| magnitude(qubo.linear(i)))
        .fold(T::ZERO, T::plus);

    qubo.quadratic_terms()
        .map(|(_, _, b)| magnitude(b))
        .fold(linear, T::plus)
}

/// The largest coefficient magnitude of `qubo`, offset excluded. Zero for an empty model.
#[must_use]
pub fn largest_coefficient<T: Coefficient>(qubo: &Qubo<T>) -> T {
    let linear = (0..qubo.num_variables()).map(|i| qubo.linear(i));
    let pairs = qubo.quadratic_terms().map(|(_, _, b)| b);

    linear
        .chain(pairs)
        .map(magnitude)
        .fold(T::ZERO, |high, v| if v > high { v } else { high })
}

/// The safe penalty weight of an encoder and the encoding at that weight.
///
/// Encodes at `P = 0` to read the objective, and then at `P = objective_spread + 1`. Each
/// infeasible assignment then has a higher energy than each feasible one, so every ground
/// state is feasible if a feasible assignment exists, and it is an optimum of the objective.
///
/// The guarantee needs integer constraint coefficients, as all encoders of the crate use:
/// only then does each violation cost at least `P`.
///
/// # Errors
///
/// Returns the encoder's error.
///
/// # Panics
///
/// For [`i64`], panics if the bound overflows.
pub fn safe<T, E, Er>(mut encode: impl FnMut(T) -> Result<E, Er>) -> Result<(T, E), Er>
where
    T: Coefficient,
    E: Encoded<T>,
{
    let bound = safe_bound(encode(T::ZERO)?.problem().qubo());

    Ok((bound, encode(bound)?))
}

/// The penalty weight that [`tune`] chose, with its encoding and the sample it found.
#[derive(Debug, Clone, PartialEq)]
pub struct Tuned<T, E> {
    /// The penalty weight of the last round.
    pub penalty: T,
    /// The encoding at `penalty`.
    pub encoding: E,
    /// The sample of the last round, all QUBO variables included.
    pub bits: Vec<bool>,
    /// Whether `bits` violates no constraint. `false` only when the safe bound was reached:
    /// the instance may be infeasible, or the sampler too weak for its budget.
    pub feasible: bool,
    /// The number of rounds (encodings and sampler runs).
    pub rounds: u32,
}

/// The error of [`tune`]: from the encoder or from the sampler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TuneError<Er, Se> {
    /// The encoder failed.
    Encode(Er),
    /// The sampler failed.
    Sample(Se),
}

impl<Er: fmt::Display, Se: fmt::Display> fmt::Display for TuneError<Er, Se> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Encode(e) => write!(f, "encoding failed: {e}"),
            Self::Sample(e) => write!(f, "sampling failed: {e}"),
        }
    }
}

impl<Er: fmt::Debug + fmt::Display, Se: fmt::Debug + fmt::Display> core::error::Error
    for TuneError<Er, Se>
{
}

/// Raises the penalty weight until `sampler` returns a feasible sample.
///
/// The first weight is the largest objective coefficient (at least 1). Each round encodes at
/// the weight, runs `sampler` with a clone of `budget`, and checks the sample against the
/// constraints. An infeasible sample doubles the weight, up to the [`safe`] bound; a round
/// at the bound ends the search either way.
///
/// The result is the smallest weight of the doubling sequence at which the sampler found a
/// feasible sample. Lower weights keep the objective visible next to the penalties, which
/// helps heuristic samplers and annealers with a limited coefficient range, but a feasible
/// sample need not be optimal.
///
/// # Errors
///
/// Returns [`TuneError`] if the encoder or the sampler fails.
///
/// # Panics
///
/// For [`i64`], panics if the bound overflows.
#[expect(
    clippy::needless_pass_by_value,
    reason = "budgets are taken by value across the planning crates; each round gets a clone"
)]
pub fn tune<T, E, Er, S, B>(
    mut encode: impl FnMut(T) -> Result<E, Er>,
    sampler: &mut S,
    budget: B,
) -> Result<Tuned<T, E>, TuneError<Er, S::Error>>
where
    T: Coefficient,
    E: Encoded<T>,
    S: for<'a> Sampler<Qubo<T>, Sample<'a> = Sample<'a, T>>,
    B: Budget<S::Stats> + Clone,
{
    let objective = encode(T::ZERO).map_err(TuneError::Encode)?;
    let bound = safe_bound(objective.problem().qubo());
    let mut penalty = first_weight(objective.problem().qubo(), bound);
    let mut rounds = 0;

    loop {
        let encoding = encode(penalty).map_err(TuneError::Encode)?;
        let run = sampler
            .sample(encoding.problem().qubo(), budget.clone())
            .map_err(TuneError::Sample)?;
        let feasible = encoding
            .problem()
            .violations(run.sample.bits)
            .next()
            .is_none();
        let bits = run.sample.bits.to_vec();
        rounds += 1;

        if feasible || penalty >= bound {
            return Ok(Tuned {
                penalty,
                encoding,
                bits,
                feasible,
                rounds,
            });
        }

        penalty = doubled(penalty, bound);
    }
}

/// `objective_spread + 1`, the weight above which every ground state is feasible.
fn safe_bound<T: Coefficient>(objective: &Qubo<T>) -> T {
    objective_spread(objective).plus(T::ONE)
}

/// The first weight of [`tune`]: the largest objective coefficient, at least 1, at most `bound`.
fn first_weight<T: Coefficient>(objective: &Qubo<T>, bound: T) -> T {
    let largest = largest_coefficient(objective);
    let start = if largest < T::ONE { T::ONE } else { largest };

    lower(start, bound)
}

/// `min(2 · penalty, bound)` for `0 < penalty < bound`, without overflowing on the way.
fn doubled<T: Coefficient>(penalty: T, bound: T) -> T {
    if bound.minus(penalty) < penalty {
        bound
    } else {
        penalty.plus(penalty)
    }
}

/// The smaller of `a` and `b`.
fn lower<T: Coefficient>(a: T, b: T) -> T {
    if b < a { b } else { a }
}

/// The magnitude `|v|`.
fn magnitude<T: Coefficient>(v: T) -> T {
    if v < T::ZERO { T::ZERO.minus(v) } else { v }
}

/// Unit tests for the bounds and the doubling.
#[cfg(test)]
mod tests {
    use super::{largest_coefficient, objective_spread, tune};
    use crate::{Builder, Exhaustive, Problem, Qubo, Unlimited};

    /// A bound above `i64::MAX / 2` caps the doubled weight instead of overflowing.
    #[test]
    fn doubling_near_the_limit_does_not_overflow() {
        let cost = -(i64::MAX / 2 + 1);
        let encode = |p: i64| -> Result<Problem<i64>, ()> {
            let mut builder = Builder::new();
            let x = builder.variable("x");
            builder.add_linear(x, cost);
            builder.exactly_one("never", &[], p);
            Ok(builder.build())
        };

        let tuned = tune(encode, &mut Exhaustive::new(), Unlimited).expect("one variable");

        assert!(!tuned.feasible);
        assert_eq!(tuned.penalty, -cost + 1);
        assert_eq!(tuned.rounds, 2);
    }

    /// The spread sums all magnitudes except the offset, and bounds the true spread.
    #[test]
    fn spread_bounds_the_energy_range() {
        let mut qubo = Qubo::<i64>::new(3);
        qubo.add_offset(100);
        qubo.add_linear(0, -2);
        qubo.add_linear(1, 3);
        qubo.add_quadratic(0, 2, -4);
        qubo.add_quadratic(1, 2, 1);

        assert_eq!(objective_spread(&qubo), 10);
        assert_eq!(largest_coefficient(&qubo), 4);

        let energies: alloc::vec::Vec<i64> = (0..8_u8)
            .map(|b| qubo.energy(&[b & 1 != 0, b & 2 != 0, b & 4 != 0]))
            .collect();
        let range = energies.iter().max().unwrap() - energies.iter().min().unwrap();
        assert!(range <= 10);
    }
}
