//! Metropolis acceptance and inverse-temperature helpers, shared by the stochastic samplers.

use rand_core::Rng;

use super::adjacency::Adjacency;
use crate::Coefficient;

/// The Metropolis rule: accept a decrease, and an increase with probability `exp(−β·Δ)`.
///
/// Draws one random word only for an increase, so a decrease uses no randomness.
pub(super) fn accepts<R: Rng>(rng: &mut R, delta: f64, beta: f64) -> bool {
    delta <= 0.0 || unit_interval(rng.next_u64()) < libm::exp(-beta * delta)
}

/// Accepts with probability `min(1, exp(exponent))`, as the replica exchange rule does.
///
/// Draws one random word only when the exponent is negative.
pub(super) fn accepts_exponent<R: Rng>(rng: &mut R, exponent: f64) -> bool {
    exponent >= 0.0 || unit_interval(rng.next_u64()) < libm::exp(exponent)
}

/// The hot and cold inverse temperatures derived from the model.
///
/// - hot: the largest possible flip delta is accepted with probability 1/2;
/// - cold: the smallest nonzero coefficient magnitude is accepted with probability 1/100.
///
/// For a model without nonzero coefficients both values are 1; any schedule is then exact.
pub(super) fn beta_range<T: Coefficient>(adjacency: &Adjacency<T>) -> (f64, f64) {
    let largest_delta = (0..adjacency.len())
        .map(|i| largest_flip_delta(adjacency, i))
        .fold(0.0, f64::max);

    let smallest = smallest_nonzero_magnitude(adjacency);

    if largest_delta <= 0.0 || !smallest.is_finite() {
        return (1.0, 1.0);
    }

    (
        core::f64::consts::LN_2 / largest_delta,
        libm::log(100.0) / smallest,
    )
}

/// An upper bound on the flip delta magnitude of variable `i`: `|aᵢ| + Σⱼ |bᵢⱼ|`.
fn largest_flip_delta<T: Coefficient>(adjacency: &Adjacency<T>, i: usize) -> f64 {
    adjacency
        .row(i)
        .iter()
        .fold(adjacency.linear(i).to_f64().abs(), |sum, &(_, b)| {
            sum + b.to_f64().abs()
        })
}

/// The smallest nonzero coefficient magnitude, or infinity if all coefficients are zero.
fn smallest_nonzero_magnitude<T: Coefficient>(adjacency: &Adjacency<T>) -> f64 {
    let linear = (0..adjacency.len()).map(|i| adjacency.linear(i));
    let pairs = (0..adjacency.len()).flat_map(|i| adjacency.row(i).iter().map(|&(_, b)| b));

    linear
        .chain(pairs)
        .map(|v| v.to_f64().abs())
        .filter(|&v| v > 0.0)
        .fold(f64::INFINITY, f64::min)
}

/// Step `step` of `steps` from `hot` to `cold` geometrically. A single step is `cold`.
pub(super) fn geometric(hot: f64, cold: f64, step: u32, steps: u32) -> f64 {
    if steps <= 1 {
        return cold;
    }

    let progress = f64::from(step) / f64::from(steps - 1);

    hot * libm::pow(cold / hot, progress)
}

/// Fills `bits` with `n` uniform random bits, reusing its capacity.
pub(super) fn random_bits<R: Rng>(rng: &mut R, bits: &mut alloc::vec::Vec<bool>, n: usize) {
    bits.clear();

    let mut word = 0_u64;

    for i in 0..n {
        if i % 64 == 0 {
            word = rng.next_u64();
        }

        bits.push(word >> (i % 64) & 1 == 1);
    }
}

/// Maps 64 random bits to a uniform value in `[0, 1)` with 53 bits of precision.
fn unit_interval(bits: u64) -> f64 {
    /// 2⁻⁵³, the spacing of 53-bit values in the unit interval.
    const SCALE: f64 = 1.0 / 9_007_199_254_740_992.0;

    #[expect(
        clippy::cast_precision_loss,
        reason = "the value is below 2^53 after the shift, so the conversion is exact"
    )]
    let mantissa = (bits >> 11) as f64;

    mantissa * SCALE
}

/// Unit tests for the schedule and acceptance helpers.
#[cfg(test)]
mod tests {
    use super::{geometric, unit_interval};

    /// The schedule starts hot and ends cold.
    #[test]
    fn schedule_ends_at_both_values() {
        assert!((geometric(0.5, 8.0, 0, 5) - 0.5).abs() < 1e-12);
        assert!((geometric(0.5, 8.0, 4, 5) - 8.0).abs() < 1e-12);
        assert!((geometric(0.5, 8.0, 0, 1) - 8.0).abs() < 1e-12);
    }

    /// Random bits map into the half-open unit interval.
    #[test]
    fn unit_interval_is_half_open() {
        assert!(unit_interval(0).abs() < f64::EPSILON);
        assert!(unit_interval(u64::MAX) < 1.0);
    }
}
