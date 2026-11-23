//! The simulated annealing sampler.

use alloc::vec::Vec;
use core::convert::Infallible;

use rand_core::Rng;

use super::adjacency::Adjacency;
use crate::{Coefficient, Qubo, Sample, Sampler};

/// Simulated annealing with single-bit Metropolis flips and a geometric schedule.
///
/// Each call starts from a uniform random assignment and runs `sweeps` sweeps. A sweep
/// visits each variable once, in index order, and flips it with the Metropolis rule. The
/// inverse temperature `β` grows geometrically from a hot to a cold value that the sampler
/// derives from the model:
///
/// - hot: the largest possible flip delta is accepted with probability 1/2;
/// - cold: the smallest nonzero coefficient magnitude is accepted with probability 1/100.
///
/// The sampler returns the lowest-energy assignment that it visits, not the last one. The
/// dynamics use [`f64`] deltas; the returned energy is exact, from
/// [`Qubo::energy`].
///
/// The RNG is the only randomness, so a seeded RNG gives reproducible samples. The sampler
/// keeps its buffers between calls.
#[derive(Debug, Clone)]
pub struct SimulatedAnnealing<R> {
    /// The random source for the initial state and the acceptance tests.
    rng: R,
    /// The number of sweeps per sample. At least 1.
    sweeps: u32,
    /// The model coefficients as `f64`, reused between calls.
    adjacency: Adjacency<f64>,
    /// The current assignment, reused between calls.
    current: Vec<bool>,
}

impl<R: Rng> SimulatedAnnealing<R> {
    /// Creates a sampler that draws from `rng` and runs `sweeps` sweeps per sample.
    ///
    /// A `sweeps` value of 0 counts as 1.
    #[must_use]
    pub fn new(rng: R, sweeps: u32) -> Self {
        Self {
            rng,
            sweeps: sweeps.max(1),
            adjacency: Adjacency::default(),
            current: Vec::new(),
        }
    }

    /// Runs the sweeps and returns the lowest-energy assignment that it visits.
    fn anneal(&mut self) -> Vec<bool> {
        self.randomize_current();

        let (hot, cold) = beta_range(&self.adjacency);
        let mut best = self.current.clone();
        let mut energy = 0.0;
        let mut best_energy = 0.0;

        for sweep in 0..self.sweeps {
            let beta = geometric(hot, cold, sweep, self.sweeps);
            energy = self.sweep(beta, energy, &mut best, &mut best_energy);
        }

        best
    }

    /// Sets the current assignment to uniform random bits.
    fn randomize_current(&mut self) {
        let n = self.adjacency.len();
        self.current.clear();

        let mut word = 0_u64;

        for i in 0..n {
            if i % 64 == 0 {
                word = self.rng.next_u64();
            }

            self.current.push(word >> (i % 64) & 1 == 1);
        }
    }

    /// Runs one Metropolis sweep at `beta` from the energy `start`, and returns the end energy.
    ///
    /// Energies are relative to the initial assignment. When the energy goes below
    /// `best_energy`, the sweep copies the current assignment to `best`.
    fn sweep(&mut self, beta: f64, start: f64, best: &mut Vec<bool>, best_energy: &mut f64) -> f64 {
        let mut energy = start;

        for i in 0..self.current.len() {
            let delta = self.adjacency.flip_delta(i, &self.current);

            if !self.accepts(delta, beta) {
                continue;
            }

            self.current[i] = !self.current[i];
            energy += delta;

            if energy < *best_energy {
                *best_energy = energy;
                best.clone_from(&self.current);
            }
        }

        energy
    }

    /// The Metropolis rule: accept a decrease, and an increase with probability `exp(−β·Δ)`.
    fn accepts(&mut self, delta: f64, beta: f64) -> bool {
        delta <= 0.0 || unit_interval(self.rng.next_u64()) < libm::exp(-beta * delta)
    }
}

impl<R: Rng, T: Coefficient> Sampler<Qubo<T>> for SimulatedAnnealing<R> {
    type Sample = Sample<T>;
    type Error = Infallible;

    fn sample(&mut self, model: &Qubo<T>) -> Result<Sample<T>, Infallible> {
        self.adjacency.rebuild(model, Coefficient::to_f64);

        let bits = self.anneal();
        let energy = model.energy(&bits);

        Ok(Sample { bits, energy })
    }
}

/// The hot and cold inverse temperatures for `adjacency`.
///
/// For a model without nonzero coefficients both values are 1; any schedule is then exact.
fn beta_range(adjacency: &Adjacency<f64>) -> (f64, f64) {
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
fn largest_flip_delta(adjacency: &Adjacency<f64>, i: usize) -> f64 {
    adjacency
        .row(i)
        .iter()
        .fold(adjacency.linear(i).abs(), |sum, &(_, b)| sum + b.abs())
}

/// The smallest nonzero coefficient magnitude, or infinity if all coefficients are zero.
fn smallest_nonzero_magnitude(adjacency: &Adjacency<f64>) -> f64 {
    let linear = (0..adjacency.len()).map(|i| adjacency.linear(i));
    let pairs = (0..adjacency.len()).flat_map(|i| adjacency.row(i).iter().map(|&(_, b)| b));

    linear
        .chain(pairs)
        .map(f64::abs)
        .filter(|&v| v > 0.0)
        .fold(f64::INFINITY, f64::min)
}

/// The inverse temperature of sweep `sweep` of `sweeps`, from `hot` to `cold` geometrically.
fn geometric(hot: f64, cold: f64, sweep: u32, sweeps: u32) -> f64 {
    if sweeps <= 1 {
        return cold;
    }

    let progress = f64::from(sweep) / f64::from(sweeps - 1);

    hot * libm::pow(cold / hot, progress)
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

/// Unit tests for the schedule helpers.
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
