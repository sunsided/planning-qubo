//! Sampler invariants on random models: reported energies are exact, sliced runs equal
//! unsliced ones, and no sampler beats the exhaustive ground state.

use planning_qubo::{
    Exhaustive, ParallelTempering, Qubo, Sampler, Schedule, SimulatedAnnealing, Status, TabuSearch,
    Unlimited,
};
use proptest::prelude::*;
use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;

/// Generates up to 8 variables with sparse pairs and coefficients from −9 to 9.
fn model() -> impl Strategy<Value = Qubo<i64>> {
    (1..=8_usize).prop_flat_map(|n| {
        let linear = prop::collection::vec(-9..=9_i64, n);
        let pairs = prop::collection::vec((0..n, 0..n, -9..=9_i64), 0..=12);

        (linear, pairs, -9..=9_i64).prop_map(move |(linear, pairs, offset)| {
            let mut qubo = Qubo::new(n);
            qubo.add_offset(offset);

            for (i, a) in linear.into_iter().enumerate() {
                qubo.add_linear(i, a);
            }

            for (i, j, b) in pairs {
                qubo.add_quadratic(i, j, b);
            }

            qubo
        })
    })
}

/// Runs `sampler` in slices of `slice` units, measured by `units`, until it finishes.
///
/// Returns the final bits and energy.
fn sliced<S, Stats>(
    sampler: &mut S,
    qubo: &Qubo<i64>,
    slice: u64,
    units: impl Fn(&Stats) -> u64 + Copy,
) -> (Vec<bool>, i64)
where
    S: for<'a> Sampler<Qubo<i64>, Stats = Stats, Sample<'a> = planning_qubo::Sample<'a, i64>>
        + 'static,
    S::Error: core::fmt::Debug,
{
    let budget = move |stats: &Stats| units(stats) >= slice;
    let mut run = sampler.sample(qubo, budget).expect("any model");

    while run.status == Status::OutOfBudget {
        run = sampler.resume(budget).expect("a run exists");
    }

    (run.sample.bits.to_vec(), run.sample.energy)
}

/// Checks a sampler's unsliced run: exact energy, not below the ground state, and equal to
/// the run in slices of `slice` units.
fn check<S, Stats>(
    make: impl Fn() -> S,
    qubo: &Qubo<i64>,
    ground: i64,
    slice: u64,
    units: impl Fn(&Stats) -> u64 + Copy,
) -> Result<(), TestCaseError>
where
    S: for<'a> Sampler<Qubo<i64>, Stats = Stats, Sample<'a> = planning_qubo::Sample<'a, i64>>
        + 'static,
    S::Error: core::fmt::Debug,
{
    let mut whole = make();
    let run = whole.sample(qubo, Unlimited).expect("any model");
    let (bits, energy) = (run.sample.bits.to_vec(), run.sample.energy);

    prop_assert_eq!(energy, qubo.energy(&bits));
    prop_assert!(energy >= ground);
    prop_assert_eq!(sliced(&mut make(), qubo, slice, units), (bits, energy));

    Ok(())
}

proptest! {
    /// Every stochastic sampler keeps the contract on random models and slice sizes.
    #[test]
    fn samplers_keep_their_contract(qubo in model(), seed in any::<u64>(), slice in 1..20_u64) {
        let ground = Exhaustive::new().sample(&qubo, Unlimited).expect("few variables").sample.energy;
        let rng = || Pcg64Mcg::seed_from_u64(seed);

        check(|| SimulatedAnnealing::new(rng(), Schedule::from_model(10)), &qubo, ground, slice, |s| s.flips)?;
        check(|| TabuSearch::new(rng(), 40, 2), &qubo, ground, slice, |s| s.moves)?;
        check(|| ParallelTempering::new(rng(), Schedule::from_model(8), 3), &qubo, ground, slice, |s| s.flips)?;
    }
}
