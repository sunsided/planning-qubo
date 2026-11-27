//! Warm sampler calls allocate nothing.
//!
//! `allocation_counter` counts the allocations of the current thread only, so tests that
//! run in parallel do not disturb the count.

use planning_qubo::{
    AnnealingStats, Exhaustive, ExhaustiveStats, ParallelTempering, Qubo, Sampler, Schedule,
    SimulatedAnnealing, TabuSearch, TabuStats, TemperingStats, Unlimited,
};
use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;

/// A model with 10 variables and a chain of pairs; `shift` changes the coefficients only.
fn model(shift: i64) -> Qubo<i64> {
    let mut qubo = Qubo::new(10);
    qubo.add_offset(shift);

    for i in 0..10 {
        qubo.add_linear(i, shift - 3);
    }

    for i in 0..9 {
        qubo.add_quadratic(i, i + 1, 5 - shift);
    }

    qubo
}

/// Warm exhaustive calls, on the same model and on a model of the same shape, allocate nothing.
#[test]
fn warm_exhaustive_calls_do_not_allocate() {
    let (first, second) = (model(0), model(1));
    let mut sampler = Exhaustive::new();
    let mut slice = |stats: &ExhaustiveStats| stats.evaluated >= 100;
    let cold = allocation_counter::measure(|| {
        let _ = sampler.sample(&first, Unlimited).expect("few variables");
    });

    assert!(cold.count_total > 0, "the counter must see the cold call");

    let info = allocation_counter::measure(|| {
        let energy = sampler
            .sample(&first, Unlimited)
            .expect("few variables")
            .sample
            .energy;
        assert_eq!(energy, -15);
        let _ = sampler.sample(&second, &mut slice).expect("few variables");
        let _ = sampler.resume(Unlimited).expect("a run exists");
    });

    assert_eq!(info.count_total, 0);
}

/// Warm annealing calls, on the same model and on a model of the same shape, allocate nothing.
#[test]
fn warm_annealing_calls_do_not_allocate() {
    let (first, second) = (model(0), model(1));
    let rng = Pcg64Mcg::seed_from_u64(1);
    let mut sampler = SimulatedAnnealing::new(rng, Schedule::from_model(20));
    let mut slice = |stats: &AnnealingStats| stats.flips >= 37;
    let cold = allocation_counter::measure(|| {
        let _ = sampler
            .sample(&first, Unlimited)
            .expect("annealing accepts any model");
    });

    assert!(cold.count_total > 0, "the counter must see the cold call");

    let info = allocation_counter::measure(|| {
        let _ = sampler
            .sample(&first, Unlimited)
            .expect("annealing accepts any model");
        let _ = sampler
            .sample(&second, &mut slice)
            .expect("annealing accepts any model");
        let _ = sampler.resume(Unlimited).expect("a run exists");
    });

    assert_eq!(info.count_total, 0);
}

/// Warm tabu calls, on the same model and on a model of the same shape, allocate nothing.
#[test]
fn warm_tabu_calls_do_not_allocate() {
    let (first, second) = (model(0), model(1));
    let mut sampler = TabuSearch::new(Pcg64Mcg::seed_from_u64(1), 100, 2);
    let mut slice = |stats: &TabuStats| stats.moves >= 37;
    let cold = allocation_counter::measure(|| {
        let _ = sampler.sample(&first, Unlimited).expect("any model");
    });

    assert!(cold.count_total > 0, "the counter must see the cold call");

    let info = allocation_counter::measure(|| {
        let _ = sampler.sample(&first, Unlimited).expect("any model");
        let _ = sampler.sample(&second, &mut slice).expect("any model");
        let _ = sampler.resume(Unlimited).expect("a run exists");
    });

    assert_eq!(info.count_total, 0);
}

/// Warm tempering calls, on the same model and on a model of the same shape, allocate nothing.
#[test]
fn warm_tempering_calls_do_not_allocate() {
    let (first, second) = (model(0), model(1));
    let rng = Pcg64Mcg::seed_from_u64(1);
    let mut sampler = ParallelTempering::new(rng, Schedule::from_model(20), 4);
    let mut slice = |stats: &TemperingStats| stats.flips >= 37;
    let cold = allocation_counter::measure(|| {
        let _ = sampler.sample(&first, Unlimited).expect("any model");
    });

    assert!(cold.count_total > 0, "the counter must see the cold call");

    let info = allocation_counter::measure(|| {
        let _ = sampler.sample(&first, Unlimited).expect("any model");
        let _ = sampler.sample(&second, &mut slice).expect("any model");
        let _ = sampler.resume(Unlimited).expect("a run exists");
    });

    assert_eq!(info.count_total, 0);
}
