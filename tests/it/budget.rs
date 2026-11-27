//! Budgets: a limit of `n` runs exactly `n` units of work, and runs in slices resume to the
//! same result as one run without a budget.

use planning_qubo::assignment;
use planning_qubo::{
    AnnealingStats, Exhaustive, ExhaustiveStats, ParallelTempering, Qubo, Sampler, Schedule,
    SimulatedAnnealing, Status, TabuSearch, TabuStats, TemperingStats, Unlimited,
};
use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;

/// An assignment model with 3 tasks and 3 agents, one of capacity 2: 11 variables with slack.
fn model() -> Qubo<i64> {
    let costs = [[4, 1, 3], [2, 0, 5], [3, 2, 2]];
    let encoding = assignment::encode(&costs, &[2, 1, 1], 30).expect("rows match");

    encoding.problem().qubo().clone()
}

/// An annealing sampler with a fixed seed and 50 sweeps.
fn annealer() -> SimulatedAnnealing<Pcg64Mcg, i64> {
    SimulatedAnnealing::new(Pcg64Mcg::seed_from_u64(42), Schedule::from_model(50))
}

/// The exhaustive sampler stops after exactly `n` evaluated assignments.
#[test]
fn exhaustive_runs_exactly_the_limit() {
    let qubo = model();
    let total = 1_u64 << qubo.num_variables();
    let mut sampler = Exhaustive::new();

    for limit in [0, 1, 5, total - 1, total, total + 3] {
        let mut budget = |stats: &ExhaustiveStats| stats.evaluated >= limit;
        let run = sampler.sample(&qubo, &mut budget).expect("few variables");

        assert_eq!(run.stats.evaluated, limit.min(total));
        assert_eq!(run.status == Status::Finished, limit >= total);
    }
}

/// Simulated annealing stops after exactly `n` flip attempts, or exactly `n` sweeps.
#[test]
fn annealing_runs_exactly_the_limit() {
    let qubo = model();
    let variables = u64::try_from(qubo.num_variables()).expect("few variables");
    let mut sampler = annealer();

    for limit in [0, 1, 7, variables, 3 * variables + 2] {
        let mut budget = |stats: &AnnealingStats| stats.flips >= limit;
        let run = sampler
            .sample(&qubo, &mut budget)
            .expect("annealing accepts any model");

        assert_eq!(run.stats.flips, limit);
        assert_eq!(run.stats.sweeps, limit / variables);
        assert_eq!(run.status, Status::OutOfBudget);
    }

    let mut budget = |stats: &AnnealingStats| stats.sweeps >= 4;
    let run = sampler
        .sample(&qubo, &mut budget)
        .expect("annealing accepts any model");

    assert_eq!(run.stats.sweeps, 4);
    assert_eq!(run.stats.flips, 4 * variables);
}

/// Slices of 7 assignments resume to the same ground state as one run.
#[test]
fn exhaustive_slices_match_one_run() {
    let qubo = model();
    let mut whole = Exhaustive::new();
    let expected = whole.sample(&qubo, Unlimited).expect("few variables");

    let mut sliced = Exhaustive::new();
    let mut slice = |stats: &ExhaustiveStats| stats.evaluated >= 7;
    let mut status = sliced
        .sample(&qubo, &mut slice)
        .expect("few variables")
        .status;
    let mut calls = 1;

    while status == Status::OutOfBudget {
        status = sliced.resume(&mut slice).expect("a run exists").status;
        calls += 1;
    }

    let run = sliced.resume(Unlimited).expect("a run exists");

    assert!(calls > 100);
    assert_eq!(run.stats.evaluated, 0);
    assert_eq!(run.sample, expected.sample);
}

/// Slices of 13 flips resume to the same result as one run with the same seed.
#[test]
fn annealing_slices_match_one_run() {
    let qubo = model();
    let mut whole = annealer();
    let expected = whole
        .sample(&qubo, Unlimited)
        .expect("annealing accepts any model");

    let mut sliced = annealer();
    let mut slice = |stats: &AnnealingStats| stats.flips >= 13;
    let mut run = sliced
        .sample(&qubo, &mut slice)
        .expect("annealing accepts any model");
    let mut flips = run.stats.flips;

    while run.status == Status::OutOfBudget {
        run = sliced.resume(&mut slice).expect("a run exists");
        flips += run.stats.flips;
    }

    assert_eq!(flips, expected.stats.flips);
    assert_eq!(run.sample, expected.sample);
}

/// A tabu sampler with a fixed seed, 200 moves and tenure 3.
fn tabu() -> TabuSearch<Pcg64Mcg, i64> {
    TabuSearch::new(Pcg64Mcg::seed_from_u64(42), 200, 3)
}

/// A parallel tempering sampler with a fixed seed, 30 rounds and 4 replicas.
fn tempering() -> ParallelTempering<Pcg64Mcg, i64> {
    ParallelTempering::new(Pcg64Mcg::seed_from_u64(42), Schedule::from_model(30), 4)
}

/// Tabu search stops after exactly `n` moves.
#[test]
fn tabu_runs_exactly_the_limit() {
    let qubo = model();
    let mut sampler = tabu();

    for limit in [0, 1, 17, 199, 200, 500] {
        let mut budget = |stats: &TabuStats| stats.moves >= limit;
        let run = sampler.sample(&qubo, &mut budget).expect("any model");

        assert_eq!(run.stats.moves, limit.min(200));
        assert_eq!(run.status == Status::Finished, limit >= 200);
    }
}

/// Parallel tempering stops after exactly `n` flip attempts.
#[test]
fn tempering_runs_exactly_the_limit() {
    let qubo = model();
    let per_round = 4 * u64::try_from(qubo.num_variables()).expect("few variables");
    let mut sampler = tempering();

    for limit in [0, 1, 7, per_round, 3 * per_round + 5] {
        let mut budget = |stats: &TemperingStats| stats.flips >= limit;
        let run = sampler.sample(&qubo, &mut budget).expect("any model");

        assert_eq!(run.stats.flips, limit);
        assert_eq!(run.stats.rounds, limit / per_round);
        assert_eq!(run.status, Status::OutOfBudget);
    }
}

/// Slices of 9 moves resume to the same result as one tabu run with the same seed.
#[test]
fn tabu_slices_match_one_run() {
    let qubo = model();
    let expected = tabu()
        .sample(&qubo, Unlimited)
        .expect("any model")
        .sample
        .bits
        .to_vec();

    let mut sliced = tabu();
    let mut slice = |stats: &TabuStats| stats.moves >= 9;
    let mut run = sliced.sample(&qubo, &mut slice).expect("any model");
    let mut moves = run.stats.moves;

    while run.status == Status::OutOfBudget {
        run = sliced.resume(&mut slice).expect("a run exists");
        moves += run.stats.moves;
    }

    assert_eq!(moves, 200);
    assert_eq!(run.sample.bits, expected);
}

/// Slices of 13 flips resume to the same result as one tempering run with the same seed.
#[test]
fn tempering_slices_match_one_run() {
    let qubo = model();
    let mut whole = tempering();
    let expected = whole.sample(&qubo, Unlimited).expect("any model");

    let mut sliced = tempering();
    let mut slice = |stats: &TemperingStats| stats.flips >= 13;
    let mut run = sliced.sample(&qubo, &mut slice).expect("any model");
    let (mut flips, mut swaps) = (run.stats.flips, run.stats.swaps_accepted);

    while run.status == Status::OutOfBudget {
        run = sliced.resume(&mut slice).expect("a run exists");
        flips += run.stats.flips;
        swaps += run.stats.swaps_accepted;
    }

    assert_eq!(flips, expected.stats.flips);
    assert_eq!(swaps, expected.stats.swaps_accepted);
    assert_eq!(run.sample, expected.sample);
}

/// Tabu search and parallel tempering reach the exhaustive ground state of the model.
#[test]
fn tabu_and_tempering_find_the_ground_state() {
    let qubo = model();
    let ground = Exhaustive::new()
        .sample(&qubo, Unlimited)
        .expect("few variables")
        .sample
        .energy;

    assert_eq!(
        tabu()
            .sample(&qubo, Unlimited)
            .expect("any model")
            .sample
            .energy,
        ground
    );
    assert_eq!(
        tempering()
            .sample(&qubo, Unlimited)
            .expect("any model")
            .sample
            .energy,
        ground
    );
}
