//! Parallel sampling: each thread's run equals the same sampler's run on one thread.

use planning_qubo::{
    AnnealingStats, Exhaustive, Qubo, Sampler, Schedule, SimulatedAnnealing, Status, Unlimited,
    assignment, resume_parallel, sample_parallel,
};
use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;

/// An assignment model with 3 tasks and 3 agents, one of capacity 2: 11 variables with slack.
fn model() -> Qubo<i64> {
    let costs = [[4, 1, 3], [2, 0, 5], [3, 2, 2]];
    let encoding = assignment::encode(&costs, &[2, 1, 1], 30).expect("rows match");

    encoding.problem().qubo().clone()
}

/// Four annealers with different seeds.
fn annealers() -> Vec<SimulatedAnnealing<Pcg64Mcg, i64>> {
    (0..4)
        .map(|seed| {
            SimulatedAnnealing::new(Pcg64Mcg::seed_from_u64(seed), Schedule::from_model(200))
        })
        .collect()
}

/// Parallel runs come back in sampler order and match sequential runs with the same seeds.
#[test]
fn parallel_runs_match_sequential_runs() {
    let qubo = model();
    let expected: Vec<(Vec<bool>, i64)> = annealers()
        .iter_mut()
        .map(|sampler| {
            let run = sampler.sample(&qubo, Unlimited).expect("any model");
            (run.sample.bits.to_vec(), run.sample.energy)
        })
        .collect();

    let mut samplers = annealers();
    let runs = sample_parallel(&mut samplers, &qubo, Unlimited);
    let actual: Vec<(Vec<bool>, i64)> = runs
        .into_iter()
        .map(|run| {
            let run = run.expect("any model");
            (run.sample.bits.to_vec(), run.sample.energy)
        })
        .collect();

    assert_eq!(actual, expected);

    let ground = Exhaustive::new()
        .sample(&qubo, Unlimited)
        .expect("few variables")
        .sample
        .energy;
    assert_eq!(actual.iter().map(|(_, e)| *e).min(), Some(ground));
}

/// Each sampler gets its own copy of the budget, and resumes in parallel to the end.
#[test]
fn parallel_runs_share_no_budget() {
    let qubo = model();
    let mut samplers = annealers();
    let budget = |stats: &AnnealingStats| stats.flips >= 50;

    for run in sample_parallel(&mut samplers, &qubo, budget) {
        let run = run.expect("any model");
        assert_eq!((run.stats.flips, run.status), (50, Status::OutOfBudget));
    }

    for run in resume_parallel(&mut samplers, Unlimited) {
        assert_eq!(run.expect("a run exists").status, Status::Finished);
    }
}
