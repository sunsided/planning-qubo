//! Penalty tuning: the safe bound makes ground states feasible optima, and tuning finds a
//! smaller weight that still gives the sampler a feasible sample.

use planning_qubo::penalty::{self, Tuned};
use planning_qubo::{Exhaustive, Sampler, Schedule, SimulatedAnnealing, Unlimited, assignment};
use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;

/// Costs for 3 tasks and 3 agents.
const COSTS: [[i64; 3]; 3] = [[4, 1, 3], [2, 0, 5], [3, 2, 2]];

/// Capacities: one agent may take two tasks.
const CAPACITIES: [u32; 3] = [2, 1, 1];

/// Encodes the instance at penalty `p`.
fn encode(p: i64) -> Result<assignment::Encoding<i64>, assignment::ShapeMismatch> {
    assignment::encode(&COSTS, &CAPACITIES, p)
}

/// The cheapest feasible cost, by brute force over the task→agent maps.
fn optimum() -> i64 {
    let mut best = i64::MAX;

    for code in 0..27_u32 {
        let agents = [code % 3, code / 3 % 3, code / 9];
        let feasible = (0..3).all(|a| {
            let load = agents.iter().filter(|&&x| x == a).count();
            load <= CAPACITIES[a as usize] as usize
        });

        if feasible {
            let cost = (0..3).map(|t| COSTS[t][agents[t] as usize]).sum();
            best = best.min(cost);
        }
    }

    best
}

/// The safe weight exceeds the objective spread, and its ground state is a feasible optimum.
#[test]
fn safe_weight_gives_a_feasible_optimum() {
    let (p, encoding) = penalty::safe(encode).expect("rows match");
    let spread: i64 = COSTS.iter().flatten().map(|c| c.abs()).sum();

    assert_eq!(p, spread + 1);

    let mut sampler = Exhaustive::new();
    let run = sampler
        .sample(encoding.problem().qubo(), Unlimited)
        .expect("few variables");
    let decoded = encoding.decode(run.sample.bits);

    assert!(decoded.is_feasible());
    assert_eq!(decoded.cost, optimum());
}

/// Tuning with the exhaustive sampler stops below the safe weight with a feasible optimum.
#[test]
fn tuning_finds_a_smaller_feasible_weight() {
    let (safe, _) = penalty::safe(encode).expect("rows match");
    let mut sampler = Exhaustive::new();
    let Tuned {
        penalty,
        encoding,
        bits,
        feasible,
        rounds,
    } = penalty::tune(encode, &mut sampler, Unlimited).expect("few variables");

    assert!(feasible);
    assert!(penalty < safe);
    assert!(rounds >= 1);
    assert_eq!(encoding.decode(&bits).cost, optimum());
}

/// Tuning works with a heuristic sampler too, and reports infeasible instances.
#[test]
fn tuning_with_annealing_and_an_infeasible_instance() {
    let mut sampler =
        SimulatedAnnealing::new(Pcg64Mcg::seed_from_u64(7), Schedule::from_model(100));
    let tuned = penalty::tune(encode, &mut sampler, Unlimited).expect("any model");
    assert!(tuned.feasible);

    // Two tasks, one agent with capacity 1: no feasible assignment exists.
    let infeasible = |p| assignment::encode(&[[1], [1]], &[1], p);
    let (safe, _) = penalty::safe(infeasible).expect("rows match");
    let tuned = penalty::tune(infeasible, &mut Exhaustive::new(), Unlimited).expect("few");

    assert!(!tuned.feasible);
    assert_eq!(tuned.penalty, safe);
}
