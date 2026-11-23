//! The assignment encoding: ground states decode to feasible optima, checked by brute force.

use planning_qubo::assignment::{self, Encoding};
use planning_qubo::{Exhaustive, Sample, Sampler, Schedule, SimulatedAnnealing, Unlimited};
use proptest::prelude::*;
use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;

/// A generated instance: costs (tasks × agents) and agent capacities.
type Instance = (Vec<Vec<i64>>, Vec<u32>);

/// Generates up to 3 tasks and 3 agents, with negative costs and capacities from 0 to 3.
fn instance() -> impl Strategy<Value = Instance> {
    (1..=3_usize, 1..=3_usize).prop_flat_map(|(tasks, agents)| {
        let row = prop::collection::vec(-9..=9_i64, agents);
        let costs = prop::collection::vec(row, tasks);
        let capacities = prop::collection::vec(0..=3_u32, agents);

        (costs, capacities)
    })
}

/// A penalty weight above `Σ|cost|`, which makes each ground state a feasible optimum.
fn safe_penalty(costs: &[Vec<i64>]) -> i64 {
    1 + costs.iter().flatten().map(|c| c.abs()).sum::<i64>()
}

/// The lowest total cost over all task-to-agent maps within the capacities, by brute force.
///
/// This uses neither the builder nor the decoder. `None` if no map fits the capacities.
fn brute_force_optimum(costs: &[Vec<i64>], capacities: &[u32]) -> Option<i64> {
    let agents = capacities.len();
    let tasks = u32::try_from(costs.len()).expect("few tasks");
    let maps = agents.pow(tasks);

    (0..maps)
        .map(|code| decode_map(code, costs.len(), agents))
        .filter(|map| fits(map, capacities))
        .map(|map| map.iter().enumerate().map(|(t, &a)| costs[t][a]).sum())
        .min()
}

/// The map with number `code`: the agent of task `t` is digit `t` of `code` in base `agents`.
fn decode_map(code: usize, tasks: usize, agents: usize) -> Vec<usize> {
    let mut rest = code;

    (0..tasks)
        .map(|_| {
            let agent = rest % agents;
            rest /= agents;
            agent
        })
        .collect()
}

/// Returns `true` if no agent gets more tasks than its capacity.
fn fits(map: &[usize], capacities: &[u32]) -> bool {
    capacities.iter().enumerate().all(|(a, &capacity)| {
        let load = map.iter().filter(|&&agent| agent == a).count();
        u32::try_from(load).is_ok_and(|load| load <= capacity)
    })
}

/// Encodes an instance with a safe penalty.
fn encode((costs, capacities): &Instance) -> Encoding<i64> {
    assignment::encode(costs, capacities, safe_penalty(costs)).expect("generated rows match")
}

/// Asserts that a sample decodes to a feasible assignment with the given optimal cost.
///
/// Feasibility is also checked without the decoder: each task has an agent, and the
/// resulting map fits the capacities.
fn assert_optimal(
    encoding: &Encoding<i64>,
    capacities: &[u32],
    sample: Sample<'_, i64>,
    optimum: i64,
) {
    let decoded = encoding.decode(sample.bits);

    assert!(
        decoded.is_feasible(),
        "violations: {:?}",
        decoded.violations
    );
    let map: Option<Vec<usize>> = decoded.agents.iter().copied().collect();
    assert!(map.is_some_and(|map| fits(&map, capacities)));
    assert_eq!(decoded.cost, optimum);
    assert_eq!(sample.energy, encoding.problem().qubo().energy(sample.bits));
}

proptest! {
    /// The exhaustive ground state decodes to a feasible optimum, or to an infeasible
    /// assignment when the capacities cannot take all tasks.
    #[test]
    fn exhaustive_ground_state_is_a_feasible_optimum(instance in instance()) {
        let encoding = encode(&instance);
        let mut sampler = Exhaustive::new();
        let run = sampler
            .sample(encoding.problem().qubo(), Unlimited)
            .expect("few variables");

        match brute_force_optimum(&instance.0, &instance.1) {
            Some(optimum) => assert_optimal(&encoding, &instance.1, run.sample, optimum),
            None => prop_assert!(!encoding.decode(run.sample.bits).is_feasible()),
        }
    }
}

/// The best of a few annealing draws with a fixed seed, as `(energy, bits)`.
fn best_annealed(encoding: &Encoding<i64>, seed: u64) -> (i64, Vec<bool>) {
    let rng = Pcg64Mcg::seed_from_u64(seed);
    let mut sampler = SimulatedAnnealing::new(rng, Schedule::from_model(1_000));
    let qubo = encoding.problem().qubo();

    (0..10)
        .map(|_| {
            let run = sampler
                .sample(qubo, Unlimited)
                .expect("annealing accepts any model");
            (run.sample.energy, run.sample.bits.to_vec())
        })
        .min()
        .expect("ten draws")
}

/// Simulated annealing finds the brute-force optimum on small instances with a fixed seed.
#[test]
fn annealing_finds_the_optimum() {
    let instances: [Instance; 3] = [
        (
            vec![vec![4, 1, 3], vec![2, 0, 5], vec![3, 2, 2]],
            vec![1, 1, 1],
        ),
        (
            vec![vec![-3, 2], vec![1, -4], vec![2, 2], vec![0, -1]],
            vec![2, 2],
        ),
        (
            vec![vec![5, 1, 4], vec![1, 6, 2], vec![7, 2, 1], vec![3, 3, 3]],
            vec![2, 1, 3],
        ),
    ];

    for (seed, instance) in (0..).zip(&instances) {
        let encoding = encode(instance);
        let optimum = brute_force_optimum(&instance.0, &instance.1).expect("feasible");

        let (energy, bits) = best_annealed(&encoding, seed);
        let sample = Sample {
            bits: &bits,
            energy,
        };

        assert_optimal(&encoding, &instance.1, sample, optimum);
    }
}

/// The same seed gives the same samples.
#[test]
fn annealing_is_reproducible() {
    let encoding = encode(&(vec![vec![4, 1], vec![2, 3]], vec![1, 1]));

    assert_eq!(best_annealed(&encoding, 7), best_annealed(&encoding, 7));
}

/// The decoder names each violated constraint and leaves ambiguous tasks without an agent.
#[test]
fn decoder_names_violations() {
    let encoding = assignment::encode(&[[1, 2], [3, 4]], &[1, 1], 20).expect("rows match");
    let mut bits = vec![false; encoding.problem().qubo().num_variables()];
    bits[encoding.variable(0, 0).index()] = true;
    bits[encoding.variable(0, 1).index()] = true;

    let decoded = encoding.decode(&bits);

    assert_eq!(decoded.agents, [None, None]);
    assert_eq!(decoded.cost, 3);
    assert_eq!(
        decoded.violations,
        ["task0.exactly_one", "task1.exactly_one"]
    );
}

/// Cost rows of the wrong length are an error.
#[test]
fn rejects_ragged_costs() {
    let result = assignment::encode(&[vec![1, 2], vec![3]], &[1, 1], 10);

    assert_eq!(result.err(), Some(assignment::ShapeMismatch { task: 1 }));
}
