//! The path finding encoding: ground states decode to conflict-free optimal plans, checked
//! by brute force over the agents' paths themselves.

use planning_qubo::mapf::{self, MapfError};
use planning_qubo::{Exhaustive, ParallelTempering, Sampler, Schedule, Unlimited, penalty};
use proptest::prelude::*;
use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;

/// A generated instance: adjacency lists, agents and horizon.
type Instance = (Vec<Vec<usize>>, Vec<(usize, usize)>, usize);

/// Generates 3 vertices with random directed edges, 2 agents and a horizon of 1 or 2.
fn instance() -> impl Strategy<Value = Instance> {
    (
        any::<u8>(),
        0..3_usize,
        0..3_usize,
        0..3_usize,
        0..3_usize,
        1..=2_usize,
    )
        .prop_filter_map("distinct starts", |(mask, s0, g0, s1, g1, horizon)| {
            let graph = (0..3)
                .map(|u| {
                    (0..3)
                        .filter(|&w| w != u && mask >> (u * 3 + w) & 1 == 1)
                        .collect()
                })
                .collect();

            (s0 != s1).then_some((graph, vec![(s0, g0), (s1, g1)], horizon))
        })
}

/// All paths of `horizon` steps from `start` to `goal`, as vertex sequences.
fn paths(graph: &[Vec<usize>], start: usize, goal: usize, horizon: usize) -> Vec<Vec<usize>> {
    let mut paths = vec![vec![start]];

    for _ in 0..horizon {
        paths = paths
            .into_iter()
            .flat_map(|path| {
                let u = *path.last().expect("non-empty");
                std::iter::once(u)
                    .chain(graph[u].iter().copied())
                    .map(move |w| {
                        let mut next = path.clone();
                        next.push(w);
                        next
                    })
            })
            .collect();
    }

    paths.retain(|path| path.last() == Some(&goal));
    paths
}

/// The cost of a path: every step except a wait at the goal.
fn path_cost(path: &[usize], goal: usize) -> i64 {
    let steps = path
        .windows(2)
        .filter(|w| !(w[0] == goal && w[1] == goal))
        .count();
    i64::try_from(steps).expect("short")
}

/// Returns `true` if two paths have no vertex or swap conflict.
fn compatible(p: &[usize], q: &[usize]) -> bool {
    let vertex = p.iter().zip(q).any(|(a, b)| a == b);
    let swap = p
        .windows(2)
        .zip(q.windows(2))
        .any(|(a, b)| a[0] != a[1] && a[0] == b[1] && a[1] == b[0]);

    !vertex && !swap
}

/// The lowest total cost of two conflict-free paths, by brute force.
fn optimum((graph, agents, horizon): &Instance) -> Option<i64> {
    let (first, second) = (agents[0], agents[1]);
    let ps = paths(graph, first.0, first.1, *horizon);
    let qs = paths(graph, second.0, second.1, *horizon);

    ps.iter()
        .flat_map(|p| qs.iter().map(move |q| (p, q)))
        .filter(|(p, q)| compatible(p, q))
        .map(|(p, q)| path_cost(p, first.1) + path_cost(q, second.1))
        .min()
}

proptest! {
    /// With the safe weight, the ground state is a conflict-free optimal plan, or violates
    /// a constraint when none exists.
    #[test]
    fn ground_state_is_an_optimal_plan(instance in instance()) {
        let (graph, agents, horizon) = &instance;
        let (_, encoding) = penalty::safe(|p: i64| mapf::encode(graph, agents, *horizon, p))
            .expect("valid instance");
        prop_assume!(encoding.problem().qubo().num_variables() <= 22);

        let mut sampler = Exhaustive::new();
        let run = sampler.sample(encoding.problem().qubo(), Unlimited).expect("few variables");
        let decoded = encoding.decode(run.sample.bits);

        match optimum(&instance) {
            Some(best) => {
                prop_assert!(decoded.is_feasible(), "{:?}", decoded.violations);
                let p = decoded.paths[0].clone().expect("a path");
                let q = decoded.paths[1].clone().expect("a path");
                prop_assert!(compatible(&p, &q));
                prop_assert_eq!(p.last(), Some(&agents[0].1));
                prop_assert_eq!(q.last(), Some(&agents[1].1));
                prop_assert_eq!(decoded.cost, best);
            }
            None => prop_assert!(!decoded.is_feasible()),
        }
    }
}

/// Two agents swapping the ends of one edge break the swap constraint, by name.
#[test]
fn swaps_are_conflicts() {
    let graph = [vec![1], vec![0]];
    let encoding = mapf::encode(&graph, &[(0, 1), (1, 0)], 1, 5_i64).expect("valid instance");
    let mut bits = vec![false; encoding.problem().qubo().num_variables()];
    bits[encoding.variable(0, 0, 0, 1).expect("on a path").index()] = true;
    bits[encoding.variable(1, 0, 1, 0).expect("on a path").index()] = true;

    let decoded = encoding.decode(&bits);

    assert_eq!(decoded.paths, [Some(vec![0, 1]), Some(vec![1, 0])]);
    assert_eq!(decoded.violations, ["0-1.t0.swap"]);
}

/// In a corridor with a side pocket, two agents swap ends by stepping aside.
#[test]
fn corridor_swap_through_a_pocket() {
    // 0 - 1 - 2, with the pocket 3 at 1.
    let graph = [vec![1], vec![0, 2, 3], vec![1], vec![1]];
    let agents = [(0, 2), (2, 0)];
    let instance = (graph.to_vec(), agents.to_vec(), 4);
    let best = optimum(&instance).expect("the pocket makes room");
    let (_, encoding) =
        penalty::safe(|p: i64| mapf::encode(&graph, &agents, 4, p)).expect("valid instance");

    let feasible_best = (0..8)
        .filter_map(|seed| {
            let rng = Pcg64Mcg::seed_from_u64(seed);
            let mut sampler = ParallelTempering::new(rng, Schedule::from_model(400), 8);
            let run = sampler
                .sample(encoding.problem().qubo(), Unlimited)
                .expect("any model");
            let decoded = encoding.decode(run.sample.bits);
            decoded.is_feasible().then_some(decoded.cost)
        })
        .min();

    assert_eq!(feasible_best, Some(best));
}

/// Invalid instances are errors.
#[test]
fn invalid_instances_are_errors() {
    let graph = [vec![1], vec![0]];

    assert_eq!(
        mapf::encode(&graph, &[(0, 2)], 1, 1_i64).err(),
        Some(MapfError::UnknownVertex { vertex: 2 })
    );
    assert_eq!(
        mapf::encode(&graph, &[(0, 1), (0, 0)], 1, 1_i64).err(),
        Some(MapfError::SharedStart { agent: 1 })
    );
    assert_eq!(
        mapf::encode(&graph, &[(0, 1)], 0, 1_i64).err(),
        Some(MapfError::ZeroHorizon)
    );
}

/// Self-loops and repeated neighbors in the adjacency lists do not duplicate moves.
#[test]
fn self_loops_and_repeated_edges_are_ignored() {
    // Vertex 0 lists itself and vertex 1 twice; the agent waits at its goal 0.
    let graph = [vec![0, 1, 1], vec![0]];
    let encoding = mapf::encode(&graph, &[(0, 0)], 1, 3_i64).expect("valid instance");

    assert_eq!(encoding.problem().qubo().num_variables(), 1);

    let mut sampler = Exhaustive::new();
    let run = sampler
        .sample(encoding.problem().qubo(), Unlimited)
        .expect("few variables");
    let decoded = encoding.decode(run.sample.bits);

    assert_eq!(run.sample.energy, 0);
    assert_eq!(decoded.paths, [Some(vec![0, 0])]);
    assert!(decoded.is_feasible());
}
