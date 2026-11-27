//! The routing encoding: ground states decode to optimal tours, checked by brute force over
//! the permutations themselves.

use planning_qubo::routing::{self, RoutingError};
use planning_qubo::{Exhaustive, Sampler, Unlimited, penalty};
use proptest::prelude::*;

/// Generates 1 to 5 cities with asymmetric distances from 0 to 9: at most 16 variables.
fn distances() -> impl Strategy<Value = Vec<Vec<i64>>> {
    (1..=5_usize).prop_flat_map(|n| prop::collection::vec(prop::collection::vec(0..=9_i64, n), n))
}

/// All permutations of `items`.
fn permutations(items: &[usize]) -> Vec<Vec<usize>> {
    if items.is_empty() {
        return vec![Vec::new()];
    }

    (0..items.len())
        .flat_map(|k| {
            let mut rest = items.to_vec();
            let first = rest.remove(k);
            permutations(&rest).into_iter().map(move |mut tail| {
                tail.insert(0, first);
                tail
            })
        })
        .collect()
}

/// The length of the closed tour `tour`, which starts at the depot.
///
/// The depot alone has length 0: the diagonal is not a leg.
fn length(distances: &[Vec<i64>], tour: &[usize]) -> i64 {
    if tour.len() == 1 {
        return 0;
    }

    let closed = tour.iter().chain(std::iter::once(&0));

    tour.iter()
        .zip(closed.skip(1))
        .map(|(&u, &v)| distances[u][v])
        .sum()
}

/// The shortest closed tour from the depot, by brute force.
fn optimum(distances: &[Vec<i64>]) -> i64 {
    let others: Vec<usize> = (1..distances.len()).collect();

    permutations(&others)
        .into_iter()
        .map(|rest| {
            let tour: Vec<usize> = std::iter::once(0).chain(rest).collect();
            length(distances, &tour)
        })
        .min()
        .expect("at least one tour")
}

proptest! {
    /// With the safe weight, the ground state decodes to a shortest tour.
    #[test]
    fn ground_state_is_a_shortest_tour(distances in distances()) {
        let (_, encoding) = penalty::safe(|p| routing::encode(&distances, p)).expect("square");
        let mut sampler = Exhaustive::new();
        let run = sampler.sample(encoding.problem().qubo(), Unlimited).expect("few variables");
        let decoded = encoding.decode(run.sample.bits);
        let tour = decoded.tour.clone().expect("a permutation");

        prop_assert!(decoded.is_feasible());
        prop_assert_eq!(tour.len(), distances.len());
        prop_assert_eq!(length(&distances, &tour), optimum(&distances));
        prop_assert_eq!(decoded.cost, optimum(&distances));
    }
}

/// A sample that puts two cities at one position is no tour, and the decoder names both
/// broken constraints.
#[test]
fn decoder_reports_a_broken_permutation() {
    let distances = vec![vec![0_i64, 2, 3], vec![2, 0, 4], vec![3, 4, 0]];
    let encoding = routing::encode(&distances, 10).expect("square");
    let mut bits = vec![false; 4];
    bits[encoding.variable(1, 1).index()] = true;
    bits[encoding.variable(2, 1).index()] = true;

    let decoded = encoding.decode(&bits);

    assert_eq!(decoded.tour, None);
    assert_eq!(decoded.violations, ["pos1.exactly_one", "pos2.exactly_one"]);
    // Both cities leave the depot first; neither returns from the last position.
    assert_eq!(decoded.cost, 2 + 3);
}

/// Invalid matrices are errors.
#[test]
fn invalid_matrices_are_errors() {
    let empty: [[i64; 0]; 0] = [];

    assert_eq!(
        routing::encode(&empty, 1).err(),
        Some(RoutingError::NoCities)
    );
    assert_eq!(
        routing::encode(&[vec![0_i64, 1], vec![1]], 1).err(),
        Some(RoutingError::ShapeMismatch { row: 1 })
    );
}

/// A city at two positions is no tour, whichever city it is and wherever it sits.
#[test]
fn a_repeated_city_is_never_a_tour() {
    let distances = vec![vec![0_i64; 4]; 4];
    let encoding = routing::encode(&distances, 10).expect("square");

    for city in 1..4 {
        let others: Vec<usize> = (1..4).filter(|&c| c != city).collect();

        // `city` at positions 1 and 2, another city at 3.
        let mut bits = vec![false; 9];
        bits[encoding.variable(city, 1).index()] = true;
        bits[encoding.variable(city, 2).index()] = true;
        bits[encoding.variable(others[0], 3).index()] = true;

        assert_eq!(encoding.decode(&bits).tour, None, "city {city}");
    }
}
