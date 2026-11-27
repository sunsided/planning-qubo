//! Integration tests: model conversions, the assignment encoding and the samplers.

mod allocations;
mod assignment;
mod budget;
mod conversion;
mod csp;
mod dwave;
mod mapf;
#[cfg(feature = "parallel")]
mod parallel;
mod penalty;
mod routing;
mod samplers;
mod scheduling;

use planning_qubo::{Ising, Problem, Qubo, assignment::Encoding};

/// All assignments of `n` variables, as bit vectors, in binary counting order.
pub fn assignments(n: usize) -> impl Iterator<Item = Vec<bool>> {
    (0..1_u32 << n).map(move |bits| (0..n).map(|i| bits >> i & 1 == 1).collect())
}

/// Models and encodings are `Send + Sync`, so one instance serves many threads.
#[test]
fn models_are_send_and_sync() {
    /// Compiles only for `Send + Sync` types.
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<Qubo<i64>>();
    assert_send_sync::<Qubo<f64>>();
    assert_send_sync::<Ising<i64>>();
    assert_send_sync::<Problem<i64>>();
    assert_send_sync::<Encoding<f64>>();
    assert_send_sync::<planning_qubo::csp::Encoding<i64>>();
    assert_send_sync::<planning_qubo::scheduling::Encoding<i64>>();
    assert_send_sync::<planning_qubo::mapf::Encoding<i64>>();
    assert_send_sync::<planning_qubo::routing::Encoding<i64>>();
    assert_send_sync::<planning_qubo::dwave::Exported>();
}
