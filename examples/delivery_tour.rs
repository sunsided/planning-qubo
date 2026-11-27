//! A delivery tour through seven German cities, as a travelling salesman QUBO.
//!
//! A van leaves the depot in Frankfurt, visits six cities and returns. Distances are
//! straight-line kilometres from the cities' coordinates. The tour encoding has 36 binary
//! variables, too many for the exhaustive sampler, so parallel tempering samples it, with
//! the penalty weight tuned against that sampler. A brute force over all 720 tours gives
//! the true optimum to compare with.
//!
//! Run with `cargo run --release --example delivery_tour`.

use planning_qubo::penalty;
use planning_qubo::routing;
use planning_qubo::{ParallelTempering, Sampler, Schedule, Unlimited};
use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;

/// The depot first, then the cities to visit: name, latitude and longitude in degrees.
const CITIES: [(&str, f64, f64); 7] = [
    ("Frankfurt", 50.110, 8.682),
    ("Cologne", 50.938, 6.960),
    ("Hamburg", 53.551, 9.994),
    ("Berlin", 52.520, 13.405),
    ("Leipzig", 51.340, 12.375),
    ("Munich", 48.137, 11.576),
    ("Stuttgart", 48.776, 9.183),
];

/// Tunes the penalty, samples ten restarts and compares the best tour with the optimum.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let distances = distance_matrix();
    let mut sampler =
        ParallelTempering::new(Pcg64Mcg::seed_from_u64(7), Schedule::from_model(400), 8);

    let tuned = penalty::tune(|p| routing::encode(&distances, p), &mut sampler, Unlimited)?;
    let (safe, _) = penalty::safe(|p| routing::encode(&distances, p))?;
    println!(
        "{} binary variables; penalty weight {} after {} rounds (safe bound {safe})",
        tuned.encoding.problem().qubo().num_variables(),
        tuned.penalty,
        tuned.rounds
    );

    let best = best_of_restarts(&tuned.encoding, &mut sampler, 10)?;
    let optimum = brute_force(&distances);

    match best {
        Some((tour, km)) => {
            let names: Vec<&str> = tour.iter().map(|&c| CITIES[c].0).collect();
            println!("sampled tour: {} -> Frankfurt, {km} km", names.join(" -> "));
        }
        None => println!("no feasible tour in 10 restarts"),
    }

    println!("optimal tour: {optimum} km (brute force over 720 tours)");

    Ok(())
}

/// The straight-line distance between every pair of cities, rounded to whole kilometres.
fn distance_matrix() -> Vec<Vec<i64>> {
    CITIES
        .iter()
        .map(|&(_, lat_a, lon_a)| {
            CITIES
                .iter()
                .map(|&(_, lat_b, lon_b)| haversine_km(lat_a, lon_a, lat_b, lon_b))
                .collect()
        })
        .collect()
}

/// The great-circle distance between two points, in whole kilometres.
#[expect(
    clippy::cast_possible_truncation,
    reason = "distances within Germany fit i64"
)]
fn haversine_km(lat_a: f64, lon_a: f64, lat_b: f64, lon_b: f64) -> i64 {
    /// The mean Earth radius in kilometres.
    const EARTH_RADIUS: f64 = 6371.0;

    let (phi_a, phi_b) = (lat_a.to_radians(), lat_b.to_radians());
    let d_phi = phi_b - phi_a;
    let d_lambda = (lon_b - lon_a).to_radians();
    let h =
        (d_phi / 2.0).sin().powi(2) + phi_a.cos() * phi_b.cos() * (d_lambda / 2.0).sin().powi(2);

    (2.0 * EARTH_RADIUS * h.sqrt().asin()).round() as i64
}

/// Samples `restarts` times and keeps the shortest feasible tour with its length.
fn best_of_restarts(
    encoding: &routing::Encoding<i64>,
    sampler: &mut ParallelTempering<Pcg64Mcg, i64>,
    restarts: usize,
) -> Result<Option<(Vec<usize>, i64)>, planning_qubo::SampleError> {
    let mut best: Option<(Vec<usize>, i64)> = None;

    for _ in 0..restarts {
        let run = sampler.sample(encoding.problem().qubo(), Unlimited)?;
        let decoded = encoding.decode(run.sample.bits);

        if let Some(tour) = decoded.tour {
            if best.as_ref().is_none_or(|(_, km)| decoded.cost < *km) {
                best = Some((tour, decoded.cost));
            }
        }
    }

    Ok(best)
}

/// The shortest closed tour from the depot, over all orders of the other cities.
fn brute_force(distances: &[Vec<i64>]) -> i64 {
    let mut order: Vec<usize> = (1..distances.len()).collect();
    let mut best = i64::MAX;

    permute(&mut order, 0, &mut |order| {
        let legs = std::iter::once(&0)
            .chain(order)
            .zip(order.iter().chain(std::iter::once(&0)));
        best = best.min(legs.map(|(&u, &v)| distances[u][v]).sum());
    });

    best
}

/// Calls `visit` with every permutation of `items[k..]`, by swapping in place.
fn permute(items: &mut [usize], k: usize, visit: &mut impl FnMut(&[usize])) {
    if k == items.len() {
        visit(items);
        return;
    }

    for i in k..items.len() {
        items.swap(k, i);
        permute(items, k + 1, visit);
        items.swap(k, i);
    }
}
