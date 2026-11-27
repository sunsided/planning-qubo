//! Parallel restarts on a tour of nine European capitals, with the `parallel` feature.
//!
//! The tour encoding has 64 binary variables. Eight parallel tempering samplers with
//! different seeds run at once, one thread each, through `sample_parallel`; the shortest
//! feasible tour among their samples wins. A brute force over all 40 320 tours gives the
//! optimum to compare with.
//!
//! Run with `cargo run --release --features parallel --example parallel_restarts`.

use std::time::Instant;

use planning_qubo::{ParallelTempering, Schedule, Unlimited, penalty, routing, sample_parallel};
use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;

/// The depot first: name, latitude and longitude in degrees.
const CITIES: [(&str, f64, f64); 9] = [
    ("Vienna", 48.208, 16.373),
    ("Prague", 50.075, 14.438),
    ("Berlin", 52.520, 13.405),
    ("Copenhagen", 55.676, 12.568),
    ("Amsterdam", 52.368, 4.904),
    ("Brussels", 50.850, 4.352),
    ("Paris", 48.857, 2.352),
    ("Bern", 46.948, 7.447),
    ("Rome", 41.903, 12.496),
];

/// The number of samplers, one thread each.
const THREADS: u64 = 8;

/// Tunes the penalty, runs eight samplers in parallel and keeps the shortest tour.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let distances: Vec<Vec<i64>> = CITIES
        .iter()
        .map(|a| CITIES.iter().map(|b| haversine_km(a, b)).collect())
        .collect();
    let mut tuning_sampler =
        ParallelTempering::new(Pcg64Mcg::seed_from_u64(0), Schedule::from_model(300), 8);
    let tuned = penalty::tune(
        |p| routing::encode(&distances, p),
        &mut tuning_sampler,
        Unlimited,
    )?;
    let encoding = &tuned.encoding;

    let mut samplers: Vec<_> = (1..=THREADS)
        .map(|seed| {
            ParallelTempering::new(
                Pcg64Mcg::seed_from_u64(seed),
                Schedule::from_model(1_500),
                8,
            )
        })
        .collect();

    let start = Instant::now();
    let runs = sample_parallel(&mut samplers, encoding.problem().qubo(), Unlimited);
    let elapsed = start.elapsed();

    let tours = runs
        .into_iter()
        .map(|run| encoding.decode(run.expect("tempering accepts any model").sample.bits))
        .filter_map(|decoded| Some((decoded.tour?, decoded.cost)));
    let best = tours.min_by_key(|&(_, km)| km);

    println!(
        "{} variables, penalty weight {}; {THREADS} samplers in {elapsed:.1?}",
        encoding.problem().qubo().num_variables(),
        tuned.penalty
    );

    match best {
        Some((tour, km)) => {
            let names: Vec<&str> = tour.iter().map(|&c| CITIES[c].0).collect();
            println!("best tour: {} -> Vienna, {km} km", names.join(" -> "));
        }
        None => println!("no sampler found a feasible tour"),
    }

    println!("optimum: {} km", brute_force(&distances));

    Ok(())
}

/// The great-circle distance between two cities, in whole kilometres.
#[expect(
    clippy::cast_possible_truncation,
    reason = "distances within Europe fit i64"
)]
fn haversine_km(
    &(_, lat_a, lon_a): &(&str, f64, f64),
    &(_, lat_b, lon_b): &(&str, f64, f64),
) -> i64 {
    /// The mean Earth radius in kilometres.
    const EARTH_RADIUS: f64 = 6371.0;

    let (phi_a, phi_b) = (lat_a.to_radians(), lat_b.to_radians());
    let half_lat = ((phi_b - phi_a) / 2.0).sin();
    let half_lon = ((lon_b - lon_a).to_radians() / 2.0).sin();
    let h = half_lat * half_lat + phi_a.cos() * phi_b.cos() * half_lon * half_lon;

    (2.0 * EARTH_RADIUS * h.sqrt().asin()).round() as i64
}

/// The shortest closed tour from the depot, over all orders of the other cities.
fn brute_force(distances: &[Vec<i64>]) -> i64 {
    let mut order: Vec<usize> = (1..distances.len()).collect();
    let mut best = i64::MAX;

    permute(&mut order, 0, &mut |order| {
        let from = std::iter::once(&0).chain(order);
        let to = order.iter().chain(std::iter::once(&0));
        best = best.min(from.zip(to).map(|(&u, &v)| distances[u][v]).sum());
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
