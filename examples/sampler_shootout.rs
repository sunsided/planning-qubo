//! The four samplers on one support-ticket assignment, against the exact optimum.
//!
//! Six tickets go to three engineers; each engineer takes at most two, and the cost is
//! the hours each engineer needs for each ticket. The encoding has 24 binary variables with
//! slack bits, small enough for the exhaustive sampler to find the optimum. Simulated
//! annealing, tabu search and parallel tempering then each draw 50 samples from different
//! seeds, and the tables show how often each hits the optimum and how long a draw takes:
//! once at the safe penalty weight, and once at the smaller weight that tuning against the
//! exhaustive sampler finds. A large weight buries the objective under the penalties.
//!
//! Run with `cargo run --release --example sampler_shootout`.

use std::time::{Duration, Instant};

use planning_qubo::assignment::{self, Encoding};
use planning_qubo::{
    Exhaustive, ParallelTempering, Sample, Sampler, Schedule, SimulatedAnnealing, TabuSearch,
    Unlimited, penalty,
};
use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;

/// Hours that each engineer (column) needs for each ticket (row).
const HOURS: [[i64; 3]; 6] = [
    [2, 5, 4],
    [3, 2, 6],
    [4, 4, 1],
    [6, 3, 3],
    [1, 4, 5],
    [5, 2, 2],
];

/// Tickets per engineer.
const CAPACITIES: [u32; 3] = [2, 2, 2];

/// Draws per sampler.
const DRAWS: u64 = 50;

/// Finds the exact optimum, then compares the heuristics at two penalty weights.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let encode = |p| assignment::encode(&HOURS, &CAPACITIES, p);
    let (safe, encoding) = penalty::safe(encode)?;
    let qubo = encoding.problem().qubo();

    let start = Instant::now();
    let mut exhaustive = Exhaustive::new();
    let exact = exhaustive.sample(qubo, Unlimited)?;
    let optimum = encoding.decode(exact.sample.bits).cost;
    println!(
        "{} variables; exhaustive optimum {optimum} hours in {:.1?}\n",
        qubo.num_variables(),
        start.elapsed()
    );
    let tuned = penalty::tune(encode, &mut exhaustive, Unlimited)?;

    for (weight, encoding) in [(safe, &encoding), (tuned.penalty, &tuned.encoding)] {
        println!("penalty weight {weight}:");
        compare(encoding, optimum)?;
    }

    Ok(())
}

/// Prints how often each heuristic sampler hits `optimum` on `encoding`.
fn compare(encoding: &Encoding<i64>, optimum: i64) -> Result<(), planning_qubo::SampleError> {
    println!("  {:<20} {:>8} {:>12}", "sampler", "optimal", "per draw");

    let annealing =
        |seed| SimulatedAnnealing::new(Pcg64Mcg::seed_from_u64(seed), Schedule::from_model(200));
    let tabu = |seed| TabuSearch::new(Pcg64Mcg::seed_from_u64(seed), 400, 4);
    let tempering =
        |seed| ParallelTempering::new(Pcg64Mcg::seed_from_u64(seed), Schedule::from_model(100), 6);

    report("simulated annealing", draws(encoding, optimum, annealing)?);
    report("tabu search", draws(encoding, optimum, tabu)?);
    report("parallel tempering", draws(encoding, optimum, tempering)?);
    println!();

    Ok(())
}

/// Draws one sample from each of [`DRAWS`] seeded samplers; returns the optimal count and
/// the mean time per draw.
fn draws<S>(
    encoding: &Encoding<i64>,
    optimum: i64,
    make: impl Fn(u64) -> S,
) -> Result<(u64, Duration), S::Error>
where
    S: for<'a> Sampler<planning_qubo::Qubo<i64>, Sample<'a> = Sample<'a, i64>> + 'static,
{
    let mut hits = 0;
    let start = Instant::now();

    for seed in 0..DRAWS {
        let mut sampler = make(seed);
        let run = sampler.sample(encoding.problem().qubo(), Unlimited)?;
        let decoded = encoding.decode(run.sample.bits);
        hits += u64::from(decoded.is_feasible() && decoded.cost == optimum);
    }

    Ok((
        hits,
        start.elapsed() / u32::try_from(DRAWS).expect("few draws"),
    ))
}

/// Prints one table row.
fn report(name: &str, (hits, per_draw): (u64, Duration)) {
    println!("  {name:<20} {hits:>5}/{DRAWS} {per_draw:>12.1?}");
}
