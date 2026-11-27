//! Scheduling a bakery's morning bakes on two ovens, in half-hour slots from 4:00.
//!
//! Each batch bakes for a number of slots, some dough is ready only later, the baguettes
//! must be out for the 6:00 delivery, and the cake base must be baked before its topping
//! goes in. The objective is the weighted completion time: batches that sell early weigh
//! more. Parallel tempering samples the time-indexed encoding with a tuned penalty weight,
//! and a brute force over all start times gives the optimum to compare with.
//!
//! Run with `cargo run --release --example bakery_ovens`.

use planning_qubo::scheduling::{self, Job};
use planning_qubo::{ParallelTempering, Sampler, Schedule, Unlimited, penalty};
use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;

/// The number of half-hour slots, 4:00 to 8:00.
const HORIZON: usize = 8;

/// The number of ovens.
const OVENS: u32 = 2;

/// The batches: name and job (slots of baking, earliest slot, latest end, weight).
const BATCHES: [(&str, Job<i64>); 6] = [
    (
        "sourdough",
        Job {
            duration: 3,
            release: 0,
            deadline: HORIZON,
            weight: 2,
        },
    ),
    (
        "rye",
        Job {
            duration: 2,
            release: 0,
            deadline: HORIZON,
            weight: 1,
        },
    ),
    (
        "croissants",
        Job {
            duration: 1,
            release: 1,
            deadline: HORIZON,
            weight: 3,
        },
    ),
    (
        "baguettes",
        Job {
            duration: 2,
            release: 0,
            deadline: 4,
            weight: 4,
        },
    ),
    (
        "cake base",
        Job {
            duration: 2,
            release: 0,
            deadline: HORIZON,
            weight: 1,
        },
    ),
    (
        "cake topping",
        Job {
            duration: 1,
            release: 0,
            deadline: HORIZON,
            weight: 1,
        },
    ),
];

/// The cake base bakes before its topping.
const PRECEDENCES: [(usize, usize); 1] = [(4, 5)];

/// Tunes the penalty, keeps the best of ten schedules and compares it with the optimum.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let jobs: Vec<Job<i64>> = BATCHES.iter().map(|&(_, job)| job).collect();
    let encode = |p| scheduling::encode(&jobs, HORIZON, OVENS, &PRECEDENCES, p);
    let mut sampler =
        ParallelTempering::new(Pcg64Mcg::seed_from_u64(3), Schedule::from_model(500), 8);

    let tuned = penalty::tune(encode, &mut sampler, Unlimited)?;
    println!(
        "{} binary variables, penalty weight {}",
        tuned.encoding.problem().qubo().num_variables(),
        tuned.penalty
    );

    let mut best = None;

    for _ in 0..10 {
        let run = sampler.sample(tuned.encoding.problem().qubo(), Unlimited)?;
        let decoded = tuned.encoding.decode(run.sample.bits);

        if decoded.is_feasible()
            && best
                .as_ref()
                .is_none_or(|b: &scheduling::Decoded<i64>| decoded.cost < b.cost)
        {
            best = Some(decoded);
        }
    }

    let Some(plan) = best else {
        println!("no feasible schedule in 10 restarts");
        return Ok(());
    };

    for ((name, job), start) in BATCHES.iter().zip(&plan.starts) {
        let start = start.expect("a feasible plan starts every batch");
        println!(
            "  {name:<13} {} to {}",
            clock(start),
            clock(start + job.duration)
        );
    }

    println!(
        "weighted completion: {} (optimum {})",
        plan.cost,
        brute_force(&jobs)
    );

    Ok(())
}

/// The wall-clock time of slot `slot`.
fn clock(slot: usize) -> String {
    format!("{}:{:02}", 4 + slot / 2, slot % 2 * 30)
}

/// The best weighted completion time over all start vectors, checked directly.
fn brute_force(jobs: &[Job<i64>]) -> i64 {
    let mut best = i64::MAX;
    let mut starts = vec![0; jobs.len()];

    loop {
        if feasible(jobs, &starts) {
            let cost = jobs
                .iter()
                .zip(&starts)
                .map(|(job, &s)| job.weight * i64::try_from(s + job.duration).expect("small"))
                .sum();
            best = best.min(cost);
        }

        if !next(&mut starts) {
            return best;
        }
    }
}

/// Advances `starts` like an odometer over `0..HORIZON`; `false` after the last vector.
fn next(starts: &mut [usize]) -> bool {
    for s in starts.iter_mut() {
        *s += 1;

        if *s < HORIZON {
            return true;
        }

        *s = 0;
    }

    false
}

/// Whether `starts` respects the windows, the oven count and the precedences.
fn feasible(jobs: &[Job<i64>], starts: &[usize]) -> bool {
    let windows = jobs
        .iter()
        .zip(starts)
        .all(|(job, &s)| s >= job.release && s + job.duration <= job.deadline);
    let ovens = (0..HORIZON).all(|slot| {
        let baking = jobs
            .iter()
            .zip(starts)
            .filter(|&(job, &s)| s <= slot && slot < s + job.duration)
            .count();
        baking <= OVENS as usize
    });
    let ordered = PRECEDENCES
        .iter()
        .all(|&(a, b)| starts[a] + jobs[a].duration <= starts[b]);

    windows && ovens && ordered
}
