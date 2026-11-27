//! Exam timetabling as constraint satisfaction: courses that share a student need
//! different exam slots.
//!
//! Eight courses, a handful of students with their enrollments, and four slots. The physics
//! lab is booked in the first slot, so physics cannot sit then. Each conflict and each
//! course's one-slot rule becomes a named penalty, so a sample that breaks a rule says
//! which. Tabu search samples the 32 one-hot variables. With only two slots, no timetable
//! exists, and the decoder names the clashes of the best sample it found.
//!
//! Run with `cargo run --release --example exam_timetable`.

use std::collections::BTreeSet;

use planning_qubo::csp::Csp;
use planning_qubo::{Sampler, TabuSearch, Unlimited};
use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;

/// The courses with an exam.
const COURSES: [&str; 8] = [
    "algebra",
    "analysis",
    "physics",
    "chemistry",
    "biology",
    "programming",
    "statistics",
    "history",
];

/// Each student's courses, by index into [`COURSES`].
const ENROLLMENTS: [&[usize]; 7] = [
    &[0, 1, 2],
    &[0, 5, 6],
    &[2, 3, 4],
    &[1, 2, 6],
    &[3, 4, 7],
    &[5, 6, 7],
    &[0, 3],
];

/// The exam slots of the week.
const SLOTS: [&str; 4] = ["Mon 9:00", "Mon 14:00", "Tue 9:00", "Tue 14:00"];

/// Schedules the exams in four slots, then shows the clashes when only two exist.
fn main() -> Result<(), planning_qubo::SampleError> {
    let csp = timetable(SLOTS.len());
    let encoding = csp.encode(1_i64);
    let mut sampler = TabuSearch::new(Pcg64Mcg::seed_from_u64(1), 2_000, 6);
    let run = sampler.sample(encoding.problem().qubo(), Unlimited)?;
    let decoded = encoding.decode(run.sample.bits);

    println!(
        "{} courses, {} slots: {} binary variables",
        COURSES.len(),
        SLOTS.len(),
        run.sample.bits.len()
    );

    for (course, value) in COURSES.iter().zip(&decoded.values) {
        let slot = value.map_or("unscheduled", |s| SLOTS[s]);
        println!("  {course:<12} {slot}");
    }

    let values: Vec<usize> = decoded
        .values
        .iter()
        .map(|v| v.unwrap_or(usize::MAX))
        .collect();
    println!(
        "valid timetable: {}",
        decoded.is_feasible() && csp.is_solution(&values)
    );

    let tight = timetable(2);
    let encoding = tight.encode(1_i64);
    let run = sampler.sample(encoding.problem().qubo(), Unlimited)?;
    let decoded = encoding.decode(run.sample.bits);
    // Leaving a course out costs as much as one clash, so the sampler may drop courses.
    println!(
        "\nwith two slots, the best sample breaks {} rules:",
        decoded.violations.len()
    );

    for name in &decoded.violations {
        println!("  {name}");
    }

    Ok(())
}

/// The timetabling problem over the first `slots` slots; variable `i` is course `i`.
fn timetable(slots: usize) -> Csp {
    let mut csp = Csp::new();
    let vars: Vec<usize> = COURSES
        .iter()
        .map(|name| csp.add_variable(name, slots))
        .collect();

    for (a, b) in conflicts() {
        let name = format!("{} vs {}", COURSES[a], COURSES[b]);
        csp.all_different(&name, &[vars[a], vars[b]]);
    }

    csp.add_constraint(
        "physics lab booked Mon 9:00",
        vars[2],
        vars[2],
        |slot, _| slot != 0,
    );

    csp
}

/// The course pairs that share at least one student, each pair once.
fn conflicts() -> BTreeSet<(usize, usize)> {
    ENROLLMENTS
        .iter()
        .flat_map(|courses| {
            courses.iter().flat_map(move |&a| {
                courses
                    .iter()
                    .filter(move |&&b| a < b)
                    .map(move |&b| (a, b))
            })
        })
        .collect()
}
