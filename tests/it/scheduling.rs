//! The scheduling encoding: ground states decode to feasible optimal schedules, checked by
//! brute force over the start times themselves.

use planning_qubo::scheduling::{self, Job, ScheduleError};
use planning_qubo::{Exhaustive, Sampler, Unlimited, penalty};
use proptest::prelude::*;

/// A generated instance: jobs, horizon and precedences, on one machine.
type Instance = (Vec<Job<i64>>, usize, Vec<(usize, usize)>);

/// Generates up to 3 jobs over a horizon of up to 4 slots: at most 12 start variables.
fn instance() -> impl Strategy<Value = Instance> {
    (1..=3_usize, 1..=4_usize).prop_flat_map(|(n, horizon)| {
        let job = (1..=2_usize, 0..=1_usize, 1..=4_usize, -3..=5_i64).prop_map(
            |(duration, release, deadline, weight)| Job {
                duration,
                release,
                deadline,
                weight,
            },
        );
        let jobs = prop::collection::vec(job, n);
        let precedences = prop::collection::vec((0..n, 0..n), 0..=2);

        (jobs, Just(horizon), precedences)
    })
}

/// Returns `true` if `starts` respects the windows, one machine and the precedences.
fn feasible((jobs, horizon, precedences): &Instance, starts: &[usize]) -> bool {
    let windows = jobs
        .iter()
        .zip(starts)
        .all(|(job, &s)| s >= job.release && s + job.duration <= job.deadline.min(*horizon));
    let disjoint = (0..jobs.len()).all(|a| {
        (a + 1..jobs.len()).all(|b| {
            starts[a] + jobs[a].duration <= starts[b] || starts[b] + jobs[b].duration <= starts[a]
        })
    });
    let ordered = precedences
        .iter()
        .all(|&(a, b)| starts[a] + jobs[a].duration <= starts[b]);

    windows && disjoint && ordered
}

/// The lowest weighted completion time over all feasible start vectors, by brute force.
fn optimum(instance: &Instance) -> Option<i64> {
    let (jobs, horizon, _) = instance;
    let total = horizon.pow(u32::try_from(jobs.len()).expect("few jobs"));

    (0..total)
        .map(|code| {
            let mut rest = code;
            (0..jobs.len())
                .map(|_| {
                    let s = rest % horizon;
                    rest /= horizon;
                    s
                })
                .collect::<Vec<usize>>()
        })
        .filter(|starts| feasible(instance, starts))
        .map(|starts| {
            jobs.iter()
                .zip(&starts)
                .map(|(job, &s)| job.weight * i64::try_from(s + job.duration).expect("small"))
                .sum()
        })
        .min()
}

proptest! {
    /// With the safe weight, the ground state is a feasible optimum, or violates a constraint.
    #[test]
    fn ground_state_is_an_optimal_schedule(instance in instance()) {
        let (jobs, horizon, precedences) = &instance;
        let (_, encoding) = penalty::safe(|p| scheduling::encode(jobs, *horizon, 1, precedences, p))
            .expect("valid instance");
        let mut sampler = Exhaustive::new();
        let run = sampler.sample(encoding.problem().qubo(), Unlimited).expect("few variables");
        let decoded = encoding.decode(run.sample.bits);

        match optimum(&instance) {
            Some(best) => {
                prop_assert!(decoded.is_feasible(), "{:?}", decoded.violations);
                let starts: Vec<usize> = decoded.starts.iter().map(|s| s.expect("a start")).collect();
                prop_assert!(feasible(&instance, &starts));
                prop_assert_eq!(decoded.cost, best);
            }
            None => prop_assert!(!decoded.is_feasible()),
        }
    }
}

/// Two machines run two jobs at once; a third waits.
#[test]
fn capacity_two_uses_slack() {
    let jobs = [Job::new(2, 1), Job::new(2, 1), Job::new(1, 1)];
    let (_, encoding) =
        penalty::safe(|p| scheduling::encode(&jobs, 3, 2, &[], p)).expect("valid instance");
    let mut sampler = Exhaustive::new();
    let run = sampler
        .sample(encoding.problem().qubo(), Unlimited)
        .expect("few variables");
    let decoded = encoding.decode(run.sample.bits);

    // The short job and one long job start at 0; the other long job starts at 1.
    assert!(decoded.is_feasible());
    assert_eq!(decoded.cost, 1 + 2 + 3);
}

/// A precedence that a sample breaks is reported by name.
#[test]
fn decoder_names_a_broken_precedence() {
    let jobs = [Job::new(1, 1), Job::new(1, 1)];
    let encoding = scheduling::encode(&jobs, 2, 2, &[(0, 1)], 10).expect("valid instance");
    let mut bits = vec![false; encoding.problem().qubo().num_variables()];
    bits[encoding.variable(0, 1).expect("allowed").index()] = true;
    bits[encoding.variable(1, 0).expect("allowed").index()] = true;

    let decoded = encoding.decode(&bits);

    assert_eq!(decoded.starts, [Some(1), Some(0)]);
    assert_eq!(decoded.violations, ["job0.before.job1"]);
}

/// Invalid instances are errors.
#[test]
fn invalid_instances_are_errors() {
    assert_eq!(
        scheduling::encode(&[Job::new(0, 1_i64)], 3, 1, &[], 1).err(),
        Some(ScheduleError::ZeroDuration { job: 0 })
    );
    assert_eq!(
        scheduling::encode(&[Job::new(1, 1_i64)], 3, 1, &[(0, 1)], 1).err(),
        Some(ScheduleError::UnknownJob { precedence: 0 })
    );
}
