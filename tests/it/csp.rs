//! The CSP encoding: the ground state energy is 0 exactly for satisfiable problems, checked
//! by brute force over the CSP's own value assignments.

use planning_qubo::csp::Csp;
use planning_qubo::{Exhaustive, Sampler, Unlimited};
use proptest::prelude::*;

/// A generated instance: domain sizes and binary constraints `(first, second, allowed mask)`.
type Instance = (Vec<usize>, Vec<(usize, usize, u16)>);

/// Generates up to 3 variables with domains of 1 to 3 values and up to 4 constraints.
///
/// Bit `a · 3 + b` of a mask allows the value pair `(a, b)`. With at most 9 one-hot
/// variables, the exhaustive sampler stays fast.
fn instance() -> impl Strategy<Value = Instance> {
    (1..=3_usize).prop_flat_map(|n| {
        let domains = prop::collection::vec(1..=3_usize, n);
        let constraint = (0..n, 0..n, any::<u16>());
        let constraints = prop::collection::vec(constraint, 0..=4);

        (domains, constraints)
    })
}

/// Builds the CSP of an instance.
fn build((domains, constraints): &Instance) -> Csp {
    let mut csp = Csp::new();

    for (i, &d) in domains.iter().enumerate() {
        csp.add_variable(&format!("v{i}"), d);
    }

    for (k, &(first, second, mask)) in constraints.iter().enumerate() {
        csp.add_constraint(&format!("c{k}"), first, second, |a, b| {
            mask >> (a * 3 + b) & 1 == 1
        });
    }

    csp
}

/// Returns `true` if some value assignment satisfies every constraint, by brute force.
///
/// This checks the masks directly and uses neither the encoding nor `Csp::is_solution`.
fn satisfiable((domains, constraints): &Instance) -> bool {
    let total: usize = domains.iter().product();

    (0..total).any(|code| {
        let mut rest = code;
        let values: Vec<usize> = domains
            .iter()
            .map(|&d| {
                let v = rest % d;
                rest /= d;
                v
            })
            .collect();

        constraints
            .iter()
            .all(|&(first, second, mask)| mask >> (values[first] * 3 + values[second]) & 1 == 1)
    })
}

proptest! {
    /// The ground state is a solution if one exists, and violates a named constraint if not.
    #[test]
    fn ground_state_decides_satisfiability(instance in instance()) {
        let csp = build(&instance);
        let encoding = csp.encode(2_i64);
        let mut sampler = Exhaustive::new();
        let run = sampler.sample(encoding.problem().qubo(), Unlimited).expect("few variables");
        let decoded = encoding.decode(run.sample.bits);

        if satisfiable(&instance) {
            prop_assert_eq!(run.sample.energy, 0);
            prop_assert!(decoded.is_feasible());

            let values: Vec<usize> = decoded.values.iter().map(|v| v.expect("one value")).collect();
            prop_assert!(csp.is_solution(&values));
        } else {
            prop_assert!(run.sample.energy >= 2);
            prop_assert!(!decoded.is_feasible());
        }
    }
}

/// Two colours cannot colour a triangle: the decoder names the broken constraint.
#[test]
fn decoder_names_the_violated_constraint() {
    let mut csp = Csp::new();
    let a = csp.add_variable("a", 2);
    let b = csp.add_variable("b", 2);
    let c = csp.add_variable("c", 2);
    csp.all_different("ab", &[a, b]);
    csp.all_different("bc", &[b, c]);
    csp.add_constraint("ac", a, c, |x, y| x != y);

    let encoding = csp.encode(1_i64);
    let mut sampler = Exhaustive::new();
    let run = sampler
        .sample(encoding.problem().qubo(), Unlimited)
        .expect("few");
    let decoded = encoding.decode(run.sample.bits);

    assert_eq!(run.sample.energy, 1);
    // One broken edge or one uncoloured vertex: both cost one penalty.
    assert_eq!(decoded.violations.len(), 1);

    // A sample with no value for `a` and two values for `b`.
    let mut bits = vec![false; 6];
    bits[encoding.variable(b, 0).index()] = true;
    bits[encoding.variable(b, 1).index()] = true;
    bits[encoding.variable(c, 0).index()] = true;
    let decoded = encoding.decode(&bits);

    assert_eq!(decoded.values, [None, None, Some(0)]);
    assert_eq!(decoded.violations, ["a.exactly_one", "b.exactly_one", "bc"]);
}

/// A constraint of a variable with itself restricts its values.
#[test]
fn self_constraints_restrict_values() {
    let mut csp = Csp::new();
    let x = csp.add_variable("x", 3);
    csp.add_constraint("odd", x, x, |a, _| a % 2 == 1);

    let encoding = csp.encode(1_i64);
    let mut sampler = Exhaustive::new();
    let run = sampler
        .sample(encoding.problem().qubo(), Unlimited)
        .expect("few");

    assert_eq!(encoding.decode(run.sample.bits).values, [Some(1)]);
}

/// Two variables with the same name keep their own one-hot bits.
#[test]
fn same_name_variables_stay_separate() {
    let mut csp = Csp::new();
    let first = csp.add_variable("x", 2);
    let second = csp.add_variable("x", 2);
    csp.all_different("differ", &[first, second]);

    let encoding = csp.encode(1_i64);
    assert_eq!(encoding.problem().qubo().num_variables(), 4);
    assert_ne!(encoding.variable(first, 0), encoding.variable(second, 0));

    let mut sampler = Exhaustive::new();
    let run = sampler
        .sample(encoding.problem().qubo(), Unlimited)
        .expect("few");
    let decoded = encoding.decode(run.sample.bits);

    assert_eq!(run.sample.energy, 0);
    assert!(decoded.values == [Some(0), Some(1)] || decoded.values == [Some(1), Some(0)]);
}
