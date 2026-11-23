//! Property tests: QUBO and Ising energies agree on every assignment, in both directions.

use planning_qubo::{Coefficient, Ising, Qubo};
use proptest::prelude::*;

use crate::assignments;

/// The largest number of variables in the generated models.
const MAX_VARIABLES: usize = 12;

/// The largest coefficient magnitude. Small enough that 4× sums cannot overflow `i64`.
///
/// Bound: one coefficient collects at most 40 terms × 1000, and the conversions multiply
/// by at most 4, so intermediate values stay below about 1.6e5.
const MAX_MAGNITUDE: i64 = 1_000;

/// Terms `(i, j, value)` over `n` variables, plus an offset. `i == j` is a linear term.
type Terms = (usize, Vec<(usize, usize, i64)>, i64);

/// Generates up to [`MAX_VARIABLES`] variables with up to 40 terms.
fn terms() -> impl Strategy<Value = Terms> {
    (0..=MAX_VARIABLES).prop_flat_map(|n| {
        let index = 0..n.max(1);
        let value = -MAX_MAGNITUDE..=MAX_MAGNITUDE;
        let term = (index.clone(), index, value.clone());

        (Just(n), prop::collection::vec(term, 0..40), value)
    })
}

/// Builds a QUBO from generated terms, with each coefficient converted by `convert`.
fn qubo_from<T: Coefficient>((n, terms, offset): &Terms, convert: impl Fn(i64) -> T) -> Qubo<T> {
    let mut qubo = Qubo::new(*n);
    qubo.add_offset(convert(*offset));

    for &(i, j, value) in terms.iter().filter(|_| *n > 0) {
        qubo.add_quadratic(i, j, convert(value));
    }

    qubo
}

/// Builds an Ising model with scale 1 from generated terms. `i == j` is a field term.
fn ising_from((n, terms, offset): &Terms) -> Ising<i64> {
    let mut ising = Ising::new(*n, 1);
    ising.add_offset(*offset);

    for &(i, j, value) in terms.iter().filter(|_| *n > 0) {
        if i == j {
            ising.add_field(i, value);
        } else {
            ising.add_coupling(i, j, value);
        }
    }

    ising
}

/// Converts an exactly representable integer to `f64`.
#[expect(
    clippy::cast_precision_loss,
    reason = "generated values stay below 2^53 in magnitude, so the conversion is exact"
)]
fn exact_f64(value: i64) -> f64 {
    value as f64
}

proptest! {
    /// QUBO to Ising with `i64`: the Ising energy is 4 times the QUBO energy, and the
    /// conversion back gives the original model.
    #[test]
    fn i64_qubo_to_ising_agrees(generated in terms()) {
        let qubo = qubo_from(&generated, |v| v);
        let ising = qubo.to_ising();

        prop_assert_eq!(ising.scale(), 4);

        for x in assignments(qubo.num_variables()) {
            prop_assert_eq!(ising.energy(&x), 4 * qubo.energy(&x));
        }

        prop_assert_eq!(ising.to_qubo(), Ok(qubo));
    }

    /// Ising to QUBO with `i64`, from an independent Ising model: energies are equal.
    #[test]
    fn i64_ising_to_qubo_agrees(generated in terms()) {
        let ising = ising_from(&generated);
        let qubo = ising.to_qubo().expect("scale 1 divides each coefficient");

        for x in assignments(ising.num_spins()) {
            prop_assert_eq!(qubo.energy(&x), ising.energy(&x));
        }
    }

    /// QUBO to Ising and back with `f64`: energies agree within rounding.
    #[test]
    fn f64_round_trip_agrees(generated in terms()) {
        let qubo = qubo_from(&generated, exact_f64);
        let ising = qubo.to_ising();
        let back = ising.to_qubo().expect("f64 division is always possible");

        for x in assignments(qubo.num_variables()) {
            let expected = qubo.energy(&x);
            prop_assert!((ising.energy(&x) - expected).abs() <= 1e-9 * expected.abs().max(1.0));
            prop_assert!((back.energy(&x) - expected).abs() <= 1e-9 * expected.abs().max(1.0));
        }
    }
}
