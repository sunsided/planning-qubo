//! D-Wave SAPI export: golden `qp` strings, the answer example from the SAPI docs, and an
//! exhaustive export → answer → decode round trip.
//!
//! The golden strings for the mock solver come from the reference encoder of
//! dwave-cloud-client (`coders.encode_problem_as_qp`), replayed with Python's `struct` and
//! `base64` on the same model and solver.

use planning_qubo::dwave::{
    self, ExportError, ParseError, ProblemType, QpAnswer, Solver, SolverRef,
};
use planning_qubo::{Ising, Qubo, assignment};

/// A solver whose qubit order is not sorted, so `lin` positions differ from qubit ids.
const QUBITS: [usize; 5] = [4, 0, 3, 1, 2];

/// The solver's couplers.
const COUPLERS: [(usize, usize); 6] = [(0, 1), (0, 3), (1, 2), (1, 3), (2, 4), (3, 4)];

/// The mock solver.
fn solver() -> Solver<'static> {
    Solver {
        qubits: &QUBITS,
        couplers: &COUPLERS,
    }
}

/// The qubit of each variable: 0 → 3, 1 → 1, 2 → 2.
const MAP: [usize; 3] = [3, 1, 2];

/// `5 + 2x₀ − 3x₁ + 4x₀x₁ − x₁x₂`.
fn model() -> Qubo<i64> {
    let mut qubo = Qubo::new(3);
    qubo.add_offset(5);
    qubo.add_linear(0, 2);
    qubo.add_linear(1, -3);
    qubo.add_quadratic(0, 1, 4);
    qubo.add_quadratic(1, 2, -1);

    qubo
}

/// The QUBO export matches the reference encoder, NaN for unused qubits included.
#[test]
fn qubo_export_matches_the_reference_encoder() {
    let exported = dwave::export_qubo(&model(), &solver(), &MAP).expect("valid map");

    assert_eq!(exported.problem_type(), ProblemType::Qubo);
    assert_eq!(
        exported.lin(),
        "AAAAAAAA+H8AAAAAAAD4fwAAAAAAAABAAAAAAAAACMAAAAAAAAAAAA=="
    );
    assert_eq!(exported.quad(), "AAAAAAAA8L8AAAAAAAAQQA==");
    assert_eq!(
        exported.submission_json(
            SolverRef::Name("Advantage_system4.1"),
            r#"{"num_reads":10}"#
        ),
        r#"[{"data":{"format":"qp","lin":"AAAAAAAA+H8AAAAAAAD4fwAAAAAAAABAAAAAAAAACMAAAAAAAAAAAA==","quad":"AAAAAAAA8L8AAAAAAAAQQA=="},"params":{"num_reads":10},"solver":"Advantage_system4.1","type":"qubo"}]"#
    );
}

/// The Ising export divides by the scale of 4 and matches the reference encoder.
#[test]
fn ising_export_removes_the_scale() {
    let ising = model().to_ising();
    let exported = dwave::export_ising(&ising, &solver(), &MAP).expect("valid map");

    assert_eq!(ising.scale(), 4);
    assert_eq!(
        exported.lin(),
        "AAAAAAAA+H8AAAAAAAD4fwAAAAAAAABAAAAAAAAA6L8AAAAAAADQvw=="
    );
    assert_eq!(exported.quad(), "AAAAAAAA0L8AAAAAAADwPw==");
    assert!((exported.offset() - 5.25).abs() < f64::EPSILON);

    let versioned = SolverRef::Versioned {
        name: "Adv\"2",
        graph_id: "01ab",
    };
    assert!(
        exported
            .submission_json(versioned, "{}")
            .contains(r#""solver":{"name":"Adv\"2","version":{"graph_id":"01ab"}},"type":"ising""#)
    );
}

/// A packed answer parses into model bits, energies with the offset, and occurrences.
#[test]
fn answer_parses_with_the_offset() {
    let qubo = model();
    let exported = dwave::export_qubo(&qubo, &solver(), &MAP).expect("valid map");
    let answer = QpAnswer {
        active_variables: "AwAAAAEAAAACAAAA",
        solutions: "oGA=",
        energies: "AAAAAAAAAEAAAAAAAAAQwA==",
        num_occurrences: Some("BwAAAAMAAAA="),
    };

    let samples = exported.parse_answer(&answer).expect("matching answer");

    assert_eq!(samples.len(), 2);
    assert_eq!(samples[0].bits, [true, false, true]);
    assert_eq!(samples[1].bits, [false, true, true]);
    assert_eq!(
        samples.iter().map(|s| s.occurrences).collect::<Vec<_>>(),
        [7, 3]
    );

    for sample in &samples {
        #[expect(clippy::cast_precision_loss, reason = "small energies")]
        let recomputed = qubo.energy(&sample.bits) as f64;
        assert!((sample.energy - recomputed).abs() < 1e-12);
    }
}

/// The two-qubit Ising example of the SAPI docs: h = (−0.5, 0.5) and J = −1 on qubits 30
/// and 31. Both returned solutions have energy −1, which only most-significant-bit-first
/// unpacking reproduces with two distinct solutions.
#[test]
fn sapi_docs_example_round_trips() {
    let mut ising = Ising::<f64>::new(2, 1.0);
    ising.add_field(0, -0.5);
    ising.add_field(1, 0.5);
    ising.add_coupling(0, 1, -1.0);
    let docs_solver = Solver {
        qubits: &[30, 31],
        couplers: &[(30, 31)],
    };

    let exported = dwave::export_ising(&ising, &docs_solver, &[30, 31]).expect("valid map");
    assert_eq!(exported.lin(), "AAAAAAAA4L8AAAAAAADgPw==");
    assert_eq!(exported.quad(), "AAAAAAAA8L8=");

    let answer = QpAnswer {
        active_variables: "HgAAAB8AAAA=",
        solutions: "AMA=",
        energies: "AAAAAAAA8L8AAAAAAADwvw==",
        num_occurrences: Some("BgAAAAQAAAA="),
    };
    let samples = exported.parse_answer(&answer).expect("matching answer");

    assert_eq!(samples[0].bits, [false, false]);
    assert_eq!(samples[1].bits, [true, true]);
    assert_eq!((samples[0].occurrences, samples[1].occurrences), (6, 4));

    for sample in &samples {
        assert!((ising.energy(&sample.bits) - sample.energy).abs() < 1e-12);
        assert!((sample.energy + 1.0).abs() < 1e-12);
    }
}

/// Standard base64 of `bytes`, written out here so the test does not share the crate's encoder.
fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();

    for chunk in bytes.chunks(3) {
        let word = chunk
            .iter()
            .enumerate()
            .fold(0_u32, |w, (k, &b)| w | u32::from(b) << (16 - 8 * k));

        for k in 0..4 {
            out.push(if k <= chunk.len() {
                char::from(ALPHABET[(word >> (18 - 6 * k) & 63) as usize])
            } else {
                '='
            });
        }
    }

    out
}

/// Every assignment of an encoded problem, sent through a simulated raw-mode answer, decodes
/// to the same plan as the original bits, and its energy matches the model's.
#[test]
fn exhaustive_round_trip_decodes_the_same_plans() {
    let encoding = assignment::encode(&[[3, 1], [2, 5]], &[1, 1], 12).expect("rows match");
    let qubo = encoding.problem().qubo();
    let n = qubo.num_variables();

    // A complete solver graph on the qubits 10, 11, …, so every pair has a coupler.
    let qubits: Vec<usize> = (10..10 + n).collect();
    let couplers: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| (i + 1..n).map(move |j| (10 + i, 10 + j)))
        .collect();
    let full = Solver {
        qubits: &qubits,
        couplers: &couplers,
    };
    let exported = dwave::export_qubo(qubo, &full, &qubits).expect("complete graph");

    // Active qubits in reverse order, to check the map back to variables.
    let active: Vec<usize> = (0..n).rev().collect();
    let width = n.div_ceil(8);
    let (mut packed, mut energies, mut all) = (Vec::new(), Vec::new(), Vec::new());

    for code in 0..1_u32 << n {
        let bits: Vec<bool> = (0..n).map(|i| code >> i & 1 == 1).collect();
        let mut bytes = vec![0_u8; width];

        for (k, &variable) in active.iter().enumerate() {
            bytes[k / 8] |= u8::from(bits[variable]) << (7 - k % 8);
        }

        packed.extend(bytes);
        #[expect(clippy::cast_precision_loss, reason = "small energies")]
        energies.extend(((qubo.energy(&bits) - qubo.offset()) as f64).to_le_bytes());
        all.push(bits);
    }

    let active_bytes: Vec<u8> = active
        .iter()
        .flat_map(|&v| i32::try_from(10 + v).expect("small").to_le_bytes())
        .collect();
    let (active_text, solutions, energy_text) =
        (base64(&active_bytes), base64(&packed), base64(&energies));
    let answer = QpAnswer {
        active_variables: &active_text,
        solutions: &solutions,
        energies: &energy_text,
        num_occurrences: None,
    };
    let samples = exported.parse_answer(&answer).expect("matching answer");

    assert_eq!(samples.len(), all.len());

    for (sample, bits) in samples.iter().zip(&all) {
        assert_eq!(&sample.bits, bits);
        assert_eq!(sample.occurrences, 1);
        assert_eq!(encoding.decode(&sample.bits), encoding.decode(bits));

        #[expect(clippy::cast_precision_loss, reason = "small energies")]
        let recomputed = qubo.energy(&sample.bits) as f64;
        assert!((sample.energy - recomputed).abs() < 1e-9);
    }
}

/// Maps and answers that do not fit are errors.
#[test]
fn mismatches_are_errors() {
    let qubo = model();

    assert_eq!(
        dwave::export_qubo(&qubo, &solver(), &[3, 1]).err(),
        Some(ExportError::MapLength {
            variables: 3,
            qubits: 2
        })
    );
    assert_eq!(
        dwave::export_qubo(&qubo, &solver(), &[3, 1, 9]).err(),
        Some(ExportError::UnknownQubit {
            variable: 2,
            qubit: 9
        })
    );
    assert_eq!(
        dwave::export_qubo(&qubo, &solver(), &[3, 1, 3]).err(),
        Some(ExportError::SharedQubit { qubit: 3 })
    );
    assert_eq!(
        dwave::export_qubo(&qubo, &solver(), &[0, 4, 2]).err(),
        Some(ExportError::MissingCoupler { i: 0, j: 1 })
    );

    let exported = dwave::export_qubo(&qubo, &solver(), &MAP).expect("valid map");
    let answer = |active, solutions, energies| QpAnswer {
        active_variables: active,
        solutions,
        energies,
        num_occurrences: None,
    };

    assert_eq!(
        exported.parse_answer(&answer(
            "AwAAAAEAAAAEAAAA",
            "oGA=",
            "AAAAAAAAAEAAAAAAAAAQwA=="
        )),
        Err(ParseError::UnknownQubit { qubit: 4 })
    );
    assert_eq!(
        exported.parse_answer(&answer("AwAAAAEAAAA=", "oGA=", "AAAAAAAAAEAAAAAAAAAQwA==")),
        Err(ParseError::MissingVariable { variable: 2 })
    );
    assert_eq!(
        exported.parse_answer(&answer(
            "AwAAAAEAAAACAAAA",
            "oA==",
            "AAAAAAAAAEAAAAAAAAAQwA=="
        )),
        Err(ParseError::CountMismatch)
    );
    assert_eq!(
        exported.parse_answer(&answer(
            "AwAAAAEAAAACAAAA",
            "oGA",
            "AAAAAAAAAEAAAAAAAAAQwA=="
        )),
        Err(ParseError::Malformed { field: "solutions" })
    );
}

/// An answer to the scaled Ising export parses back to the QUBO's energies, offset included.
#[test]
fn ising_answer_energies_match_the_qubo() {
    let qubo = model();
    let exported = dwave::export_ising(&qubo.to_ising(), &solver(), &MAP).expect("valid map");
    let solutions = [[true, false, true], [false, true, true]];

    // SAPI energies exclude the offset of 5.25: E(x) − 5.25 for each solution.
    #[expect(clippy::cast_precision_loss, reason = "small energies")]
    let energies: Vec<u8> = solutions
        .iter()
        .flat_map(|x| (qubo.energy(x) as f64 - 5.25).to_le_bytes())
        .collect();
    let energy_text = base64(&energies);
    let answer = QpAnswer {
        active_variables: "AwAAAAEAAAACAAAA",
        solutions: "oGA=",
        energies: &energy_text,
        num_occurrences: None,
    };

    let samples = exported.parse_answer(&answer).expect("matching answer");

    for (sample, x) in samples.iter().zip(&solutions) {
        assert_eq!(&sample.bits, x);

        #[expect(clippy::cast_precision_loss, reason = "small energies")]
        let recomputed = qubo.energy(x) as f64;
        assert!((sample.energy - recomputed).abs() < 1e-12);
    }
}
