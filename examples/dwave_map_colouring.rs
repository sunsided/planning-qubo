//! Colouring the map of Australia for a D-Wave annealer, without network access.
//!
//! The seven states and territories get one of three colours, neighbours differ. The CSP
//! encoding becomes an Ising model, which the `dwave` module writes as a SAPI `qp` problem
//! for a solver. The solver here is a stand-in with a fully connected qubit graph; a real
//! QPU's graph is sparse, so a real run needs a minor embedding first. Instead of a QPU,
//! tabu search draws the samples, and they are packed the way SAPI returns them (base64
//! bit-packed solutions and little-endian energies), parsed back and decoded.
//!
//! Run with `cargo run --release --example dwave_map_colouring`.

use planning_qubo::csp::Csp;
use planning_qubo::dwave::{self, QpAnswer, Solver, SolverRef};
use planning_qubo::{Sampler, TabuSearch, Unlimited};
use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;

/// The states and territories of mainland Australia and Tasmania.
const REGIONS: [&str; 7] = ["WA", "NT", "SA", "QLD", "NSW", "VIC", "TAS"];

/// The land borders, by index into [`REGIONS`]. Tasmania has none.
const BORDERS: [(usize, usize); 9] = [
    (0, 1),
    (0, 2),
    (1, 2),
    (1, 3),
    (2, 3),
    (2, 4),
    (2, 5),
    (3, 4),
    (4, 5),
];

/// The colours.
const COLOURS: [&str; 3] = ["red", "green", "blue"];

/// Exports the colouring problem, simulates an answer, parses and decodes it.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut csp = Csp::new();
    let vars: Vec<usize> = REGIONS
        .iter()
        .map(|r| csp.add_variable(r, COLOURS.len()))
        .collect();

    for &(a, b) in &BORDERS {
        csp.all_different(
            &format!("{}|{}", REGIONS[a], REGIONS[b]),
            &[vars[a], vars[b]],
        );
    }

    let encoding = csp.encode(1_i64);
    let ising = encoding.problem().qubo().to_ising();
    let n = ising.num_spins();

    let qubits: Vec<usize> = (0..n).map(|i| 100 + i).collect();
    let couplers: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| (i + 1..n).map(move |j| (100 + i, 100 + j)))
        .collect();
    let solver = Solver {
        qubits: &qubits,
        couplers: &couplers,
    };
    let exported = dwave::export_ising(&ising, &solver, &qubits)?;

    let body = exported.submission_json(SolverRef::Name("stand-in-solver"), r#"{"num_reads":20}"#);
    println!(
        "POST /problems body ({} bytes):\n{}…\n",
        body.len(),
        &body[..120]
    );

    let samples = sample_locally(encoding.problem().qubo(), 20)?;
    // SAPI energies are in the exported units without the offset: the QUBO energy minus it.
    #[expect(clippy::cast_precision_loss, reason = "small energies")]
    let energy = |bits: &[bool]| encoding.problem().qubo().energy(bits) as f64 - exported.offset();
    let answer = pack_answer(&qubits, &samples, energy);
    let parsed = exported.parse_answer(&QpAnswer {
        active_variables: &answer.0,
        solutions: &answer.1,
        energies: &answer.2,
        num_occurrences: None,
    })?;

    let best = parsed
        .iter()
        .min_by(|a, b| a.energy.total_cmp(&b.energy))
        .expect("twenty samples");
    let decoded = encoding.decode(&best.bits);

    println!(
        "best of {} parsed samples, energy {}:",
        parsed.len(),
        best.energy
    );

    for (region, value) in REGIONS.iter().zip(&decoded.values) {
        println!("  {region:<4} {}", value.map_or("?", |c| COLOURS[c]));
    }

    println!("borders respected: {}", decoded.is_feasible());

    Ok(())
}

/// Draws `count` samples with tabu search, standing in for the QPU.
fn sample_locally(
    qubo: &planning_qubo::Qubo<i64>,
    count: u64,
) -> Result<Vec<Vec<bool>>, planning_qubo::SampleError> {
    (0..count)
        .map(|seed| {
            let mut sampler = TabuSearch::new(Pcg64Mcg::seed_from_u64(seed), 200, 4);
            Ok(sampler.sample(qubo, Unlimited)?.sample.bits.to_vec())
        })
        .collect()
}

/// Packs samples as a SAPI `qp` answer: base64 active qubits (`i32`), solutions (one bit
/// per qubit, most significant first, padded per solution) and energies (`f64`).
fn pack_answer(
    qubits: &[usize],
    samples: &[Vec<bool>],
    energy: impl Fn(&[bool]) -> f64,
) -> (String, String, String) {
    let width = qubits.len().div_ceil(8);
    let active: Vec<u8> = qubits
        .iter()
        .flat_map(|&q| i32::try_from(q).expect("small qubit index").to_le_bytes())
        .collect();
    let mut solutions = Vec::new();
    let mut energies = Vec::new();

    for bits in samples {
        let mut bytes = vec![0_u8; width];

        for (k, &bit) in bits.iter().enumerate() {
            bytes[k / 8] |= u8::from(bit) << (7 - k % 8);
        }

        solutions.extend(bytes);
        energies.extend(energy(bits).to_le_bytes());
    }

    (base64(&active), base64(&solutions), base64(&energies))
}

/// Standard padded base64.
fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();

    for chunk in bytes.chunks(3) {
        let word = chunk
            .iter()
            .enumerate()
            .fold(0_u32, |w, (k, &b)| w | u32::from(b) << (16 - 8 * k));

        for k in 0..=3 {
            let sextet = (word >> (18 - 6 * k) & 63) as usize;
            out.push(if k <= chunk.len() {
                char::from(ALPHABET[sextet])
            } else {
                '='
            });
        }
    }

    out
}
