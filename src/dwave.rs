//! Export to D-Wave's Solver API (SAPI) and parsing of its sample sets.
//!
//! [`export_qubo`] and [`export_ising`] write a model in SAPI's `qp` format for a QPU
//! solver, and [`Exported::parse_answer`] reads the solver's answer back into one assignment
//! per sample, ready for the encoders' decoders. This module does no networking and needs no
//! dependencies; submitting, polling and authentication belong to a separate `std` adapter.
//! Format reference: <https://docs.dwavequantum.com/en/latest/leap_sapi/sapi_rest.html>.
//!
//! ```mermaid
//! flowchart LR
//!     model[Qubo / Ising] -->|export with qubit map| exported[Exported: qp data]
//!     exported -->|submission_json| sapi[(SAPI)]
//!     sapi -->|answer fields| parse[parse_answer]
//!     parse --> samples[samples: bits, energy, occurrences] --> decode[encoder decode]
//! ```
//!
//! # The `qp` format
//!
//! - `lin`: one little-endian `f64` per solver qubit, in the order of the solver's `qubits`
//!   property; qubits that the model does not use are NaN.
//! - `quad`: one `f64` per solver coupler whose two qubits are both used, in the order of
//!   the solver's `couplers` property, `0.0` where the model has no pair.
//! - The answer holds the used qubits (`active_variables`, `i32`), the energies (`f64`),
//!   the occurrence counts (`i32`, only in histogram mode), and the solutions packed one bit
//!   per used qubit, most significant bit first, each solution padded to a whole byte.
//!   For Ising problems, bit 1 is the spin +1.
//!
//! # Qubits and embedding
//!
//! The caller maps each model variable to one solver qubit. Each model pair must sit on a
//! solver coupler, or the export fails with [`ExportError::MissingCoupler`]. Minor embedding,
//! where one variable becomes a chain of qubits, is not done here: compute it elsewhere and
//! submit the embedded model, or use a D-Wave service that embeds on the server side. This
//! keeps the crate small and `no_std`, and leaves the choice of embedder to the caller.
//!
//! # Coefficients and precision
//!
//! Coefficients convert to [`f64`] at this boundary; [`i64`] values are exact up to 2^53 in
//! magnitude. An Ising model is divided by its [`scale`](Ising::scale), so an `i64` model
//! from [`Qubo::to_ising`] exports with the 1/2 and 1/4 factors restored. The constant
//! offset is not sent: SAPI energies exclude it, and [`Exported::parse_answer`] adds it
//! back. QPUs accept coefficients in solver-specific ranges (`h_range`, `j_range`, for
//! example [−4, 4] and [−1, 1]) and scale the model into them by default (`auto_scale`).
//! The analog hardware resolves only a few bits of each coefficient after that scaling, so
//! large penalty weights drown the objective; [`penalty::tune`](crate::penalty::tune) finds
//! smaller ones.

mod base64;

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use core::fmt::Write as _;

use crate::qubo::ordered;
use crate::{Coefficient, Ising, Qubo};

/// The SAPI problem type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProblemType {
    /// Binary variables in {0, 1}.
    Qubo,
    /// Spins in {−1, +1}.
    Ising,
}

impl ProblemType {
    /// The SAPI name: `"qubo"` or `"ising"`.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Qubo => "qubo",
            Self::Ising => "ising",
        }
    }
}

/// The qubits and couplers of a QPU solver, from its `qubits` and `couplers` properties.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Solver<'a> {
    /// The working qubits, in the order of the solver's `qubits` property.
    pub qubits: &'a [usize],
    /// The working couplers, in the order of the solver's `couplers` property.
    pub couplers: &'a [(usize, usize)],
}

/// The solver as a submission names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverRef<'a> {
    /// The solver name only.
    Name(&'a str),
    /// The name with the solver graph, as SAPI version 3 expects for QPU solvers.
    Versioned {
        /// The solver name.
        name: &'a str,
        /// The `graph_id` of the solver's working graph.
        graph_id: &'a str,
    },
}

/// The error when a model cannot be exported for a solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExportError {
    /// The qubit map does not have one qubit per model variable.
    MapLength {
        /// The number of model variables.
        variables: usize,
        /// The number of mapped qubits.
        qubits: usize,
    },
    /// A variable maps to a qubit that the solver does not have.
    UnknownQubit {
        /// The variable.
        variable: usize,
        /// The qubit.
        qubit: usize,
    },
    /// Two variables map to the same qubit.
    SharedQubit {
        /// The qubit.
        qubit: usize,
    },
    /// A model pair has no coupler between its qubits.
    MissingCoupler {
        /// The first variable of the pair.
        i: usize,
        /// The second variable of the pair.
        j: usize,
    },
}

impl fmt::Display for ExportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MapLength { variables, qubits } => {
                write!(f, "{qubits} qubits are mapped for {variables} variables")
            }
            Self::UnknownQubit { variable, qubit } => {
                write!(
                    f,
                    "variable {variable} maps to qubit {qubit}, which the solver lacks"
                )
            }
            Self::SharedQubit { qubit } => write!(f, "two variables map to qubit {qubit}"),
            Self::MissingCoupler { i, j } => {
                write!(f, "the pair ({i}, {j}) has no coupler between its qubits")
            }
        }
    }
}

impl core::error::Error for ExportError {}

/// A model in the `qp` format, with the data to parse its answers.
#[derive(Debug, Clone, PartialEq)]
pub struct Exported {
    /// The problem type.
    problem_type: ProblemType,
    /// The base64 `lin` array.
    lin: String,
    /// The base64 `quad` array.
    quad: String,
    /// The offset in the model's units, added to the answer energies.
    offset: f64,
    /// The variable of each used qubit.
    variable_of: BTreeMap<usize, usize>,
}

/// Exports `qubo` as a `qubo` problem, with `qubits[i]` the solver qubit of variable `i`.
///
/// # Errors
///
/// Returns [`ExportError`] if the map is not one-to-one onto solver qubits, or a pair has no
/// coupler.
pub fn export_qubo<T: Coefficient>(
    qubo: &Qubo<T>,
    solver: &Solver<'_>,
    qubits: &[usize],
) -> Result<Exported, ExportError> {
    let linear: Vec<f64> = (0..qubo.num_variables())
        .map(|i| qubo.linear(i).to_f64())
        .collect();
    let pairs = qubo.quadratic_terms().map(|(i, j, b)| (i, j, b.to_f64()));

    export(
        ProblemType::Qubo,
        &linear,
        pairs,
        qubo.offset().to_f64(),
        solver,
        qubits,
    )
}

/// Exports `ising` as an `ising` problem, with `qubits[i]` the solver qubit of spin `i`.
///
/// All coefficients are divided by the model's scale, so the exported energies are those of
/// the encoded problem. For an `i64` model from [`Qubo::to_ising`], they are the QUBO
/// energies.
///
/// # Errors
///
/// Returns [`ExportError`] if the map is not one-to-one onto solver qubits, or a coupling
/// has no coupler.
pub fn export_ising<T: Coefficient>(
    ising: &Ising<T>,
    solver: &Solver<'_>,
    qubits: &[usize],
) -> Result<Exported, ExportError> {
    let scale = ising.scale().to_f64();
    let field: Vec<f64> = (0..ising.num_spins())
        .map(|i| ising.field(i).to_f64() / scale)
        .collect();
    let couplings = ising
        .couplings()
        .map(|(i, j, c)| (i, j, c.to_f64() / scale));

    export(
        ProblemType::Ising,
        &field,
        couplings,
        ising.offset().to_f64() / scale,
        solver,
        qubits,
    )
}

/// Builds the `qp` arrays from the linear terms and the pairs of a model.
fn export(
    problem_type: ProblemType,
    linear: &[f64],
    pairs: impl Iterator<Item = (usize, usize, f64)>,
    offset: f64,
    solver: &Solver<'_>,
    qubits: &[usize],
) -> Result<Exported, ExportError> {
    let variable_of = invert(qubits, linear.len(), solver)?;
    let coupler_index: BTreeMap<(usize, usize), usize> = solver
        .couplers
        .iter()
        .enumerate()
        .map(|(k, &(a, b))| (ordered(a, b), k))
        .collect();

    let mut coupler_values = vec![0.0; solver.couplers.len()];

    for (i, j, value) in pairs {
        let k = coupler_index
            .get(&ordered(qubits[i], qubits[j]))
            .ok_or(ExportError::MissingCoupler { i, j })?;
        coupler_values[*k] += value;
    }

    let lin: Vec<f64> = solver
        .qubits
        .iter()
        .map(|q| variable_of.get(q).map_or(f64::NAN, |&i| linear[i]))
        .collect();
    let used = |q: &usize| variable_of.contains_key(q);
    let quad: Vec<f64> = solver
        .couplers
        .iter()
        .zip(&coupler_values)
        .filter(|((a, b), _)| used(a) && used(b))
        .map(|(_, &v)| v)
        .collect();

    Ok(Exported {
        problem_type,
        lin: base64::encode(&doubles(&lin)),
        quad: base64::encode(&doubles(&quad)),
        offset,
        variable_of,
    })
}

/// Checks that `qubits` maps each of `variables` variables to its own solver qubit, and
/// returns the variable of each qubit.
fn invert(
    qubits: &[usize],
    variables: usize,
    solver: &Solver<'_>,
) -> Result<BTreeMap<usize, usize>, ExportError> {
    if qubits.len() != variables {
        return Err(ExportError::MapLength {
            variables,
            qubits: qubits.len(),
        });
    }

    let mut variable_of = BTreeMap::new();

    for (variable, &qubit) in qubits.iter().enumerate() {
        if !solver.qubits.contains(&qubit) {
            return Err(ExportError::UnknownQubit { variable, qubit });
        }

        if variable_of.insert(qubit, variable).is_some() {
            return Err(ExportError::SharedQubit { qubit });
        }
    }

    Ok(variable_of)
}

/// The little-endian bytes of `values`.
fn doubles(values: &[f64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The answer fields of a SAPI problem in the `qp` format, as the JSON response holds them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QpAnswer<'a> {
    /// `answer.active_variables`: base64 `i32` qubit indices.
    pub active_variables: &'a str,
    /// `answer.solutions`: base64 packed bits.
    pub solutions: &'a str,
    /// `answer.energies`: base64 `f64` energies, without the offset.
    pub energies: &'a str,
    /// `answer.num_occurrences`: base64 `i32` counts; absent in raw answer mode.
    pub num_occurrences: Option<&'a str>,
}

/// One sample of a parsed answer.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedSample {
    /// The value of each model variable, by index: `true` for 1, or for the spin +1.
    pub bits: Vec<bool>,
    /// The reported energy plus the model offset, in the units of the exported model.
    pub energy: f64,
    /// How often the solver returned this sample. 1 in raw answer mode.
    pub occurrences: u32,
}

/// The error when an answer does not fit the exported model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ParseError {
    /// A field is not valid base64, or its length is not a multiple of its element size.
    Malformed {
        /// The field name.
        field: &'static str,
    },
    /// The answer uses a qubit that the exported model does not.
    UnknownQubit {
        /// The qubit.
        qubit: i32,
    },
    /// The answer lacks a qubit that the exported model uses.
    MissingVariable {
        /// The model variable.
        variable: usize,
    },
    /// The fields disagree on the number of samples.
    CountMismatch,
    /// An occurrence count is negative.
    NegativeOccurrences,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Malformed { field } => write!(f, "the answer field {field} is malformed"),
            Self::UnknownQubit { qubit } => write!(f, "the answer uses the unknown qubit {qubit}"),
            Self::MissingVariable { variable } => {
                write!(f, "the answer has no value for variable {variable}")
            }
            Self::CountMismatch => f.write_str("the answer fields disagree on the sample count"),
            Self::NegativeOccurrences => f.write_str("an occurrence count is negative"),
        }
    }
}

impl core::error::Error for ParseError {}

impl Exported {
    /// The problem type.
    #[must_use]
    pub fn problem_type(&self) -> ProblemType {
        self.problem_type
    }

    /// The base64 `lin` array.
    #[must_use]
    pub fn lin(&self) -> &str {
        &self.lin
    }

    /// The base64 `quad` array.
    #[must_use]
    pub fn quad(&self) -> &str {
        &self.quad
    }

    /// The model offset, which [`Exported::parse_answer`] adds to the energies.
    #[must_use]
    pub fn offset(&self) -> f64 {
        self.offset
    }

    /// The `data` object: `{"format":"qp","lin":"…","quad":"…"}`.
    #[must_use]
    pub fn data_json(&self) -> String {
        format!(
            r#"{{"format":"qp","lin":"{}","quad":"{}"}}"#,
            self.lin, self.quad
        )
    }

    /// The body of `POST /problems`: a JSON array with this one problem.
    ///
    /// `params` is the JSON object of solver parameters, such as `{"num_reads":100}`, and is
    /// inserted as given; pass `{}` for the defaults.
    #[must_use]
    pub fn submission_json(&self, solver: SolverRef<'_>, params: &str) -> String {
        let solver = match solver {
            SolverRef::Name(name) => json_string(name),
            SolverRef::Versioned { name, graph_id } => format!(
                r#"{{"name":{},"version":{{"graph_id":{}}}}}"#,
                json_string(name),
                json_string(graph_id)
            ),
        };

        format!(
            r#"[{{"data":{},"params":{},"solver":{},"type":"{}"}}]"#,
            self.data_json(),
            params,
            solver,
            self.problem_type.name()
        )
    }

    /// Parses an answer into one sample per returned solution, in the answer's order.
    ///
    /// # Errors
    ///
    /// Returns [`ParseError`] if a field is malformed, the answer's qubits are not the
    /// exported ones, or the field lengths disagree.
    pub fn parse_answer(&self, answer: &QpAnswer<'_>) -> Result<Vec<ParsedSample>, ParseError> {
        let active = self.active_variables(answer.active_variables)?;
        let energies = decode_f64s(answer.energies)?;
        let occurrences = match answer.num_occurrences {
            Some(text) => decode_occurrences(text)?,
            None => vec![1; energies.len()],
        };
        let solutions = decode_field(answer.solutions, "solutions")?;
        let width = active.len().div_ceil(8);

        if occurrences.len() != energies.len() || solutions.len() != energies.len() * width {
            return Err(ParseError::CountMismatch);
        }

        let samples = energies
            .iter()
            .zip(&occurrences)
            .enumerate()
            .map(|(k, (&energy, &occurrences))| ParsedSample {
                bits: unpack(
                    &solutions[k * width..(k + 1) * width],
                    &active,
                    self.variable_of.len(),
                ),
                energy: energy + self.offset,
                occurrences,
            })
            .collect();

        Ok(samples)
    }

    /// The model variable of each active qubit, checked against the exported qubits.
    fn active_variables(&self, text: &str) -> Result<Vec<usize>, ParseError> {
        let qubits = decode_i32s(text, "active_variables")?;
        let variables = qubits
            .iter()
            .map(|&q| {
                usize::try_from(q)
                    .ok()
                    .and_then(|q| self.variable_of.get(&q).copied())
                    .ok_or(ParseError::UnknownQubit { qubit: q })
            })
            .collect::<Result<Vec<usize>, ParseError>>()?;

        let mut seen = vec![false; self.variable_of.len()];
        for &v in &variables {
            seen[v] = true;
        }

        match seen.iter().position(|&s| !s) {
            Some(variable) => Err(ParseError::MissingVariable { variable }),
            None => Ok(variables),
        }
    }
}

/// Unpacks one solution: the bit of `active[k]` is bit `7 − k % 8` of byte `k / 8`.
fn unpack(bytes: &[u8], active: &[usize], variables: usize) -> Vec<bool> {
    let mut bits = vec![false; variables];

    for (k, &variable) in active.iter().enumerate() {
        bits[variable] = bytes[k / 8] >> (7 - k % 8) & 1 == 1;
    }

    bits
}

/// Decodes a base64 field, naming it in the error.
fn decode_field(text: &str, field: &'static str) -> Result<Vec<u8>, ParseError> {
    base64::decode(text).ok_or(ParseError::Malformed { field })
}

/// Decodes base64 little-endian `f64` values.
fn decode_f64s(text: &str) -> Result<Vec<f64>, ParseError> {
    let bytes = decode_field(text, "energies")?;

    if bytes.len() % 8 != 0 {
        return Err(ParseError::Malformed { field: "energies" });
    }

    Ok(bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().expect("chunks of 8 bytes")))
        .collect())
}

/// Decodes base64 little-endian `i32` values.
fn decode_i32s(text: &str, field: &'static str) -> Result<Vec<i32>, ParseError> {
    let bytes = decode_field(text, field)?;

    if bytes.len() % 4 != 0 {
        return Err(ParseError::Malformed { field });
    }

    Ok(bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().expect("chunks of 4 bytes")))
        .collect())
}

/// Decodes the occurrence counts, which must not be negative.
fn decode_occurrences(text: &str) -> Result<Vec<u32>, ParseError> {
    decode_i32s(text, "num_occurrences")?
        .into_iter()
        .map(|n| u32::try_from(n).map_err(|_| ParseError::NegativeOccurrences))
        .collect()
}

/// `text` as a JSON string literal, with quotes, backslashes and control characters escaped.
fn json_string(text: &str) -> String {
    let mut out = String::from("\"");

    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if u32::from(c) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            c => out.push(c),
        }
    }

    out.push('"');
    out
}
