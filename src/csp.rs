//! Constraint satisfaction: variables with finite domains and binary constraints.
//!
//! A [`Csp`] has named variables, each with the values `0..domain`, and named binary
//! constraints, each a relation that allows some value pairs of two variables. The encoding
//! is one-hot: one QUBO variable `{name}={value}` per variable and value, which is 1 when
//! the variable takes the value. The constraints are:
//!
//! - `{name}.exactly_one`: the variable takes exactly one value, with the penalty
//!   `P·(1 − Σ xᵥ)²`;
//! - the constraint's own name: each forbidden value pair `(a, b)` adds `P·x_a·x_b`.
//!
//! The model has no objective, so every penalty weight `P ≥ 1` works: an assignment that
//! satisfies everything has energy 0, and each violation costs at least `P`. The ground
//! state energy is 0 exactly when the problem is satisfiable. Constraints over more than two
//! variables need auxiliary variables and are not part of the encoding; write them as
//! pairwise constraints where possible, such as [`Csp::all_different`].
//!
//! ```
//! use planning_qubo::csp::Csp;
//! use planning_qubo::{Exhaustive, Sampler, Unlimited};
//!
//! // Colour a triangle with three colours.
//! let mut csp = Csp::new();
//! let vars: Vec<usize> = ["a", "b", "c"].iter().map(|n| csp.add_variable(n, 3)).collect();
//! csp.all_different("colours", &vars);
//!
//! let encoding = csp.encode(1_i64);
//! let mut sampler = Exhaustive::new();
//! let run = sampler.sample(encoding.problem().qubo(), Unlimited)?;
//! let decoded = encoding.decode(run.sample.bits);
//!
//! assert_eq!(run.sample.energy, 0);
//! assert!(decoded.is_feasible());
//! assert_eq!(decoded.values.iter().flatten().count(), 3);
//! # Ok::<(), planning_qubo::SampleError>(())
//! ```

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::{Builder, Coefficient, Encoded, Problem, Var};

/// A constraint satisfaction problem over finite domains with binary constraints.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Csp {
    /// The name of each variable, by index.
    names: Vec<String>,
    /// The domain size of each variable: its values are `0..domain`.
    domains: Vec<usize>,
    /// The binary constraints, in the order they were added.
    relations: Vec<Relation>,
}

/// A named binary constraint, stored as its forbidden value pairs.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Relation {
    /// The name that decoders report.
    name: String,
    /// The first variable.
    first: usize,
    /// The second variable.
    second: usize,
    /// The value pairs `(a, b)` that the constraint forbids for `(first, second)`.
    forbidden: Vec<(usize, usize)>,
}

impl Csp {
    /// Creates a problem with no variables.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a variable with the values `0..domain` and returns its index.
    ///
    /// A variable with an empty domain can never take a value, so its
    /// `{name}.exactly_one` constraint is always violated.
    pub fn add_variable(&mut self, name: &str, domain: usize) -> usize {
        self.names.push(String::from(name));
        self.domains.push(domain);

        self.names.len() - 1
    }

    /// The number of variables.
    #[must_use]
    pub fn num_variables(&self) -> usize {
        self.names.len()
    }

    /// The domain size of variable `var`.
    ///
    /// # Panics
    ///
    /// Panics if `var` is not a variable of the problem.
    #[must_use]
    pub fn domain(&self, var: usize) -> usize {
        self.domains[var]
    }

    /// Adds the constraint `name`: `(first, second)` may take `(a, b)` only if `allowed(a, b)`.
    ///
    /// A constraint of a variable with itself sees the same value twice, so it restricts
    /// the variable's own values: `allowed(a, a) == false` forbids `a`.
    ///
    /// # Panics
    ///
    /// Panics if `first` or `second` is not a variable of the problem.
    pub fn add_constraint(
        &mut self,
        name: &str,
        first: usize,
        second: usize,
        allowed: impl Fn(usize, usize) -> bool,
    ) {
        let (n, m) = (self.domains[first], self.domains[second]);
        let forbidden = (0..n)
            .flat_map(|a| (0..m).map(move |b| (a, b)))
            .filter(|&(a, b)| (first != second || a == b) && !allowed(a, b))
            .collect();

        self.relations.push(Relation {
            name: String::from(name),
            first,
            second,
            forbidden,
        });
    }

    /// Adds the constraint `name`: the variables `vars` take pairwise different values.
    ///
    /// # Panics
    ///
    /// Panics if a variable is not a variable of the problem.
    pub fn all_different(&mut self, name: &str, vars: &[usize]) {
        for (k, &first) in vars.iter().enumerate() {
            for &second in &vars[k + 1..] {
                self.add_constraint(name, first, second, |a, b| a != b);
            }
        }
    }

    /// Returns `true` if `values` (one value per variable) satisfies every constraint.
    ///
    /// # Panics
    ///
    /// Panics if `values.len()` is not the number of variables.
    #[must_use]
    pub fn is_solution(&self, values: &[usize]) -> bool {
        assert_eq!(values.len(), self.num_variables(), "one value per variable");

        values.iter().zip(&self.domains).all(|(&v, &d)| v < d)
            && self.relations.iter().all(|r| !r.forbids(values))
    }

    /// Encodes the problem with the penalty weight `penalty` for every constraint.
    #[must_use]
    pub fn encode<T: Coefficient>(&self, penalty: T) -> Encoding<T> {
        let mut builder = Builder::new();
        let values: Vec<Vec<Var>> = self
            .names
            .iter()
            .zip(&self.domains)
            .map(|(name, &domain)| {
                (0..domain)
                    .map(|v| builder.fresh_variable(&format!("{name}={v}")))
                    .collect()
            })
            .collect();

        for (name, vars) in self.names.iter().zip(&values) {
            builder.exactly_one(&format!("{name}.exactly_one"), vars, penalty);
        }

        for relation in &self.relations {
            for &(a, b) in &relation.forbidden {
                let pair = [values[relation.first][a], values[relation.second][b]];
                builder.at_most_one(&relation.name, &pair, penalty);
            }
        }

        Encoding {
            problem: builder.build(),
            values,
        }
    }
}

impl Relation {
    /// Returns `true` if `values` gives the two variables a forbidden pair.
    fn forbids(&self, values: &[usize]) -> bool {
        self.forbidden
            .contains(&(values[self.first], values[self.second]))
    }
}

/// An encoded [`Csp`]: the QUBO problem and the data to decode samples.
#[derive(Debug, Clone, PartialEq)]
pub struct Encoding<T> {
    /// The QUBO problem with its constraints.
    problem: Problem<T>,
    /// The one-hot variables of each CSP variable, by value.
    values: Vec<Vec<Var>>,
}

/// A sample, decoded into a value per variable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decoded {
    /// The value of each variable. `None` if the sample gives it no value or more than one.
    pub values: Vec<Option<usize>>,
    /// The names of the violated constraints, in the order they were added; the
    /// `{name}.exactly_one` constraints come first.
    pub violations: Vec<String>,
}

impl Decoded {
    /// Returns `true` if the sample violates no constraint.
    #[must_use]
    pub fn is_feasible(&self) -> bool {
        self.violations.is_empty()
    }
}

impl<T> Encoded<T> for Encoding<T> {
    fn problem(&self) -> &Problem<T> {
        &self.problem
    }
}

impl<T: Coefficient> Encoding<T> {
    /// The QUBO problem with its constraints.
    #[must_use]
    pub fn problem(&self) -> &Problem<T> {
        &self.problem
    }

    /// The QUBO variable that is 1 when variable `var` takes `value`.
    ///
    /// # Panics
    ///
    /// Panics if `var` or `value` is out of range.
    #[must_use]
    pub fn variable(&self, var: usize, value: usize) -> Var {
        self.values[var][value]
    }

    /// Decodes the assignment `bits` of all QUBO variables.
    ///
    /// # Panics
    ///
    /// Panics if `bits.len()` is not the number of QUBO variables.
    #[must_use]
    pub fn decode(&self, bits: &[bool]) -> Decoded {
        let violations = self.problem.violation_names(bits);
        let values = self.values.iter().map(|vars| one_hot(vars, bits)).collect();

        Decoded { values, violations }
    }
}

/// The index of the only set variable of `vars`, or `None` if none or several are set.
pub(crate) fn one_hot(vars: &[Var], bits: &[bool]) -> Option<usize> {
    let mut set = vars.iter().enumerate().filter(|(_, v)| bits[v.index()]);
    let first = set.next().map(|(i, _)| i);

    if set.next().is_some() { None } else { first }
}
