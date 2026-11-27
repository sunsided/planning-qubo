//! A builder for QUBO models over named variables, with penalty terms for constraints.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::{Coefficient, Qubo};

/// A binary variable of a [`Builder`] or [`Problem`]: the index into the QUBO model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Var(usize);

impl Var {
    /// The index of the variable in the QUBO model and in samples.
    #[must_use]
    pub fn index(self) -> usize {
        self.0
    }
}

/// A named constraint, kept next to the model so that decoders can check samples.
#[derive(Debug, Clone, PartialEq)]
pub struct Constraint<T> {
    /// The name that [`Problem::violations`] reports.
    pub name: String,
    /// What the constraint requires.
    pub kind: ConstraintKind<T>,
}

/// What a [`Constraint`] requires of the original variables. Slack variables are not part of it.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintKind<T> {
    /// Exactly one of the variables is 1. A variable that occurs twice counts twice.
    ExactlyOne(Vec<Var>),
    /// At most one of the variables is 1. A variable that occurs twice counts twice.
    AtMostOne(Vec<Var>),
    /// `rhs − slack_max ≤ Σ aᵢ·xᵢ ≤ rhs`, with the terms `(xᵢ, aᵢ)`.
    ///
    /// With `slack_max == 0` this is the equality `Σ aᵢ·xᵢ = rhs`.
    Linear {
        /// The terms `(xᵢ, aᵢ)` of the left side.
        terms: Vec<(Var, T)>,
        /// The right side.
        rhs: T,
        /// The largest value of the slack, which the slack bits encode.
        slack_max: u32,
    },
}

impl<T: Coefficient> Constraint<T> {
    /// Returns `true` if the assignment `x` satisfies the constraint.
    ///
    /// # Panics
    ///
    /// Panics if a variable of the constraint is not in `x`.
    #[must_use]
    pub fn is_satisfied(&self, x: &[bool]) -> bool {
        match &self.kind {
            ConstraintKind::ExactlyOne(vars) => count_set(vars, x) == 1,
            ConstraintKind::AtMostOne(vars) => count_set(vars, x) <= 1,
            ConstraintKind::Linear {
                terms,
                rhs,
                slack_max,
            } => {
                let lhs = linear_sum(terms, x);
                *rhs >= lhs && lhs >= rhs.minus(T::from_u32(*slack_max))
            }
        }
    }
}

/// The number of variables in `vars` that are 1 in `x`.
fn count_set(vars: &[Var], x: &[bool]) -> usize {
    vars.iter().filter(|v| x[v.index()]).count()
}

/// The sum `Σ aᵢ·xᵢ` over `terms`.
fn linear_sum<T: Coefficient>(terms: &[(Var, T)], x: &[bool]) -> T {
    terms
        .iter()
        .filter(|(v, _)| x[v.index()])
        .fold(T::ZERO, |sum, &(_, a)| sum.plus(a))
}

/// Builds a QUBO model over named variables from objective terms and penalized constraints.
///
/// Each constraint adds a penalty with a caller-chosen weight `P`. The penalty is 0 when
/// the constraint holds (for the best slack value) and at least `P` otherwise, if all
/// coefficients are integers. The builder does not tune the weights.
#[derive(Debug, Clone)]
pub struct Builder<T> {
    /// The variable names, in index order.
    names: Vec<String>,
    /// The variable of each name.
    lookup: BTreeMap<String, Var>,
    /// The constant energy term.
    offset: T,
    /// The terms `(i, j, value)`; `i == j` is a linear term.
    terms: Vec<(usize, usize, T)>,
    /// The constraints, in the order they were added.
    constraints: Vec<Constraint<T>>,
}

impl<T: Coefficient> Default for Builder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Coefficient> Builder<T> {
    /// Creates a builder with no variables.
    #[must_use]
    pub fn new() -> Self {
        Self {
            names: Vec::new(),
            lookup: BTreeMap::new(),
            offset: T::ZERO,
            terms: Vec::new(),
            constraints: Vec::new(),
        }
    }

    /// Returns the variable called `name`, and creates it if it does not exist.
    pub fn variable(&mut self, name: &str) -> Var {
        if let Some(&var) = self.lookup.get(name) {
            return var;
        }

        let var = Var(self.names.len());
        self.names.push(String::from(name));
        self.lookup.insert(String::from(name), var);

        var
    }

    /// Creates a new variable called `name`, also if the name exists.
    ///
    /// [`Builder::variable`] and [`Problem::variable`] find the first variable of a name.
    /// Slack bits use this, so constraints with the same name never share slack bits, and
    /// encoders use it for names that the caller chooses.
    pub(crate) fn fresh_variable(&mut self, name: &str) -> Var {
        let var = Var(self.names.len());
        self.names.push(String::from(name));
        self.lookup.entry(String::from(name)).or_insert(var);

        var
    }

    /// Adds `value` to the constant objective term.
    pub fn add_offset(&mut self, value: T) {
        self.offset = self.offset.plus(value);
    }

    /// Adds the objective term `value·x`.
    pub fn add_linear(&mut self, x: Var, value: T) {
        self.terms.push((x.index(), x.index(), value));
    }

    /// Adds the objective term `value·x·y`. For `x == y` this is `value·x`.
    pub fn add_quadratic(&mut self, x: Var, y: Var, value: T) {
        self.terms.push((x.index(), y.index(), value));
    }

    /// Adds the constraint "exactly one of `vars` is 1" with the penalty `P·(1 − Σ xᵢ)²`.
    pub fn exactly_one(&mut self, name: &str, vars: &[Var], penalty: T) {
        let terms: Vec<(Var, T)> = vars.iter().map(|&v| (v, T::ONE)).collect();
        self.add_squared_penalty(&terms, T::ONE, penalty);
        self.push_constraint(name, ConstraintKind::ExactlyOne(vars.to_vec()));
    }

    /// Adds the constraint "at most one of `vars` is 1" with the penalty `P·Σ_{i<j} xᵢ·xⱼ`.
    pub fn at_most_one(&mut self, name: &str, vars: &[Var], penalty: T) {
        for (k, &x) in vars.iter().enumerate() {
            for &y in &vars[k + 1..] {
                self.add_quadratic(x, y, penalty);
            }
        }

        self.push_constraint(name, ConstraintKind::AtMostOne(vars.to_vec()));
    }

    /// Adds the constraint `rhs − slack_max ≤ Σ aᵢ·xᵢ ≤ rhs` over `terms = [(xᵢ, aᵢ)]`.
    ///
    /// The constraint becomes the equality `Σ aᵢ·xᵢ + s = rhs` with a slack
    /// `s ∈ [0, slack_max]`, and the penalty `P·(Σ aᵢ·xᵢ + s − rhs)²`. The slack has
    /// binary digits with the weights 1, 2, 4, … and a last weight that makes the total
    /// exactly `slack_max`, so no slack value above `slack_max` exists. The slack variables
    /// are called `"{name}.slack{k}"`, and are always new variables: two constraints with
    /// the same name do not share slack bits. With `slack_max == 0` there is no slack.
    ///
    /// For an upper bound `Σ aᵢ·xᵢ ≤ rhs` with non-negative `aᵢ`, use `slack_max = rhs`.
    pub fn linear_equality(
        &mut self,
        name: &str,
        terms: &[(Var, T)],
        rhs: T,
        slack_max: u32,
        penalty: T,
    ) {
        let mut with_slack = terms.to_vec();

        for (k, weight) in slack_weights(slack_max).enumerate() {
            let slack = self.fresh_variable(&format!("{name}.slack{k}"));
            with_slack.push((slack, T::from_u32(weight)));
        }

        self.add_squared_penalty(&with_slack, rhs, penalty);
        self.push_constraint(
            name,
            ConstraintKind::Linear {
                terms: terms.to_vec(),
                rhs,
                slack_max,
            },
        );
    }

    /// Builds the QUBO model and keeps the names and constraints next to it.
    #[must_use]
    pub fn build(self) -> Problem<T> {
        let mut qubo = Qubo::new(self.names.len());
        qubo.add_offset(self.offset);

        for (i, j, value) in self.terms {
            qubo.add_quadratic(i, j, value);
        }

        Problem {
            qubo,
            names: self.names,
            lookup: self.lookup,
            constraints: self.constraints,
        }
    }

    /// Adds `P·(Σ aᵢ·yᵢ − rhs)²` over `terms = [(yᵢ, aᵢ)]`, with `yᵢ·yᵢ = yᵢ`.
    ///
    /// The expansion is `P·(Σᵢ aᵢ²·yᵢ + 2·Σ_{i<j} aᵢ·aⱼ·yᵢ·yⱼ − 2·rhs·Σᵢ aᵢ·yᵢ + rhs²)`.
    /// A variable that occurs twice gets its pair term on the diagonal.
    fn add_squared_penalty(&mut self, terms: &[(Var, T)], rhs: T, penalty: T) {
        self.add_offset(penalty.times(rhs).times(rhs));

        for (k, &(y, a)) in terms.iter().enumerate() {
            let diagonal = a.minus(rhs.plus(rhs)).times(a);
            self.add_linear(y, penalty.times(diagonal));

            for &(z, b) in &terms[k + 1..] {
                let pair = a.times(b);
                self.add_quadratic(y, z, penalty.times(pair.plus(pair)));
            }
        }
    }

    /// Records a constraint for the decoder.
    fn push_constraint(&mut self, name: &str, kind: ConstraintKind<T>) {
        self.constraints.push(Constraint {
            name: String::from(name),
            kind,
        });
    }
}

/// The slack digit weights for a slack in `[0, max]`: 1, 2, 4, …, then the remainder.
///
/// The weights sum to `max`, and each value in `[0, max]` is a subset sum. The number of
/// weights is `⌊log₂ max⌋ + 1` for `max > 0`, and 0 for `max == 0`.
fn slack_weights(max: u32) -> impl Iterator<Item = u32> {
    let mut remaining = max;
    let mut next = 1_u32;

    core::iter::from_fn(move || {
        if remaining == 0 {
            return None;
        }

        let weight = next.min(remaining);
        remaining -= weight;
        next = next.saturating_mul(2);

        Some(weight)
    })
}

/// An encoded problem: anything that holds a [`Problem`], such as an encoder's encoding.
///
/// The [`penalty`](crate::penalty) functions use it to read the model and to check samples
/// against the constraints, for every encoder.
pub trait Encoded<T> {
    /// The QUBO problem with its constraints.
    fn problem(&self) -> &Problem<T>;
}

impl<T> Encoded<T> for Problem<T> {
    fn problem(&self) -> &Problem<T> {
        self
    }
}

/// A QUBO model built by a [`Builder`], with its variable names and constraints.
#[derive(Debug, Clone, PartialEq)]
pub struct Problem<T> {
    /// The model: objective plus penalties.
    qubo: Qubo<T>,
    /// The variable names, in index order.
    names: Vec<String>,
    /// The variable of each name.
    lookup: BTreeMap<String, Var>,
    /// The constraints, in the order they were added.
    constraints: Vec<Constraint<T>>,
}

impl<T: Coefficient> Problem<T> {
    /// The QUBO model: objective plus penalties.
    #[must_use]
    pub fn qubo(&self) -> &Qubo<T> {
        &self.qubo
    }

    /// The variable called `name`, if it exists.
    #[must_use]
    pub fn variable(&self, name: &str) -> Option<Var> {
        self.lookup.get(name).copied()
    }

    /// The name of the variable `var`.
    ///
    /// # Panics
    ///
    /// Panics if `var` is not from the builder of this problem.
    #[must_use]
    pub fn name(&self, var: Var) -> &str {
        &self.names[var.index()]
    }

    /// The constraints, in the order they were added.
    #[must_use]
    pub fn constraints(&self) -> &[Constraint<T>] {
        &self.constraints
    }

    /// The constraints that the assignment `x` does not satisfy, in the order they were added.
    ///
    /// The check uses the original variables only, not the penalty energy or slack bits.
    ///
    /// # Panics
    ///
    /// Panics if `x.len()` is not the number of variables.
    pub fn violations<'a>(&'a self, x: &'a [bool]) -> impl Iterator<Item = &'a Constraint<T>> {
        assert_eq!(
            x.len(),
            self.qubo.num_variables(),
            "the assignment length must equal the number of variables"
        );

        self.constraints.iter().filter(|c| !c.is_satisfied(x))
    }

    /// The names of the violated constraints, in the order they were added.
    ///
    /// Encoders add some constraints as several pieces with one name, one after the other;
    /// such a name appears once.
    ///
    /// # Panics
    ///
    /// Panics if `x.len()` is not the number of variables.
    #[must_use]
    pub fn violation_names(&self, x: &[bool]) -> Vec<String> {
        let mut names: Vec<String> = self.violations(x).map(|c| c.name.clone()).collect();
        names.dedup();

        names
    }
}

/// Unit tests for the penalty terms of each constraint type.
#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::{Builder, Var, slack_weights};

    /// All assignments of `n` variables, as bit vectors.
    fn assignments(n: usize) -> impl Iterator<Item = Vec<bool>> {
        (0..1_u32 << n).map(move |bits| (0..n).map(|i| bits >> i & 1 == 1).collect())
    }

    /// Checks the penalty contract over all assignments of the first `original` variables.
    ///
    /// For each assignment, the minimum energy over the slack bits must be 0 if all
    /// constraints hold, and at least `penalty` otherwise.
    fn assert_penalty_contract(builder: Builder<i64>, original: usize, penalty: i64) {
        let problem = builder.build();
        let n = problem.qubo().num_variables();

        for x in assignments(original) {
            let best = assignments(n - original)
                .map(|slack| problem.qubo().energy(&[x.clone(), slack].concat()))
                .min()
                .expect("there is at least one slack assignment");

            let padded = [x.clone(), alloc::vec![false; n - original]].concat();
            let feasible = problem.violations(&padded).next().is_none();

            if feasible {
                assert_eq!(best, 0, "a feasible assignment {x:?} has a penalty");
            } else {
                assert!(
                    best >= penalty,
                    "an infeasible assignment {x:?} is too cheap"
                );
            }
        }
    }

    /// Creates a builder with the variables `x0` to `x{n-1}`.
    fn builder_with(n: usize) -> (Builder<i64>, Vec<Var>) {
        let mut builder = Builder::new();
        let vars = (0..n)
            .map(|i| builder.variable(&alloc::format!("x{i}")))
            .collect();

        (builder, vars)
    }

    /// Exactly-one penalizes zero and two or more set variables.
    #[test]
    fn exactly_one_contract() {
        let (mut builder, vars) = builder_with(4);
        builder.exactly_one("one", &vars, 3);

        assert_penalty_contract(builder, 4, 3);
    }

    /// At-most-one penalizes two or more set variables.
    #[test]
    fn at_most_one_contract() {
        let (mut builder, vars) = builder_with(4);
        builder.at_most_one("most", &vars, 3);

        assert_penalty_contract(builder, 4, 3);
    }

    /// A linear constraint with slack and a repeated variable keeps the contract.
    #[test]
    fn linear_with_slack_contract() {
        for slack_max in 0..=6 {
            let (mut builder, vars) = builder_with(3);
            let terms = [(vars[0], 2), (vars[1], 3), (vars[2], 1), (vars[0], 1)];
            builder.linear_equality("cap", &terms, 5, slack_max, 2);

            assert_penalty_contract(builder, 3, 2);
        }
    }

    /// Slack weights sum to the maximum and never exceed the next power of two.
    #[test]
    fn slack_weights_cover_the_range() {
        for max in 0..100 {
            let weights: Vec<u32> = slack_weights(max).collect();
            assert_eq!(weights.iter().sum::<u32>(), max);
            assert!(weights.iter().all(|&w| w > 0));
        }

        assert_eq!(slack_weights(5).collect::<Vec<_>>(), [1, 2, 2]);
    }

    /// Two constraints with the same name get separate slack bits.
    #[test]
    fn same_name_constraints_do_not_share_slack() {
        let (mut builder, vars) = builder_with(2);
        builder.linear_equality("cap", &[(vars[0], 1)], 1, 1, 2);
        builder.linear_equality("cap", &[(vars[1], 1)], 1, 1, 2);

        let problem = builder.build();

        assert_eq!(problem.qubo().num_variables(), 4);
        assert_eq!(problem.variable("cap.slack0").map(Var::index), Some(2));
    }

    /// The same name returns the same variable.
    #[test]
    fn variables_are_named_once() {
        let mut builder = Builder::<i64>::new();
        let a = builder.variable("a");
        let b = builder.variable("b");

        assert_eq!(builder.variable("a"), a);
        assert_ne!(a, b);

        let problem = builder.build();
        assert_eq!(problem.name(b), "b");
        assert_eq!(problem.variable("b"), Some(b));
        assert_eq!(problem.variable("c"), None);
    }
}
