//! Sparse QUBO models over binary variables.

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use crate::{Coefficient, Ising};

/// A sparse quadratic unconstrained binary optimization (QUBO) model.
///
/// The energy of an assignment `x ∈ {0, 1}ⁿ` is
///
/// ```text
/// E(x) = offset + Σᵢ aᵢ·xᵢ + Σ_{i<j} bᵢⱼ·xᵢ·xⱼ
/// ```
///
/// Storage is upper triangular: the linear terms `aᵢ` (the diagonal) are a dense vector,
/// and each unordered pair `{i, j}` has one entry `bᵢⱼ` under the key `(i, j)` with
/// `i < j`. A term added for `(j, i)` goes to `(i, j)`, and a term added for `(i, i)` goes
/// to `aᵢ`, because `xᵢ·xᵢ = xᵢ`. A symmetric matrix `Q` maps to this form with
/// `aᵢ = Qᵢᵢ` and `bᵢⱼ = Qᵢⱼ + Qⱼᵢ`.
///
/// The model has no interior mutability, so one instance serves many threads.
#[derive(Debug, Clone, PartialEq)]
pub struct Qubo<T> {
    /// The constant energy term.
    offset: T,
    /// The linear coefficient of each variable. The length is the number of variables.
    linear: Vec<T>,
    /// The pair coefficients, keyed by `(i, j)` with `i < j < linear.len()`.
    quadratic: BTreeMap<(usize, usize), T>,
}

impl<T: Coefficient> Qubo<T> {
    /// Creates a model with `variables` binary variables and all coefficients zero.
    #[must_use]
    pub fn new(variables: usize) -> Self {
        Self {
            offset: T::ZERO,
            linear: vec![T::ZERO; variables],
            quadratic: BTreeMap::new(),
        }
    }

    /// The number of binary variables.
    #[must_use]
    pub fn num_variables(&self) -> usize {
        self.linear.len()
    }

    /// The constant energy term.
    #[must_use]
    pub fn offset(&self) -> T {
        self.offset
    }

    /// The linear coefficient of variable `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i` is not a variable of the model.
    #[must_use]
    pub fn linear(&self, i: usize) -> T {
        self.linear[i]
    }

    /// The coefficient of the pair `{i, j}`, in either order. Zero for absent pairs.
    ///
    /// For `i == j` this is the linear coefficient, because `xᵢ·xᵢ = xᵢ`.
    ///
    /// # Panics
    ///
    /// Panics if `i == j` and `i` is not a variable of the model.
    #[must_use]
    pub fn quadratic(&self, i: usize, j: usize) -> T {
        if i == j {
            return self.linear(i);
        }

        self.quadratic
            .get(&ordered(i, j))
            .copied()
            .unwrap_or(T::ZERO)
    }

    /// The stored pair terms `(i, j, bᵢⱼ)` with `i < j`, in ascending key order.
    pub fn quadratic_terms(&self) -> impl Iterator<Item = (usize, usize, T)> + '_ {
        self.quadratic.iter().map(|(&(i, j), &b)| (i, j, b))
    }

    /// Adds `value` to the constant energy term.
    pub fn add_offset(&mut self, value: T) {
        self.offset = self.offset.plus(value);
    }

    /// Adds `value` to the linear coefficient of variable `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i` is not a variable of the model.
    pub fn add_linear(&mut self, i: usize, value: T) {
        let a = &mut self.linear[i];
        *a = a.plus(value);
    }

    /// Adds `value` to the coefficient of the pair `{i, j}`, in either order.
    ///
    /// For `i == j` the value goes to the linear coefficient, because `xᵢ·xᵢ = xᵢ`.
    ///
    /// # Panics
    ///
    /// Panics if `i` or `j` is not a variable of the model.
    pub fn add_quadratic(&mut self, i: usize, j: usize, value: T) {
        if i == j {
            self.add_linear(i, value);
            return;
        }

        assert!(
            i.max(j) < self.num_variables(),
            "the pair ({i}, {j}) refers to a variable outside the model"
        );

        let b = self.quadratic.entry(ordered(i, j)).or_insert(T::ZERO);
        *b = b.plus(value);
    }

    /// The energy of the assignment `x`, where `x[i]` is the value of variable `i`.
    ///
    /// # Panics
    ///
    /// Panics if `x.len()` is not the number of variables.
    #[must_use]
    pub fn energy(&self, x: &[bool]) -> T {
        assert_eq!(
            x.len(),
            self.num_variables(),
            "the assignment length must equal the number of variables"
        );

        let linear = self
            .linear
            .iter()
            .zip(x)
            .filter(|&(_, &xi)| xi)
            .fold(self.offset, |sum, (&a, _)| sum.plus(a));

        self.quadratic_terms()
            .filter(|&(i, j, _)| x[i] && x[j])
            .fold(linear, |sum, (_, _, b)| sum.plus(b))
    }

    /// Converts the model to an Ising model with spins `sᵢ = 2·xᵢ − 1`.
    ///
    /// The Ising energies are the QUBO energies times [`Coefficient::ISING_SCALE`], which
    /// [`Ising::scale`] records: `ising.energy(x) == scale · qubo.energy(x)` for each
    /// assignment `x`. For [`i64`] the scale is 4, which makes the conversion exact.
    ///
    /// # Panics
    ///
    /// For [`i64`], panics if four times a coefficient sum overflows.
    #[must_use]
    pub fn to_ising(&self) -> Ising<T> {
        let mut quadruple = Ising::new(self.num_variables(), T::ISING_SCALE);
        quadruple.add_offset(self.offset.times(four()));

        for (i, &a) in self.linear.iter().enumerate() {
            add_linear_to_quadruple(&mut quadruple, i, a);
        }

        for (i, j, b) in self.quadratic_terms() {
            add_pair_to_quadruple(&mut quadruple, i, j, b);
        }

        quadruple.map_coefficients(T::from_quadruple)
    }

    /// Applies `f` to each coefficient and the offset, and stops at the first error.
    pub(crate) fn try_map_coefficients<E>(
        mut self,
        f: impl Fn(T) -> Result<T, E>,
    ) -> Result<Self, E> {
        self.offset = f(self.offset)?;

        for a in &mut self.linear {
            *a = f(*a)?;
        }

        for b in self.quadratic.values_mut() {
            *b = f(*b)?;
        }

        Ok(self)
    }
}

/// Adds four times the Ising form of `a·xᵢ` to `quadruple`.
///
/// `4·a·xᵢ = 4·a·(1 + sᵢ)/2 = 2a + 2a·sᵢ`.
fn add_linear_to_quadruple<T: Coefficient>(quadruple: &mut Ising<T>, i: usize, a: T) {
    let twice = a.plus(a);
    quadruple.add_offset(twice);
    quadruple.add_field(i, twice);
}

/// Adds four times the Ising form of `b·xᵢ·xⱼ` to `quadruple`.
///
/// `4·b·xᵢ·xⱼ = b·(1 + sᵢ)·(1 + sⱼ) = b + b·sᵢ + b·sⱼ + b·sᵢ·sⱼ`.
fn add_pair_to_quadruple<T: Coefficient>(quadruple: &mut Ising<T>, i: usize, j: usize, b: T) {
    quadruple.add_offset(b);
    quadruple.add_field(i, b);
    quadruple.add_field(j, b);
    quadruple.add_coupling(i, j, b);
}

/// The number 4 as a coefficient.
fn four<T: Coefficient>() -> T {
    T::from_u32(4)
}

/// The pair `(i, j)` in ascending order.
pub(crate) fn ordered(i: usize, j: usize) -> (usize, usize) {
    if i < j { (i, j) } else { (j, i) }
}

/// Unit tests for [`Qubo`] storage and energy.
#[cfg(test)]
mod tests {
    use super::Qubo;

    /// Terms for `(j, i)` and `(i, i)` go to the upper-triangular slots.
    #[test]
    fn terms_normalize_to_upper_triangle() {
        let mut qubo = Qubo::<i64>::new(3);
        qubo.add_quadratic(2, 0, 5);
        qubo.add_quadratic(0, 2, 1);
        qubo.add_quadratic(1, 1, -3);

        assert_eq!(qubo.quadratic(0, 2), 6);
        assert_eq!(qubo.quadratic(2, 0), 6);
        assert_eq!(qubo.linear(1), -3);
        assert_eq!(qubo.quadratic_terms().count(), 1);
    }

    /// The energy sums the offset, the active linear terms and the active pairs.
    #[test]
    fn energy_sums_active_terms() {
        let mut qubo = Qubo::<i64>::new(3);
        qubo.add_offset(10);
        qubo.add_linear(0, -2);
        qubo.add_linear(1, -3);
        qubo.add_quadratic(0, 1, 5);

        assert_eq!(qubo.energy(&[false, false, false]), 10);
        assert_eq!(qubo.energy(&[true, false, true]), 8);
        assert_eq!(qubo.energy(&[true, true, false]), 10);
    }

    /// A pair outside the model panics.
    #[test]
    #[should_panic(expected = "outside the model")]
    fn pair_outside_the_model_panics() {
        Qubo::<i64>::new(2).add_quadratic(0, 2, 1);
    }
}
