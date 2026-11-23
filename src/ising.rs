//! Sparse Ising models over spins, and the conversion back to QUBO.

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use crate::qubo::ordered;
use crate::{Coefficient, Qubo};

/// A sparse Ising model over spins `sᵢ ∈ {−1, +1}`.
///
/// The energy of a spin configuration is
///
/// ```text
/// E(s) = offset + Σᵢ hᵢ·sᵢ + Σ_{i<j} Jᵢⱼ·sᵢ·sⱼ
/// ```
///
/// Spins are passed as `bool`, with `true` for `+1` and `false` for `−1`. With this
/// encoding, one `&[bool]` is both a QUBO assignment `x` and the spins `s = 2x − 1`.
///
/// Storage is upper triangular, as in [`Qubo`]. The model also records a [`scale`]: the
/// factor between its energies and the energies of the problem it encodes. A model from
/// [`Qubo::to_ising`] has the scale [`Coefficient::ISING_SCALE`].
///
/// [`scale`]: Ising::scale
#[derive(Debug, Clone, PartialEq)]
pub struct Ising<T> {
    /// The constant energy term.
    offset: T,
    /// The field `hᵢ` of each spin. The length is the number of spins.
    field: Vec<T>,
    /// The couplings `Jᵢⱼ`, keyed by `(i, j)` with `i < j < field.len()`.
    coupling: BTreeMap<(usize, usize), T>,
    /// The factor between these energies and the encoded problem's energies. Never zero.
    scale: T,
}

/// The error when [`Ising::to_qubo`] cannot divide a coefficient by the scale exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InexactScale;

impl fmt::Display for InexactScale {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a QUBO coefficient is not an exact multiple of the Ising scale")
    }
}

impl core::error::Error for InexactScale {}

impl<T: Coefficient> Ising<T> {
    /// Creates a model with `spins` spins, all coefficients zero, and the given `scale`.
    ///
    /// Use the scale [`Coefficient::ONE`] for a model that you build by hand.
    ///
    /// # Panics
    ///
    /// Panics if `scale` is zero.
    #[must_use]
    pub fn new(spins: usize, scale: T) -> Self {
        assert!(scale != T::ZERO, "the Ising scale must not be zero");

        Self {
            offset: T::ZERO,
            field: vec![T::ZERO; spins],
            coupling: BTreeMap::new(),
            scale,
        }
    }

    /// The number of spins.
    #[must_use]
    pub fn num_spins(&self) -> usize {
        self.field.len()
    }

    /// The constant energy term.
    #[must_use]
    pub fn offset(&self) -> T {
        self.offset
    }

    /// The factor between these energies and the energies of the encoded problem.
    #[must_use]
    pub fn scale(&self) -> T {
        self.scale
    }

    /// The field `hᵢ` of spin `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i` is not a spin of the model.
    #[must_use]
    pub fn field(&self, i: usize) -> T {
        self.field[i]
    }

    /// The coupling of the pair `{i, j}`, in either order. Zero for absent pairs and for `i == j`.
    #[must_use]
    pub fn coupling(&self, i: usize, j: usize) -> T {
        self.coupling
            .get(&ordered(i, j))
            .copied()
            .unwrap_or(T::ZERO)
    }

    /// The stored couplings `(i, j, Jᵢⱼ)` with `i < j`, in ascending key order.
    pub fn couplings(&self) -> impl Iterator<Item = (usize, usize, T)> + '_ {
        self.coupling.iter().map(|(&(i, j), &c)| (i, j, c))
    }

    /// Adds `value` to the constant energy term.
    pub fn add_offset(&mut self, value: T) {
        self.offset = self.offset.plus(value);
    }

    /// Adds `value` to the field of spin `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i` is not a spin of the model.
    pub fn add_field(&mut self, i: usize, value: T) {
        let h = &mut self.field[i];
        *h = h.plus(value);
    }

    /// Adds `value` to the coupling of the pair `{i, j}`, in either order.
    ///
    /// For `i == j` the value goes to the offset, because `sᵢ·sᵢ = 1`.
    ///
    /// # Panics
    ///
    /// Panics if `i` or `j` is not a spin of the model.
    pub fn add_coupling(&mut self, i: usize, j: usize, value: T) {
        assert!(
            i.max(j) < self.num_spins(),
            "the pair ({i}, {j}) refers to a spin outside the model"
        );

        if i == j {
            self.add_offset(value);
            return;
        }

        let c = self.coupling.entry(ordered(i, j)).or_insert(T::ZERO);
        *c = c.plus(value);
    }

    /// The energy of the spins `s`, where `s[i]` is `true` for `+1` and `false` for `−1`.
    ///
    /// # Panics
    ///
    /// Panics if `s.len()` is not the number of spins.
    #[must_use]
    pub fn energy(&self, s: &[bool]) -> T {
        assert_eq!(
            s.len(),
            self.num_spins(),
            "the spin configuration length must equal the number of spins"
        );

        let field = self
            .field
            .iter()
            .zip(s)
            .fold(self.offset, |sum, (&h, &si)| signed_add(sum, h, si));

        self.couplings()
            .fold(field, |sum, (i, j, c)| signed_add(sum, c, s[i] == s[j]))
    }

    /// Converts the model back to a QUBO over `xᵢ = (1 + sᵢ)/2`, and removes the scale.
    ///
    /// The result satisfies `scale · qubo.energy(x) == ising.energy(x)` for each `x`. For a
    /// model from [`Qubo::to_ising`], the result equals the original QUBO.
    ///
    /// # Errors
    ///
    /// Returns [`InexactScale`] if a QUBO coefficient is not an exact multiple of the
    /// scale. This happens only for [`i64`] with a scale other than ±1.
    ///
    /// # Panics
    ///
    /// For [`i64`], panics if four times a coupling overflows.
    pub fn to_qubo(&self) -> Result<Qubo<T>, InexactScale> {
        let mut scaled = Qubo::new(self.num_spins());
        scaled.add_offset(self.offset);

        for (i, &h) in self.field.iter().enumerate() {
            add_field_to_qubo(&mut scaled, i, h);
        }

        for (i, j, c) in self.couplings() {
            add_coupling_to_qubo(&mut scaled, i, j, c);
        }

        scaled.try_map_coefficients(|v| v.exact_div(self.scale).ok_or(InexactScale))
    }

    /// Applies `f` to each coefficient and the offset. The scale stays the same.
    pub(crate) fn map_coefficients(mut self, f: impl Fn(T) -> T) -> Self {
        self.offset = f(self.offset);
        self.field.iter_mut().for_each(|h| *h = f(*h));
        self.coupling.values_mut().for_each(|c| *c = f(*c));

        self
    }
}

/// Returns `sum + value` if `positive`, else `sum − value`.
fn signed_add<T: Coefficient>(sum: T, value: T, positive: bool) -> T {
    if positive {
        sum.plus(value)
    } else {
        sum.minus(value)
    }
}

/// Adds the QUBO form of `h·sᵢ` to `qubo`: `h·(2xᵢ − 1) = 2h·xᵢ − h`.
fn add_field_to_qubo<T: Coefficient>(qubo: &mut Qubo<T>, i: usize, h: T) {
    qubo.add_linear(i, h.plus(h));
    qubo.add_offset(T::ZERO.minus(h));
}

/// Adds the QUBO form of `c·sᵢ·sⱼ` to `qubo`.
///
/// `c·(2xᵢ − 1)·(2xⱼ − 1) = 4c·xᵢ·xⱼ − 2c·xᵢ − 2c·xⱼ + c`.
fn add_coupling_to_qubo<T: Coefficient>(qubo: &mut Qubo<T>, i: usize, j: usize, c: T) {
    let twice = c.plus(c);
    qubo.add_quadratic(i, j, twice.plus(twice));
    qubo.add_linear(i, T::ZERO.minus(twice));
    qubo.add_linear(j, T::ZERO.minus(twice));
    qubo.add_offset(c);
}

/// Unit tests for [`Ising`] energies and the scale handling.
#[cfg(test)]
mod tests {
    use super::{InexactScale, Ising};
    use crate::Qubo;

    /// The energy uses `+1` for `true` and `−1` for `false`.
    #[test]
    fn energy_uses_signed_spins() {
        let mut ising = Ising::<i64>::new(2, 1);
        ising.add_offset(1);
        ising.add_field(0, 2);
        ising.add_coupling(0, 1, 3);

        assert_eq!(ising.energy(&[true, true]), 1 + 2 + 3);
        assert_eq!(ising.energy(&[true, false]), 1 + 2 - 3);
        assert_eq!(ising.energy(&[false, false]), 1 - 2 + 3);
    }

    /// A QUBO converts to an `i64` Ising model with scale 4 and back to itself.
    #[test]
    fn i64_round_trip_is_exact() {
        let mut qubo = Qubo::<i64>::new(2);
        qubo.add_offset(1);
        qubo.add_linear(0, 1);
        qubo.add_quadratic(0, 1, 1);

        let ising = qubo.to_ising();

        assert_eq!(ising.scale(), 4);
        assert_eq!(ising.to_qubo(), Ok(qubo));
    }

    /// A hand-built model with a scale that does not divide the QUBO terms fails.
    #[test]
    fn inexact_scale_is_an_error() {
        let mut ising = Ising::<i64>::new(1, 4);
        ising.add_field(0, 1);

        assert_eq!(ising.to_qubo(), Err(InexactScale));
    }
}
