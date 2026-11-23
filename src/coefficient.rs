//! The numeric type of model coefficients and energies.

use core::fmt::Debug;

/// A coefficient type for QUBO and Ising models.
///
/// Two implementations exist:
///
/// - [`i64`] is exact. Use it for penalties and for brute-force checks. Arithmetic is
///   checked: an overflow panics in all build profiles and never wraps.
/// - [`f64`] follows IEEE 754 rounding. Use it for annealer export and QAOA. NaN
///   coefficients are not supported: samplers compare energies with `<`, and a NaN energy
///   is never lower than another energy.
///
/// The arithmetic is in named methods, not in `core::ops`, because the overflow policy for
/// [`i64`] must not depend on the build profile.
pub trait Coefficient: Copy + PartialOrd + Debug + Send + Sync + 'static {
    /// The additive identity.
    const ZERO: Self;

    /// The multiplicative identity.
    const ONE: Self;

    /// The factor that [`Qubo::to_ising`](crate::Qubo::to_ising) multiplies all energies by.
    ///
    /// The conversion to spins brings in the factors 1/2 and 1/4. [`i64`] cannot hold them,
    /// so its scale is 4 and the Ising model is exact. [`f64`] uses the scale 1.
    const ISING_SCALE: Self;

    /// Returns `self + rhs`.
    ///
    /// # Panics
    ///
    /// For [`i64`], panics on overflow.
    #[must_use]
    fn plus(self, rhs: Self) -> Self;

    /// Returns `self - rhs`.
    ///
    /// # Panics
    ///
    /// For [`i64`], panics on overflow.
    #[must_use]
    fn minus(self, rhs: Self) -> Self;

    /// Returns `self * rhs`.
    ///
    /// # Panics
    ///
    /// For [`i64`], panics on overflow.
    #[must_use]
    fn times(self, rhs: Self) -> Self;

    /// Converts four times a value to that value times [`Self::ISING_SCALE`].
    ///
    /// The result is `quadruple · ISING_SCALE / 4`. This is exact for both types: the
    /// identity for [`i64`], and a multiplication by a power of two for [`f64`].
    #[must_use]
    fn from_quadruple(quadruple: Self) -> Self;

    /// Divides `self` by `divisor` if the result is exact.
    ///
    /// Returns `None` for [`i64`] when the remainder is not zero, or when `divisor` is zero.
    /// Always returns `Some` for [`f64`].
    #[must_use]
    fn exact_div(self, divisor: Self) -> Option<Self>;

    /// Converts an unsigned integer without loss.
    #[must_use]
    fn from_u32(value: u32) -> Self;

    /// Converts to [`f64`], for acceptance probabilities in stochastic samplers.
    ///
    /// For [`i64`], values with a magnitude above 2^53 lose precision.
    #[must_use]
    fn to_f64(self) -> f64;
}

impl Coefficient for i64 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const ISING_SCALE: Self = 4;

    fn plus(self, rhs: Self) -> Self {
        self.checked_add(rhs)
            .expect("i64 coefficient overflow in addition")
    }

    fn minus(self, rhs: Self) -> Self {
        self.checked_sub(rhs)
            .expect("i64 coefficient overflow in subtraction")
    }

    fn times(self, rhs: Self) -> Self {
        self.checked_mul(rhs)
            .expect("i64 coefficient overflow in multiplication")
    }

    fn from_quadruple(quadruple: Self) -> Self {
        quadruple
    }

    fn exact_div(self, divisor: Self) -> Option<Self> {
        (self.checked_rem(divisor)? == 0).then(|| self / divisor)
    }

    fn from_u32(value: u32) -> Self {
        Self::from(value)
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "only acceptance probabilities use the value; the trait docs state the 2^53 limit"
    )]
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl Coefficient for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const ISING_SCALE: Self = 1.0;

    fn plus(self, rhs: Self) -> Self {
        self + rhs
    }

    fn minus(self, rhs: Self) -> Self {
        self - rhs
    }

    fn times(self, rhs: Self) -> Self {
        self * rhs
    }

    fn from_quadruple(quadruple: Self) -> Self {
        quadruple * 0.25
    }

    fn exact_div(self, divisor: Self) -> Option<Self> {
        Some(self / divisor)
    }

    fn from_u32(value: u32) -> Self {
        Self::from(value)
    }

    fn to_f64(self) -> f64 {
        self
    }
}

/// Unit tests for the two [`Coefficient`] implementations.
#[cfg(test)]
mod tests {
    use super::Coefficient;

    /// Exact division of `i64` fails on a remainder or a zero divisor.
    #[test]
    fn i64_exact_div_rejects_remainders() {
        assert_eq!(12_i64.exact_div(4), Some(3));
        assert_eq!(13_i64.exact_div(4), None);
        assert_eq!(13_i64.exact_div(0), None);
    }

    /// `i64` arithmetic panics on overflow, also in release builds.
    #[test]
    #[should_panic(expected = "overflow")]
    fn i64_overflow_panics() {
        let _ = i64::MAX.plus(1);
    }

    /// The quadruple conversion applies the Ising scale.
    #[test]
    fn quadruple_applies_the_scale() {
        assert_eq!(i64::from_quadruple(6), 6);
        assert!((f64::from_quadruple(6.0) - 1.5).abs() < f64::EPSILON);
    }
}
