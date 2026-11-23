//! A compressed adjacency view of a QUBO model for single-bit flips.

use alloc::vec::Vec;

use crate::{Coefficient, Qubo};

/// The QUBO coefficients by variable, in compressed sparse rows, for fast flip deltas.
///
/// The rows hold each pair twice, once per end. A sampler keeps one instance and calls
/// [`Adjacency::rebuild`] for each model. A rebuild for a model with the same number of
/// variables and pairs as the previous one reuses all buffers and allocates nothing.
#[derive(Debug, Clone)]
pub(crate) struct Adjacency<T> {
    /// The constant energy term.
    offset: T,
    /// The linear coefficient of each variable.
    linear: Vec<T>,
    /// Row `i` is `neighbors[starts[i]..starts[i + 1]]`. The length is the variable count plus 1.
    starts: Vec<usize>,
    /// The pairs `(j, bᵢⱼ)` of each row, rows in variable order.
    neighbors: Vec<(usize, T)>,
    /// Scratch space for [`Adjacency::fill_rows`]: the next free slot of each row.
    cursor: Vec<usize>,
}

impl<T: Coefficient> Adjacency<T> {
    /// Creates an empty view with no variables.
    pub(crate) fn new() -> Self {
        Self {
            offset: T::ZERO,
            linear: Vec::new(),
            starts: Vec::new(),
            neighbors: Vec::new(),
            cursor: Vec::new(),
        }
    }

    /// Loads `qubo` and reuses the buffers of the previous model.
    pub(crate) fn rebuild(&mut self, qubo: &Qubo<T>) {
        let n = qubo.num_variables();
        self.offset = qubo.offset();
        self.linear.clear();
        self.linear.extend((0..n).map(|i| qubo.linear(i)));

        self.fill_row_starts(qubo);
        self.fill_rows(qubo);
    }

    /// Computes the row starts from the degree of each variable.
    fn fill_row_starts(&mut self, qubo: &Qubo<T>) {
        self.starts.clear();
        self.starts.resize(qubo.num_variables() + 1, 0);

        for (i, j, _) in qubo.quadratic_terms() {
            self.starts[i + 1] += 1;
            self.starts[j + 1] += 1;
        }

        for k in 1..self.starts.len() {
            self.starts[k] += self.starts[k - 1];
        }
    }

    /// Writes each pair into the rows of both ends. Needs the row starts.
    fn fill_rows(&mut self, qubo: &Qubo<T>) {
        let total = self.starts.last().copied().unwrap_or(0);
        self.neighbors.clear();
        self.neighbors.resize(total, (0, T::ZERO));

        self.cursor.clear();
        self.cursor.extend_from_slice(&self.starts);

        for (i, j, b) in qubo.quadratic_terms() {
            self.neighbors[self.cursor[i]] = (j, b);
            self.cursor[i] += 1;
            self.neighbors[self.cursor[j]] = (i, b);
            self.cursor[j] += 1;
        }
    }

    /// The number of variables.
    pub(crate) fn len(&self) -> usize {
        self.linear.len()
    }

    /// The linear coefficient of variable `i`.
    pub(crate) fn linear(&self, i: usize) -> T {
        self.linear[i]
    }

    /// The pairs `(j, bᵢⱼ)` of variable `i`.
    pub(crate) fn row(&self, i: usize) -> &[(usize, T)] {
        &self.neighbors[self.starts[i]..self.starts[i + 1]]
    }

    /// The energy change when variable `i` flips in `x`.
    ///
    /// The local field is `aᵢ + Σⱼ bᵢⱼ·xⱼ`. Setting `xᵢ` adds it; clearing `xᵢ` removes it.
    pub(crate) fn flip_delta(&self, i: usize, x: &[bool]) -> T {
        let field = self
            .row(i)
            .iter()
            .filter(|&&(j, _)| x[j])
            .fold(self.linear[i], |sum, &(_, b)| sum.plus(b));

        if x[i] { T::ZERO.minus(field) } else { field }
    }

    /// The energy of `x`, with each pair counted once. Exact for [`i64`].
    ///
    /// The value equals [`Qubo::energy`] for [`i64`]. For [`f64`] the summation order
    /// differs, so the last bits can differ.
    pub(crate) fn energy(&self, x: &[bool]) -> T {
        (0..self.len())
            .filter(|&i| x[i])
            .fold(self.offset, |sum, i| sum.plus(self.upper_field(i, x)))
    }

    /// The linear term of `i` plus its pairs with set variables `j > i`.
    fn upper_field(&self, i: usize, x: &[bool]) -> T {
        self.row(i)
            .iter()
            .filter(|&&(j, _)| j > i && x[j])
            .fold(self.linear[i], |sum, &(_, b)| sum.plus(b))
    }
}

/// Unit tests for the flip deltas and the energy.
#[cfg(test)]
mod tests {
    use super::Adjacency;
    use crate::Qubo;

    /// A small model with linear terms, three pairs and an offset.
    fn model() -> Qubo<i64> {
        let mut qubo = Qubo::new(3);
        qubo.add_offset(7);
        qubo.add_linear(0, 2);
        qubo.add_linear(2, -1);
        qubo.add_quadratic(0, 1, -3);
        qubo.add_quadratic(1, 2, 4);
        qubo.add_quadratic(0, 2, 5);

        qubo
    }

    /// Each flip delta equals the energy difference, and the energy equals the model energy.
    #[test]
    fn flip_delta_and_energy_match_the_model() {
        let qubo = model();
        let mut adjacency = Adjacency::new();
        adjacency.rebuild(&qubo);

        for bits in 0..8_u8 {
            let x: [bool; 3] = core::array::from_fn(|i| bits >> i & 1 == 1);
            assert_eq!(adjacency.energy(&x), qubo.energy(&x));

            for i in 0..3 {
                let mut flipped = x;
                flipped[i] = !flipped[i];
                let expected = qubo.energy(&flipped) - qubo.energy(&x);

                assert_eq!(adjacency.flip_delta(i, &x), expected);
            }
        }
    }
}
