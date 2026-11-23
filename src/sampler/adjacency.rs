//! A compressed adjacency view of a QUBO model for single-bit flips.

use alloc::vec::Vec;

use crate::{Coefficient, Qubo};

/// The QUBO coefficients by variable, in compressed sparse rows, for fast flip deltas.
///
/// The rows hold each pair twice, once per end. A sampler keeps one instance and calls
/// [`Adjacency::rebuild`] for each model, which reuses the allocations.
#[derive(Debug, Clone)]
pub(crate) struct Adjacency<T> {
    /// The linear coefficient of each variable.
    linear: Vec<T>,
    /// Row `i` is `neighbors[starts[i]..starts[i + 1]]`. The length is the variable count plus 1.
    starts: Vec<usize>,
    /// The pairs `(j, bᵢⱼ)` of each row, rows in variable order.
    neighbors: Vec<(usize, T)>,
}

impl<T> Default for Adjacency<T> {
    fn default() -> Self {
        Self {
            linear: Vec::new(),
            starts: Vec::new(),
            neighbors: Vec::new(),
        }
    }
}

impl<T: Coefficient> Adjacency<T> {
    /// Loads `qubo`, with each coefficient converted by `convert`.
    pub(crate) fn rebuild<S: Coefficient>(&mut self, qubo: &Qubo<S>, convert: impl Fn(S) -> T) {
        let n = qubo.num_variables();
        self.linear.clear();
        self.linear.extend((0..n).map(|i| convert(qubo.linear(i))));

        self.fill_row_starts(qubo);
        self.fill_rows(qubo, convert);
    }

    /// Computes the row starts from the degree of each variable.
    fn fill_row_starts<S: Coefficient>(&mut self, qubo: &Qubo<S>) {
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
    fn fill_rows<S: Coefficient>(&mut self, qubo: &Qubo<S>, convert: impl Fn(S) -> T) {
        let total = self.starts.last().copied().unwrap_or(0);
        self.neighbors.clear();
        self.neighbors.resize(total, (0, T::ZERO));

        let mut cursor = self.starts.clone();

        for (i, j, b) in qubo.quadratic_terms() {
            let b = convert(b);
            self.neighbors[cursor[i]] = (j, b);
            cursor[i] += 1;
            self.neighbors[cursor[j]] = (i, b);
            cursor[j] += 1;
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
}

/// Unit tests for the flip deltas.
#[cfg(test)]
mod tests {
    use super::Adjacency;
    use crate::Qubo;

    /// Each flip delta equals the energy difference of the two assignments.
    #[test]
    fn flip_delta_matches_energy_difference() {
        let mut qubo = Qubo::<i64>::new(3);
        qubo.add_linear(0, 2);
        qubo.add_linear(2, -1);
        qubo.add_quadratic(0, 1, -3);
        qubo.add_quadratic(1, 2, 4);
        qubo.add_quadratic(0, 2, 5);

        let mut adjacency = Adjacency::default();
        adjacency.rebuild(&qubo, |v| v);

        for bits in 0..8_u8 {
            let x: [bool; 3] = core::array::from_fn(|i| bits >> i & 1 == 1);

            for i in 0..3 {
                let mut flipped = x;
                flipped[i] = !flipped[i];
                let expected = qubo.energy(&flipped) - qubo.energy(&x);

                assert_eq!(adjacency.flip_delta(i, &x), expected);
            }
        }
    }
}
