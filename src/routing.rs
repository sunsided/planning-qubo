//! Routing: the shortest tour from a depot through every city and back (travelling salesman).
//!
//! City 0 is the depot and the tour starts and ends there. The encoding fixes the depot at
//! position 0, which removes the rotations of each tour, and has one variable
//! `city{c}.pos{p}` per other city `c` and position `p` in `1..n`, which is 1 when the tour
//! visits `c` at position `p`. The objective is the tour length: `d[0][c]` for the first
//! city, `d[u][v]` for each consecutive pair, and `d[c][0]` back to the depot. Distances
//! may be asymmetric. The constraints are:
//!
//! - `city{c}.exactly_one`: each city has exactly one position;
//! - `pos{p}.exactly_one`: each position has exactly one city.
//!
//! A tour over `n` cities has `(n − 1)²` variables. Each violation costs at least the
//! penalty weight `P`; use [`penalty::safe`](crate::penalty::safe) for a weight that makes
//! each ground state an optimal tour. Several vehicles or capacities are not part of the
//! encoding.
//!
//! ```
//! use planning_qubo::routing;
//! use planning_qubo::{Exhaustive, Sampler, Unlimited, penalty};
//!
//! // Four cities on a line at 0, 1, 2 and 3: the best tour goes out and comes back.
//! let at = [0_i64, 1, 2, 3];
//! let distances: Vec<Vec<i64>> =
//!     at.iter().map(|a| at.iter().map(|b| (a - b).abs()).collect()).collect();
//! let (_, encoding) = penalty::safe(|p| routing::encode(&distances, p))?;
//!
//! let mut sampler = Exhaustive::new();
//! let run = sampler.sample(encoding.problem().qubo(), Unlimited)?;
//! let decoded = encoding.decode(run.sample.bits);
//!
//! assert_eq!(decoded.cost, 6);
//! assert!(decoded.tour == Some(vec![0, 1, 2, 3]) || decoded.tour == Some(vec![0, 3, 2, 1]));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use crate::csp::one_hot;
use crate::{Builder, Coefficient, Encoded, Problem, Var};

/// The error when a distance matrix cannot be encoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RoutingError {
    /// There are no cities, not even the depot.
    NoCities,
    /// A row of the distance matrix does not have one entry per city.
    ShapeMismatch {
        /// The first row with the wrong length.
        row: usize,
    },
}

impl fmt::Display for RoutingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoCities => f.write_str("there are no cities"),
            Self::ShapeMismatch { row } => {
                write!(f, "distance row {row} does not have one entry per city")
            }
        }
    }
}

impl core::error::Error for RoutingError {}

/// An encoded tour problem: the QUBO problem and the data to decode samples.
#[derive(Debug, Clone, PartialEq)]
pub struct Encoding<T> {
    /// The QUBO problem with its constraints.
    problem: Problem<T>,
    /// The variables and distances.
    layout: Layout<T>,
}

/// The variables of a tour problem and its distances.
#[derive(Debug, Clone, PartialEq)]
struct Layout<T> {
    /// The variable of each city `1..n` and position `1..n`: `vars[c − 1][p − 1]`.
    vars: Vec<Vec<Var>>,
    /// The distance matrix, row-major `n × n`.
    distances: Vec<T>,
    /// The number of cities, the depot included.
    cities: usize,
}

/// A sample, decoded into a tour.
#[derive(Debug, Clone, PartialEq)]
pub struct Decoded<T> {
    /// The cities in tour order, from the depot. `None` if the sample is not a permutation.
    pub tour: Option<Vec<usize>>,
    /// The tour length of the chosen variables, also for an infeasible sample.
    pub cost: T,
    /// The names of the violated constraints, in the order of [`Problem::constraints`].
    pub violations: Vec<String>,
}

impl<T> Decoded<T> {
    /// Returns `true` if the sample violates no constraint.
    #[must_use]
    pub fn is_feasible(&self) -> bool {
        self.violations.is_empty()
    }
}

/// Encodes the shortest tour from city 0 through all cities of `distances`.
///
/// `distances[u][v]` is the length of the leg from `u` to `v`; the diagonal is not used.
/// `penalty` is the weight `P` of each constraint.
///
/// # Errors
///
/// Returns [`RoutingError`] if there are no cities or the matrix is not square.
pub fn encode<T: Coefficient>(
    distances: &[impl AsRef<[T]>],
    penalty: T,
) -> Result<Encoding<T>, RoutingError> {
    let n = distances.len();

    if n == 0 {
        return Err(RoutingError::NoCities);
    }

    if let Some(row) = distances.iter().position(|r| r.as_ref().len() != n) {
        return Err(RoutingError::ShapeMismatch { row });
    }

    let flat: Vec<T> = distances
        .iter()
        .flat_map(|r| r.as_ref().iter().copied())
        .collect();
    let mut builder = Builder::new();
    let vars: Vec<Vec<Var>> = (1..n)
        .map(|c| {
            (1..n)
                .map(|p| builder.variable(&format!("city{c}.pos{p}")))
                .collect()
        })
        .collect();

    let layout = Layout {
        vars,
        distances: flat,
        cities: n,
    };
    layout.add_objective(&mut builder);
    layout.add_constraints(&mut builder, penalty);

    Ok(Encoding {
        problem: builder.build(),
        layout,
    })
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

    /// The variable that is 1 when the tour visits `city` at `position`.
    ///
    /// # Panics
    ///
    /// Panics if `city` or `position` is 0 (the depot) or not below the city count.
    #[must_use]
    pub fn variable(&self, city: usize, position: usize) -> Var {
        self.layout.variable(city, position)
    }

    /// Decodes the assignment `bits` of all QUBO variables.
    ///
    /// # Panics
    ///
    /// Panics if `bits.len()` is not the number of QUBO variables.
    #[must_use]
    pub fn decode(&self, bits: &[bool]) -> Decoded<T> {
        Decoded {
            tour: self.layout.tour(bits),
            cost: self.layout.cost(bits),
            violations: self.problem.violation_names(bits),
        }
    }
}

impl<T: Coefficient> Layout<T> {
    /// The variable of `city` at `position`. Panics for the depot or out of range.
    fn variable(&self, city: usize, position: usize) -> Var {
        assert!(city > 0 && position > 0, "the depot has no variables");

        self.vars[city - 1][position - 1]
    }

    /// The length of the leg from `u` to `v`.
    fn distance(&self, u: usize, v: usize) -> T {
        self.distances[u * self.cities + v]
    }

    /// The legs of the objective: `(city, position, next city, next position, length)`,
    /// with the depot as city 0 at the positions 0 and `n`.
    fn legs(&self) -> impl Iterator<Item = (usize, usize, usize, usize, T)> + '_ {
        let n = self.cities;
        let inner = (1..n.saturating_sub(1)).flat_map(move |p| {
            (1..n).flat_map(move |u| {
                (1..n)
                    .filter(move |&v| v != u)
                    .map(move |v| (u, p, v, p + 1, self.distance(u, v)))
            })
        });
        let ends = (1..n).flat_map(move |c| {
            [
                (0, 0, c, 1, self.distance(0, c)),
                (c, n - 1, 0, n, self.distance(c, 0)),
            ]
        });

        ends.chain(inner)
    }

    /// Adds each leg as a linear term (a leg to or from the depot) or a pair term.
    fn add_objective(&self, builder: &mut Builder<T>) {
        for (u, p, v, q, length) in self.legs() {
            match (u, v) {
                (0, _) => builder.add_linear(self.variable(v, q), length),
                (_, 0) => builder.add_linear(self.variable(u, p), length),
                _ => builder.add_quadratic(self.variable(u, p), self.variable(v, q), length),
            }
        }
    }

    /// Adds the exactly-one constraints of each city and each position.
    fn add_constraints(&self, builder: &mut Builder<T>, penalty: T) {
        for (k, row) in self.vars.iter().enumerate() {
            builder.exactly_one(&format!("city{}.exactly_one", k + 1), row, penalty);
        }

        for p in 1..self.cities {
            let column: Vec<Var> = self.vars.iter().map(|row| row[p - 1]).collect();
            builder.exactly_one(&format!("pos{p}.exactly_one"), &column, penalty);
        }
    }

    /// The cities in position order, if each position has one city and each city one position.
    fn tour(&self, bits: &[bool]) -> Option<Vec<usize>> {
        let mut tour = vec![0];
        let mut seen = vec![false; self.cities];

        for p in 1..self.cities {
            let column: Vec<Var> = self.vars.iter().map(|row| row[p - 1]).collect();
            let city = one_hot(&column, bits)? + 1;

            if core::mem::replace(&mut seen[city], true) {
                return None;
            }

            tour.push(city);
        }

        Some(tour)
    }

    /// The sum of the legs whose variables are set.
    fn cost(&self, bits: &[bool]) -> T {
        let set = |c: usize, p: usize| c == 0 || bits[self.variable(c, p).index()];

        self.legs()
            .filter(|&(u, p, v, q, _)| set(u, p) && set(v, q))
            .fold(T::ZERO, |sum, (.., length)| sum.plus(length))
    }
}
