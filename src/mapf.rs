//! Multi-agent path finding over a time-expanded graph.
//!
//! Agents move on a directed graph, given as adjacency lists, in discrete steps. In each of
//! the `horizon` steps an agent moves along an edge or waits. Each agent starts at its start
//! vertex at time 0 and must be at its goal vertex at time `horizon`. No two agents may be
//! at one vertex at the same time (vertex conflict) or traverse one edge in opposite
//! directions in the same step (swap conflict).
//!
//! The encoding uses move variables: `agent{a}.t{t}.{u}-{w}` is 1 when agent `a` goes from
//! `u` to `w` in step `t` (`u == w` is a wait). Moves are pruned to those on some path from
//! the start to the goal within the horizon. A position encoding would need a quartic term
//! for swaps; with moves, a swap is a pair. The constraints are:
//!
//! - `agent{a}.t{t}.exactly_one`: agent `a` makes exactly one move in step `t`;
//! - `agent{a}.t{t}.v{v}.flow`: the agent leaves `v` in step `t` exactly when it entered `v`
//!   in step `t − 1`, a linear equality `Σ in − Σ out = 0`;
//! - `v{v}.t{t}.vertex`: at most one agent arrives at `v` in step `t − 1`, so at most one is
//!   there at time `t`;
//! - `{u}-{w}.t{t}.swap`: agents do not traverse `u → w` and `w → u` in step `t`.
//!
//! The objective counts every move and every wait except a wait at the agent's goal. For
//! plans where agents stay at their goals once they arrive, this is the sum of costs of
//! the other planning crates. Each violation costs at least the penalty weight `P`; use
//! [`penalty::safe`](crate::penalty::safe) for a weight that makes each ground state a
//! feasible optimum.
//!
//! ```
//! use planning_qubo::mapf;
//! use planning_qubo::{Exhaustive, Sampler, Unlimited, penalty};
//!
//! // A path 0 - 1 - 2. One agent walks from 0 to 2 within 3 steps.
//! let graph = [vec![1], vec![0, 2], vec![1]];
//! let (_, encoding) = penalty::safe(|p: i64| mapf::encode(&graph, &[(0, 2)], 3, p))?;
//!
//! let mut sampler = Exhaustive::new();
//! let run = sampler.sample(encoding.problem().qubo(), Unlimited)?;
//! let decoded = encoding.decode(run.sample.bits);
//!
//! assert_eq!(decoded.paths, [Some(vec![0, 1, 2, 2])]);
//! assert_eq!(decoded.cost, 2);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use alloc::collections::{BTreeMap, VecDeque};
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use crate::{Builder, Coefficient, Encoded, Problem, Var};

/// The error when a path finding instance cannot be encoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MapfError {
    /// An adjacency list or an agent names a vertex that does not exist.
    UnknownVertex {
        /// The vertex.
        vertex: usize,
    },
    /// Two agents start at the same vertex.
    SharedStart {
        /// The second agent at that start.
        agent: usize,
    },
    /// The horizon is 0, so no agent can move.
    ZeroHorizon,
}

impl fmt::Display for MapfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownVertex { vertex } => write!(f, "vertex {vertex} does not exist"),
            Self::SharedStart { agent } => {
                write!(f, "agent {agent} starts where an earlier agent starts")
            }
            Self::ZeroHorizon => f.write_str("the horizon must be at least one step"),
        }
    }
}

impl core::error::Error for MapfError {}

/// One move variable: agent, step, from, to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Move {
    /// The vertex at the start of the step.
    from: usize,
    /// The vertex at the end of the step. Equal to `from` for a wait.
    to: usize,
    /// The QUBO variable.
    var: Var,
}

/// An encoded path finding instance: the QUBO problem and the data to decode samples.
#[derive(Debug, Clone, PartialEq)]
pub struct Encoding<T> {
    /// The QUBO problem with its constraints.
    problem: Problem<T>,
    /// The moves of each agent in each step: `moves[agent][step]`.
    moves: Vec<Vec<Vec<Move>>>,
    /// The goal of each agent, for the cost.
    goals: Vec<usize>,
    /// The start of each agent, for the decoded paths.
    starts: Vec<usize>,
}

/// A sample, decoded into paths.
#[derive(Debug, Clone, PartialEq)]
pub struct Decoded<T> {
    /// The vertex of each agent at each time `0..=horizon`. `None` if the sample does not
    /// give the agent one connected move per step.
    pub paths: Vec<Option<Vec<usize>>>,
    /// The number of moves and waits away from the goal over all chosen moves, also for an
    /// infeasible sample.
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

/// Encodes paths for `agents` (`(start, goal)` pairs) on `graph` over `horizon` steps.
///
/// `graph[u]` lists the vertices that an agent at `u` can move to; list both directions
/// for an undirected edge. `penalty` is the weight `P` of each constraint. An agent whose
/// goal is out of reach within the horizon gets no moves in some step, so that step's
/// `exactly_one` constraint is always violated.
///
/// # Errors
///
/// Returns [`MapfError`] for an unknown vertex, two agents with the same start, or a
/// horizon of 0.
pub fn encode<T: Coefficient>(
    graph: &[impl AsRef<[usize]>],
    agents: &[(usize, usize)],
    horizon: usize,
    penalty: T,
) -> Result<Encoding<T>, MapfError> {
    validate(graph, agents, horizon)?;

    let mut builder = Builder::new();
    let moves: Vec<Vec<Vec<Move>>> = agents
        .iter()
        .enumerate()
        .map(|(a, &(start, goal))| add_moves(&mut builder, graph, a, start, goal, horizon))
        .collect();

    for (a, steps) in moves.iter().enumerate() {
        add_agent_constraints(&mut builder, a, steps, graph.len(), penalty);
    }

    add_vertex_conflicts(&mut builder, &moves, graph.len(), horizon, penalty);
    add_swap_conflicts(&mut builder, &moves, horizon, penalty);

    Ok(Encoding {
        problem: builder.build(),
        moves,
        goals: agents.iter().map(|&(_, g)| g).collect(),
        starts: agents.iter().map(|&(s, _)| s).collect(),
    })
}

/// Checks the vertices, the starts and the horizon.
fn validate(
    graph: &[impl AsRef<[usize]>],
    agents: &[(usize, usize)],
    horizon: usize,
) -> Result<(), MapfError> {
    let n = graph.len();
    let edges = graph.iter().flat_map(|row| row.as_ref().iter().copied());
    let ends = agents.iter().flat_map(|&(s, g)| [s, g]);

    if let Some(vertex) = edges.chain(ends).find(|&v| v >= n) {
        return Err(MapfError::UnknownVertex { vertex });
    }

    if let Some(agent) = (1..agents.len()).find(|&a| agents[..a].iter().any(|o| o.0 == agents[a].0))
    {
        return Err(MapfError::SharedStart { agent });
    }

    if horizon == 0 {
        return Err(MapfError::ZeroHorizon);
    }

    Ok(())
}

/// Creates the moves of agent `a` that lie on a start-to-goal path within the horizon.
///
/// A move `u → w` in step `t` needs `u` within `t` steps of the start and the goal within
/// `horizon − t − 1` steps of `w`. Each move costs 1 unless it waits at the goal.
fn add_moves<T: Coefficient>(
    builder: &mut Builder<T>,
    graph: &[impl AsRef<[usize]>],
    a: usize,
    start: usize,
    goal: usize,
    horizon: usize,
) -> Vec<Vec<Move>> {
    let from_start = distances(graph.len(), start, |u| graph[u].as_ref().to_vec());
    let reverse = reverse_graph(graph);
    let to_goal = distances(graph.len(), goal, |u| reverse[u].clone());

    (0..horizon)
        .map(|t| {
            let mut step = Vec::new();

            for u in (0..graph.len()).filter(|&u| from_start[u] <= t) {
                let targets = core::iter::once(u).chain(neighbors(graph[u].as_ref(), u));

                for w in targets.filter(|&w| to_goal[w] < horizon - t) {
                    let var = builder.variable(&format!("agent{a}.t{t}.{u}-{w}"));

                    if !(u == goal && w == goal) {
                        builder.add_linear(var, T::ONE);
                    }

                    step.push(Move {
                        from: u,
                        to: w,
                        var,
                    });
                }
            }

            step
        })
        .collect()
}

/// The neighbors of `u` in ascending order, without `u` itself and without repeats, so each
/// move gets one variable.
fn neighbors(row: &[usize], u: usize) -> Vec<usize> {
    let mut targets: Vec<usize> = row.iter().copied().filter(|&w| w != u).collect();
    targets.sort_unstable();
    targets.dedup();

    targets
}

/// The edges of `graph` reversed.
fn reverse_graph(graph: &[impl AsRef<[usize]>]) -> Vec<Vec<usize>> {
    let mut reverse = vec![Vec::new(); graph.len()];

    for (u, row) in graph.iter().enumerate() {
        for &w in row.as_ref() {
            reverse[w].push(u);
        }
    }

    reverse
}

/// The breadth-first step count from `source` to each vertex, `usize::MAX` if unreachable.
fn distances(n: usize, source: usize, neighbors: impl Fn(usize) -> Vec<usize>) -> Vec<usize> {
    let mut distance = vec![usize::MAX; n];
    let mut queue = VecDeque::from([source]);
    distance[source] = 0;

    while let Some(u) = queue.pop_front() {
        for w in neighbors(u) {
            if distance[w] == usize::MAX {
                distance[w] = distance[u] + 1;
                queue.push_back(w);
            }
        }
    }

    distance
}

/// Adds the one-move-per-step and the flow constraints of agent `a`.
fn add_agent_constraints<T: Coefficient>(
    builder: &mut Builder<T>,
    a: usize,
    steps: &[Vec<Move>],
    vertices: usize,
    penalty: T,
) {
    for (t, step) in steps.iter().enumerate() {
        let vars: Vec<Var> = step.iter().map(|m| m.var).collect();
        builder.exactly_one(&format!("agent{a}.t{t}.exactly_one"), &vars, penalty);
    }

    for t in 1..steps.len() {
        for v in 0..vertices {
            let into = steps[t - 1]
                .iter()
                .filter(|m| m.to == v)
                .map(|m| (m.var, T::ONE));
            let out = steps[t]
                .iter()
                .filter(|m| m.from == v)
                .map(|m| (m.var, T::ZERO.minus(T::ONE)));
            let terms: Vec<(Var, T)> = into.chain(out).collect();

            if !terms.is_empty() {
                let name = format!("agent{a}.t{t}.v{v}.flow");
                builder.linear_equality(&name, &terms, T::ZERO, 0, penalty);
            }
        }
    }
}

/// Adds `v{v}.t{t}.vertex` for each vertex and time that two or more agents can reach.
///
/// The constraint is at-most-one over all arrivals at `v` in step `t − 1`. Two arrivals of
/// one agent also break the agent's exactly-one constraint, so they add penalty only to
/// assignments that are infeasible anyway.
fn add_vertex_conflicts<T: Coefficient>(
    builder: &mut Builder<T>,
    moves: &[Vec<Vec<Move>>],
    vertices: usize,
    horizon: usize,
    penalty: T,
) {
    for t in 1..=horizon {
        for v in 0..vertices {
            let arrives = |steps: &Vec<Vec<Move>>| steps[t - 1].iter().any(|m| m.to == v);

            if moves.iter().filter(|steps| arrives(steps)).count() < 2 {
                continue;
            }

            let arrivals: Vec<Var> = moves
                .iter()
                .flat_map(|steps| steps[t - 1].iter().filter(|m| m.to == v).map(|m| m.var))
                .collect();
            builder.at_most_one(&format!("v{v}.t{t}.vertex"), &arrivals, penalty);
        }
    }
}

/// Adds `{u}-{w}.t{t}.swap` (with `u < w`) for each step where two agents can swap on an edge.
fn add_swap_conflicts<T: Coefficient>(
    builder: &mut Builder<T>,
    moves: &[Vec<Vec<Move>>],
    horizon: usize,
    penalty: T,
) {
    for t in 0..horizon {
        let mut swaps: BTreeMap<(usize, usize), Vec<[Var; 2]>> = BTreeMap::new();

        for (a, steps) in moves.iter().enumerate() {
            for m in steps[t].iter().filter(|m| m.from < m.to) {
                let backs = moves
                    .iter()
                    .enumerate()
                    .filter(|&(b, _)| b != a)
                    .flat_map(|(_, other)| other[t].iter())
                    .filter(|o| o.from == m.to && o.to == m.from);

                for back in backs {
                    swaps
                        .entry((m.from, m.to))
                        .or_default()
                        .push([m.var, back.var]);
                }
            }
        }

        for ((u, w), pairs) in swaps {
            for pair in pairs {
                builder.at_most_one(&format!("{u}-{w}.t{t}.swap"), &pair, penalty);
            }
        }
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

    /// The variable that is 1 when agent `agent` goes from `from` to `to` in step `step`, if
    /// that move lies on a start-to-goal path within the horizon.
    ///
    /// # Panics
    ///
    /// Panics if `agent` or `step` is out of range.
    #[must_use]
    pub fn variable(&self, agent: usize, step: usize, from: usize, to: usize) -> Option<Var> {
        self.moves[agent][step]
            .iter()
            .find(|m| m.from == from && m.to == to)
            .map(|m| m.var)
    }

    /// Decodes the assignment `bits` of all QUBO variables.
    ///
    /// # Panics
    ///
    /// Panics if `bits.len()` is not the number of QUBO variables.
    #[must_use]
    pub fn decode(&self, bits: &[bool]) -> Decoded<T> {
        let paths = self
            .moves
            .iter()
            .zip(&self.starts)
            .map(|(steps, &start)| path(steps, start, bits))
            .collect();

        Decoded {
            paths,
            cost: self.cost(bits),
            violations: self.problem.violation_names(bits),
        }
    }

    /// The number of chosen moves, waits at the goal excluded.
    fn cost(&self, bits: &[bool]) -> T {
        self.moves
            .iter()
            .zip(&self.goals)
            .flat_map(|(steps, &goal)| {
                steps
                    .iter()
                    .flatten()
                    .filter(move |m| bits[m.var.index()] && !(m.from == goal && m.to == goal))
            })
            .fold(T::ZERO, |sum, _| sum.plus(T::ONE))
    }
}

/// The path of one agent: one chosen move per step, each leaving where the last one ended.
fn path(steps: &[Vec<Move>], start: usize, bits: &[bool]) -> Option<Vec<usize>> {
    let mut vertices = vec![start];

    for step in steps {
        let mut chosen = step.iter().filter(|m| bits[m.var.index()]);
        let m = chosen.next()?;

        if chosen.next().is_some() || Some(&m.from) != vertices.last() {
            return None;
        }

        vertices.push(m.to);
    }

    Some(vertices)
}
