//! The assignment problem: each task goes to one agent, and each agent has a capacity.
//!
//! The encoding has one variable `task{t}.agent{a}` per task `t` and agent `a`, which is 1
//! when the agent does the task. The objective is the sum of the costs of the chosen
//! pairs. The constraints are:
//!
//! - `task{t}.exactly_one`: task `t` goes to exactly one agent;
//! - `agent{a}.capacity`: agent `a` does at most `capacities[a]` tasks. Capacity 1 uses an
//!   at-most-one penalty, other capacities a linear constraint with slack bits. A capacity
//!   of at least the task count has no constraint.
//!
//! Each constraint uses the same penalty weight `P`. Each violation costs at least `P`,
//! and the objective of any two assignments differs by at most `Σ|cost|`. Thus with
//! integer costs, `P > Σ|cost|` makes each ground state a feasible optimum, if a feasible
//! assignment exists.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use crate::{Builder, Coefficient, Problem, Var};

/// An encoded assignment problem: the QUBO problem and the data to decode samples.
#[derive(Debug, Clone, PartialEq)]
pub struct Encoding<T> {
    /// The QUBO problem with its constraints.
    problem: Problem<T>,
    /// The variable of each pair, row-major: index `t · agents + a`.
    vars: Vec<Var>,
    /// The cost of each pair, row-major: index `t · agents + a`.
    costs: Vec<T>,
    /// The number of tasks: the number of rows of `vars` and `costs`.
    tasks: usize,
    /// The number of agents: the length of each row of `vars` and `costs`.
    agents: usize,
}

/// The error when the cost rows and the capacities do not have the same length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShapeMismatch {
    /// The first task whose cost row length is not the number of agents.
    pub task: usize,
}

impl fmt::Display for ShapeMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the cost row of task {} does not have one entry per agent",
            self.task
        )
    }
}

impl core::error::Error for ShapeMismatch {}

/// A sample, decoded into an assignment.
#[derive(Debug, Clone, PartialEq)]
pub struct Decoded<T> {
    /// The agent of each task. `None` if the sample gives the task no agent or more than one.
    pub agents: Vec<Option<usize>>,
    /// The sum of the costs of all chosen pairs, also for an infeasible sample.
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

/// Encodes the assignment of `costs.len()` tasks to `capacities.len()` agents.
///
/// `costs[t][a]` is the cost when agent `a` does task `t`. `penalty` is the weight `P` of
/// each constraint; see the [module docs](self) for a safe value.
///
/// # Errors
///
/// Returns [`ShapeMismatch`] if a cost row does not have one entry per agent.
pub fn encode<T: Coefficient>(
    costs: &[impl AsRef<[T]>],
    capacities: &[u32],
    penalty: T,
) -> Result<Encoding<T>, ShapeMismatch> {
    let agents = capacities.len();

    if let Some(task) = costs.iter().position(|row| row.as_ref().len() != agents) {
        return Err(ShapeMismatch { task });
    }

    let mut builder = Builder::new();
    let (vars, flat_costs) = add_objective(&mut builder, costs);

    add_task_constraints(&mut builder, &vars, costs.len(), penalty);
    add_capacity_constraints(&mut builder, &vars, costs.len(), capacities, penalty);

    Ok(Encoding {
        problem: builder.build(),
        vars,
        costs: flat_costs,
        tasks: costs.len(),
        agents,
    })
}

/// Creates the pair variables with their costs, and returns both in row-major order.
fn add_objective<T: Coefficient>(
    builder: &mut Builder<T>,
    costs: &[impl AsRef<[T]>],
) -> (Vec<Var>, Vec<T>) {
    let mut vars = Vec::new();
    let mut flat = Vec::new();

    for (t, row) in costs.iter().enumerate() {
        for (a, &cost) in row.as_ref().iter().enumerate() {
            let var = builder.variable(&format!("task{t}.agent{a}"));
            builder.add_linear(var, cost);
            vars.push(var);
            flat.push(cost);
        }
    }

    (vars, flat)
}

/// Adds `task{t}.exactly_one` for each of `tasks` tasks.
///
/// Without agents a task has no variables, and its constraint is always violated.
fn add_task_constraints<T: Coefficient>(
    builder: &mut Builder<T>,
    vars: &[Var],
    tasks: usize,
    penalty: T,
) {
    let agents = vars.len().checked_div(tasks).unwrap_or(0);

    for t in 0..tasks {
        let row = &vars[t * agents..(t + 1) * agents];
        builder.exactly_one(&format!("task{t}.exactly_one"), row, penalty);
    }
}

/// Adds `agent{a}.capacity` for each agent whose capacity is below the task count.
fn add_capacity_constraints<T: Coefficient>(
    builder: &mut Builder<T>,
    vars: &[Var],
    tasks: usize,
    capacities: &[u32],
    penalty: T,
) {
    let agents = capacities.len();

    for (a, &capacity) in capacities.iter().enumerate() {
        if usize::try_from(capacity).is_ok_and(|c| c >= tasks) {
            continue;
        }

        let column: Vec<Var> = vars.iter().skip(a).step_by(agents).copied().collect();
        add_capacity(
            builder,
            &format!("agent{a}.capacity"),
            &column,
            capacity,
            penalty,
        );
    }
}

/// Adds `Σ column ≤ capacity` as at-most-one for capacity 1, else as a linear constraint.
fn add_capacity<T: Coefficient>(
    builder: &mut Builder<T>,
    name: &str,
    column: &[Var],
    capacity: u32,
    penalty: T,
) {
    if capacity == 1 {
        builder.at_most_one(name, column, penalty);
        return;
    }

    let terms: Vec<(Var, T)> = column.iter().map(|&v| (v, T::ONE)).collect();
    builder.linear_equality(name, &terms, T::from_u32(capacity), capacity, penalty);
}

impl<T: Coefficient> Encoding<T> {
    /// The QUBO problem with its constraints.
    #[must_use]
    pub fn problem(&self) -> &Problem<T> {
        &self.problem
    }

    /// The number of tasks.
    #[must_use]
    pub fn tasks(&self) -> usize {
        self.tasks
    }

    /// The number of agents.
    #[must_use]
    pub fn agents(&self) -> usize {
        self.agents
    }

    /// The variable that is 1 when agent `agent` does task `task`.
    ///
    /// # Panics
    ///
    /// Panics if `task` or `agent` is out of range.
    #[must_use]
    pub fn variable(&self, task: usize, agent: usize) -> Var {
        assert!(agent < self.agents, "agent {agent} is out of range");

        self.vars[task * self.agents + agent]
    }

    /// Decodes the assignment `bits` of all QUBO variables, slack bits included.
    ///
    /// # Panics
    ///
    /// Panics if `bits.len()` is not the number of QUBO variables.
    #[must_use]
    pub fn decode(&self, bits: &[bool]) -> Decoded<T> {
        let violations = self
            .problem
            .violations(bits)
            .map(|c| c.name.clone())
            .collect();

        Decoded {
            agents: self.agent_of_each_task(bits),
            cost: self.cost(bits),
            violations,
        }
    }

    /// The agent of each task, or `None` if the task has no agent or more than one.
    fn agent_of_each_task(&self, bits: &[bool]) -> Vec<Option<usize>> {
        (0..self.tasks())
            .map(|t| {
                let mut chosen = (0..self.agents).filter(|&a| bits[self.variable(t, a).index()]);
                let first = chosen.next();

                if chosen.next().is_some() { None } else { first }
            })
            .collect()
    }

    /// The sum of the costs of the chosen pairs.
    fn cost(&self, bits: &[bool]) -> T {
        self.vars
            .iter()
            .zip(&self.costs)
            .filter(|(v, _)| bits[v.index()])
            .fold(T::ZERO, |sum, (_, &c)| sum.plus(c))
    }
}
