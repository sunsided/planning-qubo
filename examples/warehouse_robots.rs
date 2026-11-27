//! Three warehouse robots crossing a small floor with a shelf block in the middle.
//!
//! The floor is a 3 × 4 grid; the two middle cells of the centre row hold shelves. Two
//! robots swap the opposite corners, a third crosses the other diagonal, so their shortest
//! routes collide. The time-expanded encoding has one variable per robot, step and move,
//! pruned to moves on some route within the horizon. Parallel tempering samples it with a
//! tuned penalty weight; the plan is then checked for vertex and swap conflicts directly,
//! and its cost compared with the sum of the robots' shortest routes, a lower bound.
//!
//! Run with `cargo run --release --example warehouse_robots`.

use planning_qubo::{ParallelTempering, Schedule, Unlimited, mapf, penalty};
use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;

/// The floor plan: `.` is free, `#` is a shelf.
const FLOOR: [&str; 3] = ["....", ".##.", "...."];

/// The number of time steps.
const HORIZON: usize = 6;

/// Tunes the penalty against parallel tempering and checks the plan it returns.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (graph, cell) = floor_graph();
    let robots = [
        (cell(0, 0), cell(2, 3)),
        (cell(2, 3), cell(0, 0)),
        (cell(0, 3), cell(2, 0)),
    ];
    let encode = |p: i64| mapf::encode(&graph, &robots, HORIZON, p);
    let mut sampler =
        ParallelTempering::new(Pcg64Mcg::seed_from_u64(11), Schedule::from_model(1_000), 10);

    let tuned = penalty::tune(encode, &mut sampler, Unlimited)?;
    println!(
        "{} robots, {HORIZON} steps: {} binary variables, penalty weight {}",
        robots.len(),
        tuned.encoding.problem().qubo().num_variables(),
        tuned.penalty
    );

    let decoded = tuned.encoding.decode(&tuned.bits);

    if !decoded.is_feasible() {
        println!(
            "no conflict-free plan found; broken rules: {:?}",
            decoded.violations
        );
        return Ok(());
    }

    let paths: Vec<Vec<usize>> = decoded
        .paths
        .iter()
        .map(|p| p.clone().expect("a path"))
        .collect();

    for (r, path) in paths.iter().enumerate() {
        let cells: Vec<String> = path.iter().map(|&v| name(v)).collect();
        println!("  robot {r}: {}", cells.join(" "));
    }

    println!("conflict-free: {}", conflict_free(&paths));
    println!(
        "cost {} (lower bound from shortest routes: {})",
        decoded.cost,
        lower_bound(&graph, &robots)
    );

    Ok(())
}

/// The free cells as a graph of 4-neighbour moves, and the cell index of a row and column.
fn floor_graph() -> (Vec<Vec<usize>>, impl Fn(usize, usize) -> usize) {
    let cols = FLOOR[0].len();
    let free = |r: usize, c: usize| FLOOR[r].as_bytes()[c] == b'.';
    let index = move |r: usize, c: usize| r * cols + c;
    let mut graph = vec![Vec::new(); FLOOR.len() * cols];

    for r in 0..FLOOR.len() {
        for c in (0..cols).filter(|&c| free(r, c)) {
            let steps = [
                (r.wrapping_sub(1), c),
                (r + 1, c),
                (r, c.wrapping_sub(1)),
                (r, c + 1),
            ];

            for (nr, nc) in steps {
                if nr < FLOOR.len() && nc < cols && free(nr, nc) {
                    graph[index(r, c)].push(index(nr, nc));
                }
            }
        }
    }

    (graph, index)
}

/// A cell as `row:column`.
fn name(cell: usize) -> String {
    let cols = FLOOR[0].len();
    format!("{}:{}", cell / cols, cell % cols)
}

/// Whether no two paths meet in a cell or swap along an edge, checked directly.
fn conflict_free(paths: &[Vec<usize>]) -> bool {
    paths.iter().enumerate().all(|(a, p)| {
        paths[a + 1..].iter().all(|q| {
            let vertex = p.iter().zip(q).any(|(x, y)| x == y);
            let swap = p
                .windows(2)
                .zip(q.windows(2))
                .any(|(s, t)| s[0] == t[1] && s[1] == t[0]);

            !vertex && !swap
        })
    })
}

/// The sum of each robot's shortest route length, by breadth-first search.
fn lower_bound(graph: &[Vec<usize>], robots: &[(usize, usize)]) -> usize {
    robots
        .iter()
        .map(|&(start, goal)| {
            let mut distance = vec![usize::MAX; graph.len()];
            let mut queue = std::collections::VecDeque::from([start]);
            distance[start] = 0;

            while let Some(u) = queue.pop_front() {
                for &w in &graph[u] {
                    if distance[w] == usize::MAX {
                        distance[w] = distance[u] + 1;
                        queue.push_back(w);
                    }
                }
            }

            distance[goal]
        })
        .sum()
}
