# planning-qubo

[![crates.io](https://img.shields.io/crates/v/planning-qubo.svg)](https://crates.io/crates/planning-qubo)
[![docs.rs](https://img.shields.io/docsrs/planning-qubo)](https://docs.rs/planning-qubo)
[![CI](https://github.com/sunsided/planning-qubo/actions/workflows/ci.yml/badge.svg)](https://github.com/sunsided/planning-qubo/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sunsided/planning-qubo/graph/badge.svg)](https://codecov.io/gh/sunsided/planning-qubo)
[![MSRV](https://img.shields.io/crates/msrv/planning-qubo)](https://crates.io/crates/planning-qubo)
[![License](https://img.shields.io/crates/l/planning-qubo)](https://github.com/sunsided/planning-qubo#license)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)

Planning, scheduling and constraint problems as QUBO (quadratic unconstrained binary optimization) and Ising models: encoders, decoders, penalty tuning, classical samplers and export for quantum annealers.

## Who this is for

- **Anyone who wants a planning problem as a QUBO or Ising model.** Encoders turn
  assignment, constraint satisfaction, time-indexed scheduling, multi-agent path finding
  and routing problems into models, and decoders map a sample back to a plan and name
  each violated constraint. The penalty builder covers other problems with the same
  constraints (exactly-one, at-most-one, linear with slack bits), and `penalty` picks the
  weights. The models feed any QUBO solver: the samplers here, quantum-inspired
  hardware, or a quantum annealer.
- **Researchers comparing samplers on the same model.** The exhaustive sampler gives the
  exact optimum of small models. Simulated annealing, tabu search and parallel tempering,
  each with a seeded RNG, show how close a heuristic gets. All of them stop on a work
  budget and resume later, so runs can be time-sliced and measured by the work they did.
  `sample_parallel` runs several at once.
- **People with access to a D-Wave annealer.** The `dwave` module writes models in the
  Solver API's format and reads the returned samples back for the decoders. Submitting
  them over the network is left to a separate adapter.
- **Exact experiments and teaching.** `i64` models keep every penalty and every
  QUBO-to-Ising conversion exact, so results can be checked against brute force.

It is not for solving these problems fast on a classical machine. A direct solver beats a
QUBO encoding there: see [task-allocation](https://github.com/sunsided/task-allocation) for
assignment, [constraint-search](https://github.com/sunsided/constraint-search) for
constraint satisfaction, [mapf-search](https://github.com/sunsided/mapf-search) for
multi-agent path finding and [heuristic-search](https://github.com/sunsided/heuristic-search)
for pathfinding. QAOA (the quantum approximate optimization algorithm) on these models is
planned in
[quantum-planning](https://github.com/sunsided/quantum-planning).

## Scope

- QUBO and Ising models with named binary variables and a penalty builder (exactly-one, at-most-one, linear constraints with slack bits, objective terms)
- Encoders for constraint satisfaction, time-indexed scheduling, assignment and allocation, time-expanded multi-agent path finding and routing
- Decoders that map samples back to plans and report violated constraints, plus penalty-weight tuning
- A sampler interface with classical baselines (exhaustive, simulated annealing, tabu search, parallel tempering)
- Export to D-Wave's Solver API; QAOA on these models is planned in quantum-planning

Encodings fit classical solvers as well as annealers and, later, QAOA, so the crate is useful with or without quantum hardware. The problems come from the planning crates: [constraint-search](https://github.com/sunsided/constraint-search), [temporal-control](https://github.com/sunsided/temporal-control), [resource-timeline](https://github.com/sunsided/resource-timeline), [task-allocation](https://github.com/sunsided/task-allocation) and [mapf-search](https://github.com/sunsided/mapf-search).

## Usage

```sh
cargo add planning-qubo
```

Build a QUBO model from named variables, objective terms and penalized constraints, then
draw low-energy samples and check them against the constraints. Here, a job takes exactly
one of three time slots that cost 3, 1 and 2. The exactly-one constraint becomes the
penalty `P·(1 − x₀ − x₁ − x₂)²`; with `P` above the total cost, every ground state is
feasible.

```rust
use planning_qubo::{Builder, Exhaustive, Sampler, Unlimited};

let mut builder = Builder::<i64>::new();
let slots = ["early", "noon", "late"].map(|name| builder.variable(name));

for (&slot, cost) in slots.iter().zip([3, 1, 2]) {
    builder.add_linear(slot, cost);
}

builder.exactly_one("one slot", &slots, 7);
let problem = builder.build();

let mut sampler = Exhaustive::new();
let sample = sampler.sample(problem.qubo(), Unlimited)?.sample;

assert_eq!(sample.bits, [false, true, false]);
assert_eq!(sample.energy, 1);
assert_eq!(problem.violations(sample.bits).count(), 0);

// The same model as an Ising model: i64 energies are scaled by 4 to stay exact.
let ising = problem.qubo().to_ising();
assert_eq!(ising.energy(sample.bits), 4 * sample.energy);
# Ok::<(), planning_qubo::SampleError>(())
```

The `assignment` module encodes the assignment of tasks to agents with capacities, and
decodes a sample into the agent of each task plus the names of the violated constraints.
The `csp` module encodes constraint satisfaction problems over finite domains with binary
constraints, one-hot, and decodes a sample into a value per variable and the violated
constraints by name. The `scheduling` module encodes time-indexed scheduling (jobs with
durations, release times, deadlines and precedences on a resource with a capacity) and
decodes a sample into start times and the weighted completion time. The `mapf` module
encodes multi-agent path finding over a time-expanded graph with move variables, so vertex
and swap conflicts are both quadratic, and decodes a sample into one path per agent. The
`routing` module encodes the shortest tour from a depot through all cities (asymmetric
distances allowed) and decodes a sample into the tour and its length.

The `dwave` module exports a QUBO or Ising model in the `qp` format of D-Wave's Solver API
(SAPI), on a caller-given map from variables to solver qubits, and parses the returned
sample sets back into assignments for the decoders. It does no networking.
`penalty::safe` picks a penalty weight from the objective that makes every ground state a
feasible optimum, and `penalty::tune` doubles a small weight until a given sampler returns
feasible samples. Both take any encoder as a closure from the weight to an encoding.
`SimulatedAnnealing`, `TabuSearch` and `ParallelTempering` sample larger models from any
seedable `rand_core` RNG. Annealing and tempering use a schedule that they derive from the
model or that you set.

Each sampler owns a reusable workspace: a warm call allocates nothing, and the result
borrows the best state from the workspace. A `search_core::Budget` over the sampler's work
counters stops a run early, with the best state so far; `resume` continues it.

Models store coefficients as `i64` (exact, overflow is checked) or `f64`. The crate is
`no_std` with `alloc`, and depends on `libm`, `rand_core` and `search-core` only.

## Examples

Each example encodes a small but realistic instance, samples it, decodes the plan and
checks it against an exact optimum or bound where one is cheap to compute. Run them with
`cargo run --release --example <name>`, or all at once with `task examples`.

| Example | Problem | Shows |
|---|---|---|
| [`delivery_tour`](https://github.com/sunsided/planning-qubo/blob/main/examples/delivery_tour.rs) | A van's tour through seven German cities | `routing`, `penalty::tune`, parallel tempering vs. brute force |
| [`exam_timetable`](https://github.com/sunsided/planning-qubo/blob/main/examples/exam_timetable.rs) | Exam slots for courses that share students | `csp`, tabu search, named violations when no timetable exists |
| [`bakery_ovens`](https://github.com/sunsided/planning-qubo/blob/main/examples/bakery_ovens.rs) | A bakery's morning on two ovens, with a delivery deadline | `scheduling` with capacity, deadlines and a precedence |
| [`warehouse_robots`](https://github.com/sunsided/planning-qubo/blob/main/examples/warehouse_robots.rs) | Three robots crossing a floor around a shelf block | `mapf`, conflict check, lower bound |
| [`sampler_shootout`](https://github.com/sunsided/planning-qubo/blob/main/examples/sampler_shootout.rs) | Support tickets assigned to engineers | all samplers against the exhaustive optimum, safe vs. tuned penalty |
| [`dwave_map_colouring`](https://github.com/sunsided/planning-qubo/blob/main/examples/dwave_map_colouring.rs) | Colouring the map of Australia | `dwave` export, a simulated SAPI answer, parsing and decoding |
| [`parallel_restarts`](https://github.com/sunsided/planning-qubo/blob/main/examples/parallel_restarts.rs) | A tour of nine European capitals | `sample_parallel` (needs `--features parallel`) |

## Features

- `parallel`: `sample_parallel` and `resume_parallel` run several samplers on one model at
  once, one scoped `std` thread each, and return all runs in sampler order. This is the
  only feature that needs `std`.

## Minimum supported Rust version

Rust 1.85 (edition 2024). The MSRV applies to the library; tests may need a newer
toolchain. It may be raised in a minor release when a dependency or a language feature
calls for it.

## Contributing

See [CONTRIBUTING.md](https://github.com/sunsided/planning-qubo/blob/main/CONTRIBUTING.md).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/sunsided/planning-qubo/blob/main/LICENSE-APACHE))
- MIT license ([LICENSE-MIT](https://github.com/sunsided/planning-qubo/blob/main/LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
in this work by you, as defined in the Apache-2.0 license, shall be dual licensed as above,
without any additional terms or conditions.
