# planning-qubo

[![crates.io](https://img.shields.io/crates/v/planning-qubo.svg)](https://crates.io/crates/planning-qubo)
[![docs.rs](https://img.shields.io/docsrs/planning-qubo)](https://docs.rs/planning-qubo)
[![CI](https://github.com/sunsided/planning-qubo/actions/workflows/ci.yml/badge.svg)](https://github.com/sunsided/planning-qubo/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sunsided/planning-qubo/graph/badge.svg)](https://codecov.io/gh/sunsided/planning-qubo)
[![MSRV](https://img.shields.io/crates/msrv/planning-qubo)](https://crates.io/crates/planning-qubo)
[![License](https://img.shields.io/crates/l/planning-qubo)](https://github.com/sunsided/planning-qubo#license)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)

Planning, scheduling and constraint problems as QUBO and Ising models: encoders, decoders, penalty tuning, classical samplers and export for quantum annealers and QAOA.

## Who this is for

- **People with access to a quantum annealer or a QAOA setup** who need planning,
  scheduling or assignment problems as QUBO or Ising models. The penalty builder turns
  named variables and constraints (exactly-one, at-most-one, linear with slack bits) into
  a model, and the decoders map a sample back to a plan and name each violated
  constraint. The assignment problem has a ready encoder; other problems are built from
  the same constraints. Export to annealer formats is planned.
- **Researchers comparing samplers on the same model**: the exhaustive sampler gives the
  exact optimum of small models, and simulated annealing, with a seeded RNG and a
  configurable schedule, shows how close a heuristic gets. Both stop on a work budget and
  resume later, so runs can be time-sliced and measured by the work they did.
- **Exact experiments and teaching**: `i64` models keep every penalty and every
  QUBO-to-Ising conversion exact, so results can be checked against brute force.

It is not for solving these problems fast on a classical machine. A direct solver beats a
QUBO encoding there: see [task-allocation](https://github.com/sunsided/task-allocation) for
assignment, [constraint-search](https://github.com/sunsided/constraint-search) for
constraint satisfaction, [mapf-search](https://github.com/sunsided/mapf-search) for
multi-agent path finding and [heuristic-search](https://github.com/sunsided/heuristic-search)
for pathfinding. QAOA on these models is planned in
[quantum-planning](https://github.com/sunsided/quantum-planning).

## Scope

- QUBO and Ising models with named binary variables and a penalty builder (exactly-one, at-most-one, linear constraints with slack bits, objective terms)
- Encoders for constraint satisfaction, time-indexed scheduling, assignment and allocation, time-expanded multi-agent path finding and routing
- Decoders that map samples back to plans and report violated constraints, plus penalty-weight tuning
- A sampler interface with classical baselines (exhaustive, simulated annealing, tabu search, parallel tempering)
- Export to annealer formats and adapters for quantum frameworks

Encodings fit annealers and QAOA as well as classical solvers, so the crate is useful with or without quantum hardware. The problems come from the planning crates: [constraint-search](https://github.com/sunsided/constraint-search), [temporal-control](https://github.com/sunsided/temporal-control), [resource-timeline](https://github.com/sunsided/resource-timeline), [task-allocation](https://github.com/sunsided/task-allocation) and [mapf-search](https://github.com/sunsided/mapf-search).

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
`SimulatedAnnealing` samples larger models from any seedable `rand_core` RNG, on a
schedule that it derives from the model or that you set.

Each sampler owns a reusable workspace: a warm call allocates nothing, and the result
borrows the best state from the workspace. A `search_core::Budget` over the sampler's work
counters stops a run early, with the best state so far; `resume` continues it.

Models store coefficients as `i64` (exact, overflow is checked) or `f64`. The crate is
`no_std` with `alloc`, and depends on `libm`, `rand_core` and `search-core` only.

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
