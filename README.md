# planning-qubo

[![crates.io](https://img.shields.io/crates/v/planning-qubo.svg)](https://crates.io/crates/planning-qubo)
[![docs.rs](https://img.shields.io/docsrs/planning-qubo)](https://docs.rs/planning-qubo)
[![CI](https://github.com/sunsided/planning-qubo/actions/workflows/ci.yml/badge.svg)](https://github.com/sunsided/planning-qubo/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sunsided/planning-qubo/graph/badge.svg)](https://codecov.io/gh/sunsided/planning-qubo)
[![MSRV](https://img.shields.io/crates/msrv/planning-qubo)](https://crates.io/crates/planning-qubo)
[![License](https://img.shields.io/crates/l/planning-qubo)](https://github.com/sunsided/planning-qubo#license)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)

Planning, scheduling and constraint problems as QUBO and Ising models: encoders, decoders, penalty tuning, classical samplers and export for quantum annealers and QAOA.

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

A sampler draws low-energy assignments of a model's binary variables. Here, a job takes
exactly one of three time slots that cost 3, 1 and 2. The costs sit on the diagonal of the
QUBO, and the exactly-one constraint becomes the penalty `P·(1 − x₀ − x₁ − x₂)²`, which
adds `−P` to each diagonal entry and `2P` to each pair. An exhaustive sampler, the baseline
for small models, tries every assignment.

```rust
use planning_qubo::Sampler;

/// A QUBO over three binary variables, upper triangular: `E(x) = Σ q[i][j]·xᵢ·xⱼ` for `i ≤ j`.
struct Qubo {
    q: [[i64; 3]; 3],
}

impl Qubo {
    /// The energy of one assignment.
    fn energy(&self, x: [bool; 3]) -> i64 {
        let mut energy = 0;
        for i in 0..3 {
            for j in i..3 {
                if x[i] && x[j] {
                    energy += self.q[i][j];
                }
            }
        }

        energy
    }
}

/// Tries all eight assignments and keeps one with the lowest energy.
struct Exhaustive;

impl Sampler<Qubo> for Exhaustive {
    type Sample = [bool; 3];

    fn sample(&mut self, model: &Qubo) -> [bool; 3] {
        (0..8_u8)
            .map(|bits| [bits & 1 != 0, bits & 2 != 0, bits & 4 != 0])
            .min_by_key(|&x| model.energy(x))
            .expect("there are eight assignments")
    }
}

// Slot costs 3, 1, 2 with penalty weight P = 4.
let model = Qubo {
    q: [[3 - 4, 8, 8], [0, 1 - 4, 8], [0, 0, 2 - 4]],
};

let slots = Exhaustive.sample(&model);

assert_eq!(slots, [false, true, false]);

// Taking two slots breaks the constraint and costs more than the cheapest single slot.
assert!(model.energy([true, true, false]) > model.energy(slots));
```

The crate is `no_std` and has no dependencies.

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
