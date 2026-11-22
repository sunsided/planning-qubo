# planning-qubo

Planning, scheduling and constraint problems as QUBO and Ising models: encoders, decoders, penalty tuning, classical samplers and export for quantum annealers and QAOA.

## Scope

- QUBO and Ising models with named binary variables and a penalty builder (exactly-one, at-most-one, linear constraints with slack bits, objective terms)
- Encoders for constraint satisfaction, time-indexed scheduling, assignment and allocation, time-expanded multi-agent path finding and routing
- Decoders that map samples back to plans and report violated constraints, plus penalty-weight tuning
- A sampler interface with classical baselines (exhaustive, simulated annealing, tabu search, parallel tempering)
- Export to annealer formats and adapters for quantum frameworks

Encodings fit annealers and QAOA as well as classical solvers, so the crate is useful with or without quantum hardware. The problems come from the planning crates: [constraint-search](https://github.com/sunsided/constraint-search), [temporal-control](https://github.com/sunsided/temporal-control), [resource-timeline](https://github.com/sunsided/resource-timeline), [task-allocation](https://github.com/sunsided/task-allocation) and [mapf-search](https://github.com/sunsided/mapf-search).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/sunsided/planning-qubo/blob/main/LICENSE-APACHE))
- MIT license ([LICENSE-MIT](https://github.com/sunsided/planning-qubo/blob/main/LICENSE-MIT))

at your option.
