---
applyTo: "**/*.rs"
---

# Rust code

- Every module (`//!`), type, field, variant, function, constant and macro has a doc
  comment, private and `pub(crate)` items included. Docs state intent, invariants, units and
  error/panic conditions (`# Errors`, `# Panics`, `# Safety`). They do not restate the name.
- No `mod.rs`: modules with children are `foo.rs` plus `foo/`. Integration tests live under
  `tests/it/`.
- No separator comments (`// -----`, `// =====`). If a file needs sections, split it into
  submodules.
- Leave a blank line after the closing `}` of a block statement when another statement
  follows, and between logical phases of a function. No lumped code.
- One level of abstraction per function. Long loops (simulations, game ticks, solvers) are
  short drivers that call documented step functions. If a block needs a "what" comment,
  extract it into a function named after that comment.
- `clippy::pedantic` clean. Suppression only with a scoped
  `#[expect(lint, reason = "...")]`. Never `#[allow]`, and never crate-level suppression.
- No `unsafe` unless the crate opts out of `#![forbid(unsafe_code)]` with a documented
  reason. Each unsafe block has one operation and a `// SAFETY:` comment.
- The crate is `#![no_std]` with `alloc` (the CI `no-std` job). Its runtime dependencies are
  `libm` (float math), `rand_core` (the RNG trait) and `search-core` (the `Budget` trait). Flag any other new `[dependencies]`
  entry, any `std` use outside the `parallel` feature's module (`src/parallel.rs`, which
  declares `extern crate std`), and any feature-gated item without its `#[cfg]` on the
  re-export in `lib.rs`.
- Float math (`exp`, `log`, `pow`, `sqrt`) goes through `libm`; `core` has none of it.
- Model coefficients are generic over `Coefficient` (`i64` exact, `f64` for export). Its
  arithmetic methods are checked for `i64`; do not use `+`/`*` on generic coefficients.
- `Sampler::sample` draws one assignment of the model's binary variables and aims for low
  energy; lower energy is better, and a sampler need not return the optimum. It returns an
  error when it cannot handle a model (for example, too many variables for `Exhaustive`).
- Samplers own their workspace. A warm call allocates nothing (the counting-allocator tests
  in `tests/it/allocations.rs` check it), results borrow the workspace, and every loop checks
  the `Budget` before each unit of work.
- Public API changes go with a matching update to the README scope list and usage example
  and the crate docs in `lib.rs`. Name every public API change in the PR.
- Mermaid diagrams in doc comments must match the code they describe. Update them when the
  structure changes.
