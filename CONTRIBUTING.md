# Contributing

Thanks for considering a contribution. This is a research-stage project; clear, scoped help is
exactly what moves it forward.

## Getting started
1. Install a recent stable Rust toolchain (`rustup`).
2. `cargo build` and `cargo test` should pass on a fresh clone. If they don't, open an issue.
3. Read `ARCHITECTURE.md` to see how the code maps to the whitepaper.

## Picking something to work on
- New contributors: start with the `good first issue` label — self-contained, numerics-only tasks.
- Comment on the issue to claim it before starting, so we don't duplicate work.
- For anything open-ended (design, security), open a thread in **Discussions** first.

## Pull requests
- Branch from `main`; one logical change per PR.
- `cargo fmt --all`, `cargo clippy --all-targets -- -D warnings`, and `cargo test` must pass — CI enforces this.
- Reference the issue you're closing (`Closes #NN`).
- If you change the code layout, update `ARCHITECTURE.md`.

## Reporting a security/soundness problem
If you think you can break the audit, the commitment scheme, or any soundness claim, that is a
*welcome* contribution — open an issue describing the attack.

## Questions
Use Discussions, not issues, for questions. No question about the math is too basic.
