# Protocol

> A Proof-of-Useful-Work consensus where the "work" is verifiable matrix multiplication:
> exact INT8 arithmetic for bit-reproducible results, plus a cheap probabilistic audit that
> certifies a product without recomputing it.

> [!WARNING]
> **Status: research / proof-of-concept. Unaudited. Not for production use.**

---

## Why this is interesting

Quantizing to INT8 is the canonical *lossy* step in machine learning. This project uses the
same INT8 path the other way around: it reconstructs a **bit-exact** floating-point matrix
product from bounded integer slices. Exactness is the point — a result reproducible down to
the bit is the precondition for *verifying* a computation cheaply (a Freivalds-style check,
`O(n²)`) instead of recomputing it (`O(n³)`). On top of the exact product sits a commit–reveal
scheme, so a prover can generate its own challenges non-interactively without being able to
grind a passing forgery.

The decomposition technique is **not novel**: it builds on the Ozaki scheme and the line of
work putting it on integer / INT8 tensor cores. The contribution here is using it as the
compute layer of a verifiable-work consensus, with the audit and soundness analysis that requires.

## What is open

- Exact `INT8 → INT32` partial products with `FP64` reconstruction — implemented and covered by tests (bit-exact against a direct reference).
- Non-interactive commit–reveal challenge derivation — designed, not yet implemented.
- Post-quantum time-seal (delay function) — open design question.
- Data-availability layer (FRI) and private transaction layer — specified, not implemented.
- GPU kernels — current code is a CPU reference.
- **Formal security proofs.** Soundness of individual pieces is being discussed in the open
  (see [Cryptography, honestly](#cryptography-honestly)); none of it is audited.

## Quick start

```bash
git clone https://github.com/name970/Protocol
cd Protocol

cargo test                    # exactness + forgery-rejection tests
cargo run --example demo      # bit-exact INT8 matmul vs an f64 reference (adjust to your layout)
```

## Architecture

A short map from whitepaper sections to code lives in [ARCHITECTURE.md](ARCHITECTURE.md).

## Contributing

Contributors are very welcome — especially people who want to write the code.

- New here? Start with the **`good first issue`** label: self-contained tasks that touch the
  numerics and need no protocol-wide knowledge.
- Read [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup and PR conventions.
- Questions and design discussion go in [Discussions]( #community), not issues.

Every issue references the relevant whitepaper section via `ARCHITECTURE.md`, so you can see
where a task sits in the whole before picking it up.

## Community

- **Discussions:** <https://github.com/name970/Protocol/discussions>
- **Discord:** https://discord.gg/aVwCK8WEYd

## Cryptography, honestly

This is research, and the soundness arguments are being worked through in the open rather than
asserted. Public discussion so far:

- Hash-based nullifier without an elliptic-curve key tree'<https://crypto.stackexchange.com/posts/119735/revisions>'
- Commit-reveal hash binding sufficient to prevent adaptive forgeries in a Stochastic Matrix Audit<https://crypto.stackexchange.com/posts/119728/revisions>
 

If you can break something, open an issue — adversarial review is the most useful contribution.

## A note on the token

The broader protocol design includes a token. **It is not the point of this repository.**
This repo is about the technology; issues and discussion here are technical.

## License
MIT LICENSE
