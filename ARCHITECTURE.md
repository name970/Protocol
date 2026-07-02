# Architecture

This maps whitepaper sections to code, so you can see where a task sits before picking it up.

| Whitepaper | What it does | Code | Status |
|---|---|---|---|
| §2.2 Error-free decomposition | FP→INT8 slicing, exact INT8→INT32 partial products, FP64 reassembly | `src/lib.rs` | working (CPU) |
| §2.2/§2.4 SMA audit |Stochastic Matrix Audit over F_p, integer-exactness/range check, commit–reveal challenge derivation | `src/<...>` | working |
| §2.3 Aggregation | scaled FP64 summation of verified partial products | `src/<...>` | working |
| §2.4 Time-seal (VDF) | delay function + verification | `src/<...>` | open / prototype |
| §4.1 P-DAL | Reed–Solomon + FRI data-availability | — | not started |
| §4.2 zk-TWL | shielded pool, nullifiers, Dilithium | — | not started |

**Where to start:** issues labelled `good first issue` touch §2.2 and need no protocol-wide context.
