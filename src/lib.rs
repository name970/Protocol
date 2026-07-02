//! Exact matrix multiplication via INT8 slice decomposition — the Ozaki-scheme core,
//! plus a fixed-point `f32` front-end.
//!
//! The idea (whitepaper §2.2): an integer matrix is split into a short sum of
//! **bounded integer slices**, each entry fitting a signed `i8`:
//!
//! ```text
//!     A = Σ_i  A^(i) · (2^b)^i,      with every entry of A^(i) satisfying |x| < 2^b
//! ```
//!
//! Slice pairs are multiplied **exactly** (`i8` in, `i32` accumulate — no rounding),
//! then the full product is reassembled by scaling and summing the integer partial
//! products. The result is bit-for-bit exact.
//!
//! The `f32` layer ([`matmul_f32`]) maps floating-point matrices onto a fixed-point
//! grid (multiples of `2^-frac_bits`), runs them through the exact integer core, and
//! reconstructs the product in `f64`. Values already on the grid are converted
//! exactly, so for on-grid inputs the reconstructed product is bit-for-bit exact.
//!
//! A production build layers on top of this: per-tile dynamic-range scaling, ULP-based
//! pruning of negligible slice pairs, and actual INT8 tensor-core kernels for
//! `matmul_slice`. The exactness argument and the no-overflow invariant live here.

// ------------------------------------------------------------------ integer core

/// A dense, row-major integer matrix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<i64>,
}

/// One slice `A^(i)` — a matrix whose entries all fit in a signed `i8`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct I8Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<i8>,
}

/// Balanced base-`2^b` digits of `value` (the per-element slices).
///
/// Returns signed digits `d_0, d_1, …` with `value = Σ_i d_i · (2^b)^i`.
/// Every digit satisfies `|d_i| < 2^b`, so for `b <= 7` they fit in a signed `i8`.
/// Handles negative values directly (no separate sign bookkeeping needed).
pub fn decompose_balanced(mut value: i64, b: u32) -> Vec<i8> {
    assert!((1..=7).contains(&b), "b must be in 1..=7 so digits fit in i8");
    let beta: i64 = 1 << b;
    let half: i64 = beta >> 1;
    if value == 0 {
        return vec![0];
    }
    let mut digits = Vec::new();
    while value != 0 {
        // r in [0, beta); shift into the balanced range [-half, half)
        let mut r = value.rem_euclid(beta);
        if r >= half {
            r -= beta;
        }
        digits.push(r as i8);
        value = (value - r) / beta; // exact: value - r is a multiple of beta
    }
    digits
}

/// Reconstruct the integer from its balanced base-`2^b` digits (inverse of `decompose_balanced`).
pub fn recompose_balanced(digits: &[i8], b: u32) -> i64 {
    let beta: i64 = 1 << b;
    let mut acc: i64 = 0;
    let mut place: i64 = 1;
    for &d in digits {
        acc += d as i64 * place;
        place = place.wrapping_mul(beta);
    }
    acc
}

/// Decompose every entry of a matrix into slices `A^(0), A^(1), …`.
/// Shorter per-entry expansions are zero-padded so all slices share the same shape.
pub fn decompose_matrix(m: &IntMatrix, b: u32) -> Vec<I8Matrix> {
    let per_entry: Vec<Vec<i8>> = m.data.iter().map(|&v| decompose_balanced(v, b)).collect();
    let n_slices = per_entry.iter().map(Vec::len).max().unwrap_or(1);
    (0..n_slices)
        .map(|k| I8Matrix {
            rows: m.rows,
            cols: m.cols,
            data: per_entry.iter().map(|d| *d.get(k).unwrap_or(&0)).collect(),
        })
        .collect()
}

/// Direct integer matmul — the ground-truth reference (wide `i128` accumulation).
pub fn matmul_reference(a: &IntMatrix, b: &IntMatrix) -> IntMatrix {
    assert_eq!(a.cols, b.rows, "shape mismatch");
    let (n, k, m) = (a.rows, a.cols, b.cols);
    let mut data = vec![0i64; n * m];
    for i in 0..n {
        for j in 0..m {
            let mut acc: i128 = 0;
            for t in 0..k {
                acc += a.data[i * k + t] as i128 * b.data[t * m + j] as i128;
            }
            data[i * m + j] = i64::try_from(acc).expect("product entry exceeds i64 range");
        }
    }
    IntMatrix { rows: n, cols: m, data }
}

/// Exact partial product of two slices: `i8` inputs, `i32` accumulator, no rounding.
///
/// The `i32` accumulator never overflows as long as the no-overflow invariant holds:
/// `2*b + ceil(log2(n)) <= 31` for inner dimension `n` (see [`no_overflow_bits`]).
fn matmul_slice(a: &I8Matrix, b: &I8Matrix) -> IntMatrix {
    assert_eq!(a.cols, b.rows, "shape mismatch");
    let (n, k, m) = (a.rows, a.cols, b.cols);
    let mut data = vec![0i64; n * m];
    for i in 0..n {
        for j in 0..m {
            let mut acc: i32 = 0;
            for t in 0..k {
                acc += a.data[i * k + t] as i32 * b.data[t * m + j] as i32;
            }
            data[i * m + j] = acc as i64;
        }
    }
    IntMatrix { rows: n, cols: m, data }
}

/// Exact matmul via the Ozaki slice path. Bit-for-bit equal to [`matmul_reference`].
pub fn matmul_via_slices(a: &IntMatrix, b: &IntMatrix, b_bits: u32) -> IntMatrix {
    assert_eq!(a.cols, b.rows, "shape mismatch");
    let a_slices = decompose_matrix(a, b_bits);
    let b_slices = decompose_matrix(b, b_bits);
    let (n, m) = (a.rows, b.cols);
    let beta: i128 = 1 << b_bits;

    let mut acc = vec![0i128; n * m];
    for (i, sa) in a_slices.iter().enumerate() {
        for (j, sb) in b_slices.iter().enumerate() {
            let partial = matmul_slice(sa, sb); // exact
            let scale: i128 = beta.pow((i + j) as u32); // (2^b)^(i+j)
            for (idx, slot) in acc.iter_mut().enumerate() {
                *slot += scale * partial.data[idx] as i128;
            }
        }
    }
    IntMatrix {
        rows: n,
        cols: m,
        data: acc
            .into_iter()
            .map(|v| i64::try_from(v).expect("result entry exceeds i64 range"))
            .collect(),
    }
}

/// Bits needed by an `i8`×`i8` partial-product accumulator for inner dimension `n`:
/// `2*b + ceil(log2(n))`. Must be `<= 31` to stay inside `i32`.
pub fn no_overflow_bits(b: u32, n: u32) -> u32 {
    let ceil_log2_n = if n <= 1 { 0 } else { 32 - (n - 1).leading_zeros() };
    2 * b + ceil_log2_n
}

// ---------------------------------------------------------------- f32 front-end

/// A dense, row-major `f32` matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct F32Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

/// Map an `f32` onto the fixed-point grid of step `2^-frac_bits` (round to nearest).
/// Exact when `x` is already a multiple of `2^-frac_bits`.
pub fn f32_to_fixed(x: f32, frac_bits: u32) -> i64 {
    (x as f64 * (1i64 << frac_bits) as f64).round() as i64
}

/// Reconstruct an `f64` from a fixed-point integer with `frac_bits` fractional bits.
pub fn fixed_to_f64(v: i64, frac_bits: u32) -> f64 {
    v as f64 / (1i64 << frac_bits) as f64
}

/// Grid an `f32` matrix onto fixed-point integers.
pub fn f32_matrix_to_fixed(m: &F32Matrix, frac_bits: u32) -> IntMatrix {
    IntMatrix {
        rows: m.rows,
        cols: m.cols,
        data: m.data.iter().map(|&x| f32_to_fixed(x, frac_bits)).collect(),
    }
}

/// Exact product of two `f32` matrices via the INT8 slice path, reconstructed in `f64`.
///
/// Both inputs are gridded to `frac_bits`, multiplied exactly as integers, and the
/// product (scaled by `2^(2·frac_bits)`) is reconstructed. For on-grid inputs whose
/// exact product fits in `f64`, the result is bit-for-bit exact. Returns a row-major
/// `rows × cols` vector.
pub fn matmul_f32(a: &F32Matrix, b: &F32Matrix, frac_bits: u32, b_bits: u32) -> Vec<f64> {
    let ai = f32_matrix_to_fixed(a, frac_bits);
    let bi = f32_matrix_to_fixed(b, frac_bits);
    let product = matmul_via_slices(&ai, &bi, b_bits); // = 2^(2F) · (gridded product)
    product
        .data
        .iter()
        .map(|&v| fixed_to_f64(v, 2 * frac_bits))
        .collect()
}

/// Plain `f64` matrix product — the baseline you'd get without the slice trick.
pub fn matmul_f32_naive(a: &F32Matrix, b: &F32Matrix) -> Vec<f64> {
    assert_eq!(a.cols, b.rows, "shape mismatch");
    let (n, k, m) = (a.rows, a.cols, b.cols);
    let mut out = vec![0.0f64; n * m];
    for i in 0..n {
        for j in 0..m {
            let mut acc = 0.0f64;
            for t in 0..k {
                acc += a.data[i * k + t] as f64 * b.data[t * m + j] as f64;
            }
            out[i * m + j] = acc;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tiny deterministic xorshift PRNG so tests need no external crates.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn in_range(&mut self, lo: i64, hi: i64) -> i64 {
            let span = (hi - lo + 1) as u64;
            lo + (self.next() % span) as i64
        }
    }

    #[test]
    fn decompose_roundtrip_and_bound() {
        let b = 7;
        let beta = 1i64 << b;
        let mut rng = Rng(0x1234_5678_9abc_def0);
        for _ in 0..100_000 {
            let v = rng.in_range(-(1 << 40), 1 << 40);
            let d = decompose_balanced(v, b);
            assert_eq!(recompose_balanced(&d, b), v, "roundtrip must be exact");
            assert!(
                d.iter().all(|&x| (x as i64).abs() < beta),
                "every slice digit must satisfy |x| < 2^b"
            );
        }
    }

    #[test]
    fn slice_matmul_is_bit_exact() {
        let b = 7;
        let mut rng = Rng(0xdead_beef_cafe_0001);
        for _ in 0..500 {
            let n = rng.in_range(1, 6) as usize;
            let k = rng.in_range(1, 6) as usize;
            let m = rng.in_range(1, 6) as usize;
            let a = IntMatrix {
                rows: n,
                cols: k,
                data: (0..n * k).map(|_| rng.in_range(-(1 << 18), 1 << 18)).collect(),
            };
            let bb = IntMatrix {
                rows: k,
                cols: m,
                data: (0..k * m).map(|_| rng.in_range(-(1 << 18), 1 << 18)).collect(),
            };
            assert_eq!(
                matmul_via_slices(&a, &bb, b),
                matmul_reference(&a, &bb),
                "slice product must be bit-for-bit exact"
            );
        }
    }

    #[test]
    fn no_overflow_invariant_reference_config() {
        // Reference parameters b = 7, n = 512: 14 + 9 = 23 <= 31.
        assert_eq!(no_overflow_bits(7, 512), 23);
        assert!(no_overflow_bits(7, 512) <= 31);
        // Headroom: b = 7 stays safe up to n = 2^17.
        assert!(no_overflow_bits(7, 1 << 17) <= 31);
    }

    #[test]
    fn f32_pipeline_is_bit_exact() {
        let frac = 16u32;
        let b = 7u32;
        let mut rng = Rng(0xf00d_face_1234_5678);
        for _ in 0..500 {
            let n = rng.in_range(1, 5) as usize;
            let k = rng.in_range(1, 5) as usize;
            let m = rng.in_range(1, 5) as usize;
            // On-grid f32 values (integer / 2^frac), bounded so the exact product fits in f64.
            let scale = (1i64 << frac) as f64;
            let sample = |rng: &mut Rng| (rng.in_range(-(1 << 18), 1 << 18) as f64 / scale) as f32;
            let a = F32Matrix { rows: n, cols: k, data: (0..n * k).map(|_| sample(&mut rng)).collect() };
            let bb = F32Matrix { rows: k, cols: m, data: (0..k * m).map(|_| sample(&mut rng)).collect() };

            let via = matmul_f32(&a, &bb, frac, b);
            let naive = matmul_f32_naive(&a, &bb);
            // For on-grid, bounded inputs both are exact, so they must agree bit-for-bit.
            assert_eq!(via, naive, "f32 pipeline must match a direct f64 product");
        }
    }
}
