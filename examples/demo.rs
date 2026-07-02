//! Demonstrates bit-exact multiplication of real f32 matrices through the INT8 slice path.
//! Run with:  cargo run --example demo

use protocol::{matmul_f32, matmul_f32_naive, F32Matrix};

fn main() {
    let frac = 16; // fixed-point fractional bits
    let b = 7; // INT8 slice width

    // Real f32 matrices (these values land exactly on the 2^-frac grid).
    let a = F32Matrix { rows: 2, cols: 2, data: vec![1.5, -2.25, 0.125, 3.75] };
    let bb = F32Matrix { rows: 2, cols: 2, data: vec![-0.5, 8.0, 100.25, -1.125] };

    let via = matmul_f32(&a, &bb, frac, b);
    let naive = matmul_f32_naive(&a, &bb);

    println!("f32 matrices multiplied through the exact INT8 path:");
    println!("via INT8 slices : {via:?}");
    println!("direct f64      : {naive:?}");
    assert_eq!(via, naive, "the slice product must match a direct f64 product bit-for-bit");
    println!("bit-exact match — an exact FP result reconstructed from INT8, no rounding error.");
}
