//! Demonstrates bit-exact matrix multiplication through the INT8 slice path.
//! Run with:  cargo run --example demo

use protocol::{decompose_matrix, matmul_reference, matmul_via_slices, no_overflow_bits, IntMatrix};

fn main() {
    let b = 7; // INT8 slice width

    // Two small integer matrices (think: fixed-point / scaled floating-point values).
    let a = IntMatrix { rows: 2, cols: 2, data: vec![123_456, -98_765, 42, -7] };
    let bb = IntMatrix { rows: 2, cols: 2, data: vec![-31_415, 27_182, 99_999, -12_345] };

    // Decompose A into INT8 slices and confirm the bound.
    let slices = decompose_matrix(&a, b);
    println!("A splits into {} INT8 slices (every entry |x| < 2^{}).", slices.len(), b);

    // Multiply via the slice path and via a direct reference — they must match exactly.
    let via = matmul_via_slices(&a, &bb, b);
    let reference = matmul_reference(&a, &bb);

    println!("via INT8 slices : {:?}", via.data);
    println!("direct reference: {:?}", reference.data);
    assert_eq!(via, reference, "the slice product must be bit-exact");

    println!(
        "bit-exact match — no rounding error. (INT32 accumulator uses {} of 31 bits here.)",
        no_overflow_bits(b, a.cols as u32)
    );
}
