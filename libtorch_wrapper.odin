package otorch

import t "atg"

Tensor :: t.Tensor
Scalar :: t.Scalar

// Converts a number to a Torch Scalar, runs the block, and frees the Scalar.
// This is a "temporary" scalar used just for the function call.
@(private)
with_scalar :: proc(val: f64, fn: proc(s: Scalar)) {
    s := t.ats_float(val)
    defer t.ats_free(s)
    fn(s)
}

// Same for Integers
@(private)
with_scalar_int :: proc(val: i64, fn: proc(s: Scalar)) {
    s := t.ats_int(val)
    defer t.ats_free(s)
    fn(s)
}

// Common PyTorch Scalar Types (Maps to c10::ScalarType)
ScalarType :: enum i32 {
    Byte   = 0, // uint8
    Char   = 1, // int8
    Short  = 2, // int16
    Int    = 3, // int32
    Long   = 4, // int64
    Half   = 5, // float16
    Float  = 6, // float32
    Double = 7, // float64
}

// Common Device Types
DeviceType :: enum i32 {
    CPU  = 0,
    CUDA = 1,
}

manual_seed :: proc(seed: i64) {
    t.at_manual_seed(seed)
}

defined :: proc(tensor: Tensor) -> i32 {
    return t.at_defined(tensor)
}

tensor_from_data :: proc(
    data: rawptr, 
    dims: []i64, 
    element_size: int, 
    type: ScalarType
) -> Tensor {
    return t.at_tensor_of_data(
        data,
        raw_data(dims),
        len(dims),
        uint(element_size),
        i32(type),
    )
}

free_tensor :: proc(tensor: Tensor) {
    t.at_free(tensor)
}

fill_double :: proc(tensor: Tensor, value: f64) {
    t.at_fill_double(tensor, value)
}

print :: proc(tensor: Tensor) {
    t.at_print(tensor)
}


// Math Ops

// Python equivalent: torch.add(self, other)
add :: proc(self: Tensor, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_add(&out, self, other)
    return out
}

// Wrapper for atg_sub
sub :: proc(self: Tensor, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_sub(&out, self, other)
    return out
}

// Wrapper for atg_mul (element-wise multiplication)
mul :: proc(self: Tensor, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_mul(&out, self, other)
    return out
}

// Wrapper for Matrix Multiplication (mm)
mm :: proc(self: Tensor, mat2: Tensor) -> Tensor {
    out: Tensor
    t.atg_mm(&out, self, mat2)
    return out
}

// Wrapper for ReLU
relu :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_relu(&out, self)
    return out
}