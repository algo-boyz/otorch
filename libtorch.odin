package otorch

import "core:fmt"
import t "atg"

Tensor :: t.Tensor

// Common Torch Scalar Types (Maps to c10::ScalarType)
ScalarType :: enum i32 {
    Byte = 0, // uint8
    Char = 1, // int8
    Short = 2, // int16
    Int = 3, // int32
    Long = 4, // int64
    Half = 5, // float16
    Float = 6, // float32
    Double = 7, // float64
}

// 2. POOL MECHANICS (Corrected)
@thread_local 
private_tensor_pool: [dynamic]t.Tensor // Store RAW handles

@(private)
track :: proc(raw: t.Tensor) -> t.Tensor {
    append(&private_tensor_pool, raw)
    return raw
}

pool_start :: proc() -> int {
    return len(private_tensor_pool)
}

pool_end :: proc(start_index: int) {
    current_len := len(private_tensor_pool)
    for i := current_len - 1; i >= start_index; i -= 1 {
        raw := private_tensor_pool[i]
        
        // Fix: Cast raw handle to distinct Tensor for checks, or check raw directly
        if t.at_defined(raw) != 0 {
            t.at_free(raw)
        }
    }
    resize(&private_tensor_pool, start_index)
}

keep :: proc(wrapped: Tensor) -> Tensor {
    raw_target := t.Tensor(wrapped) // Unwrap
    for i := len(private_tensor_pool) - 1; i >= 0; i -= 1 {
        if private_tensor_pool[i] == raw_target {
            ordered_remove(&private_tensor_pool, i)
            return wrapped
        }
    }
    return wrapped
}

get_and_reset_last_err :: proc() -> string {
    return string(t.get_and_reset_last_err())
}

manual_seed :: proc(seed: i64) {
    t.at_manual_seed(seed)
}

defined :: proc(tensor: Tensor) -> i32 {
    return t.at_defined(tensor)
}

new_tensor :: proc() -> Tensor {
    return track(t.at_new_tensor())
}

free_tensor :: proc(tensor: Tensor) {
    if defined(tensor) != 0 {
        t.at_free(tensor) // Assuming this calls the underlying deref/free
    }
}

fill_double :: proc(tensor: Tensor, value: f64) {
    t.at_fill_double(tensor, value)
}

// Note: Ensure these are exported (no @private)
add :: proc(self: Tensor, other: Tensor) -> Tensor {
    out: Tensor
    // We must cast 'distinct Tensor' -> 't.Tensor' for the C-binding
    t.atg_add(&out, t.Tensor(self), t.Tensor(other))
    return track(out)
}

mul :: proc(self: Tensor, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_mul(&out, t.Tensor(self), t.Tensor(other))
    return track(out)
}

sub :: proc(self: Tensor, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_sub(&out, t.Tensor(self), t.Tensor(other))
    return track(out)
}

print :: proc(tensor: Tensor, label := "") {
    if label != "" {
        fmt.print(label, ": ")
    }
    t.at_print(tensor)
}

tensor_from_data :: proc(
    data: rawptr,
    dims: []i64,
    element_size: int,
    type: ScalarType
) -> Tensor {
    res := t.at_tensor_of_data(
        data,
        raw_data(dims),
        len(dims),
        uint(element_size),
        i32(type),
    )
    return track(res)
}

tensor_from_slice :: proc(data: []$T, dims: []i64,) -> Tensor {
    dtype := ScalarType.Float 
    
    res := t.at_tensor_of_data(
        raw_data(data),
        raw_data(dims),
        len(dims),
        size_of(T),
        i32(dtype),
    )
    
    // Important: Cast the result to your distinct type!
    return track(res)
}

/* Runs a block of code and frees all temporary tensors created inside it:

    otorch.scoped(proc() {
        a := otorch.rand(10, 10)
        b := otorch.rand(10, 10)

        otorch.print(otorch.add(a, b))
        // Everything cleans up automatically here
    })
*/
scoped :: proc(block: proc()) {
    idx := pool_start()
    defer pool_end(idx)

    block()
}

