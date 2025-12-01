package otorch

import "core:c"

// Define opaque ptr handle
Tensor :: distinct rawptr

foreign import lib "libtorch_wrapper.dylib"

foreign lib {
    // Bind procs from torch_api.h
    // void at_manual_seed(int64_t seed);
    at_manual_seed :: proc(seed: i64) ---
    
    // void *at_ones(int64_t *size, int size_len, int kind, int device, int requires_grad);
    at_ones :: proc(size: [^]i64, size_len: c.int, kind: c.int, device: c.int, requires_grad: c.int) -> Tensor ---
    
    // void at_print(void *tensor);
    at_print :: proc(t: Tensor) ---
}