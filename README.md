
**Downloads required:**

1.  **LibTorch (C++):** Download the **cxx11 ABI** version (Linux) or standard version (Windows) from [pytorch.org](https://pytorch.org/get-started/locally/). Unzip it to `libtorch` in the root of this project.
2.  **tch-rs C api wra[[er]]:** Download capi files from [tch-rs](https://www.google.com/search?q=https://github.com/LaurentMazare/tch-rs/tree/main/torch-sys/libtch) and put them in `capi/`.


### 3\. How to Use

```odin
package main

import "core:fmt"
import "core:c"

// Define Opaque Handle
Tensor :: distinct rawptr

// Import Wrapper Lib
// Ensure odin_torch.lib/.dll is accessible to the linker/runner
foreign import lib "build/odin_torch.lib" 

foreign lib {
    // Bind procs from torch_api.h
    // void at_manual_seed(int64_t seed);
    at_manual_seed :: proc(seed: i64) ---
    
    // void *at_ones(int64_t *size, int size_len, int kind, int device, int requires_grad);
    at_ones :: proc(size: [^]i64, size_len: c.int, kind: c.int, device: c.int, requires_grad: c.int) -> Tensor ---
    
    // void at_print(void *tensor);
    at_print :: proc(t: Tensor) ---

    // ...
}

main :: proc() {
    fmt.println("Init PyTorch via Odin...")

    // Set random seed
    at_manual_seed(42)

    // Create a Tensor: ones([3, 3])
    dims := [?]i64{3, 3}
    
    // Kind 6 = Float (usually), Device 0 = CPU, ReqGrad 0 = False
    // TODO: map constants
    tensor := at_ones(&dims[0], 2, 6, 0, 0)
    
    fmt.println("Tensor created:")
    at_print(tensor)
    fmt.println("Done.")
}
```
