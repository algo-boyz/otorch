package main

import "core:fmt"
import "../"

main :: proc() {
    fmt.println("--- Init PyTorch via Odin ---")

    // 1. Set global random seed
    otorch.at_manual_seed(42)

    // 2. Define Shape and Data in Odin
    // We want a [2, 3] tensor of Float32
    dims := [?]i64{2, 3}
    data := [?]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    }

    fmt.printf("Creating tensor of shape %v with data: %v\n", dims, data)

    // 3. Create the Tensor
    // Note: We cast to raw_data/pointers to pass to C
    tensor := otorch.at_tensor_of_data(
        raw_data(data[:]),              // void *vs
        raw_data(dims[:]),              // int64_t *dims
        len(dims),                      // size_t ndims
        size_of(f32),                   // size_t element_size_in_bytes
        i32(otorch.ScalarType.Float),          // int type
    )

    // 4. Verify properties
    if otorch.at_defined(tensor) == 1 {
        fmt.println("Tensor created successfully!")
    }

    // 5. Print using LibTorch's internal printer
    fmt.println("\n[LibTorch Output]:")
    otorch.at_print(tensor)

    // 6. Test Modifying Data
    fmt.println("\n[Filling tensor with 99.0...]")
    otorch.at_fill_double(tensor, 99.0)
    otorch.at_print(tensor)

    // 7. Cleanup
    // Important: C++ allocates memory for the Tensor object, we must free it.
    otorch.at_free(tensor)
    fmt.println("\nDone.")
}