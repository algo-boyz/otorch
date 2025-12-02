package main

import "core:fmt"
import "../"

main :: proc() {
    fmt.println("Lib Torch via Odin")

    // Set a global random seed
    otorch.manual_seed(42)

    // Define Shape and Data
    // We want a [2, 3] tensor of Float32
    dims := [?]i64{2, 3}
    data := [?]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    }
    fmt.printf("Creating tensor of shape %v with data: %v\n", dims, data)

    // Create the Tensor
    tensor := otorch.tensor_from_data(
        raw_data(data[:]), 
        dims[:], 
        size_of(f32), 
        .Float
    )
    // Verify properties
    if otorch.defined(tensor) == 1 {
        fmt.println("Tensor created successfully!")
    }
    fmt.println("\n[LibTorch Output]:")
    otorch.print(tensor)

    fmt.println("\n[Filling tensor with 99's]")
    otorch.fill_double(tensor, 99)
    otorch.print(tensor)

    // Important: C++ allocates memory for the Tensor obj, we must free it
    otorch.free_tensor(tensor)

    fmt.println("\nDone")
}