package main

import "core:fmt"
import "../../"

main :: proc() {
    fmt.println("Lib Torch via Odin")

    // Set a global random seed
    otorch.manual_seed(42)

    pool_idx := otorch.pool_start()
    defer otorch.pool_end(pool_idx)

    a := otorch.tensor_from_slice([]f32{1, 2, 3}, []i64{1, 3}) 
    b := otorch.tensor_from_slice([]f32{4, 5, 6}, []i64{1, 3})

    otorch.print(a, "Adding A")
    otorch.print(b, "to B")

    c := otorch.add(a, b)

    otorch.print(c, "Yields C")

    // We want a [2, 3] tensor of Float32
    dims := []i64{2, 3}
    data := []f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    }
    fmt.printf("\nCreating tensor of shape %v with data: %v\n", dims, data)

    // Create the Tensor
    tensor := otorch.tensor_from_slice(data, dims)

    // Verify properties
    if otorch.defined(tensor) == 1 {
        fmt.println("Tensor created successfully!")
    }
    fmt.println("\n[LibTorch Output]:")
    otorch.print(tensor)

    fmt.println("\n[Filling tensor with 99's]")
    otorch.fill_double(tensor, 99)
    otorch.print(tensor)

    fmt.println("\nDone")
}