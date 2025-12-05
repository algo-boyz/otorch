package main

import "core:fmt"
import ot "../../"

main :: proc() {
    fmt.println("Lib Torch via Odin")

    // Set a global random seed
    ot.manual_seed(42)

    pool_idx := ot.pool_start()
    defer ot.pool_end(pool_idx)

    a := ot.tensor_from_slice([]f32{1, 2, 3}, []i64{1, 3}) 
    b := ot.tensor_from_slice([]f32{4, 5, 6}, []i64{1, 3})

    ot.print(a, "Adding A")
    ot.print(b, "to B")

    c := ot.add(a, b)

    ot.print(c, "Yields C")

    // We want a [2, 3] tensor of Float32
    dims := []i64{2, 3}
    data := []f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    }
    fmt.printf("\nCreating tensor of shape %v with data: %v\n", dims, data)

    // Create the Tensor
    tensor := ot.tensor_from_slice(data, dims)

    // Verify properties
    if ot.defined(tensor) == 1 {
        fmt.println("Tensor created successfully!")
    }
    fmt.println("\n[LibTorch Output]:")
    ot.print(tensor)

    fmt.println("\n[Filling tensor with 1's]")
    ot.fill(tensor, 1)
    ot.print(tensor)

    res := ot.mul(tensor, tensor)
    fmt.println("\n[Result of tensor mul tensor:]")
    ot.print(res)

    s5 := ot.scalar(5.0)
    res2 := ot.mul(tensor, s5)
    fmt.println("\n[Result of tensor mul scalar:]")
    ot.print(res2)
    fmt.println("\nDone")
}