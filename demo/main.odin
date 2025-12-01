package main

import "core:fmt"
import "../"

main :: proc() {
    fmt.println("Init PyTorch via Odin...")

    // Set random seed
    otorch.at_manual_seed(42)

    // // Create a Tensor: ones([3, 3])
    // dims := [?]i64{3, 3}
    
    // // Kind 6 = Float (usually), Device 0 = CPU, ReqGrad 0 = False
    // // TODO: map constants
    // tensor := at_ones(&dims[0], 2, 6, 0, 0)
    
    // fmt.println("Tensor created:")
    // at_print(tensor)
    fmt.println("Done.")
}