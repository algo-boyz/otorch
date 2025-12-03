package main

import "core:fmt"
import "core:strings"

import "../../"

POLY_DEGREE :: 4
W_target: otorch.Tensor
b_target: otorch.Tensor

// Builds features [x, x^2, x^3, x^4]
make_features :: proc(x: otorch.Tensor) -> otorch.Tensor {
    // x = x.unsqueeze(1)
    x_unsqueezed := otorch.unsqueeze(x, 1)

    // Collect powers
    features := make([dynamic]otorch.Tensor)
    defer delete(features)

    for i in 1..=POLY_DEGREE {
        p := otorch.pow_tensor_scalar(x_unsqueezed, otorch.scalar_int(i64(i)))
        append(&features, p)
    }

    return otorch.cat(features[:], 1)
}

// Approximated function (Ground Truth)
f :: proc(x: otorch.Tensor) -> otorch.Tensor {
    // return x.mm(W_target) + b_target.item()
    // Note: In Odin/C-API adding a 1-element tensor acts like scalar broadcasting
    wx := otorch.mm(x, W_target)
    return otorch.add(wx, b_target) 
}

get_batch :: proc(batch_size := i64(32)) -> (otorch.Tensor, otorch.Tensor) {
    random := otorch.randn([]i64{batch_size})
    x := make_features(random)
    y := f(x)
    return x, y
}

// String description of polynomial
poly_desc :: proc(W, b: otorch.Tensor) -> string {
    // Extract data to CPU slices for formatting
    w_data := otorch.tensor_to_slice(W, f32)
    defer delete(w_data)
    b_data := otorch.tensor_to_slice(b, f32)
    defer delete(b_data)

    sb := strings.builder_make()
    
    strings.write_string(&sb, "y = ")
    for w, i in w_data {
        fmt.sbprintf(&sb, "{:+.2f} x^{} ", w, i + 1)
    }
    fmt.sbprintf(&sb, "{:+.2f}", b_data[0])

    return strings.to_string(sb)
}

main :: proc() {
    otorch.manual_seed(42)

    fmt.println("Poly Regression")

    // Setup Ground Truth
    
    // W_target = torch.randn(POLY_DEGREE, 1) * 5
    W_target = otorch.randn([]i64{POLY_DEGREE, 1})
    otorch.mul_scalar(W_target, otorch.scalar_float(5.0))

    // b_target = torch.randn(1) * 5
    b_target = otorch.randn([]i64{1})
    otorch.mul_scalar(b_target, otorch.scalar_float(-5.0))

    // Define Model (Linear Layer manually)
    fc_W := otorch.randn([]i64{POLY_DEGREE, 1})
    otorch.requires_grad_(fc_W, true)
    
    fc_b := otorch.randn([]i64{1})
    otorch.requires_grad_(fc_b, true)

    loss_val: f64
    batch_idx := 0

    // Training Loop
    loop: for {
        batch_idx += 1

        // Start a pool for this iteration. Any tensor created inside 
        // will be freed at `pool_end` UNLESS we `keep` it.
        pool_state := otorch.pool_start() // TODO: this has to use otorch.scoped

        // Get data
        batch_x, batch_y := get_batch()

        // Forward pass: fc(batch_x) -> x.mm(W) + b
        output_data := otorch.mm(batch_x, fc_W)
        // Add bias
        pred := otorch.add(output_data, fc_b)

        // Loss
        loss_tensor := otorch.smooth_l1_loss(pred, batch_y, 1, 1)
        loss_val = 1
        // loss_val = otorch.item_f64(loss_tensor)

        // Backward pass
        otorch.backward(loss_tensor)
        
        // Update W
        w_grad := otorch.grad(fc_W)
        if otorch.defined(w_grad) == 1 {
            // scale gradient: grad * -0.1
            step := otorch.mul_scalar(w_grad, otorch.scalar_float(-0.1)) 
            otorch.add(fc_W, step)
            otorch.zero_grad(fc_W) // Reset gradient
        }

        // Update b
        b_grad := otorch.grad(fc_b)
        if otorch.defined(b_grad) == 1 {
            step := otorch.mul_scalar(b_grad, otorch.scalar_float(-0.1))
            otorch.add(fc_b, step)
            otorch.zero_grad(fc_b) // Reset gradient
        }
        // Clean up all temporary tensors (batch_x, pred, loss, grads, etc.)
        // TODO: keep the model weights (fc_W, fc_b) and targets (W_target)
        otorch.pool_end(pool_state)

        if loss_val < 1e-3 {
            break loop
        }
    }

    fmt.printf("Loss: {:.6f} after {} batches\n", loss_val, batch_idx)
    
    // View(-1) is implied by our slice helper iterating linear memory
    desc_learned := poly_desc(fc_W, fc_b)
    fmt.println("==> Learned function:\t", desc_learned)
    delete(desc_learned)

    desc_actual := poly_desc(W_target, b_target)
    fmt.println("==> Actual function:\t", desc_actual)
    delete(desc_actual)
}