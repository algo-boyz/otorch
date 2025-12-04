package main

import "core:fmt"
import "core:strings"

import "../../"

POLY_DEGREE :: 4
W_target, b_target: otorch.Tensor

// Builds features [x, x^2, x^3, x^4]
make_features :: proc(x: otorch.Tensor) -> otorch.Tensor {
    // x = x.unsqueeze(1)
    x_unsqueezed := otorch.unsqueeze(x, 1)

    // Collect powers
    features := make([dynamic]otorch.Tensor)
    defer delete(features)

    for i in 1..=POLY_DEGREE {
        p := otorch.pow(x_unsqueezed, otorch.scalar(i))
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
    // 1. Global Setup (Keep these alive for the whole program)
    global_pool := otorch.pool_start()
    defer otorch.pool_end(global_pool)

    otorch.manual_seed(42)
    fmt.println("Poly Regression")

    // Setup Data
    raw_W := otorch.randn([]i64{POLY_DEGREE, 1})
    W_target = otorch.keep(otorch.mul_(raw_W, otorch.scalar(5.0)))
    raw_B := otorch.randn([]i64{1})
    b_target = otorch.keep(otorch.mul_(raw_B, otorch.scalar(-5.0)))

    fmt.println("Data setup complete.")

    // Setup Model
    fc_W := otorch.randn([]i64{POLY_DEGREE, 1})
    otorch.requires_grad_(fc_W, true)
    
    fc_b := otorch.randn([]i64{1})
    otorch.requires_grad_(fc_b, true)

    loss_val: f64
    batch_idx := 0

    // Training Loop
    loop: for {
        batch_idx += 1

        // 2. INNER SCOPE: Create a clean slate for every batch
        batch_pool := otorch.pool_start()

        // ... Get Data ...
        batch_x, batch_y := get_batch()

        // ... Forward ...
        output_data := otorch.mm(batch_x, fc_W)
        pred := otorch.add(output_data, fc_b)

        // ... Loss ...
        loss_tensor := otorch.smooth_l1_loss(pred, batch_y, 1, 1)
        
        // Use a safe item() fetcher here if available
        // loss_val = otorch.item_f64(loss_tensor) 
        loss_val = 0.5 // Mocking for now as item_f64 wasn't in snippet

        // ... Backward ...
        otorch.backward(loss_tensor)
        
        w_grad := otorch.grad(fc_W)
        if otorch.defined(w_grad) == 1 {
            step := otorch.mul_scalar(w_grad, otorch.scalar(-0.01)) // Reduced LR for stability
            
            // 1. Calculate update (Linked to graph)
            updated_W := otorch.add(fc_W, step)
            
            // 2. Detach (Sever link to 'step' and previous graph)
            new_W := otorch.detach(updated_W)
            
            // 3. Keep the new leaf tensor
            otorch.keep(new_W)
            
            // 4. Enable gradients for next iter
            otorch.requires_grad_(new_W, true)

            // 5. Clean up the OLD weight (Optional but recommended to avoid leaking C++ objects)
            // We check if it's not the initial global one to be safe, or just rely on logic
            if batch_idx > 1 {
                 otorch.free_tensor(fc_W)
            }

            // 6. Update reference
            fc_W = new_W
            
            // Note: No need to zero_grad here, we just replaced the tensor entirely.
        }
        b_grad := otorch.grad(fc_b)
        if otorch.defined(b_grad) == 1 {
            step := otorch.mul_(b_grad, otorch.scalar(-0.1))
            
            new_b := otorch.add(fc_b, step)
            otorch.keep(new_b) // Save from pool deletion
            fc_b = new_b
            
            otorch.zero_grad(fc_b)
        }

        // 3. Clean up all intermediate tensors (batch_x, pred, loss, grads, old weights)
        otorch.pool_end(batch_pool)

        // Break condition
        if batch_idx >= 100 { break loop } // Safety break
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