package main

import "core:fmt"
import "core:strings"

import ot "../../"

POLY_DEGREE :: 4
W_target, b_target: ot.Tensor

// Builds features [x, x^2, x^3, x^4]
make_features :: proc(x: ot.Tensor) -> ot.Tensor {
    x_unsqueezed := ot.unsqueeze(x, 1)

    // Collect powers
    features := make([dynamic]ot.Tensor)
    defer delete(features)

    for i in 1..=POLY_DEGREE {
        p := ot.pow(x_unsqueezed, ot.scalar(i))
        append(&features, p)
    }
    return ot.cat(features[:], 1)
}

// Approximated function (Ground Truth)
f :: proc(x: ot.Tensor) -> ot.Tensor {
    wx := ot.mm(x, W_target)
    return ot.add(wx, b_target) 
}

get_batch :: proc(batch_size := i64(32)) -> (ot.Tensor, ot.Tensor) {
    random := ot.randn([]i64{batch_size})
    x := make_features(random)
    y := f(x)
    return x, y
}

// String description of polynomial
poly_desc :: proc(W, b: ot.Tensor) -> string {
    // Extract data to CPU slices for formatting
    w_data := ot.tensor_to_slice(W, f32)
    defer delete(w_data)
    b_data := ot.tensor_to_slice(b, f32)
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
    // Global Pool keeps track of memory references
    global_pool := ot.pool_start()
    defer ot.pool_end(global_pool)

    ot.manual_seed(42)
    fmt.println("Poly Regression")

    // Setup Data
    raw_W := ot.randn([]i64{POLY_DEGREE, 1})
    W_target = ot.keep(ot.mul_(raw_W, ot.scalar(5.0)))
    raw_B := ot.randn([]i64{1})
    b_target = ot.keep(ot.mul_(raw_B, ot.scalar(-5.0)))

    fmt.println("Data setup complete.")

    // Setup Model
    fc_W := ot.randn([]i64{POLY_DEGREE, 1})
    ot.requires_grad_(fc_W, true)
    
    fc_b := ot.randn([]i64{1})
    ot.requires_grad_(fc_b, true)

    loss_val: f64
    batch_idx := 0

    // Training Loop
    loop: for {
        batch_idx += 1

        // 2. INNER SCOPE: Create a clean slate for every batch
        batch_pool := ot.pool_start()

        // Get Data
        batch_x, batch_y := get_batch()

        // Forward
        output_data := ot.mm(batch_x, fc_W)
        pred := ot.add(output_data, fc_b)

        // Loss
        loss_tensor := ot.smooth_l1_loss(pred, batch_y, 1, 1)
        
        // TODO: loss_val = ot.item_f64(loss_tensor) 
        loss_val = 0.5

        // Backward
        ot.backward(loss_tensor)
        
        w_grad := ot.grad(fc_W)
        if ot.defined(w_grad) == 1 {
            step := ot.mul(w_grad, ot.scalar(-0.01)) // Reduced LR for stability
            
            // 1. Calculate update (Linked to graph)
            updated_W := ot.add(fc_W, step)
            
            // 2. Detach (Sever link to 'step' and previous graph)
            new_W := ot.detach(updated_W)
            
            // 3. Keep the new leaf tensor
            ot.keep(new_W)
            
            // 4. Enable gradients for next iter
            ot.requires_grad_(new_W, true)

            // 5. Clean up the OLD weight
            if batch_idx > 1 {
                 ot.free_tensor(fc_W)
            }
            // 6. Update reference
            fc_W = new_W
            
            // Note: No need to zero_grad here, we just replaced the entire tensor
        }
        b_grad := ot.grad(fc_b)
        if ot.defined(b_grad) == 1 {
            step := ot.mul_(b_grad, ot.scalar(-0.1))
            
            new_b := ot.add(fc_b, step)
            ot.keep(new_b) // Save from pool deletion
            fc_b = new_b
            
            ot.zero_grad(fc_b)
        }

        // 3. Clean up all intermediate tensors (batch_x, pred, loss, grads, old weights)
        ot.pool_end(batch_pool)

        // Break condition
        if batch_idx >= 100 { break loop } // Safety break
    }
    fmt.printf("Loss: {:.6f} after {} batches\n", loss_val, batch_idx)
    
    desc_learned := poly_desc(fc_W, fc_b)
    fmt.println("==> Learned function:\t", desc_learned)
    delete(desc_learned)

    desc_actual := poly_desc(W_target, b_target)
    fmt.println("==> Actual function:\t", desc_actual)
    delete(desc_actual)
}