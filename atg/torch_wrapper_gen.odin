// AUTOMATICALLY GENERATED BINDINGS
package atg

import "core:c"
foreign import lib "torch_wrapper.dylib"

foreign lib {
	atg___and__ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg___and__tensor_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg___iand__ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg___iand__tensor_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg___ilshift__ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg___ilshift__tensor_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg___ior__ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg___ior__tensor_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg___irshift__ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg___irshift__tensor_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg___ixor__ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg___ixor__tensor_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg___lshift__ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg___lshift__scalar_out_ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg___lshift__tensor_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg___lshift__tensor_out_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg___or__ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg___or__tensor_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg___rshift__ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg___rshift__scalar_out_ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg___rshift__tensor_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg___rshift__tensor_out_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg___xor__ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg___xor__tensor_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg__adaptive_avg_pool2d :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int) ---
	atg__adaptive_avg_pool2d_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor) ---
	atg__adaptive_avg_pool2d_backward_out :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor) ---
	atg__adaptive_avg_pool2d_out :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int) ---
	atg__adaptive_avg_pool3d :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int) ---
	atg__adaptive_avg_pool3d_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor) ---
	atg__adaptive_avg_pool3d_backward_out :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor) ---
	atg__adaptive_avg_pool3d_out :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int) ---
	atg__add_batch_dim :: proc(out: ^Tensor, self: Tensor, batch_dim: i64, level: i64) ---
	atg__add_relu :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg__add_relu_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg__add_relu_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg__add_relu_scalar :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg__add_relu_scalar_ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg__add_relu_scalar_out :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg__addmm_activation :: proc(out: ^Tensor, self: Tensor, mat1: Tensor, mat2: Tensor, use_gelu: c.int) ---
	atg__addmm_activation_out :: proc(out: ^Tensor, self: Tensor, mat1: Tensor, mat2: Tensor, use_gelu: c.int) ---
	atg__aminmax :: proc(out: ^Tensor, self: Tensor) ---
	atg__aminmax_dim :: proc(out: ^Tensor, self: Tensor, dim: i64, keepdim: c.int) ---
	atg__aminmax_dim_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, self: Tensor, dim: i64, keepdim: c.int) ---
	atg__aminmax_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, self: Tensor) ---
	atg__amp_update_scale :: proc(out: ^Tensor, self: Tensor, growth_tracker: Tensor, found_inf: Tensor, scale_growth_factor: f64, scale_backoff_factor: f64, growth_interval: i64) ---
	atg__amp_update_scale_ :: proc(out: ^Tensor, self: Tensor, growth_tracker: Tensor, found_inf: Tensor, scale_growth_factor: f64, scale_backoff_factor: f64, growth_interval: i64) ---
	atg__amp_update_scale_out :: proc(out: ^Tensor, self: Tensor, growth_tracker: Tensor, found_inf: Tensor, scale_growth_factor: f64, scale_backoff_factor: f64, growth_interval: i64) ---
	atg__assert_scalar :: proc(self_scalar: Scalar, assert_msg_ptr: cstring, assert_msg_len: c.int) ---
	atg__assert_tensor_metadata :: proc(a: Tensor, size_data: [^]i64, size_len: c.int, stride_data: [^]i64, stride_len: c.int, dtype: c.int, device: c.int, layout: rawptr) ---
	atg__autocast_to_full_precision :: proc(out: ^Tensor, self: Tensor, cuda_enabled: c.int, cpu_enabled: c.int) ---
	atg__autocast_to_reduced_precision :: proc(out: ^Tensor, self: Tensor, cuda_enabled: c.int, cpu_enabled: c.int, cuda_dtype: c.int, cpu_dtype: c.int) ---
	atg__batch_norm_no_update :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor, running_mean: Tensor, running_var: Tensor, momentum: f64, eps: f64) ---
	atg__batch_norm_no_update_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, out2: Tensor, out3: Tensor, input: Tensor, weight: Tensor, bias: Tensor, running_mean: Tensor, running_var: Tensor, momentum: f64, eps: f64) ---
	atg__batch_norm_with_update :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor, running_mean: Tensor, running_var: Tensor, momentum: f64, eps: f64) ---
	atg__batch_norm_with_update_functional :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor, running_mean: Tensor, running_var: Tensor, momentum: f64, eps: f64) ---
	atg__batch_norm_with_update_out :: proc(out: ^Tensor, save_mean: Tensor, save_invstd: Tensor, reserve: Tensor, input: Tensor, weight: Tensor, bias: Tensor, running_mean: Tensor, running_var: Tensor, momentum: f64, eps: f64) ---
	atg__cast_byte :: proc(out: ^Tensor, self: Tensor, non_blocking: c.int) ---
	atg__cast_char :: proc(out: ^Tensor, self: Tensor, non_blocking: c.int) ---
	atg__cast_double :: proc(out: ^Tensor, self: Tensor, non_blocking: c.int) ---
	atg__cast_float :: proc(out: ^Tensor, self: Tensor, non_blocking: c.int) ---
	atg__cast_half :: proc(out: ^Tensor, self: Tensor, non_blocking: c.int) ---
	atg__cast_int :: proc(out: ^Tensor, self: Tensor, non_blocking: c.int) ---
	atg__cast_long :: proc(out: ^Tensor, self: Tensor, non_blocking: c.int) ---
	atg__cast_short :: proc(out: ^Tensor, self: Tensor, non_blocking: c.int) ---
	atg__cdist_backward :: proc(out: ^Tensor, grad: Tensor, x1: Tensor, x2: Tensor, p: f64, cdist: Tensor) ---
	atg__cdist_backward_out :: proc(out: ^Tensor, grad: Tensor, x1: Tensor, x2: Tensor, p: f64, cdist: Tensor) ---
	atg__cholesky_solve_helper :: proc(out: ^Tensor, self: Tensor, A: Tensor, upper: c.int) ---
	atg__cholesky_solve_helper_out :: proc(out: ^Tensor, self: Tensor, A: Tensor, upper: c.int) ---
	atg__chunk_cat :: proc(out: ^Tensor, tensors_data: ^Tensor, tensors_len: c.int, dim: i64, num_chunks: i64) ---
	atg__chunk_cat_out :: proc(out: ^Tensor, tensors_data: ^Tensor, tensors_len: c.int, dim: i64, num_chunks: i64) ---
	atg__coalesce :: proc(out: ^Tensor, self: Tensor) ---
	atg__coalesce_out :: proc(out: ^Tensor, self: Tensor) ---
	atg__coalesced :: proc(out: ^Tensor, self: Tensor, coalesced: c.int) ---
	atg__coalesced_ :: proc(out: ^Tensor, self: Tensor, coalesced: c.int) ---
	atg__coalesced_out :: proc(out: ^Tensor, self: Tensor, coalesced: c.int) ---
	atg__compute_linear_combination :: proc(out: ^Tensor, input: Tensor, coefficients: Tensor) ---
	atg__compute_linear_combination_out :: proc(out: ^Tensor, input: Tensor, coefficients: Tensor) ---
	atg__conj :: proc(out: ^Tensor, self: Tensor) ---
	atg__conj_copy :: proc(out: ^Tensor, self: Tensor) ---
	atg__conj_copy_out :: proc(out: ^Tensor, self: Tensor) ---
	atg__conj_physical :: proc(out: ^Tensor, self: Tensor) ---
	atg__conj_physical_out :: proc(out: ^Tensor, self: Tensor) ---
	atg__conv_depthwise2d :: proc(out: ^Tensor, self: Tensor, weight: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int) ---
	atg__conv_depthwise2d_out :: proc(out: ^Tensor, self: Tensor, weight: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int) ---
	atg__convert_indices_from_coo_to_csr :: proc(out: ^Tensor, self: Tensor, size: i64, out_int32: c.int) ---
	atg__convert_indices_from_coo_to_csr_out :: proc(out: ^Tensor, self: Tensor, size: i64, out_int32: c.int) ---
	atg__convert_indices_from_csr_to_coo :: proc(out: ^Tensor, crow_indices: Tensor, col_indices: Tensor, out_int32: c.int, transpose: c.int) ---
	atg__convert_indices_from_csr_to_coo_out :: proc(out: ^Tensor, crow_indices: Tensor, col_indices: Tensor, out_int32: c.int, transpose: c.int) ---
	atg__convert_weight_to_int4pack :: proc(out: ^Tensor, self: Tensor, innerKTiles: i64) ---
	atg__convert_weight_to_int4pack_for_cpu :: proc(out: ^Tensor, self: Tensor, innerKTiles: i64) ---
	atg__convolution :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, transposed: c.int, output_padding_data: [^]i64, output_padding_len: c.int, groups: i64, benchmark: c.int, deterministic: c.int, cudnn_enabled: c.int, allow_tf32: c.int) ---
	atg__convolution_deprecated :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, transposed: c.int, output_padding_data: [^]i64, output_padding_len: c.int, groups: i64, benchmark: c.int, deterministic: c.int, cudnn_enabled: c.int) ---
	atg__convolution_mode :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_ptr: cstring, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, groups: i64) ---
	atg__convolution_out :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, transposed: c.int, output_padding_data: [^]i64, output_padding_len: c.int, groups: i64, benchmark: c.int, deterministic: c.int, cudnn_enabled: c.int, allow_tf32: c.int) ---
	atg__copy_from :: proc(out: ^Tensor, self: Tensor, dst: Tensor, non_blocking: c.int) ---
	atg__copy_from_and_resize :: proc(out: ^Tensor, self: Tensor, dst: Tensor) ---
	atg__copy_from_and_resize_out :: proc(out: ^Tensor, self: Tensor, dst: Tensor) ---
	atg__copy_from_out :: proc(out: ^Tensor, self: Tensor, dst: Tensor, non_blocking: c.int) ---
	atg__cslt_compress :: proc(out: ^Tensor, input: Tensor) ---
	atg__cslt_sparse_mm :: proc(out: ^Tensor, compressed_A: Tensor, dense_B: Tensor, bias: Tensor, alpha: Tensor, out_dtype: c.int, transpose_result: c.int, alg_id: i64, split_k: i64, split_k_mode: i64) ---
	atg__ctc_loss :: proc(out: ^Tensor, log_probs: Tensor, targets: Tensor, input_lengths_data: [^]i64, input_lengths_len: c.int, target_lengths_data: [^]i64, target_lengths_len: c.int, blank: i64, zero_infinity: c.int) ---
	atg__ctc_loss_backward :: proc(out: ^Tensor, grad: Tensor, log_probs: Tensor, targets: Tensor, input_lengths_data: [^]i64, input_lengths_len: c.int, target_lengths_data: [^]i64, target_lengths_len: c.int, neg_log_likelihood: Tensor, log_alpha: Tensor, blank: i64, zero_infinity: c.int) ---
	atg__ctc_loss_backward_out :: proc(out: ^Tensor, grad: Tensor, log_probs: Tensor, targets: Tensor, input_lengths_data: [^]i64, input_lengths_len: c.int, target_lengths_data: [^]i64, target_lengths_len: c.int, neg_log_likelihood: Tensor, log_alpha: Tensor, blank: i64, zero_infinity: c.int) ---
	atg__ctc_loss_backward_tensor :: proc(out: ^Tensor, grad: Tensor, log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor, neg_log_likelihood: Tensor, log_alpha: Tensor, blank: i64, zero_infinity: c.int) ---
	atg__ctc_loss_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, log_probs: Tensor, targets: Tensor, input_lengths_data: [^]i64, input_lengths_len: c.int, target_lengths_data: [^]i64, target_lengths_len: c.int, blank: i64, zero_infinity: c.int) ---
	atg__ctc_loss_tensor :: proc(out: ^Tensor, log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor, blank: i64, zero_infinity: c.int) ---
	atg__ctc_loss_tensor_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor, blank: i64, zero_infinity: c.int) ---
	atg__cudnn_attention_backward :: proc(out: ^Tensor, grad_out: Tensor, query: Tensor, key: Tensor, value: Tensor, logsumexp: Tensor, philox_seed: Tensor, philox_offset: Tensor, attn_bias: Tensor, cum_seq_q: Tensor, cum_seq_k: Tensor, max_q: i64, max_k: i64, dropout_p: f64, is_causal: c.int, scale_v: f64, scale_null: rawptr) ---
	atg__cudnn_ctc_loss :: proc(out: ^Tensor, log_probs: Tensor, targets: Tensor, input_lengths_data: [^]i64, input_lengths_len: c.int, target_lengths_data: [^]i64, target_lengths_len: c.int, blank: i64, deterministic: c.int, zero_infinity: c.int) ---
	atg__cudnn_ctc_loss_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, log_probs: Tensor, targets: Tensor, input_lengths_data: [^]i64, input_lengths_len: c.int, target_lengths_data: [^]i64, target_lengths_len: c.int, blank: i64, deterministic: c.int, zero_infinity: c.int) ---
	atg__cudnn_ctc_loss_tensor :: proc(out: ^Tensor, log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor, blank: i64, deterministic: c.int, zero_infinity: c.int) ---
	atg__cudnn_init_dropout_state :: proc(out: ^Tensor, dropout: f64, train: c.int, dropout_seed: i64, options_kind: c.int, options_device: c.int) ---
	atg__cudnn_init_dropout_state_out :: proc(out: ^Tensor, dropout: f64, train: c.int, dropout_seed: i64) ---
	atg__cudnn_rnn :: proc(out: ^Tensor, input: Tensor, weight_data: ^Tensor, weight_len: c.int, weight_stride0: i64, weight_buf: Tensor, hx: Tensor, cx: Tensor, mode: i64, hidden_size: i64, proj_size: i64, num_layers: i64, batch_first: c.int, dropout: f64, train: c.int, bidirectional: c.int, batch_sizes_data: [^]i64, batch_sizes_len: c.int, dropout_state: Tensor) ---
	atg__cudnn_rnn_flatten_weight :: proc(out: ^Tensor, weight_arr_data: ^Tensor, weight_arr_len: c.int, weight_stride0: i64, input_size: i64, mode: i64, hidden_size: i64, proj_size: i64, num_layers: i64, batch_first: c.int, bidirectional: c.int) ---
	atg__cudnn_rnn_flatten_weight_out :: proc(out: ^Tensor, weight_arr_data: ^Tensor, weight_arr_len: c.int, weight_stride0: i64, input_size: i64, mode: i64, hidden_size: i64, proj_size: i64, num_layers: i64, batch_first: c.int, bidirectional: c.int) ---
	atg__cudnn_rnn_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, out2: Tensor, out3: Tensor, out4: Tensor, input: Tensor, weight_data: ^Tensor, weight_len: c.int, weight_stride0: i64, weight_buf: Tensor, hx: Tensor, cx: Tensor, mode: i64, hidden_size: i64, proj_size: i64, num_layers: i64, batch_first: c.int, dropout: f64, train: c.int, bidirectional: c.int, batch_sizes_data: [^]i64, batch_sizes_len: c.int, dropout_state: Tensor) ---
	atg__dim_arange :: proc(out: ^Tensor, like: Tensor, dim: i64) ---
	atg__dirichlet_grad :: proc(out: ^Tensor, x: Tensor, alpha: Tensor, total: Tensor) ---
	atg__dirichlet_grad_out :: proc(out: ^Tensor, x: Tensor, alpha: Tensor, total: Tensor) ---
	atg__dyn_quant_matmul_4bit :: proc(out: ^Tensor, inp: Tensor, packed_weights: Tensor, block_size: i64, in_features: i64, out_features: i64) ---
	atg__dyn_quant_pack_4bit_weight :: proc(out: ^Tensor, weights: Tensor, scales_zeros: Tensor, bias: Tensor, block_size: i64, in_features: i64, out_features: i64) ---
	atg__efficient_attention_backward :: proc(out: ^Tensor, grad_out_: Tensor, query: Tensor, key: Tensor, value: Tensor, bias: Tensor, cu_seqlens_q: Tensor, cu_seqlens_k: Tensor, max_seqlen_q: i64, max_seqlen_k: i64, logsumexp: Tensor, dropout_p: f64, philox_seed: Tensor, philox_offset: Tensor, custom_mask_type: i64, bias_requires_grad: c.int, scale_v: f64, scale_null: rawptr, num_splits_key_v: i64, num_splits_key_null: rawptr, window_size_v: i64, window_size_null: rawptr, shared_storage_dqdkdv: c.int) ---
	atg__efficientzerotensor :: proc(out: ^Tensor, size_data: [^]i64, size_len: c.int, options_kind: c.int, options_device: c.int) ---
	atg__efficientzerotensor_out :: proc(out: ^Tensor, size_data: [^]i64, size_len: c.int) ---
	atg__embedding_bag :: proc(out: ^Tensor, weight: Tensor, indices: Tensor, offsets: Tensor, scale_grad_by_freq: c.int, mode: i64, sparse: c.int, per_sample_weights: Tensor, include_last_offset: c.int, padding_idx: i64) ---
	atg__embedding_bag_backward :: proc(out: ^Tensor, grad: Tensor, indices: Tensor, offsets: Tensor, offset2bag: Tensor, bag_size: Tensor, maximum_indices: Tensor, num_weights: i64, scale_grad_by_freq: c.int, mode: i64, sparse: c.int, per_sample_weights: Tensor, padding_idx: i64) ---
	atg__embedding_bag_dense_backward :: proc(out: ^Tensor, grad: Tensor, indices: Tensor, offset2bag: Tensor, bag_size: Tensor, maximum_indices: Tensor, num_weights: i64, scale_grad_by_freq: c.int, mode: i64, per_sample_weights: Tensor, padding_idx: i64) ---
	atg__embedding_bag_dense_backward_out :: proc(out: ^Tensor, grad: Tensor, indices: Tensor, offset2bag: Tensor, bag_size: Tensor, maximum_indices: Tensor, num_weights: i64, scale_grad_by_freq: c.int, mode: i64, per_sample_weights: Tensor, padding_idx: i64) ---
	atg__embedding_bag_forward_only :: proc(out: ^Tensor, weight: Tensor, indices: Tensor, offsets: Tensor, scale_grad_by_freq: c.int, mode: i64, sparse: c.int, per_sample_weights: Tensor, include_last_offset: c.int, padding_idx: i64) ---
	atg__embedding_bag_forward_only_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, out2: Tensor, out3: Tensor, weight: Tensor, indices: Tensor, offsets: Tensor, scale_grad_by_freq: c.int, mode: i64, sparse: c.int, per_sample_weights: Tensor, include_last_offset: c.int, padding_idx: i64) ---
	atg__embedding_bag_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, out2: Tensor, out3: Tensor, weight: Tensor, indices: Tensor, offsets: Tensor, scale_grad_by_freq: c.int, mode: i64, sparse: c.int, per_sample_weights: Tensor, include_last_offset: c.int, padding_idx: i64) ---
	atg__embedding_bag_per_sample_weights_backward :: proc(out: ^Tensor, grad: Tensor, weight: Tensor, indices: Tensor, offsets: Tensor, offset2bag: Tensor, mode: i64, padding_idx: i64) ---
	atg__embedding_bag_per_sample_weights_backward_out :: proc(out: ^Tensor, grad: Tensor, weight: Tensor, indices: Tensor, offsets: Tensor, offset2bag: Tensor, mode: i64, padding_idx: i64) ---
	atg__embedding_bag_sparse_backward :: proc(out: ^Tensor, grad: Tensor, indices: Tensor, offsets: Tensor, offset2bag: Tensor, bag_size: Tensor, num_weights: i64, scale_grad_by_freq: c.int, mode: i64, per_sample_weights: Tensor, padding_idx: i64) ---
	atg__empty_affine_quantized :: proc(out: ^Tensor, size_data: [^]i64, size_len: c.int, options_kind: c.int, options_device: c.int, scale: f64, zero_point: i64) ---
	atg__empty_affine_quantized_out :: proc(out: ^Tensor, size_data: [^]i64, size_len: c.int, scale: f64, zero_point: i64) ---
	atg__empty_per_channel_affine_quantized :: proc(out: ^Tensor, size_data: [^]i64, size_len: c.int, scales: Tensor, zero_points: Tensor, axis: i64, options_kind: c.int, options_device: c.int) ---
	atg__empty_per_channel_affine_quantized_out :: proc(out: ^Tensor, size_data: [^]i64, size_len: c.int, scales: Tensor, zero_points: Tensor, axis: i64) ---
	atg__euclidean_dist :: proc(out: ^Tensor, x1: Tensor, x2: Tensor) ---
	atg__euclidean_dist_out :: proc(out: ^Tensor, x1: Tensor, x2: Tensor) ---
	atg__fake_quantize_learnable_per_channel_affine :: proc(out: ^Tensor, self: Tensor, scale: Tensor, zero_point: Tensor, axis: i64, quant_min: i64, quant_max: i64, grad_factor: f64) ---
	atg__fake_quantize_learnable_per_channel_affine_backward :: proc(out: ^Tensor, grad: Tensor, self: Tensor, scale: Tensor, zero_point: Tensor, axis: i64, quant_min: i64, quant_max: i64, grad_factor: f64) ---
	atg__fake_quantize_learnable_per_channel_affine_out :: proc(out: ^Tensor, self: Tensor, scale: Tensor, zero_point: Tensor, axis: i64, quant_min: i64, quant_max: i64, grad_factor: f64) ---
	atg__fake_quantize_learnable_per_tensor_affine :: proc(out: ^Tensor, self: Tensor, scale: Tensor, zero_point: Tensor, quant_min: i64, quant_max: i64, grad_factor: f64) ---
	atg__fake_quantize_learnable_per_tensor_affine_backward :: proc(out: ^Tensor, grad: Tensor, self: Tensor, scale: Tensor, zero_point: Tensor, quant_min: i64, quant_max: i64, grad_factor: f64) ---
	atg__fake_quantize_learnable_per_tensor_affine_out :: proc(out: ^Tensor, self: Tensor, scale: Tensor, zero_point: Tensor, quant_min: i64, quant_max: i64, grad_factor: f64) ---
	atg__fake_quantize_per_tensor_affine_cachemask_tensor_qparams :: proc(out: ^Tensor, self: Tensor, scale: Tensor, zero_point: Tensor, fake_quant_enabled: Tensor, quant_min: i64, quant_max: i64) ---
	atg__fake_quantize_per_tensor_affine_cachemask_tensor_qparams_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, self: Tensor, scale: Tensor, zero_point: Tensor, fake_quant_enabled: Tensor, quant_min: i64, quant_max: i64) ---
	atg__fft_c2c :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, normalization: i64, forward: c.int) ---
	atg__fft_c2c_out :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, normalization: i64, forward: c.int) ---
	atg__fft_c2r :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, normalization: i64, last_dim_size: i64) ---
	atg__fft_c2r_out :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, normalization: i64, last_dim_size: i64) ---
	atg__fft_r2c :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, normalization: i64, onesided: c.int) ---
	atg__fft_r2c_out :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, normalization: i64, onesided: c.int) ---
	atg__fill_mem_eff_dropout_mask_ :: proc(out: ^Tensor, self: Tensor, dropout_p: f64, seed: i64, offset: i64) ---
	atg__flash_attention_backward :: proc(out: ^Tensor, grad_out: Tensor, query: Tensor, key: Tensor, value: Tensor, logsumexp: Tensor, cum_seq_q: Tensor, cum_seq_k: Tensor, max_q: i64, max_k: i64, dropout_p: f64, is_causal: c.int, rng_state: Tensor, unused: Tensor, scale_v: f64, scale_null: rawptr, window_size_left_v: i64, window_size_left_null: rawptr, window_size_right_v: i64, window_size_right_null: rawptr) ---
	atg__foobar :: proc(out: ^Tensor, self: Tensor, arg1: c.int, arg2: c.int, arg3: c.int) ---
	atg__foobar_out :: proc(out: ^Tensor, self: Tensor, arg1: c.int, arg2: c.int, arg3: c.int) ---
	atg__functional_assert_async :: proc(out: ^Tensor, self: Tensor, assert_msg_ptr: cstring, assert_msg_len: c.int, dep_token: Tensor) ---
	atg__functional_assert_scalar :: proc(out: ^Tensor, self_scalar: Scalar, assert_msg_ptr: cstring, assert_msg_len: c.int, dep_token: Tensor) ---
	atg__functional_sym_constrain_range :: proc(out: ^Tensor, size: Scalar, min_v: i64, min_null: rawptr, max_v: i64, max_null: rawptr, dep_token: Tensor) ---
	atg__functional_sym_constrain_range_for_size :: proc(out: ^Tensor, size: Scalar, min_v: i64, min_null: rawptr, max_v: i64, max_null: rawptr, dep_token: Tensor) ---
	atg__fused_dropout :: proc(out: ^Tensor, self: Tensor, p: f64) ---
	atg__fused_dropout_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, self: Tensor, p: f64) ---
	atg__fused_moving_avg_obs_fq_helper :: proc(out: ^Tensor, self: Tensor, observer_on: Tensor, fake_quant_on: Tensor, running_min: Tensor, running_max: Tensor, scale: Tensor, zero_point: Tensor, averaging_const: f64, quant_min: i64, quant_max: i64, ch_axis: i64, per_row_fake_quant: c.int, symmetric_quant: c.int) ---
	atg__fused_moving_avg_obs_fq_helper_functional :: proc(out: ^Tensor, self: Tensor, observer_on: Tensor, fake_quant_on: Tensor, running_min: Tensor, running_max: Tensor, scale: Tensor, zero_point: Tensor, averaging_const: f64, quant_min: i64, quant_max: i64, ch_axis: i64, per_row_fake_quant: c.int, symmetric_quant: c.int) ---
	atg__fused_moving_avg_obs_fq_helper_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, self: Tensor, observer_on: Tensor, fake_quant_on: Tensor, running_min: Tensor, running_max: Tensor, scale: Tensor, zero_point: Tensor, averaging_const: f64, quant_min: i64, quant_max: i64, ch_axis: i64, per_row_fake_quant: c.int, symmetric_quant: c.int) ---
	atg__fused_rms_norm :: proc(out: ^Tensor, input: Tensor, normalized_shape_data: [^]i64, normalized_shape_len: c.int, weight: Tensor, eps_v: f64, eps_null: rawptr) ---
	atg__fw_primal :: proc(out: ^Tensor, self: Tensor, level: i64) ---
	atg__fw_primal_copy :: proc(out: ^Tensor, self: Tensor, level: i64) ---
	atg__fw_primal_copy_out :: proc(out: ^Tensor, self: Tensor, level: i64) ---
	atg__gather_sparse_backward :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor, grad: Tensor) ---
	atg__grid_sampler_2d_cpu_fallback :: proc(out: ^Tensor, input: Tensor, grid: Tensor, interpolation_mode: i64, padding_mode: i64, align_corners: c.int) ---
	atg__grid_sampler_2d_cpu_fallback_backward :: proc(out: ^Tensor, grad_output: Tensor, input: Tensor, grid: Tensor, interpolation_mode: i64, padding_mode: i64, align_corners: c.int) ---
	atg__grid_sampler_2d_cpu_fallback_out :: proc(out: ^Tensor, input: Tensor, grid: Tensor, interpolation_mode: i64, padding_mode: i64, align_corners: c.int) ---
	atg__grouped_mm :: proc(out: ^Tensor, self: Tensor, mat2: Tensor, offs: Tensor, bias: Tensor, out_dtype: c.int) ---
	atg__histogramdd_bin_edges_out :: proc(out_data: ^Tensor, out_len: c.int, self: Tensor, bins_data: [^]i64, bins_len: c.int, range_data: [^]f64, range_len: c.int, weight: Tensor, density: c.int) ---
	atg__histogramdd_from_bin_cts :: proc(out: ^Tensor, self: Tensor, bins_data: [^]i64, bins_len: c.int, range_data: [^]f64, range_len: c.int, weight: Tensor, density: c.int) ---
	atg__histogramdd_from_bin_cts_out :: proc(out: ^Tensor, self: Tensor, bins_data: [^]i64, bins_len: c.int, range_data: [^]f64, range_len: c.int, weight: Tensor, density: c.int) ---
	atg__histogramdd_from_bin_tensors :: proc(out: ^Tensor, self: Tensor, bins_data: ^Tensor, bins_len: c.int, weight: Tensor, density: c.int) ---
	atg__histogramdd_from_bin_tensors_out :: proc(out: ^Tensor, self: Tensor, bins_data: ^Tensor, bins_len: c.int, weight: Tensor, density: c.int) ---
	atg__index_put_impl :: proc(out: ^Tensor, self: Tensor, indices_data: ^Tensor, indices_len: c.int, values: Tensor, accumulate: c.int, unsafe: c.int) ---
	atg__index_put_impl_ :: proc(out: ^Tensor, self: Tensor, indices_data: ^Tensor, indices_len: c.int, values: Tensor, accumulate: c.int, unsafe: c.int) ---
	atg__index_put_impl_out :: proc(out: ^Tensor, self: Tensor, indices_data: ^Tensor, indices_len: c.int, values: Tensor, accumulate: c.int, unsafe: c.int) ---
	atg__indices :: proc(out: ^Tensor, self: Tensor) ---
	atg__indices_copy :: proc(out: ^Tensor, self: Tensor) ---
	atg__indices_copy_out :: proc(out: ^Tensor, self: Tensor) ---
	atg__int_mm :: proc(out: ^Tensor, self: Tensor, mat2: Tensor) ---
	atg__int_mm_out :: proc(out: ^Tensor, self: Tensor, mat2: Tensor) ---
	atg__is_all_true :: proc(out: ^Tensor, self: Tensor) ---
	atg__is_any_true :: proc(out: ^Tensor, self: Tensor) ---
	atg__lazy_clone :: proc(out: ^Tensor, self: Tensor) ---
	atg__linalg_check_errors :: proc(info: Tensor, api_name_ptr: cstring, api_name_len: c.int, is_matrix: c.int) ---
	atg__linalg_det :: proc(out: ^Tensor, A: Tensor) ---
	atg__linalg_det_result :: proc(out: ^Tensor, result: Tensor, LU: Tensor, pivots: Tensor, A: Tensor) ---
	atg__linalg_eigh :: proc(out: ^Tensor, A: Tensor, UPLO_ptr: cstring, UPLO_len: c.int, compute_v: c.int) ---
	atg__linalg_eigh_eigenvalues :: proc(out: ^Tensor, eigenvalues: Tensor, eigenvectors: Tensor, A: Tensor, UPLO_ptr: cstring, UPLO_len: c.int, compute_v: c.int) ---
	atg__linalg_eigvals :: proc(out: ^Tensor, self: Tensor) ---
	atg__linalg_slogdet :: proc(out: ^Tensor, A: Tensor) ---
	atg__linalg_slogdet_sign :: proc(out: ^Tensor, sign: Tensor, logabsdet: Tensor, LU: Tensor, pivots: Tensor, A: Tensor) ---
	atg__linalg_solve_ex :: proc(out: ^Tensor, A: Tensor, B: Tensor, left: c.int, check_errors: c.int) ---
	atg__linalg_solve_ex_result :: proc(out: ^Tensor, result: Tensor, LU: Tensor, pivots: Tensor, info: Tensor, A: Tensor, B: Tensor, left: c.int, check_errors: c.int) ---
	atg__linalg_svd :: proc(out: ^Tensor, A: Tensor, full_matrices: c.int, compute_uv: c.int, driver_ptr: cstring, driver_len: c.int) ---
	atg__linalg_svd_u :: proc(out: ^Tensor, U: Tensor, S: Tensor, Vh: Tensor, A: Tensor, full_matrices: c.int, compute_uv: c.int, driver_ptr: cstring, driver_len: c.int) ---
	atg__log_softmax :: proc(out: ^Tensor, self: Tensor, dim: i64, half_to_float: c.int) ---
	atg__log_softmax_backward_data :: proc(out: ^Tensor, grad_output: Tensor, output: Tensor, dim: i64, input_dtype: c.int) ---
	atg__log_softmax_backward_data_out :: proc(out: ^Tensor, grad_output: Tensor, output: Tensor, dim: i64, input_dtype: c.int) ---
	atg__log_softmax_out :: proc(out: ^Tensor, self: Tensor, dim: i64, half_to_float: c.int) ---
	atg__logcumsumexp :: proc(out: ^Tensor, self: Tensor, dim: i64) ---
	atg__logcumsumexp_out :: proc(out: ^Tensor, self: Tensor, dim: i64) ---
	atg__lstm_mps :: proc(out: ^Tensor, input: Tensor, hx_data: ^Tensor, hx_len: c.int, params_data: ^Tensor, params_len: c.int, has_biases: c.int, num_layers: i64, dropout: f64, train: c.int, bidirectional: c.int, batch_first: c.int) ---
	atg__lstm_mps_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, out2: Tensor, out3: Tensor, out4: Tensor, out5: Tensor, input: Tensor, hx_data: ^Tensor, hx_len: c.int, params_data: ^Tensor, params_len: c.int, has_biases: c.int, num_layers: i64, dropout: f64, train: c.int, bidirectional: c.int, batch_first: c.int) ---
	atg__lu_with_info :: proc(out: ^Tensor, self: Tensor, pivot: c.int, check_errors: c.int) ---
	atg__make_dep_token :: proc(out: ^Tensor, options_kind: c.int, options_device: c.int) ---
	atg__make_dual :: proc(out: ^Tensor, primal: Tensor, tangent: Tensor, level: i64) ---
	atg__make_dual_copy :: proc(out: ^Tensor, primal: Tensor, tangent: Tensor, level: i64) ---
	atg__make_dual_copy_out :: proc(out: ^Tensor, primal: Tensor, tangent: Tensor, level: i64) ---
	atg__make_per_channel_quantized_tensor :: proc(out: ^Tensor, self: Tensor, scale: Tensor, zero_point: Tensor, axis: i64) ---
	atg__make_per_channel_quantized_tensor_out :: proc(out: ^Tensor, self: Tensor, scale: Tensor, zero_point: Tensor, axis: i64) ---
	atg__make_per_tensor_quantized_tensor :: proc(out: ^Tensor, self: Tensor, scale: f64, zero_point: i64) ---
	atg__make_per_tensor_quantized_tensor_out :: proc(out: ^Tensor, self: Tensor, scale: f64, zero_point: i64) ---
	atg__masked_scale :: proc(out: ^Tensor, self: Tensor, mask: Tensor, scale: f64) ---
	atg__masked_scale_out :: proc(out: ^Tensor, self: Tensor, mask: Tensor, scale: f64) ---
	atg__masked_softmax :: proc(out: ^Tensor, self: Tensor, mask: Tensor, dim_v: i64, dim_null: rawptr, mask_type_v: i64, mask_type_null: rawptr) ---
	atg__masked_softmax_backward :: proc(out: ^Tensor, grad_output: Tensor, output: Tensor, mask: Tensor, dim_v: i64, dim_null: rawptr) ---
	atg__masked_softmax_backward_out :: proc(out: ^Tensor, grad_output: Tensor, output: Tensor, mask: Tensor, dim_v: i64, dim_null: rawptr) ---
	atg__masked_softmax_out :: proc(out: ^Tensor, self: Tensor, mask: Tensor, dim_v: i64, dim_null: rawptr, mask_type_v: i64, mask_type_null: rawptr) ---
	atg__mixed_dtypes_linear :: proc(out: ^Tensor, input: Tensor, weight: Tensor, scale: Tensor, bias: Tensor, activation_ptr: cstring, activation_len: c.int) ---
	atg__mkldnn_reshape :: proc(out: ^Tensor, self: Tensor, shape_data: [^]i64, shape_len: c.int) ---
	atg__mkldnn_reshape_out :: proc(out: ^Tensor, self: Tensor, shape_data: [^]i64, shape_len: c.int) ---
	atg__mkldnn_transpose :: proc(out: ^Tensor, self: Tensor, dim0: i64, dim1: i64) ---
	atg__mkldnn_transpose_ :: proc(out: ^Tensor, self: Tensor, dim0: i64, dim1: i64) ---
	atg__mkldnn_transpose_out :: proc(out: ^Tensor, self: Tensor, dim0: i64, dim1: i64) ---
	atg__mps_convolution :: proc(out: ^Tensor, self: Tensor, weight: Tensor, bias: Tensor, padding_data: [^]i64, padding_len: c.int, stride_data: [^]i64, stride_len: c.int, dilation_data: [^]i64, dilation_len: c.int, groups: i64) ---
	atg__mps_convolution_out :: proc(out: ^Tensor, self: Tensor, weight: Tensor, bias: Tensor, padding_data: [^]i64, padding_len: c.int, stride_data: [^]i64, stride_len: c.int, dilation_data: [^]i64, dilation_len: c.int, groups: i64) ---
	atg__mps_convolution_transpose :: proc(out: ^Tensor, self: Tensor, weight: Tensor, padding_data: [^]i64, padding_len: c.int, output_padding_data: [^]i64, output_padding_len: c.int, stride_data: [^]i64, stride_len: c.int, dilation_data: [^]i64, dilation_len: c.int, groups: i64) ---
	atg__mps_convolution_transpose_out :: proc(out: ^Tensor, self: Tensor, weight: Tensor, padding_data: [^]i64, padding_len: c.int, output_padding_data: [^]i64, output_padding_len: c.int, stride_data: [^]i64, stride_len: c.int, dilation_data: [^]i64, dilation_len: c.int, groups: i64) ---
	atg__native_batch_norm_legit :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor, running_mean: Tensor, running_var: Tensor, training: c.int, momentum: f64, eps: f64) ---
	atg__native_batch_norm_legit_functional :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor, running_mean: Tensor, running_var: Tensor, training: c.int, momentum: f64, eps: f64) ---
	atg__native_batch_norm_legit_no_stats :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor, training: c.int, momentum: f64, eps: f64) ---
	atg__native_batch_norm_legit_no_stats_out :: proc(out: ^Tensor, save_mean: Tensor, save_invstd: Tensor, input: Tensor, weight: Tensor, bias: Tensor, training: c.int, momentum: f64, eps: f64) ---
	atg__native_batch_norm_legit_no_training :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor, running_mean: Tensor, running_var: Tensor, momentum: f64, eps: f64) ---
	atg__native_batch_norm_legit_no_training_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, out2: Tensor, input: Tensor, weight: Tensor, bias: Tensor, running_mean: Tensor, running_var: Tensor, momentum: f64, eps: f64) ---
	atg__native_batch_norm_legit_out :: proc(out: ^Tensor, save_mean: Tensor, save_invstd: Tensor, input: Tensor, weight: Tensor, bias: Tensor, running_mean: Tensor, running_var: Tensor, training: c.int, momentum: f64, eps: f64) ---
	atg__native_multi_head_attention :: proc(out: ^Tensor, query: Tensor, key: Tensor, value: Tensor, embed_dim: i64, num_head: i64, qkv_weight: Tensor, qkv_bias: Tensor, proj_weight: Tensor, proj_bias: Tensor, mask: Tensor, need_weights: c.int, average_attn_weights: c.int, mask_type_v: i64, mask_type_null: rawptr) ---
	atg__native_multi_head_attention_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, query: Tensor, key: Tensor, value: Tensor, embed_dim: i64, num_head: i64, qkv_weight: Tensor, qkv_bias: Tensor, proj_weight: Tensor, proj_bias: Tensor, mask: Tensor, need_weights: c.int, average_attn_weights: c.int, mask_type_v: i64, mask_type_null: rawptr) ---
	atg__neg_view :: proc(out: ^Tensor, self: Tensor) ---
	atg__neg_view_copy :: proc(out: ^Tensor, self: Tensor) ---
	atg__neg_view_copy_out :: proc(out: ^Tensor, self: Tensor) ---
	atg__nested_compute_contiguous_strides_offsets :: proc(out: ^Tensor, nested_size: Tensor) ---
	atg__nested_from_padded :: proc(out: ^Tensor, padded: Tensor, cpu_nested_shape_example: Tensor, fuse_transform_0213: c.int) ---
	atg__nested_from_padded_and_nested_example :: proc(out: ^Tensor, padded: Tensor, nt_example: Tensor) ---
	atg__nested_from_padded_and_nested_example_out :: proc(out: ^Tensor, padded: Tensor, nt_example: Tensor) ---
	atg__nested_from_padded_out :: proc(out: ^Tensor, padded: Tensor, cpu_nested_shape_example: Tensor, fuse_transform_0213: c.int) ---
	atg__nested_from_padded_tensor :: proc(out: ^Tensor, padded: Tensor, offsets: Tensor, dummy: Tensor, ragged_idx: i64, min_seqlen: Tensor, max_seqlen: Tensor, sum_S_v: i64, sum_S_null: rawptr) ---
	atg__nested_get_jagged_dummy :: proc(out: ^Tensor, any: Tensor) ---
	atg__nested_get_lengths :: proc(out: ^Tensor, self: Tensor) ---
	atg__nested_get_max_seqlen :: proc(out: ^Tensor, self: Tensor) ---
	atg__nested_get_min_seqlen :: proc(out: ^Tensor, self: Tensor) ---
	atg__nested_get_offsets :: proc(out: ^Tensor, self: Tensor) ---
	atg__nested_get_values :: proc(out: ^Tensor, self: Tensor) ---
	atg__nested_get_values_copy :: proc(out: ^Tensor, self: Tensor) ---
	atg__nested_get_values_copy_out :: proc(out: ^Tensor, self: Tensor) ---
	atg__nested_select_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, dim: i64, index: i64) ---
	atg__nested_sum_backward :: proc(out: ^Tensor, grad: Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, keepdim: c.int) ---
	atg__nested_view_from_buffer :: proc(out: ^Tensor, self: Tensor, nested_size: Tensor, nested_strides: Tensor, offsets: Tensor) ---
	atg__nested_view_from_buffer_copy :: proc(out: ^Tensor, self: Tensor, nested_size: Tensor, nested_strides: Tensor, offsets: Tensor) ---
	atg__nested_view_from_buffer_copy_out :: proc(out: ^Tensor, self: Tensor, nested_size: Tensor, nested_strides: Tensor, offsets: Tensor) ---
	atg__nested_view_from_jagged :: proc(out: ^Tensor, self: Tensor, offsets: Tensor, dummy: Tensor, lengths: Tensor, ragged_idx: i64, min_seqlen: Tensor, max_seqlen: Tensor) ---
	atg__nested_view_from_jagged_copy :: proc(out: ^Tensor, self: Tensor, offsets: Tensor, dummy: Tensor, lengths: Tensor, ragged_idx: i64, min_seqlen: Tensor, max_seqlen: Tensor) ---
	atg__nested_view_from_jagged_copy_out :: proc(out: ^Tensor, self: Tensor, offsets: Tensor, dummy: Tensor, lengths: Tensor, ragged_idx: i64, min_seqlen: Tensor, max_seqlen: Tensor) ---
	atg__new_zeros_with_same_feature_meta :: proc(out: ^Tensor, self: Tensor, other: Tensor, self_num_batch_dims: i64) ---
	atg__new_zeros_with_same_feature_meta_out :: proc(out: ^Tensor, self: Tensor, other: Tensor, self_num_batch_dims: i64) ---
	atg__nnpack_spatial_convolution :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor, padding_data: [^]i64, padding_len: c.int, stride_data: [^]i64, stride_len: c.int) ---
	atg__nnpack_spatial_convolution_out :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor, padding_data: [^]i64, padding_len: c.int, stride_data: [^]i64, stride_len: c.int) ---
	atg__pack_padded_sequence :: proc(out: ^Tensor, input: Tensor, lengths: Tensor, batch_first: c.int) ---
	atg__pack_padded_sequence_backward :: proc(out: ^Tensor, grad: Tensor, input_size_data: [^]i64, input_size_len: c.int, batch_sizes: Tensor, batch_first: c.int) ---
	atg__pack_padded_sequence_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, input: Tensor, lengths: Tensor, batch_first: c.int) ---
	atg__pad_circular :: proc(out: ^Tensor, self: Tensor, pad_data: [^]i64, pad_len: c.int) ---
	atg__pad_enum :: proc(out: ^Tensor, self: Tensor, pad_data: [^]i64, pad_len: c.int, mode: i64, value_v: f64, value_null: rawptr) ---
	atg__pad_packed_sequence :: proc(out: ^Tensor, data: Tensor, batch_sizes: Tensor, batch_first: c.int, padding_value: Scalar, total_length: i64) ---
	atg__pdist_backward :: proc(out: ^Tensor, grad: Tensor, self: Tensor, p: f64, pdist: Tensor) ---
	atg__pdist_backward_out :: proc(out: ^Tensor, grad: Tensor, self: Tensor, p: f64, pdist: Tensor) ---
	atg__pin_memory :: proc(out: ^Tensor, self: Tensor, device: c.int) ---
	atg__pin_memory_out :: proc(out: ^Tensor, self: Tensor, device: c.int) ---
	atg__prelu_kernel :: proc(out: ^Tensor, self: Tensor, weight: Tensor) ---
	atg__prelu_kernel_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, weight: Tensor) ---
	atg__print :: proc(s_ptr: cstring, s_len: c.int) ---
	atg__propagate_xla_data :: proc(input: Tensor, output: Tensor) ---
	atg__remove_batch_dim :: proc(out: ^Tensor, self: Tensor, level: i64, batch_size: i64, out_dim: i64) ---
	atg__reshape_alias :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int, stride_data: [^]i64, stride_len: c.int) ---
	atg__reshape_alias_copy :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int, stride_data: [^]i64, stride_len: c.int) ---
	atg__reshape_alias_copy_out :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int, stride_data: [^]i64, stride_len: c.int) ---
	atg__reshape_copy :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int) ---
	atg__reshape_from_tensor :: proc(out: ^Tensor, self: Tensor, shape: Tensor) ---
	atg__resize_output :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int, device: c.int) ---
	atg__resize_output_ :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int, device: c.int) ---
	atg__resize_output_out :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int, device: c.int) ---
	atg__rowwise_prune :: proc(out: ^Tensor, weight: Tensor, mask: Tensor, compressed_indices_dtype: c.int) ---
	atg__safe_softmax :: proc(out: ^Tensor, self: Tensor, dim: i64, dtype: c.int) ---
	atg__sample_dirichlet :: proc(out: ^Tensor, self: Tensor) ---
	atg__sample_dirichlet_out :: proc(out: ^Tensor, self: Tensor) ---
	atg__saturate_weight_to_fp16 :: proc(out: ^Tensor, weight: Tensor) ---
	atg__scaled_dot_product_attention_math :: proc(out: ^Tensor, query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor, dropout_p: f64, is_causal: c.int, dropout_mask: Tensor, scale_v: f64, scale_null: rawptr, enable_gqa: c.int) ---
	atg__scaled_dot_product_attention_math_for_mps :: proc(out: ^Tensor, query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor, dropout_p: f64, is_causal: c.int, dropout_mask: Tensor, scale_v: f64, scale_null: rawptr) ---
	atg__scaled_dot_product_cudnn_attention_backward :: proc(out: ^Tensor, grad_out: Tensor, query: Tensor, key: Tensor, value: Tensor, logsumexp: Tensor, philox_seed: Tensor, philox_offset: Tensor, attn_bias: Tensor, cum_seq_q: Tensor, cum_seq_k: Tensor, max_q: i64, max_k: i64, dropout_p: f64, is_causal: c.int, scale_v: f64, scale_null: rawptr) ---
	atg__scaled_dot_product_efficient_attention :: proc(out: ^Tensor, query: Tensor, key: Tensor, value: Tensor, attn_bias: Tensor, compute_log_sumexp: c.int, dropout_p: f64, is_causal: c.int, scale_v: f64, scale_null: rawptr) ---
	atg__scaled_dot_product_flash_attention_backward :: proc(out: ^Tensor, grad_out: Tensor, query: Tensor, key: Tensor, value: Tensor, logsumexp: Tensor, cum_seq_q: Tensor, cum_seq_k: Tensor, max_q: i64, max_k: i64, dropout_p: f64, is_causal: c.int, philox_seed: Tensor, philox_offset: Tensor, scale_v: f64, scale_null: rawptr) ---
	atg__scaled_dot_product_flash_attention_for_cpu :: proc(out: ^Tensor, query: Tensor, key: Tensor, value: Tensor, dropout_p: f64, is_causal: c.int, attn_mask: Tensor, scale_v: f64, scale_null: rawptr) ---
	atg__scaled_dot_product_flash_attention_for_cpu_backward :: proc(out: ^Tensor, grad_out: Tensor, query: Tensor, key: Tensor, value: Tensor, logsumexp: Tensor, dropout_p: f64, is_causal: c.int, attn_mask: Tensor, scale_v: f64, scale_null: rawptr) ---
	atg__scaled_grouped_mm :: proc(out: ^Tensor, self: Tensor, mat2: Tensor, scale_a: Tensor, scale_b: Tensor, offs: Tensor, bias: Tensor, scale_result: Tensor, out_dtype: c.int, use_fast_accum: c.int) ---
	atg__scaled_mm :: proc(out: ^Tensor, self: Tensor, mat2: Tensor, scale_a: Tensor, scale_b: Tensor, bias: Tensor, scale_result: Tensor, out_dtype: c.int, use_fast_accum: c.int) ---
	atg__scaled_mm_out :: proc(out: ^Tensor, self: Tensor, mat2: Tensor, scale_a: Tensor, scale_b: Tensor, bias: Tensor, scale_result: Tensor, out_dtype: c.int, use_fast_accum: c.int) ---
	atg__scatter_reduce :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor, src: Tensor, reduce_ptr: cstring, reduce_len: c.int, include_self: c.int) ---
	atg__scatter_reduce_ :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor, src: Tensor, reduce_ptr: cstring, reduce_len: c.int, include_self: c.int) ---
	atg__scatter_reduce_two_out :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor, src: Tensor, reduce_ptr: cstring, reduce_len: c.int, include_self: c.int) ---
	atg__segment_reduce_backward :: proc(out: ^Tensor, grad: Tensor, output: Tensor, data: Tensor, reduce_ptr: cstring, reduce_len: c.int, lengths: Tensor, offsets: Tensor, axis: i64, initial: Scalar) ---
	atg__segment_reduce_backward_out :: proc(out: ^Tensor, grad: Tensor, output: Tensor, data: Tensor, reduce_ptr: cstring, reduce_len: c.int, lengths: Tensor, offsets: Tensor, axis: i64, initial: Scalar) ---
	atg__shape_as_tensor :: proc(out: ^Tensor, self: Tensor) ---
	atg__slow_conv2d_backward :: proc(out: ^Tensor, grad_input: Tensor, grad_weight: Tensor, grad_bias: Tensor, grad_output: Tensor, self: Tensor, weight: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int) ---
	atg__sobol_engine_draw :: proc(out: ^Tensor, quasi: Tensor, n: i64, sobolstate: Tensor, dimension: i64, num_generated: i64, dtype: c.int) ---
	atg__sobol_engine_ff_ :: proc(out: ^Tensor, self: Tensor, n: i64, sobolstate: Tensor, dimension: i64, num_generated: i64) ---
	atg__sobol_engine_initialize_state_ :: proc(out: ^Tensor, self: Tensor, dimension: i64) ---
	atg__sobol_engine_scramble_ :: proc(out: ^Tensor, self: Tensor, ltm: Tensor, dimension: i64) ---
	atg__softmax :: proc(out: ^Tensor, self: Tensor, dim: i64, half_to_float: c.int) ---
	atg__softmax_backward_data :: proc(out: ^Tensor, grad_output: Tensor, output: Tensor, dim: i64, input_dtype: c.int) ---
	atg__softmax_backward_data_out :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, output: Tensor, dim: i64, input_dtype: c.int) ---
	atg__softmax_out :: proc(out: ^Tensor, self: Tensor, dim: i64, half_to_float: c.int) ---
	atg__sparse_addmm :: proc(out: ^Tensor, self: Tensor, mat1: Tensor, mat2: Tensor) ---
	atg__sparse_addmm_out :: proc(out: ^Tensor, self: Tensor, mat1: Tensor, mat2: Tensor) ---
	atg__sparse_broadcast_to :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int) ---
	atg__sparse_broadcast_to_copy :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int) ---
	atg__sparse_broadcast_to_copy_out :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int) ---
	atg__sparse_bsc_tensor_unsafe :: proc(out: ^Tensor, ccol_indices: Tensor, row_indices: Tensor, values: Tensor, size_data: [^]i64, size_len: c.int, options_kind: c.int, options_device: c.int) ---
	atg__sparse_bsr_tensor_unsafe :: proc(out: ^Tensor, crow_indices: Tensor, col_indices: Tensor, values: Tensor, size_data: [^]i64, size_len: c.int, options_kind: c.int, options_device: c.int) ---
	atg__sparse_compressed_tensor_unsafe :: proc(out: ^Tensor, compressed_indices: Tensor, plain_indices: Tensor, values: Tensor, size_data: [^]i64, size_len: c.int, options_kind: c.int, options_device: c.int) ---
	atg__sparse_compressed_tensor_with_dims :: proc(out: ^Tensor, nnz: i64, dense_dim: i64, size_data: [^]i64, size_len: c.int, blocksize_data: [^]i64, blocksize_len: c.int, index_dtype: c.int, options_kind: c.int, options_device: c.int) ---
	atg__sparse_coo_tensor_unsafe :: proc(out: ^Tensor, indices: Tensor, values: Tensor, size_data: [^]i64, size_len: c.int, options_kind: c.int, options_device: c.int, is_coalesced: c.int) ---
	atg__sparse_coo_tensor_with_dims :: proc(out: ^Tensor, sparse_dim: i64, dense_dim: i64, size_data: [^]i64, size_len: c.int, options_kind: c.int, options_device: c.int) ---
	atg__sparse_coo_tensor_with_dims_and_tensors :: proc(out: ^Tensor, sparse_dim: i64, dense_dim: i64, size_data: [^]i64, size_len: c.int, indices: Tensor, values: Tensor, options_kind: c.int, options_device: c.int, is_coalesced: c.int) ---
	atg__sparse_coo_tensor_with_dims_and_tensors_out :: proc(out: ^Tensor, sparse_dim: i64, dense_dim: i64, size_data: [^]i64, size_len: c.int, indices: Tensor, values: Tensor, is_coalesced: c.int) ---
	atg__sparse_coo_tensor_with_dims_out :: proc(out: ^Tensor, sparse_dim: i64, dense_dim: i64, size_data: [^]i64, size_len: c.int) ---
	atg__sparse_csc_tensor_unsafe :: proc(out: ^Tensor, ccol_indices: Tensor, row_indices: Tensor, values: Tensor, size_data: [^]i64, size_len: c.int, options_kind: c.int, options_device: c.int) ---
	atg__sparse_csr_prod :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, keepdim: c.int, dtype: c.int) ---
	atg__sparse_csr_prod_dim_dtype_out :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, keepdim: c.int, dtype: c.int) ---
	atg__sparse_csr_sum :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, keepdim: c.int, dtype: c.int) ---
	atg__sparse_csr_sum_dim_dtype_out :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, keepdim: c.int, dtype: c.int) ---
	atg__sparse_csr_tensor_unsafe :: proc(out: ^Tensor, crow_indices: Tensor, col_indices: Tensor, values: Tensor, size_data: [^]i64, size_len: c.int, options_kind: c.int, options_device: c.int) ---
	atg__sparse_log_softmax :: proc(out: ^Tensor, self: Tensor, dim: i64, half_to_float: c.int) ---
	atg__sparse_log_softmax_backward_data :: proc(out: ^Tensor, grad_output: Tensor, output: Tensor, dim: i64, self: Tensor) ---
	atg__sparse_log_softmax_backward_data_out :: proc(out: ^Tensor, grad_output: Tensor, output: Tensor, dim: i64, self: Tensor) ---
	atg__sparse_log_softmax_int :: proc(out: ^Tensor, self: Tensor, dim: i64, dtype: c.int) ---
	atg__sparse_log_softmax_out :: proc(out: ^Tensor, self: Tensor, dim: i64, half_to_float: c.int) ---
	atg__sparse_mask_projection :: proc(out: ^Tensor, self: Tensor, mask: Tensor, accumulate_matches: c.int) ---
	atg__sparse_mask_projection_out :: proc(out: ^Tensor, self: Tensor, mask: Tensor, accumulate_matches: c.int) ---
	atg__sparse_mm :: proc(out: ^Tensor, sparse: Tensor, dense: Tensor) ---
	atg__sparse_mm_reduce :: proc(out: ^Tensor, sparse: Tensor, dense: Tensor, reduce_ptr: cstring, reduce_len: c.int) ---
	atg__sparse_mm_reduce_impl :: proc(out: ^Tensor, self: Tensor, other: Tensor, reduce_ptr: cstring, reduce_len: c.int) ---
	atg__sparse_semi_structured_apply :: proc(out: ^Tensor, input: Tensor, thread_masks: Tensor) ---
	atg__sparse_semi_structured_apply_dense :: proc(out: ^Tensor, input: Tensor, thread_masks: Tensor) ---
	atg__sparse_semi_structured_linear :: proc(out: ^Tensor, input: Tensor, weight: Tensor, meta: Tensor, bias: Tensor, activation_ptr: cstring, activation_len: c.int, out_dtype: c.int) ---
	atg__sparse_semi_structured_mm :: proc(out: ^Tensor, mat1: Tensor, mat1_meta: Tensor, mat2: Tensor, out_dtype: c.int) ---
	atg__sparse_semi_structured_tile :: proc(out: ^Tensor, input: Tensor, algorithm_ptr: cstring, algorithm_len: c.int, use_cutlass: c.int) ---
	atg__sparse_softmax :: proc(out: ^Tensor, self: Tensor, dim: i64, half_to_float: c.int) ---
	atg__sparse_softmax_backward_data :: proc(out: ^Tensor, grad_output: Tensor, output: Tensor, dim: i64, self: Tensor) ---
	atg__sparse_softmax_backward_data_out :: proc(out: ^Tensor, grad_output: Tensor, output: Tensor, dim: i64, self: Tensor) ---
	atg__sparse_softmax_int :: proc(out: ^Tensor, self: Tensor, dim: i64, dtype: c.int) ---
	atg__sparse_softmax_out :: proc(out: ^Tensor, self: Tensor, dim: i64, half_to_float: c.int) ---
	atg__sparse_sparse_matmul :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg__sparse_sparse_matmul_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg__sparse_sum :: proc(out: ^Tensor, self: Tensor) ---
	atg__sparse_sum_backward :: proc(out: ^Tensor, grad: Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int) ---
	atg__sparse_sum_backward_out :: proc(out: ^Tensor, grad: Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int) ---
	atg__sparse_sum_dim :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int) ---
	atg__sparse_sum_dim_dtype :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, dtype: c.int) ---
	atg__sparse_sum_dim_out :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int) ---
	atg__sparse_sum_dtype :: proc(out: ^Tensor, self: Tensor, dtype: c.int) ---
	atg__spdiags :: proc(out: ^Tensor, diagonals: Tensor, offsets: Tensor, shape_data: [^]i64, shape_len: c.int, layout: rawptr) ---
	atg__spdiags_out :: proc(out: ^Tensor, diagonals: Tensor, offsets: Tensor, shape_data: [^]i64, shape_len: c.int, layout: rawptr) ---
	atg__spsolve :: proc(out: ^Tensor, A: Tensor, B: Tensor, left: c.int) ---
	atg__stack :: proc(out: ^Tensor, tensors_data: ^Tensor, tensors_len: c.int, dim: i64) ---
	atg__stack_out :: proc(out: ^Tensor, tensors_data: ^Tensor, tensors_len: c.int, dim: i64) ---
	atg__standard_gamma :: proc(out: ^Tensor, self: Tensor) ---
	atg__standard_gamma_grad :: proc(out: ^Tensor, self: Tensor, output: Tensor) ---
	atg__standard_gamma_grad_out :: proc(out: ^Tensor, self: Tensor, output: Tensor) ---
	atg__standard_gamma_out :: proc(out: ^Tensor, self: Tensor) ---
	atg__test_ambiguous_defaults :: proc(out: ^Tensor, dummy: Tensor, a: i64, b: i64) ---
	atg__test_ambiguous_defaults_b :: proc(out: ^Tensor, dummy: Tensor, a: i64, b_ptr: cstring, b_len: c.int) ---
	atg__test_autograd_multiple_dispatch :: proc(out: ^Tensor, self: Tensor) ---
	atg__test_autograd_multiple_dispatch_fullcoverage_out :: proc(out: ^Tensor, self: Tensor) ---
	atg__test_autograd_multiple_dispatch_ntonly :: proc(out: ^Tensor, self: Tensor, b: c.int) ---
	atg__test_autograd_multiple_dispatch_view :: proc(out: ^Tensor, self: Tensor) ---
	atg__test_autograd_multiple_dispatch_view_copy :: proc(out: ^Tensor, self: Tensor) ---
	atg__test_autograd_multiple_dispatch_view_copy_out :: proc(out: ^Tensor, self: Tensor) ---
	atg__test_check_tensor :: proc(out: ^Tensor, self: Tensor) ---
	atg__test_functorch_fallback :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg__test_functorch_fallback_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg__test_optional_filled_intlist :: proc(out: ^Tensor, values: Tensor, addends_data: [^]i64, addends_len: c.int) ---
	atg__test_optional_filled_intlist_out :: proc(out: ^Tensor, values: Tensor, addends_data: [^]i64, addends_len: c.int) ---
	atg__test_optional_floatlist :: proc(out: ^Tensor, values: Tensor, addends_data: [^]f64, addends_len: c.int) ---
	atg__test_optional_floatlist_out :: proc(out: ^Tensor, values: Tensor, addends_data: [^]f64, addends_len: c.int) ---
	atg__test_optional_intlist :: proc(out: ^Tensor, values: Tensor, addends_data: [^]i64, addends_len: c.int) ---
	atg__test_optional_intlist_out :: proc(out: ^Tensor, values: Tensor, addends_data: [^]i64, addends_len: c.int) ---
	atg__test_parallel_materialize :: proc(out: ^Tensor, self: Tensor, num_parallel: i64, skip_first: c.int) ---
	atg__test_serialization_subcmul :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg__test_string_default :: proc(out: ^Tensor, dummy: Tensor, a_ptr: cstring, a_len: c.int, b_ptr: cstring, b_len: c.int) ---
	atg__test_warn_in_autograd :: proc(out: ^Tensor, self: Tensor) ---
	atg__test_warn_in_autograd_out :: proc(out: ^Tensor, self: Tensor) ---
	atg__to_copy :: proc(out: ^Tensor, self: Tensor, options_kind: c.int, options_device: c.int, non_blocking: c.int) ---
	atg__to_copy_out :: proc(out: ^Tensor, self: Tensor, non_blocking: c.int) ---
	atg__to_dense :: proc(out: ^Tensor, self: Tensor, dtype: c.int, masked_grad: c.int) ---
	atg__to_dense_out :: proc(out: ^Tensor, self: Tensor, dtype: c.int, masked_grad: c.int) ---
	atg__to_sparse :: proc(out: ^Tensor, self: Tensor, layout: rawptr, blocksize_data: [^]i64, blocksize_len: c.int, dense_dim_v: i64, dense_dim_null: rawptr) ---
	atg__to_sparse_bsc :: proc(out: ^Tensor, self: Tensor, blocksize_data: [^]i64, blocksize_len: c.int, dense_dim_v: i64, dense_dim_null: rawptr) ---
	atg__to_sparse_bsc_out :: proc(out: ^Tensor, self: Tensor, blocksize_data: [^]i64, blocksize_len: c.int, dense_dim_v: i64, dense_dim_null: rawptr) ---
	atg__to_sparse_bsr :: proc(out: ^Tensor, self: Tensor, blocksize_data: [^]i64, blocksize_len: c.int, dense_dim_v: i64, dense_dim_null: rawptr) ---
	atg__to_sparse_bsr_out :: proc(out: ^Tensor, self: Tensor, blocksize_data: [^]i64, blocksize_len: c.int, dense_dim_v: i64, dense_dim_null: rawptr) ---
	atg__to_sparse_csc :: proc(out: ^Tensor, self: Tensor, dense_dim_v: i64, dense_dim_null: rawptr) ---
	atg__to_sparse_csc_out :: proc(out: ^Tensor, self: Tensor, dense_dim_v: i64, dense_dim_null: rawptr) ---
	atg__to_sparse_csr :: proc(out: ^Tensor, self: Tensor, dense_dim_v: i64, dense_dim_null: rawptr) ---
	atg__to_sparse_csr_out :: proc(out: ^Tensor, self: Tensor, dense_dim_v: i64, dense_dim_null: rawptr) ---
	atg__to_sparse_out :: proc(out: ^Tensor, self: Tensor, layout: rawptr, blocksize_data: [^]i64, blocksize_len: c.int, dense_dim_v: i64, dense_dim_null: rawptr) ---
	atg__to_sparse_semi_structured :: proc(out: ^Tensor, dense: Tensor) ---
	atg__to_sparse_sparse_dim :: proc(out: ^Tensor, self: Tensor, sparse_dim: i64) ---
	atg__to_sparse_sparse_dim_out :: proc(out: ^Tensor, self: Tensor, sparse_dim: i64) ---
	atg__transform_bias_rescale_qkv :: proc(out: ^Tensor, qkv: Tensor, qkv_bias: Tensor, num_heads: i64) ---
	atg__transform_bias_rescale_qkv_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, out2: Tensor, qkv: Tensor, qkv_bias: Tensor, num_heads: i64) ---
	atg__transformer_encoder_layer_fwd :: proc(out: ^Tensor, src: Tensor, embed_dim: i64, num_heads: i64, qkv_weight: Tensor, qkv_bias: Tensor, proj_weight: Tensor, proj_bias: Tensor, use_gelu: c.int, norm_first: c.int, eps: f64, norm_weight_1: Tensor, norm_bias_1: Tensor, norm_weight_2: Tensor, norm_bias_2: Tensor, ffn_weight_1: Tensor, ffn_bias_1: Tensor, ffn_weight_2: Tensor, ffn_bias_2: Tensor, mask: Tensor, mask_type_v: i64, mask_type_null: rawptr) ---
	atg__transformer_encoder_layer_fwd_out :: proc(out: ^Tensor, src: Tensor, embed_dim: i64, num_heads: i64, qkv_weight: Tensor, qkv_bias: Tensor, proj_weight: Tensor, proj_bias: Tensor, use_gelu: c.int, norm_first: c.int, eps: f64, norm_weight_1: Tensor, norm_bias_1: Tensor, norm_weight_2: Tensor, norm_bias_2: Tensor, ffn_weight_1: Tensor, ffn_bias_1: Tensor, ffn_weight_2: Tensor, ffn_bias_2: Tensor, mask: Tensor, mask_type_v: i64, mask_type_null: rawptr) ---
	atg__trilinear :: proc(out: ^Tensor, i1: Tensor, i2: Tensor, i3: Tensor, expand1_data: [^]i64, expand1_len: c.int, expand2_data: [^]i64, expand2_len: c.int, expand3_data: [^]i64, expand3_len: c.int, sumdim_data: [^]i64, sumdim_len: c.int, unroll_dim: i64) ---
	atg__trilinear_out :: proc(out: ^Tensor, i1: Tensor, i2: Tensor, i3: Tensor, expand1_data: [^]i64, expand1_len: c.int, expand2_data: [^]i64, expand2_len: c.int, expand3_data: [^]i64, expand3_len: c.int, sumdim_data: [^]i64, sumdim_len: c.int, unroll_dim: i64) ---
	atg__triton_multi_head_attention :: proc(out: ^Tensor, query: Tensor, key: Tensor, value: Tensor, embed_dim: i64, num_head: i64, qkv_weight: Tensor, qkv_bias: Tensor, proj_weight: Tensor, proj_bias: Tensor, mask: Tensor) ---
	atg__triton_multi_head_attention_out :: proc(out: ^Tensor, query: Tensor, key: Tensor, value: Tensor, embed_dim: i64, num_head: i64, qkv_weight: Tensor, qkv_bias: Tensor, proj_weight: Tensor, proj_bias: Tensor, mask: Tensor) ---
	atg__triton_scaled_dot_attention :: proc(out: ^Tensor, q: Tensor, k: Tensor, v: Tensor, dropout_p: f64) ---
	atg__triton_scaled_dot_attention_out :: proc(out: ^Tensor, q: Tensor, k: Tensor, v: Tensor, dropout_p: f64) ---
	atg__unique :: proc(out: ^Tensor, self: Tensor, sorted: c.int, return_inverse: c.int) ---
	atg__unique2 :: proc(out: ^Tensor, self: Tensor, sorted: c.int, return_inverse: c.int, return_counts: c.int) ---
	atg__unique2_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, out2: Tensor, self: Tensor, sorted: c.int, return_inverse: c.int, return_counts: c.int) ---
	atg__unique_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, self: Tensor, sorted: c.int, return_inverse: c.int) ---
	atg__unpack_dual :: proc(out: ^Tensor, dual: Tensor, level: i64) ---
	atg__unsafe_index :: proc(out: ^Tensor, self: Tensor, indices_data: ^Tensor, indices_len: c.int) ---
	atg__unsafe_index_put :: proc(out: ^Tensor, self: Tensor, indices_data: ^Tensor, indices_len: c.int, values: Tensor, accumulate: c.int) ---
	atg__unsafe_masked_index :: proc(out: ^Tensor, self: Tensor, mask: Tensor, indices_data: ^Tensor, indices_len: c.int, fill: Scalar) ---
	atg__unsafe_masked_index_put_accumulate :: proc(out: ^Tensor, self: Tensor, mask: Tensor, indices_data: ^Tensor, indices_len: c.int, values: Tensor) ---
	atg__unsafe_view :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int) ---
	atg__unsafe_view_out :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int) ---
	atg__upsample_bicubic2d_aa :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int, align_corners: c.int, scales_h_v: f64, scales_h_null: rawptr, scales_w_v: f64, scales_w_null: rawptr) ---
	atg__upsample_bicubic2d_aa_backward :: proc(out: ^Tensor, grad_output: Tensor, output_size_data: [^]i64, output_size_len: c.int, input_size_data: [^]i64, input_size_len: c.int, align_corners: c.int, scales_h_v: f64, scales_h_null: rawptr, scales_w_v: f64, scales_w_null: rawptr) ---
	atg__upsample_bicubic2d_aa_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, output_size_data: [^]i64, output_size_len: c.int, input_size_data: [^]i64, input_size_len: c.int, align_corners: c.int, scales_h_v: f64, scales_h_null: rawptr, scales_w_v: f64, scales_w_null: rawptr) ---
	atg__upsample_bicubic2d_aa_out :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int, align_corners: c.int, scales_h_v: f64, scales_h_null: rawptr, scales_w_v: f64, scales_w_null: rawptr) ---
	atg__upsample_bicubic2d_aa_vec :: proc(out: ^Tensor, input: Tensor, output_size_data: [^]i64, output_size_len: c.int, align_corners: c.int, scale_factors_data: [^]f64, scale_factors_len: c.int) ---
	atg__upsample_bilinear2d_aa :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int, align_corners: c.int, scales_h_v: f64, scales_h_null: rawptr, scales_w_v: f64, scales_w_null: rawptr) ---
	atg__upsample_bilinear2d_aa_backward :: proc(out: ^Tensor, grad_output: Tensor, output_size_data: [^]i64, output_size_len: c.int, input_size_data: [^]i64, input_size_len: c.int, align_corners: c.int, scales_h_v: f64, scales_h_null: rawptr, scales_w_v: f64, scales_w_null: rawptr) ---
	atg__upsample_bilinear2d_aa_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, output_size_data: [^]i64, output_size_len: c.int, input_size_data: [^]i64, input_size_len: c.int, align_corners: c.int, scales_h_v: f64, scales_h_null: rawptr, scales_w_v: f64, scales_w_null: rawptr) ---
	atg__upsample_bilinear2d_aa_out :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int, align_corners: c.int, scales_h_v: f64, scales_h_null: rawptr, scales_w_v: f64, scales_w_null: rawptr) ---
	atg__upsample_bilinear2d_aa_vec :: proc(out: ^Tensor, input: Tensor, output_size_data: [^]i64, output_size_len: c.int, align_corners: c.int, scale_factors_data: [^]f64, scale_factors_len: c.int) ---
	atg__upsample_nearest_exact1d :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int, scales_v: f64, scales_null: rawptr) ---
	atg__upsample_nearest_exact1d_backward :: proc(out: ^Tensor, grad_output: Tensor, output_size_data: [^]i64, output_size_len: c.int, input_size_data: [^]i64, input_size_len: c.int, scales_v: f64, scales_null: rawptr) ---
	atg__upsample_nearest_exact1d_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, output_size_data: [^]i64, output_size_len: c.int, input_size_data: [^]i64, input_size_len: c.int, scales_v: f64, scales_null: rawptr) ---
	atg__upsample_nearest_exact1d_out :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int, scales_v: f64, scales_null: rawptr) ---
	atg__upsample_nearest_exact1d_vec :: proc(out: ^Tensor, input: Tensor, output_size_data: [^]i64, output_size_len: c.int, scale_factors_data: [^]f64, scale_factors_len: c.int) ---
	atg__upsample_nearest_exact2d :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int, scales_h_v: f64, scales_h_null: rawptr, scales_w_v: f64, scales_w_null: rawptr) ---
	atg__upsample_nearest_exact2d_backward :: proc(out: ^Tensor, grad_output: Tensor, output_size_data: [^]i64, output_size_len: c.int, input_size_data: [^]i64, input_size_len: c.int, scales_h_v: f64, scales_h_null: rawptr, scales_w_v: f64, scales_w_null: rawptr) ---
	atg__upsample_nearest_exact2d_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, output_size_data: [^]i64, output_size_len: c.int, input_size_data: [^]i64, input_size_len: c.int, scales_h_v: f64, scales_h_null: rawptr, scales_w_v: f64, scales_w_null: rawptr) ---
	atg__upsample_nearest_exact2d_out :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int, scales_h_v: f64, scales_h_null: rawptr, scales_w_v: f64, scales_w_null: rawptr) ---
	atg__upsample_nearest_exact2d_vec :: proc(out: ^Tensor, input: Tensor, output_size_data: [^]i64, output_size_len: c.int, scale_factors_data: [^]f64, scale_factors_len: c.int) ---
	atg__upsample_nearest_exact3d :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int, scales_d_v: f64, scales_d_null: rawptr, scales_h_v: f64, scales_h_null: rawptr, scales_w_v: f64, scales_w_null: rawptr) ---
	atg__upsample_nearest_exact3d_backward :: proc(out: ^Tensor, grad_output: Tensor, output_size_data: [^]i64, output_size_len: c.int, input_size_data: [^]i64, input_size_len: c.int, scales_d_v: f64, scales_d_null: rawptr, scales_h_v: f64, scales_h_null: rawptr, scales_w_v: f64, scales_w_null: rawptr) ---
	atg__upsample_nearest_exact3d_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, output_size_data: [^]i64, output_size_len: c.int, input_size_data: [^]i64, input_size_len: c.int, scales_d_v: f64, scales_d_null: rawptr, scales_h_v: f64, scales_h_null: rawptr, scales_w_v: f64, scales_w_null: rawptr) ---
	atg__upsample_nearest_exact3d_out :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int, scales_d_v: f64, scales_d_null: rawptr, scales_h_v: f64, scales_h_null: rawptr, scales_w_v: f64, scales_w_null: rawptr) ---
	atg__upsample_nearest_exact3d_vec :: proc(out: ^Tensor, input: Tensor, output_size_data: [^]i64, output_size_len: c.int, scale_factors_data: [^]f64, scale_factors_len: c.int) ---
	atg__validate_compressed_sparse_indices :: proc(is_crow: c.int, compressed_idx: Tensor, plain_idx: Tensor, cdim: i64, dim: i64, nnz: i64) ---
	atg__validate_sparse_bsc_tensor_args :: proc(ccol_indices: Tensor, row_indices: Tensor, values: Tensor, size_data: [^]i64, size_len: c.int, check_pinning: c.int) ---
	atg__validate_sparse_bsr_tensor_args :: proc(crow_indices: Tensor, col_indices: Tensor, values: Tensor, size_data: [^]i64, size_len: c.int, check_pinning: c.int) ---
	atg__validate_sparse_compressed_tensor_args :: proc(compressed_indices: Tensor, plain_indices: Tensor, values: Tensor, size_data: [^]i64, size_len: c.int, layout: rawptr, check_pinning: c.int) ---
	atg__validate_sparse_csc_tensor_args :: proc(ccol_indices: Tensor, row_indices: Tensor, values: Tensor, size_data: [^]i64, size_len: c.int, check_pinning: c.int) ---
	atg__validate_sparse_csr_tensor_args :: proc(crow_indices: Tensor, col_indices: Tensor, values: Tensor, size_data: [^]i64, size_len: c.int, check_pinning: c.int) ---
	atg__values :: proc(out: ^Tensor, self: Tensor) ---
	atg__values_copy :: proc(out: ^Tensor, self: Tensor) ---
	atg__values_copy_out :: proc(out: ^Tensor, self: Tensor) ---
	atg__weight_int4pack_mm :: proc(out: ^Tensor, self: Tensor, mat2: Tensor, qGroupSize: i64, qScaleAndZeros: Tensor) ---
	atg__weight_int4pack_mm_for_cpu :: proc(out: ^Tensor, self: Tensor, mat2: Tensor, qGroupSize: i64, qScaleAndZeros: Tensor) ---
	atg__weight_int4pack_mm_with_scales_and_zeros :: proc(out: ^Tensor, self: Tensor, mat2: Tensor, qGroupSize: i64, qScale: Tensor, qZeros: Tensor) ---
	atg__weight_int8pack_mm :: proc(out: ^Tensor, self: Tensor, mat2: Tensor, scales: Tensor) ---
	atg__weight_norm :: proc(out: ^Tensor, v: Tensor, g: Tensor, dim: i64) ---
	atg__weight_norm_differentiable_backward :: proc(out: ^Tensor, grad_w: Tensor, saved_v: Tensor, saved_g: Tensor, saved_norms: Tensor, dim: i64) ---
	atg__weight_norm_interface :: proc(out: ^Tensor, v: Tensor, g: Tensor, dim: i64) ---
	atg__weight_norm_interface_backward :: proc(out: ^Tensor, grad_w: Tensor, saved_v: Tensor, saved_g: Tensor, saved_norms: Tensor, dim: i64) ---
	atg__weight_norm_interface_backward_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, grad_w: Tensor, saved_v: Tensor, saved_g: Tensor, saved_norms: Tensor, dim: i64) ---
	atg__weight_norm_interface_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, v: Tensor, g: Tensor, dim: i64) ---
	atg__wrapped_linear_prepack :: proc(out: ^Tensor, weight: Tensor, weight_scale: Tensor, weight_zero_point: Tensor, bias: Tensor) ---
	atg__wrapped_quantized_linear_prepacked :: proc(out: ^Tensor, input: Tensor, input_scale: Tensor, input_zero_point: Tensor, packed_weight: Tensor, output_scale: Tensor, output_zero_point: Tensor, out_channel: i64) ---
	atg_abs :: proc(out: ^Tensor, self: Tensor) ---
	atg_abs_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_abs_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_absolute :: proc(out: ^Tensor, self: Tensor) ---
	atg_absolute_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_absolute_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_acos :: proc(out: ^Tensor, self: Tensor) ---
	atg_acos_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_acos_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_acosh :: proc(out: ^Tensor, self: Tensor) ---
	atg_acosh_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_acosh_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_adaptive_avg_pool1d :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int) ---
	atg_adaptive_avg_pool1d_out :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int) ---
	atg_adaptive_avg_pool2d :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int) ---
	atg_adaptive_avg_pool2d_out :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int) ---
	atg_adaptive_avg_pool3d :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int) ---
	atg_adaptive_avg_pool3d_backward :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, self: Tensor) ---
	atg_adaptive_avg_pool3d_out :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int) ---
	atg_adaptive_max_pool1d :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int) ---
	atg_adaptive_max_pool2d :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int) ---
	atg_adaptive_max_pool2d_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, indices: Tensor) ---
	atg_adaptive_max_pool2d_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, self: Tensor, indices: Tensor) ---
	atg_adaptive_max_pool2d_out :: proc(out: ^Tensor, indices: Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int) ---
	atg_adaptive_max_pool3d :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int) ---
	atg_adaptive_max_pool3d_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, indices: Tensor) ---
	atg_adaptive_max_pool3d_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, self: Tensor, indices: Tensor) ---
	atg_adaptive_max_pool3d_out :: proc(out: ^Tensor, indices: Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int) ---
	atg_add :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_add_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_add_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_add_scalar :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_add_scalar_ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_add_scalar_out :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_addbmm :: proc(out: ^Tensor, self: Tensor, batch1: Tensor, batch2: Tensor) ---
	atg_addbmm_ :: proc(out: ^Tensor, self: Tensor, batch1: Tensor, batch2: Tensor) ---
	atg_addbmm_out :: proc(out: ^Tensor, self: Tensor, batch1: Tensor, batch2: Tensor) ---
	atg_addcdiv :: proc(out: ^Tensor, self: Tensor, tensor1: Tensor, tensor2: Tensor) ---
	atg_addcdiv_ :: proc(out: ^Tensor, self: Tensor, tensor1: Tensor, tensor2: Tensor) ---
	atg_addcdiv_out :: proc(out: ^Tensor, self: Tensor, tensor1: Tensor, tensor2: Tensor) ---
	atg_addcmul :: proc(out: ^Tensor, self: Tensor, tensor1: Tensor, tensor2: Tensor) ---
	atg_addcmul_ :: proc(out: ^Tensor, self: Tensor, tensor1: Tensor, tensor2: Tensor) ---
	atg_addcmul_out :: proc(out: ^Tensor, self: Tensor, tensor1: Tensor, tensor2: Tensor) ---
	atg_addmm :: proc(out: ^Tensor, self: Tensor, mat1: Tensor, mat2: Tensor) ---
	atg_addmm_ :: proc(out: ^Tensor, self: Tensor, mat1: Tensor, mat2: Tensor) ---
	atg_addmm_dtype :: proc(out: ^Tensor, self: Tensor, mat1: Tensor, mat2: Tensor, out_dtype: c.int) ---
	atg_addmm_dtype_out :: proc(out: ^Tensor, self: Tensor, mat1: Tensor, mat2: Tensor, out_dtype: c.int) ---
	atg_addmm_out :: proc(out: ^Tensor, self: Tensor, mat1: Tensor, mat2: Tensor) ---
	atg_addmv :: proc(out: ^Tensor, self: Tensor, mat: Tensor, vec: Tensor) ---
	atg_addmv_ :: proc(out: ^Tensor, self: Tensor, mat: Tensor, vec: Tensor) ---
	atg_addmv_out :: proc(out: ^Tensor, self: Tensor, mat: Tensor, vec: Tensor) ---
	atg_addr :: proc(out: ^Tensor, self: Tensor, vec1: Tensor, vec2: Tensor) ---
	atg_addr_ :: proc(out: ^Tensor, self: Tensor, vec1: Tensor, vec2: Tensor) ---
	atg_addr_out :: proc(out: ^Tensor, self: Tensor, vec1: Tensor, vec2: Tensor) ---
	atg_adjoint :: proc(out: ^Tensor, self: Tensor) ---
	atg_affine_grid_generator :: proc(out: ^Tensor, theta: Tensor, size_data: [^]i64, size_len: c.int, align_corners: c.int) ---
	atg_affine_grid_generator_backward :: proc(out: ^Tensor, grad: Tensor, size_data: [^]i64, size_len: c.int, align_corners: c.int) ---
	atg_affine_grid_generator_out :: proc(out: ^Tensor, theta: Tensor, size_data: [^]i64, size_len: c.int, align_corners: c.int) ---
	atg_alias :: proc(out: ^Tensor, self: Tensor) ---
	atg_alias_copy :: proc(out: ^Tensor, self: Tensor) ---
	atg_alias_copy_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_align_as :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_all :: proc(out: ^Tensor, self: Tensor) ---
	atg_all_all_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_all_dim :: proc(out: ^Tensor, self: Tensor, dim: i64, keepdim: c.int) ---
	atg_all_dims :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, keepdim: c.int) ---
	atg_all_dims_out :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, keepdim: c.int) ---
	atg_all_out :: proc(out: ^Tensor, self: Tensor, dim: i64, keepdim: c.int) ---
	atg_alpha_dropout :: proc(out: ^Tensor, input: Tensor, p: f64, train: c.int) ---
	atg_alpha_dropout_ :: proc(out: ^Tensor, self: Tensor, p: f64, train: c.int) ---
	atg_amax :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, keepdim: c.int) ---
	atg_amax_out :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, keepdim: c.int) ---
	atg_amin :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, keepdim: c.int) ---
	atg_amin_out :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, keepdim: c.int) ---
	atg_aminmax :: proc(out: ^Tensor, self: Tensor, dim_v: i64, dim_null: rawptr, keepdim: c.int) ---
	atg_aminmax_out :: proc(out: ^Tensor, min: Tensor, max: Tensor, self: Tensor, dim_v: i64, dim_null: rawptr, keepdim: c.int) ---
	atg_angle :: proc(out: ^Tensor, self: Tensor) ---
	atg_angle_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_any :: proc(out: ^Tensor, self: Tensor) ---
	atg_any_all_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_any_dim :: proc(out: ^Tensor, self: Tensor, dim: i64, keepdim: c.int) ---
	atg_any_dims :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, keepdim: c.int) ---
	atg_any_dims_out :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, keepdim: c.int) ---
	atg_any_out :: proc(out: ^Tensor, self: Tensor, dim: i64, keepdim: c.int) ---
	atg_arange :: proc(out: ^Tensor, end: Scalar, options_kind: c.int, options_device: c.int) ---
	atg_arange_start :: proc(out: ^Tensor, start: Scalar, end: Scalar, options_kind: c.int, options_device: c.int) ---
	atg_arange_start_step :: proc(out: ^Tensor, start: Scalar, end: Scalar, step: Scalar, options_kind: c.int, options_device: c.int) ---
	atg_arccos :: proc(out: ^Tensor, self: Tensor) ---
	atg_arccos_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_arccos_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_arccosh :: proc(out: ^Tensor, self: Tensor) ---
	atg_arccosh_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_arccosh_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_arcsin :: proc(out: ^Tensor, self: Tensor) ---
	atg_arcsin_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_arcsin_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_arcsinh :: proc(out: ^Tensor, self: Tensor) ---
	atg_arcsinh_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_arcsinh_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_arctan :: proc(out: ^Tensor, self: Tensor) ---
	atg_arctan2 :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_arctan2_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_arctan2_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_arctan_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_arctan_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_arctanh :: proc(out: ^Tensor, self: Tensor) ---
	atg_arctanh_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_arctanh_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_argmax :: proc(out: ^Tensor, self: Tensor, dim_v: i64, dim_null: rawptr, keepdim: c.int) ---
	atg_argmax_out :: proc(out: ^Tensor, self: Tensor, dim_v: i64, dim_null: rawptr, keepdim: c.int) ---
	atg_argmin :: proc(out: ^Tensor, self: Tensor, dim_v: i64, dim_null: rawptr, keepdim: c.int) ---
	atg_argmin_out :: proc(out: ^Tensor, self: Tensor, dim_v: i64, dim_null: rawptr, keepdim: c.int) ---
	atg_argsort :: proc(out: ^Tensor, self: Tensor, dim: i64, descending: c.int) ---
	atg_argsort_stable :: proc(out: ^Tensor, self: Tensor, stable: c.int, dim: i64, descending: c.int) ---
	atg_argsort_stable_out :: proc(out: ^Tensor, self: Tensor, stable: c.int, dim: i64, descending: c.int) ---
	atg_argwhere :: proc(out: ^Tensor, self: Tensor) ---
	atg_as_strided :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int, stride_data: [^]i64, stride_len: c.int, storage_offset_v: i64, storage_offset_null: rawptr) ---
	atg_as_strided_ :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int, stride_data: [^]i64, stride_len: c.int, storage_offset_v: i64, storage_offset_null: rawptr) ---
	atg_as_strided_copy :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int, stride_data: [^]i64, stride_len: c.int, storage_offset_v: i64, storage_offset_null: rawptr) ---
	atg_as_strided_copy_out :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int, stride_data: [^]i64, stride_len: c.int, storage_offset_v: i64, storage_offset_null: rawptr) ---
	atg_as_strided_scatter :: proc(out: ^Tensor, self: Tensor, src: Tensor, size_data: [^]i64, size_len: c.int, stride_data: [^]i64, stride_len: c.int, storage_offset_v: i64, storage_offset_null: rawptr) ---
	atg_as_strided_scatter_out :: proc(out: ^Tensor, self: Tensor, src: Tensor, size_data: [^]i64, size_len: c.int, stride_data: [^]i64, stride_len: c.int, storage_offset_v: i64, storage_offset_null: rawptr) ---
	atg_asin :: proc(out: ^Tensor, self: Tensor) ---
	atg_asin_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_asin_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_asinh :: proc(out: ^Tensor, self: Tensor) ---
	atg_asinh_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_asinh_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_atan :: proc(out: ^Tensor, self: Tensor) ---
	atg_atan2 :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_atan2_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_atan2_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_atan_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_atan_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_atanh :: proc(out: ^Tensor, self: Tensor) ---
	atg_atanh_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_atanh_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_atleast_1d :: proc(out: ^Tensor, self: Tensor) ---
	atg_atleast_2d :: proc(out: ^Tensor, self: Tensor) ---
	atg_atleast_3d :: proc(out: ^Tensor, self: Tensor) ---
	atg_avg_pool1d :: proc(out: ^Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, ceil_mode: c.int, count_include_pad: c.int) ---
	atg_avg_pool1d_out :: proc(out: ^Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, ceil_mode: c.int, count_include_pad: c.int) ---
	atg_avg_pool2d :: proc(out: ^Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, ceil_mode: c.int, count_include_pad: c.int, divisor_override_v: i64, divisor_override_null: rawptr) ---
	atg_avg_pool2d_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, ceil_mode: c.int, count_include_pad: c.int, divisor_override_v: i64, divisor_override_null: rawptr) ---
	atg_avg_pool2d_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, ceil_mode: c.int, count_include_pad: c.int, divisor_override_v: i64, divisor_override_null: rawptr) ---
	atg_avg_pool2d_out :: proc(out: ^Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, ceil_mode: c.int, count_include_pad: c.int, divisor_override_v: i64, divisor_override_null: rawptr) ---
	atg_avg_pool3d :: proc(out: ^Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, ceil_mode: c.int, count_include_pad: c.int, divisor_override_v: i64, divisor_override_null: rawptr) ---
	atg_avg_pool3d_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, ceil_mode: c.int, count_include_pad: c.int, divisor_override_v: i64, divisor_override_null: rawptr) ---
	atg_avg_pool3d_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, ceil_mode: c.int, count_include_pad: c.int, divisor_override_v: i64, divisor_override_null: rawptr) ---
	atg_avg_pool3d_out :: proc(out: ^Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, ceil_mode: c.int, count_include_pad: c.int, divisor_override_v: i64, divisor_override_null: rawptr) ---
	atg_baddbmm :: proc(out: ^Tensor, self: Tensor, batch1: Tensor, batch2: Tensor, beta: Scalar, alpha: Scalar) ---
	atg_baddbmm_ :: proc(out: ^Tensor, self: Tensor, batch1: Tensor, batch2: Tensor) ---
	atg_baddbmm_dtype :: proc(out: ^Tensor, self: Tensor, batch1: Tensor, batch2: Tensor, out_dtype: c.int, beta: Scalar, alpha: Scalar) ---
	atg_baddbmm_dtype_out :: proc(out: ^Tensor, self: Tensor, batch1: Tensor, batch2: Tensor, out_dtype: c.int) ---
	atg_baddbmm_out :: proc(out: ^Tensor, self: Tensor, batch1: Tensor, batch2: Tensor) ---
	atg_bartlett_window :: proc(out: ^Tensor, window_length: i64, options_kind: c.int, options_device: c.int) ---
	atg_bartlett_window_out :: proc(out: ^Tensor, window_length: i64) ---
	atg_bartlett_window_periodic :: proc(out: ^Tensor, window_length: i64, periodic: c.int, options_kind: c.int, options_device: c.int) ---
	atg_bartlett_window_periodic_out :: proc(out: ^Tensor, window_length: i64, periodic: c.int) ---
	atg_batch_norm :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor, running_mean: Tensor, running_var: Tensor, training: c.int, momentum: f64, eps: f64, cudnn_enabled: c.int) ---
	atg_batch_norm_backward_elemt :: proc(out: ^Tensor, grad_out: Tensor, input: Tensor, mean: Tensor, invstd: Tensor, weight: Tensor, sum_dy: Tensor, sum_dy_xmu: Tensor, count: Tensor) ---
	atg_batch_norm_backward_elemt_out :: proc(out: ^Tensor, grad_out: Tensor, input: Tensor, mean: Tensor, invstd: Tensor, weight: Tensor, sum_dy: Tensor, sum_dy_xmu: Tensor, count: Tensor) ---
	atg_batch_norm_backward_reduce :: proc(out: ^Tensor, grad_out: Tensor, input: Tensor, mean: Tensor, invstd: Tensor, weight: Tensor, input_g: c.int, weight_g: c.int, bias_g: c.int) ---
	atg_batch_norm_backward_reduce_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, out2: Tensor, out3: Tensor, grad_out: Tensor, input: Tensor, mean: Tensor, invstd: Tensor, weight: Tensor, input_g: c.int, weight_g: c.int, bias_g: c.int) ---
	atg_batch_norm_elemt :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor, mean: Tensor, invstd: Tensor, eps: f64) ---
	atg_batch_norm_elemt_out :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor, mean: Tensor, invstd: Tensor, eps: f64) ---
	atg_batch_norm_gather_stats :: proc(out: ^Tensor, input: Tensor, mean: Tensor, invstd: Tensor, running_mean: Tensor, running_var: Tensor, momentum: f64, eps: f64, count: i64) ---
	atg_batch_norm_gather_stats_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, input: Tensor, mean: Tensor, invstd: Tensor, running_mean: Tensor, running_var: Tensor, momentum: f64, eps: f64, count: i64) ---
	atg_batch_norm_gather_stats_with_counts :: proc(out: ^Tensor, input: Tensor, mean: Tensor, invstd: Tensor, running_mean: Tensor, running_var: Tensor, momentum: f64, eps: f64, counts: Tensor) ---
	atg_batch_norm_gather_stats_with_counts_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, input: Tensor, mean: Tensor, invstd: Tensor, running_mean: Tensor, running_var: Tensor, momentum: f64, eps: f64, counts: Tensor) ---
	atg_batch_norm_stats :: proc(out: ^Tensor, input: Tensor, eps: f64) ---
	atg_batch_norm_stats_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, input: Tensor, eps: f64) ---
	atg_batch_norm_update_stats :: proc(out: ^Tensor, input: Tensor, running_mean: Tensor, running_var: Tensor, momentum: f64) ---
	atg_batch_norm_update_stats_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, input: Tensor, running_mean: Tensor, running_var: Tensor, momentum: f64) ---
	atg_bernoulli :: proc(out: ^Tensor, self: Tensor) ---
	atg_bernoulli_ :: proc(out: ^Tensor, self: Tensor, p: Tensor) ---
	atg_bernoulli_float_ :: proc(out: ^Tensor, self: Tensor, p: f64) ---
	atg_bernoulli_p :: proc(out: ^Tensor, self: Tensor, p: f64) ---
	atg_bernoulli_tensor :: proc(out: ^Tensor, self: Tensor, p: Tensor) ---
	atg_bilinear :: proc(out: ^Tensor, input1: Tensor, input2: Tensor, weight: Tensor, bias: Tensor) ---
	atg_binary_cross_entropy :: proc(out: ^Tensor, self: Tensor, target: Tensor, weight: Tensor, reduction: i64) ---
	atg_binary_cross_entropy_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, target: Tensor, weight: Tensor, reduction: i64) ---
	atg_binary_cross_entropy_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, self: Tensor, target: Tensor, weight: Tensor, reduction: i64) ---
	atg_binary_cross_entropy_out :: proc(out: ^Tensor, self: Tensor, target: Tensor, weight: Tensor, reduction: i64) ---
	atg_binary_cross_entropy_with_logits :: proc(out: ^Tensor, self: Tensor, target: Tensor, weight: Tensor, pos_weight: Tensor, reduction: i64) ---
	atg_binary_cross_entropy_with_logits_out :: proc(out: ^Tensor, self: Tensor, target: Tensor, weight: Tensor, pos_weight: Tensor, reduction: i64) ---
	atg_bincount :: proc(out: ^Tensor, self: Tensor, weights: Tensor, minlength: i64) ---
	atg_bincount_out :: proc(out: ^Tensor, self: Tensor, weights: Tensor, minlength: i64) ---
	atg_binomial :: proc(out: ^Tensor, count: Tensor, prob: Tensor) ---
	atg_binomial_out :: proc(out: ^Tensor, count: Tensor, prob: Tensor) ---
	atg_bitwise_and :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_bitwise_and_ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_bitwise_and_scalar_out :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_bitwise_and_scalar_tensor :: proc(out: ^Tensor, self_scalar: Scalar, other: Tensor) ---
	atg_bitwise_and_scalar_tensor_out :: proc(out: ^Tensor, self_scalar: Scalar, other: Tensor) ---
	atg_bitwise_and_tensor :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_bitwise_and_tensor_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_bitwise_and_tensor_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_bitwise_left_shift :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_bitwise_left_shift_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_bitwise_left_shift_scalar_tensor :: proc(out: ^Tensor, self_scalar: Scalar, other: Tensor) ---
	atg_bitwise_left_shift_scalar_tensor_out :: proc(out: ^Tensor, self_scalar: Scalar, other: Tensor) ---
	atg_bitwise_left_shift_tensor_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_bitwise_left_shift_tensor_scalar :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_bitwise_left_shift_tensor_scalar_ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_bitwise_left_shift_tensor_scalar_out :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_bitwise_not :: proc(out: ^Tensor, self: Tensor) ---
	atg_bitwise_not_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_bitwise_not_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_bitwise_or :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_bitwise_or_ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_bitwise_or_scalar_out :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_bitwise_or_scalar_tensor :: proc(out: ^Tensor, self_scalar: Scalar, other: Tensor) ---
	atg_bitwise_or_scalar_tensor_out :: proc(out: ^Tensor, self_scalar: Scalar, other: Tensor) ---
	atg_bitwise_or_tensor :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_bitwise_or_tensor_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_bitwise_or_tensor_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_bitwise_right_shift :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_bitwise_right_shift_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_bitwise_right_shift_scalar_tensor :: proc(out: ^Tensor, self_scalar: Scalar, other: Tensor) ---
	atg_bitwise_right_shift_scalar_tensor_out :: proc(out: ^Tensor, self_scalar: Scalar, other: Tensor) ---
	atg_bitwise_right_shift_tensor_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_bitwise_right_shift_tensor_scalar :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_bitwise_right_shift_tensor_scalar_ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_bitwise_right_shift_tensor_scalar_out :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_bitwise_xor :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_bitwise_xor_ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_bitwise_xor_scalar_out :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_bitwise_xor_scalar_tensor :: proc(out: ^Tensor, self_scalar: Scalar, other: Tensor) ---
	atg_bitwise_xor_scalar_tensor_out :: proc(out: ^Tensor, self_scalar: Scalar, other: Tensor) ---
	atg_bitwise_xor_tensor :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_bitwise_xor_tensor_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_bitwise_xor_tensor_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_blackman_window :: proc(out: ^Tensor, window_length: i64, options_kind: c.int, options_device: c.int) ---
	atg_blackman_window_out :: proc(out: ^Tensor, window_length: i64) ---
	atg_blackman_window_periodic :: proc(out: ^Tensor, window_length: i64, periodic: c.int, options_kind: c.int, options_device: c.int) ---
	atg_blackman_window_periodic_out :: proc(out: ^Tensor, window_length: i64, periodic: c.int) ---
	atg_block_diag :: proc(out: ^Tensor, tensors_data: ^Tensor, tensors_len: c.int) ---
	atg_block_diag_out :: proc(out: ^Tensor, tensors_data: ^Tensor, tensors_len: c.int) ---
	atg_bmm :: proc(out: ^Tensor, self: Tensor, mat2: Tensor) ---
	atg_bmm_dtype :: proc(out: ^Tensor, self: Tensor, mat2: Tensor, out_dtype: c.int) ---
	atg_bmm_dtype_out :: proc(out: ^Tensor, self: Tensor, mat2: Tensor, out_dtype: c.int) ---
	atg_bmm_out :: proc(out: ^Tensor, self: Tensor, mat2: Tensor) ---
	atg_broadcast_to :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int) ---
	atg_bucketize :: proc(out: ^Tensor, self: Tensor, boundaries: Tensor, out_int32: c.int, right: c.int) ---
	atg_bucketize_scalar :: proc(out: ^Tensor, self_scalar: Scalar, boundaries: Tensor, out_int32: c.int, right: c.int) ---
	atg_bucketize_scalar_out :: proc(out: ^Tensor, self_scalar: Scalar, boundaries: Tensor, out_int32: c.int, right: c.int) ---
	atg_bucketize_tensor_out :: proc(out: ^Tensor, self: Tensor, boundaries: Tensor, out_int32: c.int, right: c.int) ---
	atg_cartesian_prod :: proc(out: ^Tensor, tensors_data: ^Tensor, tensors_len: c.int) ---
	atg_cat :: proc(out: ^Tensor, tensors_data: ^Tensor, tensors_len: c.int, dim: i64) ---
	atg_cat_out :: proc(out: ^Tensor, tensors_data: ^Tensor, tensors_len: c.int, dim: i64) ---
	atg_cauchy :: proc(out: ^Tensor, self: Tensor, median: f64, sigma: f64) ---
	atg_cauchy_ :: proc(out: ^Tensor, self: Tensor, median: f64, sigma: f64) ---
	atg_cauchy_out :: proc(out: ^Tensor, self: Tensor, median: f64, sigma: f64) ---
	atg_ccol_indices :: proc(out: ^Tensor, self: Tensor) ---
	atg_ccol_indices_copy :: proc(out: ^Tensor, self: Tensor) ---
	atg_ccol_indices_copy_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_cdist :: proc(out: ^Tensor, x1: Tensor, x2: Tensor, p: f64, compute_mode_v: i64, compute_mode_null: rawptr) ---
	atg_ceil :: proc(out: ^Tensor, self: Tensor) ---
	atg_ceil_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_ceil_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_celu :: proc(out: ^Tensor, self: Tensor) ---
	atg_celu_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_celu_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_chain_matmul :: proc(out: ^Tensor, matrices_data: ^Tensor, matrices_len: c.int) ---
	atg_chain_matmul_out :: proc(out: ^Tensor, matrices_data: ^Tensor, matrices_len: c.int) ---
	atg_chalf :: proc(out: ^Tensor, self: Tensor) ---
	atg_channel_shuffle :: proc(out: ^Tensor, self: Tensor, groups: i64) ---
	atg_channel_shuffle_out :: proc(out: ^Tensor, self: Tensor, groups: i64) ---
	atg_cholesky :: proc(out: ^Tensor, self: Tensor, upper: c.int) ---
	atg_cholesky_inverse :: proc(out: ^Tensor, self: Tensor, upper: c.int) ---
	atg_cholesky_inverse_out :: proc(out: ^Tensor, self: Tensor, upper: c.int) ---
	atg_cholesky_out :: proc(out: ^Tensor, self: Tensor, upper: c.int) ---
	atg_cholesky_solve :: proc(out: ^Tensor, self: Tensor, input2: Tensor, upper: c.int) ---
	atg_cholesky_solve_out :: proc(out: ^Tensor, self: Tensor, input2: Tensor, upper: c.int) ---
	atg_choose_qparams_optimized :: proc(out: ^Tensor, input: Tensor, numel: i64, n_bins: i64, ratio: f64, bit_width: i64) ---
	atg_clamp :: proc(out: ^Tensor, self: Tensor, min: Scalar, max: Scalar) ---
	atg_clamp_ :: proc(out: ^Tensor, self: Tensor, min: Scalar, max: Scalar) ---
	atg_clamp_max :: proc(out: ^Tensor, self: Tensor, max: Scalar) ---
	atg_clamp_max_ :: proc(out: ^Tensor, self: Tensor, max: Scalar) ---
	atg_clamp_max_out :: proc(out: ^Tensor, self: Tensor, max: Scalar) ---
	atg_clamp_max_tensor :: proc(out: ^Tensor, self: Tensor, max: Tensor) ---
	atg_clamp_max_tensor_ :: proc(out: ^Tensor, self: Tensor, max: Tensor) ---
	atg_clamp_max_tensor_out :: proc(out: ^Tensor, self: Tensor, max: Tensor) ---
	atg_clamp_min :: proc(out: ^Tensor, self: Tensor, min: Scalar) ---
	atg_clamp_min_ :: proc(out: ^Tensor, self: Tensor, min: Scalar) ---
	atg_clamp_min_out :: proc(out: ^Tensor, self: Tensor, min: Scalar) ---
	atg_clamp_min_tensor :: proc(out: ^Tensor, self: Tensor, min: Tensor) ---
	atg_clamp_min_tensor_ :: proc(out: ^Tensor, self: Tensor, min: Tensor) ---
	atg_clamp_min_tensor_out :: proc(out: ^Tensor, self: Tensor, min: Tensor) ---
	atg_clamp_out :: proc(out: ^Tensor, self: Tensor, min: Scalar, max: Scalar) ---
	atg_clamp_tensor :: proc(out: ^Tensor, self: Tensor, min: Tensor, max: Tensor) ---
	atg_clamp_tensor_ :: proc(out: ^Tensor, self: Tensor, min: Tensor, max: Tensor) ---
	atg_clamp_tensor_out :: proc(out: ^Tensor, self: Tensor, min: Tensor, max: Tensor) ---
	atg_clip :: proc(out: ^Tensor, self: Tensor, min: Scalar, max: Scalar) ---
	atg_clip_ :: proc(out: ^Tensor, self: Tensor, min: Scalar, max: Scalar) ---
	atg_clip_out :: proc(out: ^Tensor, self: Tensor, min: Scalar, max: Scalar) ---
	atg_clip_tensor :: proc(out: ^Tensor, self: Tensor, min: Tensor, max: Tensor) ---
	atg_clip_tensor_ :: proc(out: ^Tensor, self: Tensor, min: Tensor, max: Tensor) ---
	atg_clip_tensor_out :: proc(out: ^Tensor, self: Tensor, min: Tensor, max: Tensor) ---
	atg_clone :: proc(out: ^Tensor, self: Tensor) ---
	atg_coalesce :: proc(out: ^Tensor, self: Tensor) ---
	atg_col2im :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int, kernel_size_data: [^]i64, kernel_size_len: c.int, dilation_data: [^]i64, dilation_len: c.int, padding_data: [^]i64, padding_len: c.int, stride_data: [^]i64, stride_len: c.int) ---
	atg_col2im_out :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int, kernel_size_data: [^]i64, kernel_size_len: c.int, dilation_data: [^]i64, dilation_len: c.int, padding_data: [^]i64, padding_len: c.int, stride_data: [^]i64, stride_len: c.int) ---
	atg_col_indices :: proc(out: ^Tensor, self: Tensor) ---
	atg_col_indices_copy :: proc(out: ^Tensor, self: Tensor) ---
	atg_col_indices_copy_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_column_stack :: proc(out: ^Tensor, tensors_data: ^Tensor, tensors_len: c.int) ---
	atg_column_stack_out :: proc(out: ^Tensor, tensors_data: ^Tensor, tensors_len: c.int) ---
	atg_combinations :: proc(out: ^Tensor, self: Tensor, r: i64, with_replacement: c.int) ---
	atg_complex :: proc(out: ^Tensor, real: Tensor, imag: Tensor) ---
	atg_complex_out :: proc(out: ^Tensor, real: Tensor, imag: Tensor) ---
	atg_concat :: proc(out: ^Tensor, tensors_data: ^Tensor, tensors_len: c.int, dim: i64) ---
	atg_concat_out :: proc(out: ^Tensor, tensors_data: ^Tensor, tensors_len: c.int, dim: i64) ---
	atg_concatenate :: proc(out: ^Tensor, tensors_data: ^Tensor, tensors_len: c.int, dim: i64) ---
	atg_concatenate_out :: proc(out: ^Tensor, tensors_data: ^Tensor, tensors_len: c.int, dim: i64) ---
	atg_conj :: proc(out: ^Tensor, self: Tensor) ---
	atg_conj_physical :: proc(out: ^Tensor, self: Tensor) ---
	atg_conj_physical_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_conj_physical_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_constant_pad_nd :: proc(out: ^Tensor, self: Tensor, pad_data: [^]i64, pad_len: c.int) ---
	atg_constant_pad_nd_out :: proc(out: ^Tensor, self: Tensor, pad_data: [^]i64, pad_len: c.int) ---
	atg_contiguous :: proc(out: ^Tensor, self: Tensor) ---
	atg_conv1d :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, groups: i64) ---
	atg_conv1d_padding :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_ptr: cstring, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, groups: i64) ---
	atg_conv2d :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, groups: i64) ---
	atg_conv2d_padding :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_ptr: cstring, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, groups: i64) ---
	atg_conv3d :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, groups: i64) ---
	atg_conv3d_padding :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_ptr: cstring, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, groups: i64) ---
	atg_conv_depthwise3d :: proc(out: ^Tensor, self: Tensor, weight: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int) ---
	atg_conv_depthwise3d_out :: proc(out: ^Tensor, self: Tensor, weight: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int) ---
	atg_conv_tbc :: proc(out: ^Tensor, self: Tensor, weight: Tensor, bias: Tensor, pad: i64) ---
	atg_conv_tbc_backward :: proc(out: ^Tensor, self: Tensor, input: Tensor, weight: Tensor, bias: Tensor, pad: i64) ---
	atg_conv_tbc_out :: proc(out: ^Tensor, self: Tensor, weight: Tensor, bias: Tensor, pad: i64) ---
	atg_conv_transpose1d :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, output_padding_data: [^]i64, output_padding_len: c.int, groups: i64, dilation_data: [^]i64, dilation_len: c.int) ---
	atg_conv_transpose2d :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, output_padding_data: [^]i64, output_padding_len: c.int, groups: i64, dilation_data: [^]i64, dilation_len: c.int) ---
	atg_conv_transpose3d :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, output_padding_data: [^]i64, output_padding_len: c.int, groups: i64, dilation_data: [^]i64, dilation_len: c.int) ---
	atg_convolution :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, transposed: c.int, output_padding_data: [^]i64, output_padding_len: c.int, groups: i64) ---
	atg_convolution_out :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, transposed: c.int, output_padding_data: [^]i64, output_padding_len: c.int, groups: i64) ---
	atg_convolution_overrideable :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, transposed: c.int, output_padding_data: [^]i64, output_padding_len: c.int, groups: i64) ---
	atg_convolution_overrideable_out :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, transposed: c.int, output_padding_data: [^]i64, output_padding_len: c.int, groups: i64) ---
	atg_copy_sparse_to_sparse :: proc(out: ^Tensor, self: Tensor, src: Tensor, non_blocking: c.int) ---
	atg_copy_sparse_to_sparse_ :: proc(out: ^Tensor, self: Tensor, src: Tensor, non_blocking: c.int) ---
	atg_copy_sparse_to_sparse_out :: proc(out: ^Tensor, self: Tensor, src: Tensor, non_blocking: c.int) ---
	atg_copysign :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_copysign_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_copysign_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_copysign_scalar :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_copysign_scalar_ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_copysign_scalar_out :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_corrcoef :: proc(out: ^Tensor, self: Tensor) ---
	atg_cos :: proc(out: ^Tensor, self: Tensor) ---
	atg_cos_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_cos_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_cosh :: proc(out: ^Tensor, self: Tensor) ---
	atg_cosh_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_cosh_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_cosine_embedding_loss :: proc(out: ^Tensor, input1: Tensor, input2: Tensor, target: Tensor, margin: f64, reduction: i64) ---
	atg_cosine_similarity :: proc(out: ^Tensor, x1: Tensor, x2: Tensor, dim: i64, eps: f64) ---
	atg_count_nonzero :: proc(out: ^Tensor, self: Tensor, dim_v: i64, dim_null: rawptr) ---
	atg_count_nonzero_dim_intlist :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int) ---
	atg_count_nonzero_dim_intlist_out :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int) ---
	atg_count_nonzero_out :: proc(out: ^Tensor, self: Tensor, dim_v: i64, dim_null: rawptr) ---
	atg_cov :: proc(out: ^Tensor, self: Tensor, correction: i64, fweights: Tensor, aweights: Tensor) ---
	atg_cross :: proc(out: ^Tensor, self: Tensor, other: Tensor, dim_v: i64, dim_null: rawptr) ---
	atg_cross_entropy_loss :: proc(out: ^Tensor, self: Tensor, target: Tensor, weight: Tensor, reduction: i64, ignore_index: i64, label_smoothing: f64) ---
	atg_cross_out :: proc(out: ^Tensor, self: Tensor, other: Tensor, dim_v: i64, dim_null: rawptr) ---
	atg_crow_indices :: proc(out: ^Tensor, self: Tensor) ---
	atg_crow_indices_copy :: proc(out: ^Tensor, self: Tensor) ---
	atg_crow_indices_copy_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_ctc_loss :: proc(out: ^Tensor, log_probs: Tensor, targets: Tensor, input_lengths_data: [^]i64, input_lengths_len: c.int, target_lengths_data: [^]i64, target_lengths_len: c.int, blank: i64, reduction: i64, zero_infinity: c.int) ---
	atg_ctc_loss_tensor :: proc(out: ^Tensor, log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor, blank: i64, reduction: i64, zero_infinity: c.int) ---
	atg_cudnn_affine_grid_generator :: proc(out: ^Tensor, theta: Tensor, n: i64, C: i64, H: i64, W: i64) ---
	atg_cudnn_affine_grid_generator_backward :: proc(out: ^Tensor, grad: Tensor, n: i64, C: i64, H: i64, W: i64) ---
	atg_cudnn_affine_grid_generator_backward_out :: proc(out: ^Tensor, grad: Tensor, n: i64, C: i64, H: i64, W: i64) ---
	atg_cudnn_affine_grid_generator_out :: proc(out: ^Tensor, theta: Tensor, n: i64, C: i64, H: i64, W: i64) ---
	atg_cudnn_batch_norm :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor, running_mean: Tensor, running_var: Tensor, training: c.int, exponential_average_factor: f64, epsilon: f64) ---
	atg_cudnn_batch_norm_backward :: proc(out: ^Tensor, input: Tensor, grad_output: Tensor, weight: Tensor, running_mean: Tensor, running_var: Tensor, save_mean: Tensor, save_var: Tensor, epsilon: f64, reserveSpace: Tensor) ---
	atg_cudnn_batch_norm_backward_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, out2: Tensor, input: Tensor, grad_output: Tensor, weight: Tensor, running_mean: Tensor, running_var: Tensor, save_mean: Tensor, save_var: Tensor, epsilon: f64, reserveSpace: Tensor) ---
	atg_cudnn_batch_norm_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, out2: Tensor, out3: Tensor, input: Tensor, weight: Tensor, bias: Tensor, running_mean: Tensor, running_var: Tensor, training: c.int, exponential_average_factor: f64, epsilon: f64) ---
	atg_cudnn_convolution :: proc(out: ^Tensor, self: Tensor, weight: Tensor, padding_data: [^]i64, padding_len: c.int, stride_data: [^]i64, stride_len: c.int, dilation_data: [^]i64, dilation_len: c.int, groups: i64, benchmark: c.int, deterministic: c.int, allow_tf32: c.int) ---
	atg_cudnn_convolution_add_relu :: proc(out: ^Tensor, self: Tensor, weight: Tensor, z: Tensor, alpha: Scalar, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, groups: i64) ---
	atg_cudnn_convolution_add_relu_out :: proc(out: ^Tensor, self: Tensor, weight: Tensor, z: Tensor, alpha: Scalar, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, groups: i64) ---
	atg_cudnn_convolution_out :: proc(out: ^Tensor, self: Tensor, weight: Tensor, padding_data: [^]i64, padding_len: c.int, stride_data: [^]i64, stride_len: c.int, dilation_data: [^]i64, dilation_len: c.int, groups: i64, benchmark: c.int, deterministic: c.int, allow_tf32: c.int) ---
	atg_cudnn_convolution_relu :: proc(out: ^Tensor, self: Tensor, weight: Tensor, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, groups: i64) ---
	atg_cudnn_convolution_relu_out :: proc(out: ^Tensor, self: Tensor, weight: Tensor, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, groups: i64) ---
	atg_cudnn_convolution_transpose :: proc(out: ^Tensor, self: Tensor, weight: Tensor, padding_data: [^]i64, padding_len: c.int, output_padding_data: [^]i64, output_padding_len: c.int, stride_data: [^]i64, stride_len: c.int, dilation_data: [^]i64, dilation_len: c.int, groups: i64, benchmark: c.int, deterministic: c.int, allow_tf32: c.int) ---
	atg_cudnn_convolution_transpose_out :: proc(out: ^Tensor, self: Tensor, weight: Tensor, padding_data: [^]i64, padding_len: c.int, output_padding_data: [^]i64, output_padding_len: c.int, stride_data: [^]i64, stride_len: c.int, dilation_data: [^]i64, dilation_len: c.int, groups: i64, benchmark: c.int, deterministic: c.int, allow_tf32: c.int) ---
	atg_cudnn_grid_sampler :: proc(out: ^Tensor, self: Tensor, grid: Tensor) ---
	atg_cudnn_grid_sampler_backward :: proc(out: ^Tensor, self: Tensor, grid: Tensor, grad_output: Tensor) ---
	atg_cudnn_grid_sampler_backward_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, self: Tensor, grid: Tensor, grad_output: Tensor) ---
	atg_cudnn_grid_sampler_out :: proc(out: ^Tensor, self: Tensor, grid: Tensor) ---
	atg_cummax :: proc(out: ^Tensor, self: Tensor, dim: i64) ---
	atg_cummax_out :: proc(out: ^Tensor, values: Tensor, indices: Tensor, self: Tensor, dim: i64) ---
	atg_cummaxmin_backward :: proc(out: ^Tensor, grad: Tensor, input: Tensor, indices: Tensor, dim: i64) ---
	atg_cummin :: proc(out: ^Tensor, self: Tensor, dim: i64) ---
	atg_cummin_out :: proc(out: ^Tensor, values: Tensor, indices: Tensor, self: Tensor, dim: i64) ---
	atg_cumprod :: proc(out: ^Tensor, self: Tensor, dim: i64, dtype: c.int) ---
	atg_cumprod_ :: proc(out: ^Tensor, self: Tensor, dim: i64, dtype: c.int) ---
	atg_cumprod_backward :: proc(out: ^Tensor, grad: Tensor, input: Tensor, dim: i64, output: Tensor) ---
	atg_cumprod_out :: proc(out: ^Tensor, self: Tensor, dim: i64, dtype: c.int) ---
	atg_cumsum :: proc(out: ^Tensor, self: Tensor, dim: i64, dtype: c.int) ---
	atg_cumsum_ :: proc(out: ^Tensor, self: Tensor, dim: i64, dtype: c.int) ---
	atg_cumsum_out :: proc(out: ^Tensor, self: Tensor, dim: i64, dtype: c.int) ---
	atg_cumulative_trapezoid :: proc(out: ^Tensor, y: Tensor, dim: i64) ---
	atg_cumulative_trapezoid_x :: proc(out: ^Tensor, y: Tensor, x: Tensor, dim: i64) ---
	atg_data :: proc(out: ^Tensor, self: Tensor) ---
	atg_deg2rad :: proc(out: ^Tensor, self: Tensor) ---
	atg_deg2rad_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_deg2rad_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_dequantize :: proc(out: ^Tensor, self: Tensor) ---
	atg_dequantize_self_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_dequantize_tensors_out :: proc(out_data: ^Tensor, out_len: c.int, tensors_data: ^Tensor, tensors_len: c.int) ---
	atg_det :: proc(out: ^Tensor, self: Tensor) ---
	atg_detach :: proc(out: ^Tensor, self: Tensor) ---
	atg_detach_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_detach_copy :: proc(out: ^Tensor, self: Tensor) ---
	atg_detach_copy_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_diag :: proc(out: ^Tensor, self: Tensor, diagonal: i64) ---
	atg_diag_embed :: proc(out: ^Tensor, self: Tensor, offset: i64, dim1: i64, dim2: i64) ---
	atg_diag_embed_out :: proc(out: ^Tensor, self: Tensor, offset: i64, dim1: i64, dim2: i64) ---
	atg_diag_out :: proc(out: ^Tensor, self: Tensor, diagonal: i64) ---
	atg_diagflat :: proc(out: ^Tensor, self: Tensor, offset: i64) ---
	atg_diagonal :: proc(out: ^Tensor, self: Tensor, offset: i64, dim1: i64, dim2: i64) ---
	atg_diagonal_backward :: proc(out: ^Tensor, grad_output: Tensor, input_sizes_data: [^]i64, input_sizes_len: c.int, offset: i64, dim1: i64, dim2: i64) ---
	atg_diagonal_backward_out :: proc(out: ^Tensor, grad_output: Tensor, input_sizes_data: [^]i64, input_sizes_len: c.int, offset: i64, dim1: i64, dim2: i64) ---
	atg_diagonal_copy :: proc(out: ^Tensor, self: Tensor, offset: i64, dim1: i64, dim2: i64) ---
	atg_diagonal_copy_out :: proc(out: ^Tensor, self: Tensor, offset: i64, dim1: i64, dim2: i64) ---
	atg_diagonal_scatter :: proc(out: ^Tensor, self: Tensor, src: Tensor, offset: i64, dim1: i64, dim2: i64) ---
	atg_diagonal_scatter_out :: proc(out: ^Tensor, self: Tensor, src: Tensor, offset: i64, dim1: i64, dim2: i64) ---
	atg_diff :: proc(out: ^Tensor, self: Tensor, n: i64, dim: i64, prepend: Tensor, append: Tensor) ---
	atg_diff_out :: proc(out: ^Tensor, self: Tensor, n: i64, dim: i64, prepend: Tensor, append: Tensor) ---
	atg_digamma :: proc(out: ^Tensor, self: Tensor) ---
	atg_digamma_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_digamma_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_dist :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_dist_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_div :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_div_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_div_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_div_out_mode :: proc(out: ^Tensor, self: Tensor, other: Tensor, rounding_mode_ptr: cstring, rounding_mode_len: c.int) ---
	atg_div_scalar :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_div_scalar_ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_div_scalar_mode :: proc(out: ^Tensor, self: Tensor, other: Scalar, rounding_mode_ptr: cstring, rounding_mode_len: c.int) ---
	atg_div_scalar_mode_ :: proc(out: ^Tensor, self: Tensor, other: Scalar, rounding_mode_ptr: cstring, rounding_mode_len: c.int) ---
	atg_div_scalar_mode_out :: proc(out: ^Tensor, self: Tensor, other: Scalar, rounding_mode_ptr: cstring, rounding_mode_len: c.int) ---
	atg_div_scalar_out :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_div_tensor_mode :: proc(out: ^Tensor, self: Tensor, other: Tensor, rounding_mode_ptr: cstring, rounding_mode_len: c.int) ---
	atg_div_tensor_mode_ :: proc(out: ^Tensor, self: Tensor, other: Tensor, rounding_mode_ptr: cstring, rounding_mode_len: c.int) ---
	atg_divide :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_divide_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_divide_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_divide_out_mode :: proc(out: ^Tensor, self: Tensor, other: Tensor, rounding_mode_ptr: cstring, rounding_mode_len: c.int) ---
	atg_divide_scalar :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_divide_scalar_ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_divide_scalar_mode :: proc(out: ^Tensor, self: Tensor, other: Scalar, rounding_mode_ptr: cstring, rounding_mode_len: c.int) ---
	atg_divide_scalar_mode_ :: proc(out: ^Tensor, self: Tensor, other: Scalar, rounding_mode_ptr: cstring, rounding_mode_len: c.int) ---
	atg_divide_tensor_mode :: proc(out: ^Tensor, self: Tensor, other: Tensor, rounding_mode_ptr: cstring, rounding_mode_len: c.int) ---
	atg_divide_tensor_mode_ :: proc(out: ^Tensor, self: Tensor, other: Tensor, rounding_mode_ptr: cstring, rounding_mode_len: c.int) ---
	atg_dot :: proc(out: ^Tensor, self: Tensor, tensor: Tensor) ---
	atg_dot_out :: proc(out: ^Tensor, self: Tensor, tensor: Tensor) ---
	atg_dropout :: proc(out: ^Tensor, input: Tensor, p: f64, train: c.int) ---
	atg_dropout_ :: proc(out: ^Tensor, self: Tensor, p: f64, train: c.int) ---
	atg_dstack :: proc(out: ^Tensor, tensors_data: ^Tensor, tensors_len: c.int) ---
	atg_dstack_out :: proc(out: ^Tensor, tensors_data: ^Tensor, tensors_len: c.int) ---
	atg_einsum :: proc(out: ^Tensor, equation_ptr: cstring, equation_len: c.int, tensors_data: ^Tensor, tensors_len: c.int, path_data: [^]i64, path_len: c.int) ---
	atg_elu :: proc(out: ^Tensor, self: Tensor) ---
	atg_elu_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_elu_backward :: proc(out: ^Tensor, grad_output: Tensor, alpha: Scalar, scale: Scalar, input_scale: Scalar, is_result: c.int, self_or_result: Tensor) ---
	atg_elu_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, alpha: Scalar, scale: Scalar, input_scale: Scalar, is_result: c.int, self_or_result: Tensor) ---
	atg_elu_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_embedding :: proc(out: ^Tensor, weight: Tensor, indices: Tensor, padding_idx: i64, scale_grad_by_freq: c.int, sparse: c.int) ---
	atg_embedding_backward :: proc(out: ^Tensor, grad: Tensor, indices: Tensor, num_weights: i64, padding_idx: i64, scale_grad_by_freq: c.int, sparse: c.int) ---
	atg_embedding_bag :: proc(out: ^Tensor, weight: Tensor, indices: Tensor, offsets: Tensor, scale_grad_by_freq: c.int, mode: i64, sparse: c.int, per_sample_weights: Tensor, include_last_offset: c.int) ---
	atg_embedding_bag_padding_idx :: proc(out: ^Tensor, weight: Tensor, indices: Tensor, offsets: Tensor, scale_grad_by_freq: c.int, mode: i64, sparse: c.int, per_sample_weights: Tensor, include_last_offset: c.int, padding_idx_v: i64, padding_idx_null: rawptr) ---
	atg_embedding_dense_backward :: proc(out: ^Tensor, grad_output: Tensor, indices: Tensor, num_weights: i64, padding_idx: i64, scale_grad_by_freq: c.int) ---
	atg_embedding_dense_backward_out :: proc(out: ^Tensor, grad_output: Tensor, indices: Tensor, num_weights: i64, padding_idx: i64, scale_grad_by_freq: c.int) ---
	atg_embedding_out :: proc(out: ^Tensor, weight: Tensor, indices: Tensor, padding_idx: i64, scale_grad_by_freq: c.int, sparse: c.int) ---
	atg_embedding_renorm :: proc(out: ^Tensor, self: Tensor, indices: Tensor, max_norm: f64, norm_type: f64) ---
	atg_embedding_renorm_ :: proc(out: ^Tensor, self: Tensor, indices: Tensor, max_norm: f64, norm_type: f64) ---
	atg_embedding_renorm_out :: proc(out: ^Tensor, self: Tensor, indices: Tensor, max_norm: f64, norm_type: f64) ---
	atg_embedding_sparse_backward :: proc(out: ^Tensor, grad: Tensor, indices: Tensor, num_weights: i64, padding_idx: i64, scale_grad_by_freq: c.int) ---
	atg_empty :: proc(out: ^Tensor, size_data: [^]i64, size_len: c.int, options_kind: c.int, options_device: c.int) ---
	atg_empty_like :: proc(out: ^Tensor, self: Tensor) ---
	atg_empty_like_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_empty_out :: proc(out: ^Tensor, size_data: [^]i64, size_len: c.int) ---
	atg_empty_permuted :: proc(out: ^Tensor, size_data: [^]i64, size_len: c.int, physical_layout_data: [^]i64, physical_layout_len: c.int, options_kind: c.int, options_device: c.int) ---
	atg_empty_permuted_out :: proc(out: ^Tensor, size_data: [^]i64, size_len: c.int, physical_layout_data: [^]i64, physical_layout_len: c.int) ---
	atg_empty_quantized :: proc(out: ^Tensor, size_data: [^]i64, size_len: c.int, qtensor: Tensor, options_kind: c.int, options_device: c.int) ---
	atg_empty_quantized_out :: proc(out: ^Tensor, size_data: [^]i64, size_len: c.int, qtensor: Tensor) ---
	atg_empty_strided :: proc(out: ^Tensor, size_data: [^]i64, size_len: c.int, stride_data: [^]i64, stride_len: c.int, options_kind: c.int, options_device: c.int) ---
	atg_empty_strided_out :: proc(out: ^Tensor, size_data: [^]i64, size_len: c.int, stride_data: [^]i64, stride_len: c.int) ---
	atg_eq :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_eq_ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_eq_scalar_out :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_eq_tensor :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_eq_tensor_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_eq_tensor_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_erf :: proc(out: ^Tensor, self: Tensor) ---
	atg_erf_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_erf_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_erfc :: proc(out: ^Tensor, self: Tensor) ---
	atg_erfc_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_erfc_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_erfinv :: proc(out: ^Tensor, self: Tensor) ---
	atg_erfinv_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_erfinv_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_exp :: proc(out: ^Tensor, self: Tensor) ---
	atg_exp2 :: proc(out: ^Tensor, self: Tensor) ---
	atg_exp2_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_exp2_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_exp_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_exp_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_expand :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int, implicit: c.int) ---
	atg_expand_as :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_expand_copy :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int, implicit: c.int) ---
	atg_expand_copy_out :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int, implicit: c.int) ---
	atg_expm1 :: proc(out: ^Tensor, self: Tensor) ---
	atg_expm1_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_expm1_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_exponential :: proc(out: ^Tensor, self: Tensor, lambd: f64) ---
	atg_exponential_ :: proc(out: ^Tensor, self: Tensor, lambd: f64) ---
	atg_exponential_out :: proc(out: ^Tensor, self: Tensor, lambd: f64) ---
	atg_eye :: proc(out: ^Tensor, n: i64, options_kind: c.int, options_device: c.int) ---
	atg_eye_m :: proc(out: ^Tensor, n: i64, m: i64, options_kind: c.int, options_device: c.int) ---
	atg_eye_m_out :: proc(out: ^Tensor, n: i64, m: i64) ---
	atg_eye_out :: proc(out: ^Tensor, n: i64) ---
	atg_fake_quantize_per_channel_affine :: proc(out: ^Tensor, self: Tensor, scale: Tensor, zero_point: Tensor, axis: i64, quant_min: i64, quant_max: i64) ---
	atg_fake_quantize_per_channel_affine_cachemask :: proc(out: ^Tensor, self: Tensor, scale: Tensor, zero_point: Tensor, axis: i64, quant_min: i64, quant_max: i64) ---
	atg_fake_quantize_per_channel_affine_cachemask_backward :: proc(out: ^Tensor, grad: Tensor, mask: Tensor) ---
	atg_fake_quantize_per_channel_affine_cachemask_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, self: Tensor, scale: Tensor, zero_point: Tensor, axis: i64, quant_min: i64, quant_max: i64) ---
	atg_fake_quantize_per_tensor_affine :: proc(out: ^Tensor, self: Tensor, scale: f64, zero_point: i64, quant_min: i64, quant_max: i64) ---
	atg_fake_quantize_per_tensor_affine_cachemask :: proc(out: ^Tensor, self: Tensor, scale: f64, zero_point: i64, quant_min: i64, quant_max: i64) ---
	atg_fake_quantize_per_tensor_affine_cachemask_backward :: proc(out: ^Tensor, grad: Tensor, mask: Tensor) ---
	atg_fake_quantize_per_tensor_affine_cachemask_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, self: Tensor, scale: f64, zero_point: i64, quant_min: i64, quant_max: i64) ---
	atg_fake_quantize_per_tensor_affine_tensor_qparams :: proc(out: ^Tensor, self: Tensor, scale: Tensor, zero_point: Tensor, quant_min: i64, quant_max: i64) ---
	atg_feature_alpha_dropout :: proc(out: ^Tensor, input: Tensor, p: f64, train: c.int) ---
	atg_feature_alpha_dropout_ :: proc(out: ^Tensor, self: Tensor, p: f64, train: c.int) ---
	atg_feature_dropout :: proc(out: ^Tensor, input: Tensor, p: f64, train: c.int) ---
	atg_feature_dropout_ :: proc(out: ^Tensor, self: Tensor, p: f64, train: c.int) ---
	atg_fft_fft :: proc(out: ^Tensor, self: Tensor, n_v: i64, n_null: rawptr, dim: i64, norm_ptr: cstring, norm_len: c.int) ---
	atg_fft_fft2 :: proc(out: ^Tensor, self: Tensor, s_data: [^]i64, s_len: c.int, dim_data: [^]i64, dim_len: c.int, norm_ptr: cstring, norm_len: c.int) ---
	atg_fft_fft2_out :: proc(out: ^Tensor, self: Tensor, s_data: [^]i64, s_len: c.int, dim_data: [^]i64, dim_len: c.int, norm_ptr: cstring, norm_len: c.int) ---
	atg_fft_fft_out :: proc(out: ^Tensor, self: Tensor, n_v: i64, n_null: rawptr, dim: i64, norm_ptr: cstring, norm_len: c.int) ---
	atg_fft_fftfreq :: proc(out: ^Tensor, n: i64, d: f64, options_kind: c.int, options_device: c.int) ---
	atg_fft_fftfreq_out :: proc(out: ^Tensor, n: i64, d: f64) ---
	atg_fft_fftn :: proc(out: ^Tensor, self: Tensor, s_data: [^]i64, s_len: c.int, dim_data: [^]i64, dim_len: c.int, norm_ptr: cstring, norm_len: c.int) ---
	atg_fft_fftn_out :: proc(out: ^Tensor, self: Tensor, s_data: [^]i64, s_len: c.int, dim_data: [^]i64, dim_len: c.int, norm_ptr: cstring, norm_len: c.int) ---
	atg_fft_fftshift :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int) ---
	atg_fft_hfft :: proc(out: ^Tensor, self: Tensor, n_v: i64, n_null: rawptr, dim: i64, norm_ptr: cstring, norm_len: c.int) ---
	atg_fft_hfft2 :: proc(out: ^Tensor, self: Tensor, s_data: [^]i64, s_len: c.int, dim_data: [^]i64, dim_len: c.int, norm_ptr: cstring, norm_len: c.int) ---
	atg_fft_hfft2_out :: proc(out: ^Tensor, self: Tensor, s_data: [^]i64, s_len: c.int, dim_data: [^]i64, dim_len: c.int, norm_ptr: cstring, norm_len: c.int) ---
	atg_fft_hfft_out :: proc(out: ^Tensor, self: Tensor, n_v: i64, n_null: rawptr, dim: i64, norm_ptr: cstring, norm_len: c.int) ---
	atg_fft_hfftn :: proc(out: ^Tensor, self: Tensor, s_data: [^]i64, s_len: c.int, dim_data: [^]i64, dim_len: c.int, norm_ptr: cstring, norm_len: c.int) ---
	atg_fft_hfftn_out :: proc(out: ^Tensor, self: Tensor, s_data: [^]i64, s_len: c.int, dim_data: [^]i64, dim_len: c.int, norm_ptr: cstring, norm_len: c.int) ---
	atg_fft_ifft :: proc(out: ^Tensor, self: Tensor, n_v: i64, n_null: rawptr, dim: i64, norm_ptr: cstring, norm_len: c.int) ---
	atg_fft_ifft2 :: proc(out: ^Tensor, self: Tensor, s_data: [^]i64, s_len: c.int, dim_data: [^]i64, dim_len: c.int, norm_ptr: cstring, norm_len: c.int) ---
	atg_fft_ifft2_out :: proc(out: ^Tensor, self: Tensor, s_data: [^]i64, s_len: c.int, dim_data: [^]i64, dim_len: c.int, norm_ptr: cstring, norm_len: c.int) ---
	atg_fft_ifft_out :: proc(out: ^Tensor, self: Tensor, n_v: i64, n_null: rawptr, dim: i64, norm_ptr: cstring, norm_len: c.int) ---
	atg_fft_ifftn :: proc(out: ^Tensor, self: Tensor, s_data: [^]i64, s_len: c.int, dim_data: [^]i64, dim_len: c.int, norm_ptr: cstring, norm_len: c.int) ---
	atg_fft_ifftn_out :: proc(out: ^Tensor, self: Tensor, s_data: [^]i64, s_len: c.int, dim_data: [^]i64, dim_len: c.int, norm_ptr: cstring, norm_len: c.int) ---
	atg_fft_ifftshift :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int) ---
	atg_fft_ihfft :: proc(out: ^Tensor, self: Tensor, n_v: i64, n_null: rawptr, dim: i64, norm_ptr: cstring, norm_len: c.int) ---
	atg_fft_ihfft2 :: proc(out: ^Tensor, self: Tensor, s_data: [^]i64, s_len: c.int, dim_data: [^]i64, dim_len: c.int, norm_ptr: cstring, norm_len: c.int) ---
	atg_fft_ihfft2_out :: proc(out: ^Tensor, self: Tensor, s_data: [^]i64, s_len: c.int, dim_data: [^]i64, dim_len: c.int, norm_ptr: cstring, norm_len: c.int) ---
	atg_fft_ihfft_out :: proc(out: ^Tensor, self: Tensor, n_v: i64, n_null: rawptr, dim: i64, norm_ptr: cstring, norm_len: c.int) ---
	atg_fft_ihfftn :: proc(out: ^Tensor, self: Tensor, s_data: [^]i64, s_len: c.int, dim_data: [^]i64, dim_len: c.int, norm_ptr: cstring, norm_len: c.int) ---
	atg_fft_ihfftn_out :: proc(out: ^Tensor, self: Tensor, s_data: [^]i64, s_len: c.int, dim_data: [^]i64, dim_len: c.int, norm_ptr: cstring, norm_len: c.int) ---
	atg_fft_irfft :: proc(out: ^Tensor, self: Tensor, n_v: i64, n_null: rawptr, dim: i64, norm_ptr: cstring, norm_len: c.int) ---
	atg_fft_irfft2 :: proc(out: ^Tensor, self: Tensor, s_data: [^]i64, s_len: c.int, dim_data: [^]i64, dim_len: c.int, norm_ptr: cstring, norm_len: c.int) ---
	atg_fft_irfft2_out :: proc(out: ^Tensor, self: Tensor, s_data: [^]i64, s_len: c.int, dim_data: [^]i64, dim_len: c.int, norm_ptr: cstring, norm_len: c.int) ---
	atg_fft_irfft_out :: proc(out: ^Tensor, self: Tensor, n_v: i64, n_null: rawptr, dim: i64, norm_ptr: cstring, norm_len: c.int) ---
	atg_fft_irfftn :: proc(out: ^Tensor, self: Tensor, s_data: [^]i64, s_len: c.int, dim_data: [^]i64, dim_len: c.int, norm_ptr: cstring, norm_len: c.int) ---
	atg_fft_irfftn_out :: proc(out: ^Tensor, self: Tensor, s_data: [^]i64, s_len: c.int, dim_data: [^]i64, dim_len: c.int, norm_ptr: cstring, norm_len: c.int) ---
	atg_fft_rfft :: proc(out: ^Tensor, self: Tensor, n_v: i64, n_null: rawptr, dim: i64, norm_ptr: cstring, norm_len: c.int) ---
	atg_fft_rfft2 :: proc(out: ^Tensor, self: Tensor, s_data: [^]i64, s_len: c.int, dim_data: [^]i64, dim_len: c.int, norm_ptr: cstring, norm_len: c.int) ---
	atg_fft_rfft2_out :: proc(out: ^Tensor, self: Tensor, s_data: [^]i64, s_len: c.int, dim_data: [^]i64, dim_len: c.int, norm_ptr: cstring, norm_len: c.int) ---
	atg_fft_rfft_out :: proc(out: ^Tensor, self: Tensor, n_v: i64, n_null: rawptr, dim: i64, norm_ptr: cstring, norm_len: c.int) ---
	atg_fft_rfftfreq :: proc(out: ^Tensor, n: i64, d: f64, options_kind: c.int, options_device: c.int) ---
	atg_fft_rfftfreq_out :: proc(out: ^Tensor, n: i64, d: f64) ---
	atg_fft_rfftn :: proc(out: ^Tensor, self: Tensor, s_data: [^]i64, s_len: c.int, dim_data: [^]i64, dim_len: c.int, norm_ptr: cstring, norm_len: c.int) ---
	atg_fft_rfftn_out :: proc(out: ^Tensor, self: Tensor, s_data: [^]i64, s_len: c.int, dim_data: [^]i64, dim_len: c.int, norm_ptr: cstring, norm_len: c.int) ---
	atg_fill :: proc(out: ^Tensor, self: Tensor, value: Scalar) ---
	atg_fill_ :: proc(out: ^Tensor, self: Tensor, value: Scalar) ---
	atg_fill_diagonal_ :: proc(out: ^Tensor, self: Tensor, fill_value: Scalar, wrap: c.int) ---
	atg_fill_scalar_out :: proc(out: ^Tensor, self: Tensor, value: Scalar) ---
	atg_fill_tensor :: proc(out: ^Tensor, self: Tensor, value: Tensor) ---
	atg_fill_tensor_ :: proc(out: ^Tensor, self: Tensor, value: Tensor) ---
	atg_fill_tensor_out :: proc(out: ^Tensor, self: Tensor, value: Tensor) ---
	atg_fix :: proc(out: ^Tensor, self: Tensor) ---
	atg_fix_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_fix_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_flatten :: proc(out: ^Tensor, self: Tensor, start_dim: i64, end_dim: i64) ---
	atg_flatten_dense_tensors :: proc(out: ^Tensor, tensors_data: ^Tensor, tensors_len: c.int) ---
	atg_flip :: proc(out: ^Tensor, self: Tensor, dims_data: [^]i64, dims_len: c.int) ---
	atg_flip_out :: proc(out: ^Tensor, self: Tensor, dims_data: [^]i64, dims_len: c.int) ---
	atg_fliplr :: proc(out: ^Tensor, self: Tensor) ---
	atg_flipud :: proc(out: ^Tensor, self: Tensor) ---
	atg_float_power :: proc(out: ^Tensor, self: Tensor, exponent: Tensor) ---
	atg_float_power_ :: proc(out: ^Tensor, self: Tensor, exponent: Scalar) ---
	atg_float_power_scalar :: proc(out: ^Tensor, self_scalar: Scalar, exponent: Tensor) ---
	atg_float_power_scalar_out :: proc(out: ^Tensor, self_scalar: Scalar, exponent: Tensor) ---
	atg_float_power_tensor_ :: proc(out: ^Tensor, self: Tensor, exponent: Tensor) ---
	atg_float_power_tensor_scalar :: proc(out: ^Tensor, self: Tensor, exponent: Scalar) ---
	atg_float_power_tensor_scalar_out :: proc(out: ^Tensor, self: Tensor, exponent: Scalar) ---
	atg_float_power_tensor_tensor_out :: proc(out: ^Tensor, self: Tensor, exponent: Tensor) ---
	atg_floor :: proc(out: ^Tensor, self: Tensor) ---
	atg_floor_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_floor_divide :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_floor_divide_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_floor_divide_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_floor_divide_scalar :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_floor_divide_scalar_ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_floor_divide_scalar_out :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_floor_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_fmax :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_fmax_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_fmin :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_fmin_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_fmod :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_fmod_ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_fmod_scalar_out :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_fmod_tensor :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_fmod_tensor_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_fmod_tensor_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_frac :: proc(out: ^Tensor, self: Tensor) ---
	atg_frac_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_frac_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_fractional_max_pool2d :: proc(out: ^Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, output_size_data: [^]i64, output_size_len: c.int, random_samples: Tensor) ---
	atg_fractional_max_pool2d_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, output_size_data: [^]i64, output_size_len: c.int, indices: Tensor) ---
	atg_fractional_max_pool2d_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, output_size_data: [^]i64, output_size_len: c.int, indices: Tensor) ---
	atg_fractional_max_pool2d_output :: proc(out: ^Tensor, output: Tensor, indices: Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, output_size_data: [^]i64, output_size_len: c.int, random_samples: Tensor) ---
	atg_fractional_max_pool3d :: proc(out: ^Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, output_size_data: [^]i64, output_size_len: c.int, random_samples: Tensor) ---
	atg_fractional_max_pool3d_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, output_size_data: [^]i64, output_size_len: c.int, indices: Tensor) ---
	atg_fractional_max_pool3d_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, output_size_data: [^]i64, output_size_len: c.int, indices: Tensor) ---
	atg_fractional_max_pool3d_output :: proc(out: ^Tensor, output: Tensor, indices: Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, output_size_data: [^]i64, output_size_len: c.int, random_samples: Tensor) ---
	atg_frexp :: proc(out: ^Tensor, self: Tensor) ---
	atg_frexp_tensor_out :: proc(out: ^Tensor, mantissa: Tensor, exponent: Tensor, self: Tensor) ---
	atg_frobenius_norm :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, keepdim: c.int) ---
	atg_frobenius_norm_out :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, keepdim: c.int) ---
	atg_from_file :: proc(out: ^Tensor, filename_ptr: cstring, filename_len: c.int, shared: c.int, size_v: i64, size_null: rawptr, options_kind: c.int, options_device: c.int) ---
	atg_from_file_out :: proc(out: ^Tensor, filename_ptr: cstring, filename_len: c.int, shared: c.int, size_v: i64, size_null: rawptr) ---
	atg_full :: proc(out: ^Tensor, size_data: [^]i64, size_len: c.int, fill_value: Scalar, options_kind: c.int, options_device: c.int) ---
	atg_full_like :: proc(out: ^Tensor, self: Tensor, fill_value: Scalar) ---
	atg_full_like_out :: proc(out: ^Tensor, self: Tensor, fill_value: Scalar) ---
	atg_full_out :: proc(out: ^Tensor, size_data: [^]i64, size_len: c.int, fill_value: Scalar) ---
	atg_fused_moving_avg_obs_fake_quant :: proc(out: ^Tensor, self: Tensor, observer_on: Tensor, fake_quant_on: Tensor, running_min: Tensor, running_max: Tensor, scale: Tensor, zero_point: Tensor, averaging_const: f64, quant_min: i64, quant_max: i64, ch_axis: i64, per_row_fake_quant: c.int, symmetric_quant: c.int) ---
	atg_gather :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor, sparse_grad: c.int) ---
	atg_gather_backward :: proc(out: ^Tensor, grad: Tensor, self: Tensor, dim: i64, index: Tensor, sparse_grad: c.int) ---
	atg_gather_out :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor, sparse_grad: c.int) ---
	atg_gcd :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_gcd_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_gcd_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_ge :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_ge_ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_ge_scalar_out :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_ge_tensor :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_ge_tensor_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_ge_tensor_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_gelu :: proc(out: ^Tensor, self: Tensor, approximate_ptr: cstring, approximate_len: c.int) ---
	atg_gelu_ :: proc(out: ^Tensor, self: Tensor, approximate_ptr: cstring, approximate_len: c.int) ---
	atg_gelu_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, approximate_ptr: cstring, approximate_len: c.int) ---
	atg_gelu_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, self: Tensor, approximate_ptr: cstring, approximate_len: c.int) ---
	atg_gelu_out :: proc(out: ^Tensor, self: Tensor, approximate_ptr: cstring, approximate_len: c.int) ---
	atg_geometric :: proc(out: ^Tensor, self: Tensor, p: f64) ---
	atg_geometric_ :: proc(out: ^Tensor, self: Tensor, p: f64) ---
	atg_geometric_out :: proc(out: ^Tensor, self: Tensor, p: f64) ---
	atg_geqrf :: proc(out: ^Tensor, self: Tensor) ---
	atg_geqrf_a :: proc(out: ^Tensor, a: Tensor, tau: Tensor, self: Tensor) ---
	atg_ger :: proc(out: ^Tensor, self: Tensor, vec2: Tensor) ---
	atg_ger_out :: proc(out: ^Tensor, self: Tensor, vec2: Tensor) ---
	atg_glu :: proc(out: ^Tensor, self: Tensor, dim: i64) ---
	atg_glu_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, dim: i64) ---
	atg_glu_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, self: Tensor, dim: i64) ---
	atg_glu_backward_jvp :: proc(out: ^Tensor, grad_x: Tensor, grad_glu: Tensor, x: Tensor, dgrad_glu: Tensor, dx: Tensor, dim: i64) ---
	atg_glu_backward_jvp_out :: proc(out: ^Tensor, grad_x: Tensor, grad_glu: Tensor, x: Tensor, dgrad_glu: Tensor, dx: Tensor, dim: i64) ---
	atg_glu_jvp :: proc(out: ^Tensor, glu: Tensor, x: Tensor, dx: Tensor, dim: i64) ---
	atg_glu_jvp_out :: proc(out: ^Tensor, glu: Tensor, x: Tensor, dx: Tensor, dim: i64) ---
	atg_glu_out :: proc(out: ^Tensor, self: Tensor, dim: i64) ---
	atg_grad :: proc(out: ^Tensor, self: Tensor) ---
	atg_greater :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_greater_ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_greater_equal :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_greater_equal_ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_greater_equal_scalar_out :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_greater_equal_tensor :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_greater_equal_tensor_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_greater_equal_tensor_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_greater_scalar_out :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_greater_tensor :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_greater_tensor_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_greater_tensor_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_grid_sampler :: proc(out: ^Tensor, input: Tensor, grid: Tensor, interpolation_mode: i64, padding_mode: i64, align_corners: c.int) ---
	atg_grid_sampler_2d :: proc(out: ^Tensor, input: Tensor, grid: Tensor, interpolation_mode: i64, padding_mode: i64, align_corners: c.int) ---
	atg_grid_sampler_2d_out :: proc(out: ^Tensor, input: Tensor, grid: Tensor, interpolation_mode: i64, padding_mode: i64, align_corners: c.int) ---
	atg_grid_sampler_3d :: proc(out: ^Tensor, input: Tensor, grid: Tensor, interpolation_mode: i64, padding_mode: i64, align_corners: c.int) ---
	atg_grid_sampler_3d_out :: proc(out: ^Tensor, input: Tensor, grid: Tensor, interpolation_mode: i64, padding_mode: i64, align_corners: c.int) ---
	atg_group_norm :: proc(out: ^Tensor, input: Tensor, num_groups: i64, weight: Tensor, bias: Tensor, eps: f64, cudnn_enabled: c.int) ---
	atg_gru :: proc(out: ^Tensor, input: Tensor, hx: Tensor, params_data: ^Tensor, params_len: c.int, has_biases: c.int, num_layers: i64, dropout: f64, train: c.int, bidirectional: c.int, batch_first: c.int) ---
	atg_gru_cell :: proc(out: ^Tensor, input: Tensor, hx: Tensor, w_ih: Tensor, w_hh: Tensor, b_ih: Tensor, b_hh: Tensor) ---
	atg_gru_data :: proc(out: ^Tensor, data: Tensor, batch_sizes: Tensor, hx: Tensor, params_data: ^Tensor, params_len: c.int, has_biases: c.int, num_layers: i64, dropout: f64, train: c.int, bidirectional: c.int) ---
	atg_gt :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_gt_ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_gt_scalar_out :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_gt_tensor :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_gt_tensor_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_gt_tensor_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_hamming_window :: proc(out: ^Tensor, window_length: i64, options_kind: c.int, options_device: c.int) ---
	atg_hamming_window_out :: proc(out: ^Tensor, window_length: i64) ---
	atg_hamming_window_periodic :: proc(out: ^Tensor, window_length: i64, periodic: c.int, options_kind: c.int, options_device: c.int) ---
	atg_hamming_window_periodic_alpha :: proc(out: ^Tensor, window_length: i64, periodic: c.int, alpha: f64, options_kind: c.int, options_device: c.int) ---
	atg_hamming_window_periodic_alpha_beta :: proc(out: ^Tensor, window_length: i64, periodic: c.int, alpha: f64, beta: f64, options_kind: c.int, options_device: c.int) ---
	atg_hamming_window_periodic_alpha_beta_out :: proc(out: ^Tensor, window_length: i64, periodic: c.int, alpha: f64, beta: f64) ---
	atg_hamming_window_periodic_alpha_out :: proc(out: ^Tensor, window_length: i64, periodic: c.int, alpha: f64) ---
	atg_hamming_window_periodic_out :: proc(out: ^Tensor, window_length: i64, periodic: c.int) ---
	atg_hann_window :: proc(out: ^Tensor, window_length: i64, options_kind: c.int, options_device: c.int) ---
	atg_hann_window_out :: proc(out: ^Tensor, window_length: i64) ---
	atg_hann_window_periodic :: proc(out: ^Tensor, window_length: i64, periodic: c.int, options_kind: c.int, options_device: c.int) ---
	atg_hann_window_periodic_out :: proc(out: ^Tensor, window_length: i64, periodic: c.int) ---
	atg_hardshrink :: proc(out: ^Tensor, self: Tensor) ---
	atg_hardshrink_backward :: proc(out: ^Tensor, grad_out: Tensor, self: Tensor, lambd: Scalar) ---
	atg_hardshrink_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_out: Tensor, self: Tensor, lambd: Scalar) ---
	atg_hardshrink_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_hardsigmoid :: proc(out: ^Tensor, self: Tensor) ---
	atg_hardsigmoid_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_hardsigmoid_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor) ---
	atg_hardsigmoid_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, self: Tensor) ---
	atg_hardsigmoid_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_hardswish :: proc(out: ^Tensor, self: Tensor) ---
	atg_hardswish_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_hardswish_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor) ---
	atg_hardswish_backward_out :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor) ---
	atg_hardswish_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_hardtanh :: proc(out: ^Tensor, self: Tensor) ---
	atg_hardtanh_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_hardtanh_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, min_val: Scalar, max_val: Scalar) ---
	atg_hardtanh_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, self: Tensor, min_val: Scalar, max_val: Scalar) ---
	atg_hardtanh_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_hash_tensor :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, keepdim: c.int, mode: i64) ---
	atg_hash_tensor_out :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, keepdim: c.int, mode: i64) ---
	atg_heaviside :: proc(out: ^Tensor, self: Tensor, values: Tensor) ---
	atg_heaviside_ :: proc(out: ^Tensor, self: Tensor, values: Tensor) ---
	atg_heaviside_out :: proc(out: ^Tensor, self: Tensor, values: Tensor) ---
	atg_hinge_embedding_loss :: proc(out: ^Tensor, self: Tensor, target: Tensor, margin: f64, reduction: i64) ---
	atg_histc :: proc(out: ^Tensor, self: Tensor, bins: i64) ---
	atg_histc_out :: proc(out: ^Tensor, self: Tensor, bins: i64) ---
	atg_histogram :: proc(out: ^Tensor, self: Tensor, bins: Tensor, weight: Tensor, density: c.int) ---
	atg_histogram_bin_ct :: proc(out: ^Tensor, self: Tensor, bins: i64, range_data: [^]f64, range_len: c.int, weight: Tensor, density: c.int) ---
	atg_histogram_bin_ct_out :: proc(out: ^Tensor, hist: Tensor, bin_edges: Tensor, self: Tensor, bins: i64, range_data: [^]f64, range_len: c.int, weight: Tensor, density: c.int) ---
	atg_histogram_bins_tensor_out :: proc(out: ^Tensor, hist: Tensor, bin_edges: Tensor, self: Tensor, bins: Tensor, weight: Tensor, density: c.int) ---
	atg_hspmm :: proc(out: ^Tensor, mat1: Tensor, mat2: Tensor) ---
	atg_hspmm_out :: proc(out: ^Tensor, mat1: Tensor, mat2: Tensor) ---
	atg_hstack :: proc(out: ^Tensor, tensors_data: ^Tensor, tensors_len: c.int) ---
	atg_hstack_out :: proc(out: ^Tensor, tensors_data: ^Tensor, tensors_len: c.int) ---
	atg_huber_loss :: proc(out: ^Tensor, self: Tensor, target: Tensor, reduction: i64, delta: f64) ---
	atg_huber_loss_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, target: Tensor, reduction: i64, delta: f64) ---
	atg_huber_loss_backward_out :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, self: Tensor, target: Tensor, reduction: i64, delta: f64) ---
	atg_huber_loss_out :: proc(out: ^Tensor, self: Tensor, target: Tensor, reduction: i64, delta: f64) ---
	atg_hypot :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_hypot_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_hypot_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_i0 :: proc(out: ^Tensor, self: Tensor) ---
	atg_i0_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_i0_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_igamma :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_igamma_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_igamma_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_igammac :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_igammac_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_igammac_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_im2col :: proc(out: ^Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, dilation_data: [^]i64, dilation_len: c.int, padding_data: [^]i64, padding_len: c.int, stride_data: [^]i64, stride_len: c.int) ---
	atg_im2col_out :: proc(out: ^Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, dilation_data: [^]i64, dilation_len: c.int, padding_data: [^]i64, padding_len: c.int, stride_data: [^]i64, stride_len: c.int) ---
	atg_imag :: proc(out: ^Tensor, self: Tensor) ---
	atg_index :: proc(out: ^Tensor, self: Tensor, indices_data: ^Tensor, indices_len: c.int) ---
	atg_index_add :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor, source: Tensor) ---
	atg_index_add_ :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor, source: Tensor) ---
	atg_index_add_out :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor, source: Tensor) ---
	atg_index_copy :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor, source: Tensor) ---
	atg_index_copy_ :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor, source: Tensor) ---
	atg_index_copy_out :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor, source: Tensor) ---
	atg_index_fill :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor, value: Scalar) ---
	atg_index_fill_ :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor, value: Scalar) ---
	atg_index_fill_int_scalar_out :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor, value: Scalar) ---
	atg_index_fill_int_tensor :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor, value: Tensor) ---
	atg_index_fill_int_tensor_ :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor, value: Tensor) ---
	atg_index_fill_int_tensor_out :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor, value: Tensor) ---
	atg_index_put :: proc(out: ^Tensor, self: Tensor, indices_data: ^Tensor, indices_len: c.int, values: Tensor, accumulate: c.int) ---
	atg_index_put_ :: proc(out: ^Tensor, self: Tensor, indices_data: ^Tensor, indices_len: c.int, values: Tensor, accumulate: c.int) ---
	atg_index_put_out :: proc(out: ^Tensor, self: Tensor, indices_data: ^Tensor, indices_len: c.int, values: Tensor, accumulate: c.int) ---
	atg_index_reduce :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor, source: Tensor, reduce_ptr: cstring, reduce_len: c.int, include_self: c.int) ---
	atg_index_reduce_ :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor, source: Tensor, reduce_ptr: cstring, reduce_len: c.int, include_self: c.int) ---
	atg_index_reduce_out :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor, source: Tensor, reduce_ptr: cstring, reduce_len: c.int, include_self: c.int) ---
	atg_index_select :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor) ---
	atg_index_select_backward :: proc(out: ^Tensor, grad: Tensor, self_sizes_data: [^]i64, self_sizes_len: c.int, dim: i64, index: Tensor) ---
	atg_index_select_out :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor) ---
	atg_index_tensor_out :: proc(out: ^Tensor, self: Tensor, indices_data: ^Tensor, indices_len: c.int) ---
	atg_indices :: proc(out: ^Tensor, self: Tensor) ---
	atg_indices_copy :: proc(out: ^Tensor, self: Tensor) ---
	atg_indices_copy_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_infinitely_differentiable_gelu_backward :: proc(out: ^Tensor, grad: Tensor, self: Tensor) ---
	atg_inner :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_inner_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_instance_norm :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor, running_mean: Tensor, running_var: Tensor, use_input_stats: c.int, momentum: f64, eps: f64, cudnn_enabled: c.int) ---
	atg_int_repr :: proc(out: ^Tensor, self: Tensor) ---
	atg_int_repr_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_inverse :: proc(out: ^Tensor, self: Tensor) ---
	atg_inverse_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_isclose :: proc(out: ^Tensor, self: Tensor, other: Tensor, rtol: f64, atol: f64, equal_nan: c.int) ---
	atg_isfinite :: proc(out: ^Tensor, self: Tensor) ---
	atg_isin :: proc(out: ^Tensor, elements: Tensor, test_elements: Tensor, assume_unique: c.int, invert: c.int) ---
	atg_isin_scalar_tensor :: proc(out: ^Tensor, element: Scalar, test_elements: Tensor, assume_unique: c.int, invert: c.int) ---
	atg_isin_scalar_tensor_out :: proc(out: ^Tensor, element: Scalar, test_elements: Tensor, assume_unique: c.int, invert: c.int) ---
	atg_isin_tensor_scalar :: proc(out: ^Tensor, elements: Tensor, test_element: Scalar, assume_unique: c.int, invert: c.int) ---
	atg_isin_tensor_scalar_out :: proc(out: ^Tensor, elements: Tensor, test_element: Scalar, assume_unique: c.int, invert: c.int) ---
	atg_isin_tensor_tensor_out :: proc(out: ^Tensor, elements: Tensor, test_elements: Tensor, assume_unique: c.int, invert: c.int) ---
	atg_isinf :: proc(out: ^Tensor, self: Tensor) ---
	atg_isinf_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_isnan :: proc(out: ^Tensor, self: Tensor) ---
	atg_isnan_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_isneginf :: proc(out: ^Tensor, self: Tensor) ---
	atg_isneginf_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_isposinf :: proc(out: ^Tensor, self: Tensor) ---
	atg_isposinf_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_isreal :: proc(out: ^Tensor, self: Tensor) ---
	atg_istft :: proc(out: ^Tensor, self: Tensor, n_fft: i64, hop_length_v: i64, hop_length_null: rawptr, win_length_v: i64, win_length_null: rawptr, window: Tensor, center: c.int, normalized: c.int, onesided: c.int, length_v: i64, length_null: rawptr, return_complex: c.int) ---
	atg_kaiser_window :: proc(out: ^Tensor, window_length: i64, options_kind: c.int, options_device: c.int) ---
	atg_kaiser_window_beta :: proc(out: ^Tensor, window_length: i64, periodic: c.int, beta: f64, options_kind: c.int, options_device: c.int) ---
	atg_kaiser_window_beta_out :: proc(out: ^Tensor, window_length: i64, periodic: c.int, beta: f64) ---
	atg_kaiser_window_out :: proc(out: ^Tensor, window_length: i64) ---
	atg_kaiser_window_periodic :: proc(out: ^Tensor, window_length: i64, periodic: c.int, options_kind: c.int, options_device: c.int) ---
	atg_kaiser_window_periodic_out :: proc(out: ^Tensor, window_length: i64, periodic: c.int) ---
	atg_kl_div :: proc(out: ^Tensor, self: Tensor, target: Tensor, reduction: i64, log_target: c.int) ---
	atg_kron :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_kron_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_kthvalue :: proc(out: ^Tensor, self: Tensor, k: i64, dim: i64, keepdim: c.int) ---
	atg_kthvalue_values :: proc(out: ^Tensor, values: Tensor, indices: Tensor, self: Tensor, k: i64, dim: i64, keepdim: c.int) ---
	atg_l1_loss :: proc(out: ^Tensor, self: Tensor, target: Tensor, reduction: i64) ---
	atg_layer_norm :: proc(out: ^Tensor, input: Tensor, normalized_shape_data: [^]i64, normalized_shape_len: c.int, weight: Tensor, bias: Tensor, eps: f64, cudnn_enable: c.int) ---
	atg_lcm :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_lcm_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_lcm_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_ldexp :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_ldexp_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_ldexp_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_le :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_le_ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_le_scalar_out :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_le_tensor :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_le_tensor_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_le_tensor_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_leaky_relu :: proc(out: ^Tensor, self: Tensor) ---
	atg_leaky_relu_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_leaky_relu_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, negative_slope: Scalar, self_is_result: c.int) ---
	atg_leaky_relu_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, self: Tensor, negative_slope: Scalar, self_is_result: c.int) ---
	atg_leaky_relu_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_lerp :: proc(out: ^Tensor, self: Tensor, end: Tensor, weight: Scalar) ---
	atg_lerp_ :: proc(out: ^Tensor, self: Tensor, end: Tensor, weight: Scalar) ---
	atg_lerp_scalar_out :: proc(out: ^Tensor, self: Tensor, end: Tensor, weight: Scalar) ---
	atg_lerp_tensor :: proc(out: ^Tensor, self: Tensor, end: Tensor, weight: Tensor) ---
	atg_lerp_tensor_ :: proc(out: ^Tensor, self: Tensor, end: Tensor, weight: Tensor) ---
	atg_lerp_tensor_out :: proc(out: ^Tensor, self: Tensor, end: Tensor, weight: Tensor) ---
	atg_less :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_less_ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_less_equal :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_less_equal_ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_less_equal_scalar_out :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_less_equal_tensor :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_less_equal_tensor_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_less_equal_tensor_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_less_scalar_out :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_less_tensor :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_less_tensor_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_less_tensor_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_lgamma :: proc(out: ^Tensor, self: Tensor) ---
	atg_lgamma_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_lgamma_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_lift :: proc(out: ^Tensor, self: Tensor) ---
	atg_lift_fresh :: proc(out: ^Tensor, self: Tensor) ---
	atg_lift_fresh_copy :: proc(out: ^Tensor, self: Tensor) ---
	atg_lift_fresh_copy_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_lift_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_linalg_cholesky :: proc(out: ^Tensor, self: Tensor, upper: c.int) ---
	atg_linalg_cholesky_ex :: proc(out: ^Tensor, self: Tensor, upper: c.int, check_errors: c.int) ---
	atg_linalg_cholesky_ex_l :: proc(out: ^Tensor, L: Tensor, info: Tensor, self: Tensor, upper: c.int, check_errors: c.int) ---
	atg_linalg_cholesky_out :: proc(out: ^Tensor, self: Tensor, upper: c.int) ---
	atg_linalg_cond :: proc(out: ^Tensor, self: Tensor, p: Scalar) ---
	atg_linalg_cond_out :: proc(out: ^Tensor, self: Tensor, p: Scalar) ---
	atg_linalg_cond_p_str :: proc(out: ^Tensor, self: Tensor, p_ptr: cstring, p_len: c.int) ---
	atg_linalg_cond_p_str_out :: proc(out: ^Tensor, self: Tensor, p_ptr: cstring, p_len: c.int) ---
	atg_linalg_cross :: proc(out: ^Tensor, self: Tensor, other: Tensor, dim: i64) ---
	atg_linalg_cross_out :: proc(out: ^Tensor, self: Tensor, other: Tensor, dim: i64) ---
	atg_linalg_det :: proc(out: ^Tensor, A: Tensor) ---
	atg_linalg_det_out :: proc(out: ^Tensor, A: Tensor) ---
	atg_linalg_diagonal :: proc(out: ^Tensor, A: Tensor, offset: i64, dim1: i64, dim2: i64) ---
	atg_linalg_eig :: proc(out: ^Tensor, self: Tensor) ---
	atg_linalg_eig_out :: proc(out: ^Tensor, eigenvalues: Tensor, eigenvectors: Tensor, self: Tensor) ---
	atg_linalg_eigh :: proc(out: ^Tensor, self: Tensor, UPLO_ptr: cstring, UPLO_len: c.int) ---
	atg_linalg_eigh_eigvals :: proc(out: ^Tensor, eigvals: Tensor, eigvecs: Tensor, self: Tensor, UPLO_ptr: cstring, UPLO_len: c.int) ---
	atg_linalg_eigvals :: proc(out: ^Tensor, self: Tensor) ---
	atg_linalg_eigvals_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_linalg_eigvalsh :: proc(out: ^Tensor, self: Tensor, UPLO_ptr: cstring, UPLO_len: c.int) ---
	atg_linalg_eigvalsh_out :: proc(out: ^Tensor, self: Tensor, UPLO_ptr: cstring, UPLO_len: c.int) ---
	atg_linalg_householder_product :: proc(out: ^Tensor, input: Tensor, tau: Tensor) ---
	atg_linalg_householder_product_out :: proc(out: ^Tensor, input: Tensor, tau: Tensor) ---
	atg_linalg_inv :: proc(out: ^Tensor, A: Tensor) ---
	atg_linalg_inv_ex :: proc(out: ^Tensor, A: Tensor, check_errors: c.int) ---
	atg_linalg_inv_ex_inverse :: proc(out: ^Tensor, inverse: Tensor, info: Tensor, A: Tensor, check_errors: c.int) ---
	atg_linalg_inv_out :: proc(out: ^Tensor, A: Tensor) ---
	atg_linalg_ldl_factor :: proc(out: ^Tensor, self: Tensor, hermitian: c.int) ---
	atg_linalg_ldl_factor_ex :: proc(out: ^Tensor, self: Tensor, hermitian: c.int, check_errors: c.int) ---
	atg_linalg_ldl_factor_ex_out :: proc(out: ^Tensor, LD: Tensor, pivots: Tensor, info: Tensor, self: Tensor, hermitian: c.int, check_errors: c.int) ---
	atg_linalg_ldl_factor_out :: proc(out: ^Tensor, LD: Tensor, pivots: Tensor, self: Tensor, hermitian: c.int) ---
	atg_linalg_ldl_solve :: proc(out: ^Tensor, LD: Tensor, pivots: Tensor, B: Tensor, hermitian: c.int) ---
	atg_linalg_ldl_solve_out :: proc(out: ^Tensor, LD: Tensor, pivots: Tensor, B: Tensor, hermitian: c.int) ---
	atg_linalg_lstsq :: proc(out: ^Tensor, self: Tensor, b: Tensor, rcond_v: f64, rcond_null: rawptr, driver_ptr: cstring, driver_len: c.int) ---
	atg_linalg_lstsq_out :: proc(out: ^Tensor, solution: Tensor, residuals: Tensor, rank: Tensor, singular_values: Tensor, self: Tensor, b: Tensor, rcond_v: f64, rcond_null: rawptr, driver_ptr: cstring, driver_len: c.int) ---
	atg_linalg_lu :: proc(out: ^Tensor, A: Tensor, pivot: c.int) ---
	atg_linalg_lu_factor :: proc(out: ^Tensor, A: Tensor, pivot: c.int) ---
	atg_linalg_lu_factor_ex :: proc(out: ^Tensor, A: Tensor, pivot: c.int, check_errors: c.int) ---
	atg_linalg_lu_factor_ex_out :: proc(out: ^Tensor, LU: Tensor, pivots: Tensor, info: Tensor, A: Tensor, pivot: c.int, check_errors: c.int) ---
	atg_linalg_lu_factor_out :: proc(out: ^Tensor, LU: Tensor, pivots: Tensor, A: Tensor, pivot: c.int) ---
	atg_linalg_lu_out :: proc(out: ^Tensor, P: Tensor, L: Tensor, U: Tensor, A: Tensor, pivot: c.int) ---
	atg_linalg_lu_solve :: proc(out: ^Tensor, LU: Tensor, pivots: Tensor, B: Tensor, left: c.int, adjoint: c.int) ---
	atg_linalg_lu_solve_out :: proc(out: ^Tensor, LU: Tensor, pivots: Tensor, B: Tensor, left: c.int, adjoint: c.int) ---
	atg_linalg_matmul :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_linalg_matmul_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_linalg_matrix_exp :: proc(out: ^Tensor, self: Tensor) ---
	atg_linalg_matrix_exp_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_linalg_matrix_power :: proc(out: ^Tensor, self: Tensor, n: i64) ---
	atg_linalg_matrix_power_out :: proc(out: ^Tensor, self: Tensor, n: i64) ---
	atg_linalg_matrix_rank :: proc(out: ^Tensor, self: Tensor, tol: f64, hermitian: c.int) ---
	atg_linalg_matrix_rank_atol_rtol_float :: proc(out: ^Tensor, self: Tensor, atol_v: f64, atol_null: rawptr, rtol_v: f64, rtol_null: rawptr, hermitian: c.int) ---
	atg_linalg_matrix_rank_atol_rtol_float_out :: proc(out: ^Tensor, self: Tensor, atol_v: f64, atol_null: rawptr, rtol_v: f64, rtol_null: rawptr, hermitian: c.int) ---
	atg_linalg_matrix_rank_atol_rtol_tensor :: proc(out: ^Tensor, input: Tensor, atol: Tensor, rtol: Tensor, hermitian: c.int) ---
	atg_linalg_matrix_rank_atol_rtol_tensor_out :: proc(out: ^Tensor, input: Tensor, atol: Tensor, rtol: Tensor, hermitian: c.int) ---
	atg_linalg_matrix_rank_out :: proc(out: ^Tensor, self: Tensor, tol: f64, hermitian: c.int) ---
	atg_linalg_matrix_rank_out_tol_tensor :: proc(out: ^Tensor, input: Tensor, tol: Tensor, hermitian: c.int) ---
	atg_linalg_matrix_rank_tol_tensor :: proc(out: ^Tensor, input: Tensor, tol: Tensor, hermitian: c.int) ---
	atg_linalg_multi_dot :: proc(out: ^Tensor, tensors_data: ^Tensor, tensors_len: c.int) ---
	atg_linalg_multi_dot_out :: proc(out: ^Tensor, tensors_data: ^Tensor, tensors_len: c.int) ---
	atg_linalg_norm :: proc(out: ^Tensor, self: Tensor, ord: Scalar, dim_data: [^]i64, dim_len: c.int, keepdim: c.int, dtype: c.int) ---
	atg_linalg_norm_ord_str :: proc(out: ^Tensor, self: Tensor, ord_ptr: cstring, ord_len: c.int, dim_data: [^]i64, dim_len: c.int, keepdim: c.int, dtype: c.int) ---
	atg_linalg_norm_ord_str_out :: proc(out: ^Tensor, self: Tensor, ord_ptr: cstring, ord_len: c.int, dim_data: [^]i64, dim_len: c.int, keepdim: c.int, dtype: c.int) ---
	atg_linalg_norm_out :: proc(out: ^Tensor, self: Tensor, ord: Scalar, dim_data: [^]i64, dim_len: c.int, keepdim: c.int, dtype: c.int) ---
	atg_linalg_pinv :: proc(out: ^Tensor, self: Tensor, rcond: f64, hermitian: c.int) ---
	atg_linalg_pinv_atol_rtol_float :: proc(out: ^Tensor, self: Tensor, atol_v: f64, atol_null: rawptr, rtol_v: f64, rtol_null: rawptr, hermitian: c.int) ---
	atg_linalg_pinv_atol_rtol_float_out :: proc(out: ^Tensor, self: Tensor, atol_v: f64, atol_null: rawptr, rtol_v: f64, rtol_null: rawptr, hermitian: c.int) ---
	atg_linalg_pinv_atol_rtol_tensor :: proc(out: ^Tensor, self: Tensor, atol: Tensor, rtol: Tensor, hermitian: c.int) ---
	atg_linalg_pinv_atol_rtol_tensor_out :: proc(out: ^Tensor, self: Tensor, atol: Tensor, rtol: Tensor, hermitian: c.int) ---
	atg_linalg_pinv_out :: proc(out: ^Tensor, self: Tensor, rcond: f64, hermitian: c.int) ---
	atg_linalg_pinv_out_rcond_tensor :: proc(out: ^Tensor, self: Tensor, rcond: Tensor, hermitian: c.int) ---
	atg_linalg_pinv_rcond_tensor :: proc(out: ^Tensor, self: Tensor, rcond: Tensor, hermitian: c.int) ---
	atg_linalg_qr :: proc(out: ^Tensor, A: Tensor, mode_ptr: cstring, mode_len: c.int) ---
	atg_linalg_qr_out :: proc(out: ^Tensor, Q: Tensor, R: Tensor, A: Tensor, mode_ptr: cstring, mode_len: c.int) ---
	atg_linalg_slogdet :: proc(out: ^Tensor, A: Tensor) ---
	atg_linalg_slogdet_out :: proc(out: ^Tensor, sign: Tensor, logabsdet: Tensor, A: Tensor) ---
	atg_linalg_solve :: proc(out: ^Tensor, A: Tensor, B: Tensor, left: c.int) ---
	atg_linalg_solve_ex :: proc(out: ^Tensor, A: Tensor, B: Tensor, left: c.int, check_errors: c.int) ---
	atg_linalg_solve_ex_out :: proc(out: ^Tensor, result: Tensor, info: Tensor, A: Tensor, B: Tensor, left: c.int, check_errors: c.int) ---
	atg_linalg_solve_out :: proc(out: ^Tensor, A: Tensor, B: Tensor, left: c.int) ---
	atg_linalg_solve_triangular :: proc(out: ^Tensor, self: Tensor, B: Tensor, upper: c.int, left: c.int, unitriangular: c.int) ---
	atg_linalg_solve_triangular_out :: proc(out: ^Tensor, self: Tensor, B: Tensor, upper: c.int, left: c.int, unitriangular: c.int) ---
	atg_linalg_svd :: proc(out: ^Tensor, A: Tensor, full_matrices: c.int, driver_ptr: cstring, driver_len: c.int) ---
	atg_linalg_svd_u :: proc(out: ^Tensor, U: Tensor, S: Tensor, Vh: Tensor, A: Tensor, full_matrices: c.int, driver_ptr: cstring, driver_len: c.int) ---
	atg_linalg_svdvals :: proc(out: ^Tensor, A: Tensor, driver_ptr: cstring, driver_len: c.int) ---
	atg_linalg_svdvals_out :: proc(out: ^Tensor, A: Tensor, driver_ptr: cstring, driver_len: c.int) ---
	atg_linalg_tensorinv :: proc(out: ^Tensor, self: Tensor, ind: i64) ---
	atg_linalg_tensorinv_out :: proc(out: ^Tensor, self: Tensor, ind: i64) ---
	atg_linalg_tensorsolve :: proc(out: ^Tensor, self: Tensor, other: Tensor, dims_data: [^]i64, dims_len: c.int) ---
	atg_linalg_tensorsolve_out :: proc(out: ^Tensor, self: Tensor, other: Tensor, dims_data: [^]i64, dims_len: c.int) ---
	atg_linalg_vander :: proc(out: ^Tensor, x: Tensor, n_v: i64, n_null: rawptr) ---
	atg_linalg_vecdot :: proc(out: ^Tensor, x: Tensor, y: Tensor, dim: i64) ---
	atg_linalg_vecdot_out :: proc(out: ^Tensor, x: Tensor, y: Tensor, dim: i64) ---
	atg_linear :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor) ---
	atg_linear_out :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor) ---
	atg_linspace :: proc(out: ^Tensor, start: Scalar, end: Scalar, steps: i64, options_kind: c.int, options_device: c.int) ---
	atg_linspace_out :: proc(out: ^Tensor, start: Scalar, end: Scalar, steps: i64) ---
	atg_linspace_scalar_tensor :: proc(out: ^Tensor, start: Scalar, end: Tensor, steps: i64, options_kind: c.int, options_device: c.int) ---
	atg_linspace_scalar_tensor_out :: proc(out: ^Tensor, start: Scalar, end: Tensor, steps: i64) ---
	atg_linspace_tensor_scalar :: proc(out: ^Tensor, start: Tensor, end: Scalar, steps: i64, options_kind: c.int, options_device: c.int) ---
	atg_linspace_tensor_scalar_out :: proc(out: ^Tensor, start: Tensor, end: Scalar, steps: i64) ---
	atg_linspace_tensor_tensor :: proc(out: ^Tensor, start: Tensor, end: Tensor, steps: i64, options_kind: c.int, options_device: c.int) ---
	atg_linspace_tensor_tensor_out :: proc(out: ^Tensor, start: Tensor, end: Tensor, steps: i64) ---
	atg_log :: proc(out: ^Tensor, self: Tensor) ---
	atg_log10 :: proc(out: ^Tensor, self: Tensor) ---
	atg_log10_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_log10_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_log1p :: proc(out: ^Tensor, self: Tensor) ---
	atg_log1p_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_log1p_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_log2 :: proc(out: ^Tensor, self: Tensor) ---
	atg_log2_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_log2_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_log_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_log_normal :: proc(out: ^Tensor, self: Tensor, mean: f64, std: f64) ---
	atg_log_normal_ :: proc(out: ^Tensor, self: Tensor, mean: f64, std: f64) ---
	atg_log_normal_out :: proc(out: ^Tensor, self: Tensor, mean: f64, std: f64) ---
	atg_log_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_log_sigmoid :: proc(out: ^Tensor, self: Tensor) ---
	atg_log_sigmoid_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, buffer: Tensor) ---
	atg_log_sigmoid_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, self: Tensor, buffer: Tensor) ---
	atg_log_sigmoid_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_log_softmax :: proc(out: ^Tensor, self: Tensor, dim: i64, dtype: c.int) ---
	atg_log_softmax_int_out :: proc(out: ^Tensor, self: Tensor, dim: i64, dtype: c.int) ---
	atg_logaddexp :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_logaddexp2 :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_logaddexp2_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_logaddexp_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_logcumsumexp :: proc(out: ^Tensor, self: Tensor, dim: i64) ---
	atg_logcumsumexp_out :: proc(out: ^Tensor, self: Tensor, dim: i64) ---
	atg_logdet :: proc(out: ^Tensor, self: Tensor) ---
	atg_logical_and :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_logical_and_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_logical_and_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_logical_not :: proc(out: ^Tensor, self: Tensor) ---
	atg_logical_not_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_logical_not_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_logical_or :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_logical_or_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_logical_or_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_logical_xor :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_logical_xor_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_logical_xor_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_logit :: proc(out: ^Tensor, self: Tensor, eps_v: f64, eps_null: rawptr) ---
	atg_logit_ :: proc(out: ^Tensor, self: Tensor, eps_v: f64, eps_null: rawptr) ---
	atg_logit_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, eps_v: f64, eps_null: rawptr) ---
	atg_logit_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, self: Tensor, eps_v: f64, eps_null: rawptr) ---
	atg_logit_out :: proc(out: ^Tensor, self: Tensor, eps_v: f64, eps_null: rawptr) ---
	atg_logspace :: proc(out: ^Tensor, start: Scalar, end: Scalar, steps: i64, base: f64, options_kind: c.int, options_device: c.int) ---
	atg_logspace_out :: proc(out: ^Tensor, start: Scalar, end: Scalar, steps: i64, base: f64) ---
	atg_logspace_scalar_tensor :: proc(out: ^Tensor, start: Scalar, end: Tensor, steps: i64, base: f64, options_kind: c.int, options_device: c.int) ---
	atg_logspace_scalar_tensor_out :: proc(out: ^Tensor, start: Scalar, end: Tensor, steps: i64, base: f64) ---
	atg_logspace_tensor_scalar :: proc(out: ^Tensor, start: Tensor, end: Scalar, steps: i64, base: f64, options_kind: c.int, options_device: c.int) ---
	atg_logspace_tensor_scalar_out :: proc(out: ^Tensor, start: Tensor, end: Scalar, steps: i64, base: f64) ---
	atg_logspace_tensor_tensor :: proc(out: ^Tensor, start: Tensor, end: Tensor, steps: i64, base: f64, options_kind: c.int, options_device: c.int) ---
	atg_logspace_tensor_tensor_out :: proc(out: ^Tensor, start: Tensor, end: Tensor, steps: i64, base: f64) ---
	atg_logsumexp :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, keepdim: c.int) ---
	atg_logsumexp_out :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, keepdim: c.int) ---
	atg_lstm :: proc(out: ^Tensor, input: Tensor, hx_data: ^Tensor, hx_len: c.int, params_data: ^Tensor, params_len: c.int, has_biases: c.int, num_layers: i64, dropout: f64, train: c.int, bidirectional: c.int, batch_first: c.int) ---
	atg_lstm_cell :: proc(out: ^Tensor, input: Tensor, hx_data: ^Tensor, hx_len: c.int, w_ih: Tensor, w_hh: Tensor, b_ih: Tensor, b_hh: Tensor) ---
	atg_lstm_data :: proc(out: ^Tensor, data: Tensor, batch_sizes: Tensor, hx_data: ^Tensor, hx_len: c.int, params_data: ^Tensor, params_len: c.int, has_biases: c.int, num_layers: i64, dropout: f64, train: c.int, bidirectional: c.int) ---
	atg_lstm_mps_backward :: proc(out0: Tensor, out1_data: ^Tensor, out1_len: c.int, out2_data: ^Tensor, out2_len: c.int, grad_y: Tensor, grad_hy: Tensor, grad_cy: Tensor, z_state: Tensor, cell_state_fwd: Tensor, input: Tensor, layersOutputs: Tensor, hx_data: ^Tensor, hx_len: c.int, params_data: ^Tensor, params_len: c.int, has_biases: c.int, num_layers: i64, dropout: f64, train: c.int, bidirectional: c.int, batch_first: c.int) ---
	atg_lt :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_lt_ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_lt_scalar_out :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_lt_tensor :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_lt_tensor_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_lt_tensor_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_lu_solve :: proc(out: ^Tensor, self: Tensor, LU_data: Tensor, LU_pivots: Tensor) ---
	atg_lu_solve_out :: proc(out: ^Tensor, self: Tensor, LU_data: Tensor, LU_pivots: Tensor) ---
	atg_lu_unpack :: proc(out: ^Tensor, LU_data: Tensor, LU_pivots: Tensor, unpack_data: c.int, unpack_pivots: c.int) ---
	atg_lu_unpack_out :: proc(out: ^Tensor, P: Tensor, L: Tensor, U: Tensor, LU_data: Tensor, LU_pivots: Tensor, unpack_data: c.int, unpack_pivots: c.int) ---
	atg_margin_ranking_loss :: proc(out: ^Tensor, input1: Tensor, input2: Tensor, target: Tensor, margin: f64, reduction: i64) ---
	atg_masked_fill :: proc(out: ^Tensor, self: Tensor, mask: Tensor, value: Scalar) ---
	atg_masked_fill_ :: proc(out: ^Tensor, self: Tensor, mask: Tensor, value: Scalar) ---
	atg_masked_fill_scalar_out :: proc(out: ^Tensor, self: Tensor, mask: Tensor, value: Scalar) ---
	atg_masked_fill_tensor :: proc(out: ^Tensor, self: Tensor, mask: Tensor, value: Tensor) ---
	atg_masked_fill_tensor_ :: proc(out: ^Tensor, self: Tensor, mask: Tensor, value: Tensor) ---
	atg_masked_fill_tensor_out :: proc(out: ^Tensor, self: Tensor, mask: Tensor, value: Tensor) ---
	atg_masked_scatter :: proc(out: ^Tensor, self: Tensor, mask: Tensor, source: Tensor) ---
	atg_masked_scatter_ :: proc(out: ^Tensor, self: Tensor, mask: Tensor, source: Tensor) ---
	atg_masked_scatter_backward :: proc(out: ^Tensor, grad_output: Tensor, mask: Tensor, sizes_data: [^]i64, sizes_len: c.int) ---
	atg_masked_scatter_out :: proc(out: ^Tensor, self: Tensor, mask: Tensor, source: Tensor) ---
	atg_masked_select :: proc(out: ^Tensor, self: Tensor, mask: Tensor) ---
	atg_masked_select_backward :: proc(out: ^Tensor, grad: Tensor, input: Tensor, mask: Tensor) ---
	atg_masked_select_out :: proc(out: ^Tensor, self: Tensor, mask: Tensor) ---
	atg_matmul :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_matmul_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_matrix_exp :: proc(out: ^Tensor, self: Tensor) ---
	atg_matrix_exp_backward :: proc(out: ^Tensor, self: Tensor, grad: Tensor) ---
	atg_matrix_h :: proc(out: ^Tensor, self: Tensor) ---
	atg_matrix_power :: proc(out: ^Tensor, self: Tensor, n: i64) ---
	atg_matrix_power_out :: proc(out: ^Tensor, self: Tensor, n: i64) ---
	atg_max :: proc(out: ^Tensor, self: Tensor) ---
	atg_max_dim :: proc(out: ^Tensor, self: Tensor, dim: i64, keepdim: c.int) ---
	atg_max_dim_max :: proc(out: ^Tensor, max: Tensor, max_values: Tensor, self: Tensor, dim: i64, keepdim: c.int) ---
	atg_max_other :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_max_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_max_pool1d :: proc(out: ^Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, ceil_mode: c.int) ---
	atg_max_pool1d_with_indices :: proc(out: ^Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, ceil_mode: c.int) ---
	atg_max_pool2d :: proc(out: ^Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, ceil_mode: c.int) ---
	atg_max_pool2d_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, ceil_mode: c.int) ---
	atg_max_pool2d_backward_out :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, ceil_mode: c.int) ---
	atg_max_pool2d_with_indices :: proc(out: ^Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, ceil_mode: c.int) ---
	atg_max_pool2d_with_indices_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, ceil_mode: c.int, indices: Tensor) ---
	atg_max_pool2d_with_indices_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, ceil_mode: c.int, indices: Tensor) ---
	atg_max_pool2d_with_indices_out :: proc(out: ^Tensor, indices: Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, ceil_mode: c.int) ---
	atg_max_pool3d :: proc(out: ^Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, ceil_mode: c.int) ---
	atg_max_pool3d_with_indices :: proc(out: ^Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, ceil_mode: c.int) ---
	atg_max_pool3d_with_indices_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, ceil_mode: c.int, indices: Tensor) ---
	atg_max_pool3d_with_indices_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, ceil_mode: c.int, indices: Tensor) ---
	atg_max_pool3d_with_indices_out :: proc(out: ^Tensor, indices: Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, ceil_mode: c.int) ---
	atg_max_unary_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_max_unpool2d :: proc(out: ^Tensor, self: Tensor, indices: Tensor, output_size_data: [^]i64, output_size_len: c.int) ---
	atg_max_unpool2d_out :: proc(out: ^Tensor, self: Tensor, indices: Tensor, output_size_data: [^]i64, output_size_len: c.int) ---
	atg_max_unpool3d :: proc(out: ^Tensor, self: Tensor, indices: Tensor, output_size_data: [^]i64, output_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int) ---
	atg_max_unpool3d_out :: proc(out: ^Tensor, self: Tensor, indices: Tensor, output_size_data: [^]i64, output_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int) ---
	atg_maximum :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_maximum_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_mean :: proc(out: ^Tensor, self: Tensor, dtype: c.int) ---
	atg_mean_dim :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, keepdim: c.int, dtype: c.int) ---
	atg_mean_dtype_out :: proc(out: ^Tensor, self: Tensor, dtype: c.int) ---
	atg_mean_out :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, keepdim: c.int, dtype: c.int) ---
	atg_median :: proc(out: ^Tensor, self: Tensor) ---
	atg_median_dim :: proc(out: ^Tensor, self: Tensor, dim: i64, keepdim: c.int) ---
	atg_median_dim_values :: proc(out: ^Tensor, values: Tensor, indices: Tensor, self: Tensor, dim: i64, keepdim: c.int) ---
	atg_median_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_mh :: proc(out: ^Tensor, self: Tensor) ---
	atg_min :: proc(out: ^Tensor, self: Tensor) ---
	atg_min_dim :: proc(out: ^Tensor, self: Tensor, dim: i64, keepdim: c.int) ---
	atg_min_dim_min :: proc(out: ^Tensor, min: Tensor, min_indices: Tensor, self: Tensor, dim: i64, keepdim: c.int) ---
	atg_min_other :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_min_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_min_unary_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_minimum :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_minimum_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_miopen_batch_norm :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor, running_mean: Tensor, running_var: Tensor, training: c.int, exponential_average_factor: f64, epsilon: f64) ---
	atg_miopen_batch_norm_backward :: proc(out: ^Tensor, input: Tensor, grad_output: Tensor, weight: Tensor, running_mean: Tensor, running_var: Tensor, save_mean: Tensor, save_var: Tensor, epsilon: f64) ---
	atg_miopen_batch_norm_backward_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, out2: Tensor, input: Tensor, grad_output: Tensor, weight: Tensor, running_mean: Tensor, running_var: Tensor, save_mean: Tensor, save_var: Tensor, epsilon: f64) ---
	atg_miopen_batch_norm_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, out2: Tensor, input: Tensor, weight: Tensor, bias: Tensor, running_mean: Tensor, running_var: Tensor, training: c.int, exponential_average_factor: f64, epsilon: f64) ---
	atg_miopen_convolution :: proc(out: ^Tensor, self: Tensor, weight: Tensor, bias: Tensor, padding_data: [^]i64, padding_len: c.int, stride_data: [^]i64, stride_len: c.int, dilation_data: [^]i64, dilation_len: c.int, groups: i64, benchmark: c.int, deterministic: c.int) ---
	atg_miopen_convolution_add_relu :: proc(out: ^Tensor, self: Tensor, weight: Tensor, z: Tensor, alpha: Scalar, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, groups: i64) ---
	atg_miopen_convolution_out :: proc(out: ^Tensor, self: Tensor, weight: Tensor, bias: Tensor, padding_data: [^]i64, padding_len: c.int, stride_data: [^]i64, stride_len: c.int, dilation_data: [^]i64, dilation_len: c.int, groups: i64, benchmark: c.int, deterministic: c.int) ---
	atg_miopen_convolution_relu :: proc(out: ^Tensor, self: Tensor, weight: Tensor, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, groups: i64) ---
	atg_miopen_convolution_transpose :: proc(out: ^Tensor, self: Tensor, weight: Tensor, bias: Tensor, padding_data: [^]i64, padding_len: c.int, output_padding_data: [^]i64, output_padding_len: c.int, stride_data: [^]i64, stride_len: c.int, dilation_data: [^]i64, dilation_len: c.int, groups: i64, benchmark: c.int, deterministic: c.int) ---
	atg_miopen_convolution_transpose_out :: proc(out: ^Tensor, self: Tensor, weight: Tensor, bias: Tensor, padding_data: [^]i64, padding_len: c.int, output_padding_data: [^]i64, output_padding_len: c.int, stride_data: [^]i64, stride_len: c.int, dilation_data: [^]i64, dilation_len: c.int, groups: i64, benchmark: c.int, deterministic: c.int) ---
	atg_miopen_depthwise_convolution :: proc(out: ^Tensor, self: Tensor, weight: Tensor, bias: Tensor, padding_data: [^]i64, padding_len: c.int, stride_data: [^]i64, stride_len: c.int, dilation_data: [^]i64, dilation_len: c.int, groups: i64, benchmark: c.int, deterministic: c.int) ---
	atg_miopen_depthwise_convolution_out :: proc(out: ^Tensor, self: Tensor, weight: Tensor, bias: Tensor, padding_data: [^]i64, padding_len: c.int, stride_data: [^]i64, stride_len: c.int, dilation_data: [^]i64, dilation_len: c.int, groups: i64, benchmark: c.int, deterministic: c.int) ---
	atg_miopen_rnn :: proc(out: ^Tensor, input: Tensor, weight_data: ^Tensor, weight_len: c.int, weight_stride0: i64, hx: Tensor, cx: Tensor, mode: i64, hidden_size: i64, num_layers: i64, batch_first: c.int, dropout: f64, train: c.int, bidirectional: c.int, batch_sizes_data: [^]i64, batch_sizes_len: c.int, dropout_state: Tensor) ---
	atg_miopen_rnn_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, out2: Tensor, out3: Tensor, out4: Tensor, input: Tensor, weight_data: ^Tensor, weight_len: c.int, weight_stride0: i64, hx: Tensor, cx: Tensor, mode: i64, hidden_size: i64, num_layers: i64, batch_first: c.int, dropout: f64, train: c.int, bidirectional: c.int, batch_sizes_data: [^]i64, batch_sizes_len: c.int, dropout_state: Tensor) ---
	atg_mish :: proc(out: ^Tensor, self: Tensor) ---
	atg_mish_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_mish_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor) ---
	atg_mish_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_mkldnn_adaptive_avg_pool2d :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int) ---
	atg_mkldnn_adaptive_avg_pool2d_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor) ---
	atg_mkldnn_adaptive_avg_pool2d_backward_out :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor) ---
	atg_mkldnn_adaptive_avg_pool2d_out :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int) ---
	atg_mkldnn_convolution :: proc(out: ^Tensor, self: Tensor, weight: Tensor, bias: Tensor, padding_data: [^]i64, padding_len: c.int, stride_data: [^]i64, stride_len: c.int, dilation_data: [^]i64, dilation_len: c.int, groups: i64) ---
	atg_mkldnn_convolution_out :: proc(out: ^Tensor, self: Tensor, weight: Tensor, bias: Tensor, padding_data: [^]i64, padding_len: c.int, stride_data: [^]i64, stride_len: c.int, dilation_data: [^]i64, dilation_len: c.int, groups: i64) ---
	atg_mkldnn_linear :: proc(out: ^Tensor, self: Tensor, weight: Tensor, bias: Tensor) ---
	atg_mkldnn_linear_backward_input :: proc(out: ^Tensor, input_size_data: [^]i64, input_size_len: c.int, grad_output: Tensor, weight: Tensor) ---
	atg_mkldnn_linear_backward_input_out :: proc(out: ^Tensor, input_size_data: [^]i64, input_size_len: c.int, grad_output: Tensor, weight: Tensor) ---
	atg_mkldnn_linear_backward_weights :: proc(out: ^Tensor, grad_output: Tensor, input: Tensor, weight: Tensor, bias_defined: c.int) ---
	atg_mkldnn_linear_backward_weights_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, grad_output: Tensor, input: Tensor, weight: Tensor, bias_defined: c.int) ---
	atg_mkldnn_linear_out :: proc(out: ^Tensor, self: Tensor, weight: Tensor, bias: Tensor) ---
	atg_mkldnn_max_pool2d :: proc(out: ^Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, ceil_mode: c.int) ---
	atg_mkldnn_max_pool2d_backward :: proc(out: ^Tensor, grad_output: Tensor, output: Tensor, input: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, ceil_mode: c.int) ---
	atg_mkldnn_max_pool2d_backward_out :: proc(out: ^Tensor, grad_output: Tensor, output: Tensor, input: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, ceil_mode: c.int) ---
	atg_mkldnn_max_pool2d_out :: proc(out: ^Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, ceil_mode: c.int) ---
	atg_mkldnn_max_pool3d :: proc(out: ^Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, ceil_mode: c.int) ---
	atg_mkldnn_max_pool3d_backward :: proc(out: ^Tensor, grad_output: Tensor, output: Tensor, input: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, ceil_mode: c.int) ---
	atg_mkldnn_max_pool3d_backward_out :: proc(out: ^Tensor, grad_output: Tensor, output: Tensor, input: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, ceil_mode: c.int) ---
	atg_mkldnn_max_pool3d_out :: proc(out: ^Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, ceil_mode: c.int) ---
	atg_mkldnn_reorder_conv2d_weight :: proc(out: ^Tensor, self: Tensor, padding_data: [^]i64, padding_len: c.int, stride_data: [^]i64, stride_len: c.int, dilation_data: [^]i64, dilation_len: c.int, groups: i64, input_size_data: [^]i64, input_size_len: c.int) ---
	atg_mkldnn_reorder_conv2d_weight_out :: proc(out: ^Tensor, self: Tensor, padding_data: [^]i64, padding_len: c.int, stride_data: [^]i64, stride_len: c.int, dilation_data: [^]i64, dilation_len: c.int, groups: i64, input_size_data: [^]i64, input_size_len: c.int) ---
	atg_mkldnn_reorder_conv3d_weight :: proc(out: ^Tensor, self: Tensor, padding_data: [^]i64, padding_len: c.int, stride_data: [^]i64, stride_len: c.int, dilation_data: [^]i64, dilation_len: c.int, groups: i64, input_size_data: [^]i64, input_size_len: c.int) ---
	atg_mkldnn_reorder_conv3d_weight_out :: proc(out: ^Tensor, self: Tensor, padding_data: [^]i64, padding_len: c.int, stride_data: [^]i64, stride_len: c.int, dilation_data: [^]i64, dilation_len: c.int, groups: i64, input_size_data: [^]i64, input_size_len: c.int) ---
	atg_mkldnn_rnn_layer :: proc(out: ^Tensor, input: Tensor, weight0: Tensor, weight1: Tensor, weight2: Tensor, weight3: Tensor, hx_: Tensor, cx_: Tensor, reverse: c.int, batch_sizes_data: [^]i64, batch_sizes_len: c.int, mode: i64, hidden_size: i64, num_layers: i64, has_biases: c.int, bidirectional: c.int, batch_first: c.int, train: c.int) ---
	atg_mkldnn_rnn_layer_backward :: proc(out: ^Tensor, input: Tensor, weight1: Tensor, weight2: Tensor, weight3: Tensor, weight4: Tensor, hx_: Tensor, cx_tmp: Tensor, output: Tensor, hy_: Tensor, cy_: Tensor, grad_output: Tensor, grad_hy: Tensor, grad_cy: Tensor, reverse: c.int, mode: i64, hidden_size: i64, num_layers: i64, has_biases: c.int, train: c.int, bidirectional: c.int, batch_sizes_data: [^]i64, batch_sizes_len: c.int, batch_first: c.int, workspace: Tensor) ---
	atg_mkldnn_rnn_layer_backward_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, out2: Tensor, out3: Tensor, out4: Tensor, out5: Tensor, out6: Tensor, input: Tensor, weight1: Tensor, weight2: Tensor, weight3: Tensor, weight4: Tensor, hx_: Tensor, cx_tmp: Tensor, output: Tensor, hy_: Tensor, cy_: Tensor, grad_output: Tensor, grad_hy: Tensor, grad_cy: Tensor, reverse: c.int, mode: i64, hidden_size: i64, num_layers: i64, has_biases: c.int, train: c.int, bidirectional: c.int, batch_sizes_data: [^]i64, batch_sizes_len: c.int, batch_first: c.int, workspace: Tensor) ---
	atg_mkldnn_rnn_layer_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, out2: Tensor, out3: Tensor, input: Tensor, weight0: Tensor, weight1: Tensor, weight2: Tensor, weight3: Tensor, hx_: Tensor, cx_: Tensor, reverse: c.int, batch_sizes_data: [^]i64, batch_sizes_len: c.int, mode: i64, hidden_size: i64, num_layers: i64, has_biases: c.int, bidirectional: c.int, batch_first: c.int, train: c.int) ---
	atg_mm :: proc(out: ^Tensor, self: Tensor, mat2: Tensor) ---
	atg_mm_dtype :: proc(out: ^Tensor, self: Tensor, mat2: Tensor, out_dtype: c.int) ---
	atg_mm_dtype_out :: proc(out: ^Tensor, self: Tensor, mat2: Tensor, out_dtype: c.int) ---
	atg_mm_out :: proc(out: ^Tensor, self: Tensor, mat2: Tensor) ---
	atg_mode :: proc(out: ^Tensor, self: Tensor, dim: i64, keepdim: c.int) ---
	atg_mode_values :: proc(out: ^Tensor, values: Tensor, indices: Tensor, self: Tensor, dim: i64, keepdim: c.int) ---
	atg_moveaxis :: proc(out: ^Tensor, self: Tensor, source_data: [^]i64, source_len: c.int, destination_data: [^]i64, destination_len: c.int) ---
	atg_moveaxis_int :: proc(out: ^Tensor, self: Tensor, source: i64, destination: i64) ---
	atg_movedim :: proc(out: ^Tensor, self: Tensor, source_data: [^]i64, source_len: c.int, destination_data: [^]i64, destination_len: c.int) ---
	atg_movedim_int :: proc(out: ^Tensor, self: Tensor, source: i64, destination: i64) ---
	atg_mse_loss :: proc(out: ^Tensor, self: Tensor, target: Tensor, reduction: i64) ---
	atg_mse_loss_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, target: Tensor, reduction: i64) ---
	atg_mse_loss_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, self: Tensor, target: Tensor, reduction: i64) ---
	atg_mse_loss_out :: proc(out: ^Tensor, self: Tensor, target: Tensor, reduction: i64) ---
	atg_msort :: proc(out: ^Tensor, self: Tensor) ---
	atg_msort_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_mt :: proc(out: ^Tensor, self: Tensor) ---
	atg_mul :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_mul_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_mul_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_mul_scalar :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_mul_scalar_ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_mul_scalar_out :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_multi_margin_loss_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, target: Tensor, p: Scalar, margin: Scalar, weight: Tensor, reduction: i64) ---
	atg_multi_margin_loss_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, self: Tensor, target: Tensor, p: Scalar, margin: Scalar, weight: Tensor, reduction: i64) ---
	atg_multilabel_margin_loss :: proc(out: ^Tensor, self: Tensor, target: Tensor, reduction: i64) ---
	atg_multilabel_margin_loss_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, target: Tensor, reduction: i64, is_target: Tensor) ---
	atg_multilabel_margin_loss_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, self: Tensor, target: Tensor, reduction: i64, is_target: Tensor) ---
	atg_multilabel_margin_loss_out :: proc(out: ^Tensor, self: Tensor, target: Tensor, reduction: i64) ---
	atg_multinomial :: proc(out: ^Tensor, self: Tensor, num_samples: i64, replacement: c.int) ---
	atg_multinomial_out :: proc(out: ^Tensor, self: Tensor, num_samples: i64, replacement: c.int) ---
	atg_multiply :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_multiply_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_multiply_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_multiply_scalar :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_multiply_scalar_ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_mv :: proc(out: ^Tensor, self: Tensor, vec: Tensor) ---
	atg_mv_out :: proc(out: ^Tensor, self: Tensor, vec: Tensor) ---
	atg_mvlgamma :: proc(out: ^Tensor, self: Tensor, p: i64) ---
	atg_mvlgamma_ :: proc(out: ^Tensor, self: Tensor, p: i64) ---
	atg_mvlgamma_out :: proc(out: ^Tensor, self: Tensor, p: i64) ---
	atg_nan_to_num :: proc(out: ^Tensor, self: Tensor, nan_v: f64, nan_null: rawptr, posinf_v: f64, posinf_null: rawptr, neginf_v: f64, neginf_null: rawptr) ---
	atg_nan_to_num_ :: proc(out: ^Tensor, self: Tensor, nan_v: f64, nan_null: rawptr, posinf_v: f64, posinf_null: rawptr, neginf_v: f64, neginf_null: rawptr) ---
	atg_nan_to_num_out :: proc(out: ^Tensor, self: Tensor, nan_v: f64, nan_null: rawptr, posinf_v: f64, posinf_null: rawptr, neginf_v: f64, neginf_null: rawptr) ---
	atg_nanmean :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, keepdim: c.int, dtype: c.int) ---
	atg_nanmean_out :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, keepdim: c.int, dtype: c.int) ---
	atg_nanmedian :: proc(out: ^Tensor, self: Tensor) ---
	atg_nanmedian_dim :: proc(out: ^Tensor, self: Tensor, dim: i64, keepdim: c.int) ---
	atg_nanmedian_dim_values :: proc(out: ^Tensor, values: Tensor, indices: Tensor, self: Tensor, dim: i64, keepdim: c.int) ---
	atg_nanmedian_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_nanquantile :: proc(out: ^Tensor, self: Tensor, q: Tensor, dim_v: i64, dim_null: rawptr, keepdim: c.int, interpolation_ptr: cstring, interpolation_len: c.int) ---
	atg_nanquantile_out :: proc(out: ^Tensor, self: Tensor, q: Tensor, dim_v: i64, dim_null: rawptr, keepdim: c.int, interpolation_ptr: cstring, interpolation_len: c.int) ---
	atg_nanquantile_scalar :: proc(out: ^Tensor, self: Tensor, q: f64, dim_v: i64, dim_null: rawptr, keepdim: c.int, interpolation_ptr: cstring, interpolation_len: c.int) ---
	atg_nanquantile_scalar_out :: proc(out: ^Tensor, self: Tensor, q: f64, dim_v: i64, dim_null: rawptr, keepdim: c.int, interpolation_ptr: cstring, interpolation_len: c.int) ---
	atg_nansum :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, keepdim: c.int, dtype: c.int) ---
	atg_nansum_out :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, keepdim: c.int, dtype: c.int) ---
	atg_narrow :: proc(out: ^Tensor, self: Tensor, dim: i64, start: i64, length: i64) ---
	atg_narrow_copy :: proc(out: ^Tensor, self: Tensor, dim: i64, start: i64, length: i64) ---
	atg_narrow_copy_out :: proc(out: ^Tensor, self: Tensor, dim: i64, start: i64, length: i64) ---
	atg_narrow_tensor :: proc(out: ^Tensor, self: Tensor, dim: i64, start: Tensor, length: i64) ---
	atg_native_batch_norm :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor, running_mean: Tensor, running_var: Tensor, training: c.int, momentum: f64, eps: f64) ---
	atg_native_batch_norm_out :: proc(out: ^Tensor, save_mean: Tensor, save_invstd: Tensor, input: Tensor, weight: Tensor, bias: Tensor, running_mean: Tensor, running_var: Tensor, training: c.int, momentum: f64, eps: f64) ---
	atg_native_channel_shuffle :: proc(out: ^Tensor, self: Tensor, groups: i64) ---
	atg_native_dropout :: proc(out: ^Tensor, input: Tensor, p: f64, train: c.int) ---
	atg_native_dropout_backward :: proc(out: ^Tensor, grad_output: Tensor, mask: Tensor, scale: f64) ---
	atg_native_dropout_backward_out :: proc(out: ^Tensor, grad_output: Tensor, mask: Tensor, scale: f64) ---
	atg_native_dropout_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, input: Tensor, p: f64, train: c.int) ---
	atg_native_group_norm :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor, n: i64, C: i64, HxW: i64, group: i64, eps: f64) ---
	atg_native_group_norm_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, out2: Tensor, input: Tensor, weight: Tensor, bias: Tensor, n: i64, C: i64, HxW: i64, group: i64, eps: f64) ---
	atg_native_layer_norm :: proc(out: ^Tensor, input: Tensor, normalized_shape_data: [^]i64, normalized_shape_len: c.int, weight: Tensor, bias: Tensor, eps: f64) ---
	atg_native_layer_norm_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, out2: Tensor, input: Tensor, normalized_shape_data: [^]i64, normalized_shape_len: c.int, weight: Tensor, bias: Tensor, eps: f64) ---
	atg_native_norm :: proc(out: ^Tensor, self: Tensor) ---
	atg_native_norm_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_native_norm_scalaropt_dim_dtype :: proc(out: ^Tensor, self: Tensor, p: Scalar, dim_data: [^]i64, dim_len: c.int, keepdim: c.int, dtype: c.int) ---
	atg_native_norm_scalaropt_dim_dtype_out :: proc(out: ^Tensor, self: Tensor, p: Scalar, dim_data: [^]i64, dim_len: c.int, keepdim: c.int, dtype: c.int) ---
	atg_ne :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_ne_ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_ne_scalar_out :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_ne_tensor :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_ne_tensor_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_ne_tensor_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_neg :: proc(out: ^Tensor, self: Tensor) ---
	atg_neg_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_neg_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_negative :: proc(out: ^Tensor, self: Tensor) ---
	atg_negative_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_negative_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_nested_to_padded_tensor :: proc(out: ^Tensor, self: Tensor, padding: f64, output_size_data: [^]i64, output_size_len: c.int) ---
	atg_new_empty :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int, options_kind: c.int, options_device: c.int) ---
	atg_new_empty_out :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int) ---
	atg_new_empty_strided :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int, stride_data: [^]i64, stride_len: c.int, options_kind: c.int, options_device: c.int) ---
	atg_new_empty_strided_out :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int, stride_data: [^]i64, stride_len: c.int) ---
	atg_new_full :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int, fill_value: Scalar, options_kind: c.int, options_device: c.int) ---
	atg_new_full_out :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int, fill_value: Scalar) ---
	atg_new_ones :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int, options_kind: c.int, options_device: c.int) ---
	atg_new_ones_out :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int) ---
	atg_new_zeros :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int, options_kind: c.int, options_device: c.int) ---
	atg_new_zeros_out :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int) ---
	atg_nextafter :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_nextafter_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_nextafter_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_nll_loss :: proc(out: ^Tensor, self: Tensor, target: Tensor, weight: Tensor, reduction: i64, ignore_index: i64) ---
	atg_nll_loss2d :: proc(out: ^Tensor, self: Tensor, target: Tensor, weight: Tensor, reduction: i64, ignore_index: i64) ---
	atg_nll_loss2d_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, target: Tensor, weight: Tensor, reduction: i64, ignore_index: i64, total_weight: Tensor) ---
	atg_nll_loss2d_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, self: Tensor, target: Tensor, weight: Tensor, reduction: i64, ignore_index: i64, total_weight: Tensor) ---
	atg_nll_loss2d_out :: proc(out: ^Tensor, self: Tensor, target: Tensor, weight: Tensor, reduction: i64, ignore_index: i64) ---
	atg_nll_loss_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, target: Tensor, weight: Tensor, reduction: i64, ignore_index: i64, total_weight: Tensor) ---
	atg_nll_loss_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, self: Tensor, target: Tensor, weight: Tensor, reduction: i64, ignore_index: i64, total_weight: Tensor) ---
	atg_nll_loss_nd :: proc(out: ^Tensor, self: Tensor, target: Tensor, weight: Tensor, reduction: i64, ignore_index: i64) ---
	atg_nll_loss_out :: proc(out: ^Tensor, self: Tensor, target: Tensor, weight: Tensor, reduction: i64, ignore_index: i64) ---
	atg_nonzero :: proc(out: ^Tensor, self: Tensor) ---
	atg_nonzero_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_nonzero_static :: proc(out: ^Tensor, self: Tensor, size: i64, fill_value: i64) ---
	atg_nonzero_static_out :: proc(out: ^Tensor, self: Tensor, size: i64, fill_value: i64) ---
	atg_norm :: proc(out: ^Tensor, self: Tensor) ---
	atg_norm_dtype_out :: proc(out: ^Tensor, self: Tensor, p: Scalar, dim_data: [^]i64, dim_len: c.int, keepdim: c.int, dtype: c.int) ---
	atg_norm_except_dim :: proc(out: ^Tensor, v: Tensor, pow: i64, dim: i64) ---
	atg_norm_out :: proc(out: ^Tensor, self: Tensor, p: Scalar, dim_data: [^]i64, dim_len: c.int, keepdim: c.int) ---
	atg_norm_scalar_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_norm_scalaropt_dim :: proc(out: ^Tensor, self: Tensor, p: Scalar, dim_data: [^]i64, dim_len: c.int, keepdim: c.int) ---
	atg_norm_scalaropt_dim_dtype :: proc(out: ^Tensor, self: Tensor, p: Scalar, dim_data: [^]i64, dim_len: c.int, keepdim: c.int, dtype: c.int) ---
	atg_norm_scalaropt_dtype :: proc(out: ^Tensor, self: Tensor, p: Scalar, dtype: c.int) ---
	atg_norm_scalaropt_dtype_out :: proc(out: ^Tensor, self: Tensor, p: Scalar, dtype: c.int) ---
	atg_normal_ :: proc(out: ^Tensor, self: Tensor, mean: f64, std: f64) ---
	atg_normal_functional :: proc(out: ^Tensor, self: Tensor, mean: f64, std: f64) ---
	atg_not_equal :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_not_equal_ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_not_equal_scalar_out :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_not_equal_tensor :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_not_equal_tensor_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_not_equal_tensor_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_nuclear_norm :: proc(out: ^Tensor, self: Tensor, keepdim: c.int) ---
	atg_nuclear_norm_dim :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, keepdim: c.int) ---
	atg_nuclear_norm_dim_out :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, keepdim: c.int) ---
	atg_nuclear_norm_out :: proc(out: ^Tensor, self: Tensor, keepdim: c.int) ---
	atg_numpy_t :: proc(out: ^Tensor, self: Tensor) ---
	atg_one_hot :: proc(out: ^Tensor, self: Tensor, num_classes: i64) ---
	atg_ones :: proc(out: ^Tensor, size_data: [^]i64, size_len: c.int, options_kind: c.int, options_device: c.int) ---
	atg_ones_like :: proc(out: ^Tensor, self: Tensor) ---
	atg_ones_like_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_ones_out :: proc(out: ^Tensor, size_data: [^]i64, size_len: c.int) ---
	atg_orgqr :: proc(out: ^Tensor, self: Tensor, input2: Tensor) ---
	atg_orgqr_out :: proc(out: ^Tensor, self: Tensor, input2: Tensor) ---
	atg_ormqr :: proc(out: ^Tensor, self: Tensor, input2: Tensor, input3: Tensor, left: c.int, transpose: c.int) ---
	atg_ormqr_out :: proc(out: ^Tensor, self: Tensor, input2: Tensor, input3: Tensor, left: c.int, transpose: c.int) ---
	atg_outer :: proc(out: ^Tensor, self: Tensor, vec2: Tensor) ---
	atg_outer_out :: proc(out: ^Tensor, self: Tensor, vec2: Tensor) ---
	atg_pad :: proc(out: ^Tensor, self: Tensor, pad_data: [^]i64, pad_len: c.int, mode_ptr: cstring, mode_len: c.int, value_v: f64, value_null: rawptr) ---
	atg_pad_sequence :: proc(out: ^Tensor, sequences_data: ^Tensor, sequences_len: c.int, batch_first: c.int, padding_value: f64, padding_side_ptr: cstring, padding_side_len: c.int) ---
	atg_pairwise_distance :: proc(out: ^Tensor, x1: Tensor, x2: Tensor, p: f64, eps: f64, keepdim: c.int) ---
	atg_pdist :: proc(out: ^Tensor, self: Tensor, p: f64) ---
	atg_permute :: proc(out: ^Tensor, self: Tensor, dims_data: [^]i64, dims_len: c.int) ---
	atg_permute_copy :: proc(out: ^Tensor, self: Tensor, dims_data: [^]i64, dims_len: c.int) ---
	atg_permute_copy_out :: proc(out: ^Tensor, self: Tensor, dims_data: [^]i64, dims_len: c.int) ---
	atg_pin_memory :: proc(out: ^Tensor, self: Tensor, device: c.int) ---
	atg_pinverse :: proc(out: ^Tensor, self: Tensor, rcond: f64) ---
	atg_pixel_shuffle :: proc(out: ^Tensor, self: Tensor, upscale_factor: i64) ---
	atg_pixel_shuffle_out :: proc(out: ^Tensor, self: Tensor, upscale_factor: i64) ---
	atg_pixel_unshuffle :: proc(out: ^Tensor, self: Tensor, downscale_factor: i64) ---
	atg_pixel_unshuffle_out :: proc(out: ^Tensor, self: Tensor, downscale_factor: i64) ---
	atg_poisson :: proc(out: ^Tensor, self: Tensor) ---
	atg_poisson_nll_loss :: proc(out: ^Tensor, input: Tensor, target: Tensor, log_input: c.int, full: c.int, eps: f64, reduction: i64) ---
	atg_poisson_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_polar :: proc(out: ^Tensor, abs: Tensor, angle: Tensor) ---
	atg_polar_out :: proc(out: ^Tensor, abs: Tensor, angle: Tensor) ---
	atg_polygamma :: proc(out: ^Tensor, n: i64, self: Tensor) ---
	atg_polygamma_ :: proc(out: ^Tensor, self: Tensor, n: i64) ---
	atg_polygamma_out :: proc(out: ^Tensor, n: i64, self: Tensor) ---
	atg_positive :: proc(out: ^Tensor, self: Tensor) ---
	atg_pow :: proc(out: ^Tensor, self: Tensor, exponent: Tensor) ---
	atg_pow_ :: proc(out: ^Tensor, self: Tensor, exponent: Scalar) ---
	atg_pow_scalar :: proc(out: ^Tensor, self_scalar: Scalar, exponent: Tensor) ---
	atg_pow_scalar_out :: proc(out: ^Tensor, self_scalar: Scalar, exponent: Tensor) ---
	atg_pow_tensor_ :: proc(out: ^Tensor, self: Tensor, exponent: Tensor) ---
	atg_pow_tensor_scalar :: proc(out: ^Tensor, self: Tensor, exponent: Scalar) ---
	atg_pow_tensor_scalar_out :: proc(out: ^Tensor, self: Tensor, exponent: Scalar) ---
	atg_pow_tensor_tensor_out :: proc(out: ^Tensor, self: Tensor, exponent: Tensor) ---
	atg_prelu :: proc(out: ^Tensor, self: Tensor, weight: Tensor) ---
	atg_prod :: proc(out: ^Tensor, self: Tensor, dtype: c.int) ---
	atg_prod_dim_int :: proc(out: ^Tensor, self: Tensor, dim: i64, keepdim: c.int, dtype: c.int) ---
	atg_prod_int_out :: proc(out: ^Tensor, self: Tensor, dim: i64, keepdim: c.int, dtype: c.int) ---
	atg_prod_out :: proc(out: ^Tensor, self: Tensor, dtype: c.int) ---
	atg_put :: proc(out: ^Tensor, self: Tensor, index: Tensor, source: Tensor, accumulate: c.int) ---
	atg_put_ :: proc(out: ^Tensor, self: Tensor, index: Tensor, source: Tensor, accumulate: c.int) ---
	atg_put_out :: proc(out: ^Tensor, self: Tensor, index: Tensor, source: Tensor, accumulate: c.int) ---
	atg_q_per_channel_scales :: proc(out: ^Tensor, self: Tensor) ---
	atg_q_per_channel_scales_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_q_per_channel_zero_points :: proc(out: ^Tensor, self: Tensor) ---
	atg_q_per_channel_zero_points_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_qr :: proc(out: ^Tensor, self: Tensor, some: c.int) ---
	atg_qr_q :: proc(out: ^Tensor, Q: Tensor, R: Tensor, self: Tensor, some: c.int) ---
	atg_quantile :: proc(out: ^Tensor, self: Tensor, q: Tensor, dim_v: i64, dim_null: rawptr, keepdim: c.int, interpolation_ptr: cstring, interpolation_len: c.int) ---
	atg_quantile_out :: proc(out: ^Tensor, self: Tensor, q: Tensor, dim_v: i64, dim_null: rawptr, keepdim: c.int, interpolation_ptr: cstring, interpolation_len: c.int) ---
	atg_quantile_scalar :: proc(out: ^Tensor, self: Tensor, q: f64, dim_v: i64, dim_null: rawptr, keepdim: c.int, interpolation_ptr: cstring, interpolation_len: c.int) ---
	atg_quantile_scalar_out :: proc(out: ^Tensor, self: Tensor, q: f64, dim_v: i64, dim_null: rawptr, keepdim: c.int, interpolation_ptr: cstring, interpolation_len: c.int) ---
	atg_quantize_per_channel :: proc(out: ^Tensor, self: Tensor, scales: Tensor, zero_points: Tensor, axis: i64, dtype: c.int) ---
	atg_quantize_per_channel_out :: proc(out: ^Tensor, self: Tensor, scales: Tensor, zero_points: Tensor, axis: i64, dtype: c.int) ---
	atg_quantize_per_tensor :: proc(out: ^Tensor, self: Tensor, scale: f64, zero_point: i64, dtype: c.int) ---
	atg_quantize_per_tensor_dynamic :: proc(out: ^Tensor, self: Tensor, dtype: c.int, reduce_range: c.int) ---
	atg_quantize_per_tensor_dynamic_out :: proc(out: ^Tensor, self: Tensor, dtype: c.int, reduce_range: c.int) ---
	atg_quantize_per_tensor_out :: proc(out: ^Tensor, self: Tensor, scale: f64, zero_point: i64, dtype: c.int) ---
	atg_quantize_per_tensor_tensor_qparams :: proc(out: ^Tensor, self: Tensor, scale: Tensor, zero_point: Tensor, dtype: c.int) ---
	atg_quantize_per_tensor_tensor_qparams_out :: proc(out: ^Tensor, self: Tensor, scale: Tensor, zero_point: Tensor, dtype: c.int) ---
	atg_quantize_per_tensor_tensors_out :: proc(out_data: ^Tensor, out_len: c.int, tensors_data: ^Tensor, tensors_len: c.int, scales: Tensor, zero_points: Tensor, dtype: c.int) ---
	atg_quantized_batch_norm :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor, mean: Tensor, var: Tensor, eps: f64, output_scale: f64, output_zero_point: i64) ---
	atg_quantized_batch_norm_out :: proc(out: ^Tensor, input: Tensor, weight: Tensor, bias: Tensor, mean: Tensor, var: Tensor, eps: f64, output_scale: f64, output_zero_point: i64) ---
	atg_quantized_gru_cell :: proc(out: ^Tensor, input: Tensor, hx: Tensor, w_ih: Tensor, w_hh: Tensor, b_ih: Tensor, b_hh: Tensor, packed_ih: Tensor, packed_hh: Tensor, col_offsets_ih: Tensor, col_offsets_hh: Tensor, scale_ih: Scalar, scale_hh: Scalar, zero_point_ih: Scalar, zero_point_hh: Scalar) ---
	atg_quantized_lstm_cell :: proc(out: ^Tensor, input: Tensor, hx_data: ^Tensor, hx_len: c.int, w_ih: Tensor, w_hh: Tensor, b_ih: Tensor, b_hh: Tensor, packed_ih: Tensor, packed_hh: Tensor, col_offsets_ih: Tensor, col_offsets_hh: Tensor, scale_ih: Scalar, scale_hh: Scalar, zero_point_ih: Scalar, zero_point_hh: Scalar) ---
	atg_quantized_max_pool1d :: proc(out: ^Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, ceil_mode: c.int) ---
	atg_quantized_max_pool1d_out :: proc(out: ^Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, ceil_mode: c.int) ---
	atg_quantized_max_pool2d :: proc(out: ^Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, ceil_mode: c.int) ---
	atg_quantized_max_pool2d_out :: proc(out: ^Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, ceil_mode: c.int) ---
	atg_quantized_max_pool3d :: proc(out: ^Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, ceil_mode: c.int) ---
	atg_quantized_max_pool3d_out :: proc(out: ^Tensor, self: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int, ceil_mode: c.int) ---
	atg_quantized_rnn_relu_cell :: proc(out: ^Tensor, input: Tensor, hx: Tensor, w_ih: Tensor, w_hh: Tensor, b_ih: Tensor, b_hh: Tensor, packed_ih: Tensor, packed_hh: Tensor, col_offsets_ih: Tensor, col_offsets_hh: Tensor, scale_ih: Scalar, scale_hh: Scalar, zero_point_ih: Scalar, zero_point_hh: Scalar) ---
	atg_quantized_rnn_tanh_cell :: proc(out: ^Tensor, input: Tensor, hx: Tensor, w_ih: Tensor, w_hh: Tensor, b_ih: Tensor, b_hh: Tensor, packed_ih: Tensor, packed_hh: Tensor, col_offsets_ih: Tensor, col_offsets_hh: Tensor, scale_ih: Scalar, scale_hh: Scalar, zero_point_ih: Scalar, zero_point_hh: Scalar) ---
	atg_rad2deg :: proc(out: ^Tensor, self: Tensor) ---
	atg_rad2deg_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_rad2deg_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_rand :: proc(out: ^Tensor, size_data: [^]i64, size_len: c.int, options_kind: c.int, options_device: c.int) ---
	atg_rand_like :: proc(out: ^Tensor, self: Tensor) ---
	atg_rand_like_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_rand_out :: proc(out: ^Tensor, size_data: [^]i64, size_len: c.int) ---
	atg_randint :: proc(out: ^Tensor, high: i64, size_data: [^]i64, size_len: c.int, options_kind: c.int, options_device: c.int) ---
	atg_randint_like :: proc(out: ^Tensor, self: Tensor, high: i64) ---
	atg_randint_like_low_dtype :: proc(out: ^Tensor, self: Tensor, low: i64, high: i64) ---
	atg_randint_like_low_dtype_out :: proc(out: ^Tensor, self: Tensor, low: i64, high: i64) ---
	atg_randint_like_out :: proc(out: ^Tensor, self: Tensor, high: i64) ---
	atg_randint_like_tensor :: proc(out: ^Tensor, self: Tensor, high: Tensor) ---
	atg_randint_like_tensor_out :: proc(out: ^Tensor, self: Tensor, high: Tensor) ---
	atg_randint_low :: proc(out: ^Tensor, low: i64, high: i64, size_data: [^]i64, size_len: c.int, options_kind: c.int, options_device: c.int) ---
	atg_randint_low_out :: proc(out: ^Tensor, low: i64, high: i64, size_data: [^]i64, size_len: c.int) ---
	atg_randint_out :: proc(out: ^Tensor, high: i64, size_data: [^]i64, size_len: c.int) ---
	atg_randn :: proc(out: ^Tensor, size_data: [^]i64, size_len: c.int, options_kind: c.int, options_device: c.int) ---
	atg_randn_like :: proc(out: ^Tensor, self: Tensor) ---
	atg_randn_like_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_randn_out :: proc(out: ^Tensor, size_data: [^]i64, size_len: c.int) ---
	atg_random :: proc(out: ^Tensor, self: Tensor) ---
	atg_random_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_random_from :: proc(out: ^Tensor, self: Tensor, from: i64, to_v: i64, to_null: rawptr) ---
	atg_random_from_ :: proc(out: ^Tensor, self: Tensor, from: i64, to_v: i64, to_null: rawptr) ---
	atg_random_from_out :: proc(out: ^Tensor, self: Tensor, from: i64, to_v: i64, to_null: rawptr) ---
	atg_random_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_random_to :: proc(out: ^Tensor, self: Tensor, to: i64) ---
	atg_random_to_ :: proc(out: ^Tensor, self: Tensor, to: i64) ---
	atg_random_to_out :: proc(out: ^Tensor, self: Tensor, to: i64) ---
	atg_randperm :: proc(out: ^Tensor, n: i64, options_kind: c.int, options_device: c.int) ---
	atg_randperm_out :: proc(out: ^Tensor, n: i64) ---
	atg_range :: proc(out: ^Tensor, start: Scalar, end: Scalar, options_kind: c.int, options_device: c.int) ---
	atg_range_out :: proc(out: ^Tensor, start: Scalar, end: Scalar) ---
	atg_range_out_ :: proc(out: ^Tensor, start: Scalar, end: Scalar) ---
	atg_range_step :: proc(out: ^Tensor, start: Scalar, end: Scalar, options_kind: c.int, options_device: c.int) ---
	atg_ravel :: proc(out: ^Tensor, self: Tensor) ---
	atg_real :: proc(out: ^Tensor, self: Tensor) ---
	atg_reciprocal :: proc(out: ^Tensor, self: Tensor) ---
	atg_reciprocal_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_reciprocal_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_reflection_pad1d :: proc(out: ^Tensor, self: Tensor, padding_data: [^]i64, padding_len: c.int) ---
	atg_reflection_pad1d_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, padding_data: [^]i64, padding_len: c.int) ---
	atg_reflection_pad1d_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, self: Tensor, padding_data: [^]i64, padding_len: c.int) ---
	atg_reflection_pad1d_out :: proc(out: ^Tensor, self: Tensor, padding_data: [^]i64, padding_len: c.int) ---
	atg_reflection_pad2d :: proc(out: ^Tensor, self: Tensor, padding_data: [^]i64, padding_len: c.int) ---
	atg_reflection_pad2d_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, padding_data: [^]i64, padding_len: c.int) ---
	atg_reflection_pad2d_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, self: Tensor, padding_data: [^]i64, padding_len: c.int) ---
	atg_reflection_pad2d_out :: proc(out: ^Tensor, self: Tensor, padding_data: [^]i64, padding_len: c.int) ---
	atg_reflection_pad3d :: proc(out: ^Tensor, self: Tensor, padding_data: [^]i64, padding_len: c.int) ---
	atg_reflection_pad3d_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, padding_data: [^]i64, padding_len: c.int) ---
	atg_reflection_pad3d_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, self: Tensor, padding_data: [^]i64, padding_len: c.int) ---
	atg_reflection_pad3d_out :: proc(out: ^Tensor, self: Tensor, padding_data: [^]i64, padding_len: c.int) ---
	atg_relu :: proc(out: ^Tensor, self: Tensor) ---
	atg_relu6 :: proc(out: ^Tensor, self: Tensor) ---
	atg_relu6_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_relu_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_relu_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_remainder :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_remainder_ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_remainder_scalar_out :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_remainder_scalar_tensor :: proc(out: ^Tensor, self_scalar: Scalar, other: Tensor) ---
	atg_remainder_scalar_tensor_out :: proc(out: ^Tensor, self_scalar: Scalar, other: Tensor) ---
	atg_remainder_tensor :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_remainder_tensor_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_remainder_tensor_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_renorm :: proc(out: ^Tensor, self: Tensor, p: Scalar, dim: i64, maxnorm: Scalar) ---
	atg_renorm_ :: proc(out: ^Tensor, self: Tensor, p: Scalar, dim: i64, maxnorm: Scalar) ---
	atg_renorm_out :: proc(out: ^Tensor, self: Tensor, p: Scalar, dim: i64, maxnorm: Scalar) ---
	atg_repeat :: proc(out: ^Tensor, self: Tensor, repeats_data: [^]i64, repeats_len: c.int) ---
	atg_repeat_interleave :: proc(out: ^Tensor, repeats: Tensor, output_size_v: i64, output_size_null: rawptr) ---
	atg_repeat_interleave_self_int :: proc(out: ^Tensor, self: Tensor, repeats: i64, dim_v: i64, dim_null: rawptr, output_size_v: i64, output_size_null: rawptr) ---
	atg_repeat_interleave_self_tensor :: proc(out: ^Tensor, self: Tensor, repeats: Tensor, dim_v: i64, dim_null: rawptr, output_size_v: i64, output_size_null: rawptr) ---
	atg_repeat_interleave_tensor_out :: proc(out: ^Tensor, repeats: Tensor, output_size_v: i64, output_size_null: rawptr) ---
	atg_repeat_out :: proc(out: ^Tensor, self: Tensor, repeats_data: [^]i64, repeats_len: c.int) ---
	atg_replication_pad1d :: proc(out: ^Tensor, self: Tensor, padding_data: [^]i64, padding_len: c.int) ---
	atg_replication_pad1d_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, padding_data: [^]i64, padding_len: c.int) ---
	atg_replication_pad1d_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, self: Tensor, padding_data: [^]i64, padding_len: c.int) ---
	atg_replication_pad1d_out :: proc(out: ^Tensor, self: Tensor, padding_data: [^]i64, padding_len: c.int) ---
	atg_replication_pad2d :: proc(out: ^Tensor, self: Tensor, padding_data: [^]i64, padding_len: c.int) ---
	atg_replication_pad2d_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, padding_data: [^]i64, padding_len: c.int) ---
	atg_replication_pad2d_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, self: Tensor, padding_data: [^]i64, padding_len: c.int) ---
	atg_replication_pad2d_out :: proc(out: ^Tensor, self: Tensor, padding_data: [^]i64, padding_len: c.int) ---
	atg_replication_pad3d :: proc(out: ^Tensor, self: Tensor, padding_data: [^]i64, padding_len: c.int) ---
	atg_replication_pad3d_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, padding_data: [^]i64, padding_len: c.int) ---
	atg_replication_pad3d_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, self: Tensor, padding_data: [^]i64, padding_len: c.int) ---
	atg_replication_pad3d_out :: proc(out: ^Tensor, self: Tensor, padding_data: [^]i64, padding_len: c.int) ---
	atg_requires_grad_ :: proc(out: ^Tensor, self: Tensor, requires_grad: c.int) ---
	atg_reshape :: proc(out: ^Tensor, self: Tensor, shape_data: [^]i64, shape_len: c.int) ---
	atg_reshape_as :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_resize :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int) ---
	atg_resize_ :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int) ---
	atg_resize_as :: proc(out: ^Tensor, self: Tensor, the_template: Tensor) ---
	atg_resize_as_ :: proc(out: ^Tensor, self: Tensor, the_template: Tensor) ---
	atg_resize_as_out :: proc(out: ^Tensor, self: Tensor, the_template: Tensor) ---
	atg_resize_as_sparse :: proc(out: ^Tensor, self: Tensor, the_template: Tensor) ---
	atg_resize_as_sparse_ :: proc(out: ^Tensor, self: Tensor, the_template: Tensor) ---
	atg_resize_as_sparse_out :: proc(out: ^Tensor, self: Tensor, the_template: Tensor) ---
	atg_resize_out :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int) ---
	atg_resolve_conj :: proc(out: ^Tensor, self: Tensor) ---
	atg_resolve_neg :: proc(out: ^Tensor, self: Tensor) ---
	atg_rms_norm :: proc(out: ^Tensor, input: Tensor, normalized_shape_data: [^]i64, normalized_shape_len: c.int, weight: Tensor, eps_v: f64, eps_null: rawptr) ---
	atg_rnn_relu :: proc(out: ^Tensor, input: Tensor, hx: Tensor, params_data: ^Tensor, params_len: c.int, has_biases: c.int, num_layers: i64, dropout: f64, train: c.int, bidirectional: c.int, batch_first: c.int) ---
	atg_rnn_relu_cell :: proc(out: ^Tensor, input: Tensor, hx: Tensor, w_ih: Tensor, w_hh: Tensor, b_ih: Tensor, b_hh: Tensor) ---
	atg_rnn_relu_data :: proc(out: ^Tensor, data: Tensor, batch_sizes: Tensor, hx: Tensor, params_data: ^Tensor, params_len: c.int, has_biases: c.int, num_layers: i64, dropout: f64, train: c.int, bidirectional: c.int) ---
	atg_rnn_tanh :: proc(out: ^Tensor, input: Tensor, hx: Tensor, params_data: ^Tensor, params_len: c.int, has_biases: c.int, num_layers: i64, dropout: f64, train: c.int, bidirectional: c.int, batch_first: c.int) ---
	atg_rnn_tanh_cell :: proc(out: ^Tensor, input: Tensor, hx: Tensor, w_ih: Tensor, w_hh: Tensor, b_ih: Tensor, b_hh: Tensor) ---
	atg_rnn_tanh_data :: proc(out: ^Tensor, data: Tensor, batch_sizes: Tensor, hx: Tensor, params_data: ^Tensor, params_len: c.int, has_biases: c.int, num_layers: i64, dropout: f64, train: c.int, bidirectional: c.int) ---
	atg_roll :: proc(out: ^Tensor, self: Tensor, shifts_data: [^]i64, shifts_len: c.int, dims_data: [^]i64, dims_len: c.int) ---
	atg_roll_out :: proc(out: ^Tensor, self: Tensor, shifts_data: [^]i64, shifts_len: c.int, dims_data: [^]i64, dims_len: c.int) ---
	atg_rot90 :: proc(out: ^Tensor, self: Tensor, k: i64, dims_data: [^]i64, dims_len: c.int) ---
	atg_rot90_out :: proc(out: ^Tensor, self: Tensor, k: i64, dims_data: [^]i64, dims_len: c.int) ---
	atg_round :: proc(out: ^Tensor, self: Tensor) ---
	atg_round_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_round_decimals :: proc(out: ^Tensor, self: Tensor, decimals: i64) ---
	atg_round_decimals_ :: proc(out: ^Tensor, self: Tensor, decimals: i64) ---
	atg_round_decimals_out :: proc(out: ^Tensor, self: Tensor, decimals: i64) ---
	atg_round_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_row_indices :: proc(out: ^Tensor, self: Tensor) ---
	atg_row_indices_copy :: proc(out: ^Tensor, self: Tensor) ---
	atg_row_indices_copy_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_row_stack :: proc(out: ^Tensor, tensors_data: ^Tensor, tensors_len: c.int) ---
	atg_row_stack_out :: proc(out: ^Tensor, tensors_data: ^Tensor, tensors_len: c.int) ---
	atg_rrelu :: proc(out: ^Tensor, self: Tensor, training: c.int) ---
	atg_rrelu_ :: proc(out: ^Tensor, self: Tensor, training: c.int) ---
	atg_rrelu_with_noise :: proc(out: ^Tensor, self: Tensor, noise: Tensor, training: c.int) ---
	atg_rrelu_with_noise_ :: proc(out: ^Tensor, self: Tensor, noise: Tensor, training: c.int) ---
	atg_rrelu_with_noise_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, noise: Tensor, lower: Scalar, upper: Scalar, training: c.int, self_is_result: c.int) ---
	atg_rrelu_with_noise_backward_out :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, noise: Tensor, lower: Scalar, upper: Scalar, training: c.int, self_is_result: c.int) ---
	atg_rrelu_with_noise_functional :: proc(out: ^Tensor, self: Tensor, noise: Tensor, training: c.int) ---
	atg_rrelu_with_noise_out :: proc(out: ^Tensor, self: Tensor, noise: Tensor, training: c.int) ---
	atg_rsqrt :: proc(out: ^Tensor, self: Tensor) ---
	atg_rsqrt_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_rsqrt_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_rsub :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_rsub_scalar :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_rsub_scalar_out :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_rsub_tensor_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_scalar_tensor :: proc(out: ^Tensor, s: Scalar, options_kind: c.int, options_device: c.int) ---
	atg_scalar_tensor_out :: proc(out: ^Tensor, s: Scalar) ---
	atg_scaled_dot_product_attention :: proc(out: ^Tensor, query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor, dropout_p: f64, is_causal: c.int, scale_v: f64, scale_null: rawptr, enable_gqa: c.int) ---
	atg_scatter :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor, src: Tensor) ---
	atg_scatter_ :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor, src: Tensor) ---
	atg_scatter_add :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor, src: Tensor) ---
	atg_scatter_add_ :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor, src: Tensor) ---
	atg_scatter_add_out :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor, src: Tensor) ---
	atg_scatter_reduce :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor, src: Tensor, reduce_ptr: cstring, reduce_len: c.int) ---
	atg_scatter_reduce_ :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor, src: Tensor, reduce_ptr: cstring, reduce_len: c.int) ---
	atg_scatter_reduce_out :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor, src: Tensor, reduce_ptr: cstring, reduce_len: c.int) ---
	atg_scatter_src_out :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor, src: Tensor) ---
	atg_scatter_value :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor, value: Scalar) ---
	atg_scatter_value_ :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor, value: Scalar) ---
	atg_scatter_value_out :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor, value: Scalar) ---
	atg_scatter_value_reduce :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor, value: Scalar, reduce_ptr: cstring, reduce_len: c.int) ---
	atg_scatter_value_reduce_ :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor, value: Scalar, reduce_ptr: cstring, reduce_len: c.int) ---
	atg_scatter_value_reduce_out :: proc(out: ^Tensor, self: Tensor, dim: i64, index: Tensor, value: Scalar, reduce_ptr: cstring, reduce_len: c.int) ---
	atg_searchsorted :: proc(out: ^Tensor, sorted_sequence: Tensor, self: Tensor, out_int32: c.int, right: c.int, side_ptr: cstring, side_len: c.int, sorter: Tensor) ---
	atg_searchsorted_scalar :: proc(out: ^Tensor, sorted_sequence: Tensor, self_scalar: Scalar, out_int32: c.int, right: c.int, side_ptr: cstring, side_len: c.int, sorter: Tensor) ---
	atg_searchsorted_scalar_out :: proc(out: ^Tensor, sorted_sequence: Tensor, self_scalar: Scalar, out_int32: c.int, right: c.int, side_ptr: cstring, side_len: c.int, sorter: Tensor) ---
	atg_searchsorted_tensor_out :: proc(out: ^Tensor, sorted_sequence: Tensor, self: Tensor, out_int32: c.int, right: c.int, side_ptr: cstring, side_len: c.int, sorter: Tensor) ---
	atg_segment_reduce :: proc(out: ^Tensor, data: Tensor, reduce_ptr: cstring, reduce_len: c.int, lengths: Tensor, indices: Tensor, offsets: Tensor, axis: i64, unsafe: c.int, initial: Scalar) ---
	atg_segment_reduce_out :: proc(out: ^Tensor, data: Tensor, reduce_ptr: cstring, reduce_len: c.int, lengths: Tensor, indices: Tensor, offsets: Tensor, axis: i64, unsafe: c.int, initial: Scalar) ---
	atg_select :: proc(out: ^Tensor, self: Tensor, dim: i64, index: i64) ---
	atg_select_backward :: proc(out: ^Tensor, grad_output: Tensor, input_sizes_data: [^]i64, input_sizes_len: c.int, dim: i64, index: i64) ---
	atg_select_backward_out :: proc(out: ^Tensor, grad_output: Tensor, input_sizes_data: [^]i64, input_sizes_len: c.int, dim: i64, index: i64) ---
	atg_select_copy :: proc(out: ^Tensor, self: Tensor, dim: i64, index: i64) ---
	atg_select_copy_int_out :: proc(out: ^Tensor, self: Tensor, dim: i64, index: i64) ---
	atg_select_scatter :: proc(out: ^Tensor, self: Tensor, src: Tensor, dim: i64, index: i64) ---
	atg_select_scatter_out :: proc(out: ^Tensor, self: Tensor, src: Tensor, dim: i64, index: i64) ---
	atg_selu :: proc(out: ^Tensor, self: Tensor) ---
	atg_selu_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_set :: proc(out: ^Tensor, self: Tensor) ---
	atg_set_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_set_data :: proc(self: Tensor, new_data: Tensor) ---
	atg_set_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_set_requires_grad :: proc(out: ^Tensor, self: Tensor, r: c.int) ---
	atg_set_source_tensor :: proc(out: ^Tensor, self: Tensor, source: Tensor) ---
	atg_set_source_tensor_ :: proc(out: ^Tensor, self: Tensor, source: Tensor) ---
	atg_set_source_tensor_out :: proc(out: ^Tensor, self: Tensor, source: Tensor) ---
	atg_set_source_tensor_storage_offset_ :: proc(out: ^Tensor, self: Tensor, source: Tensor, storage_offset: i64, size_data: [^]i64, size_len: c.int, stride_data: [^]i64, stride_len: c.int) ---
	atg_sgn :: proc(out: ^Tensor, self: Tensor) ---
	atg_sgn_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_sgn_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_sigmoid :: proc(out: ^Tensor, self: Tensor) ---
	atg_sigmoid_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_sigmoid_backward :: proc(out: ^Tensor, grad_output: Tensor, output: Tensor) ---
	atg_sigmoid_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, output: Tensor) ---
	atg_sigmoid_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_sign :: proc(out: ^Tensor, self: Tensor) ---
	atg_sign_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_sign_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_signbit :: proc(out: ^Tensor, self: Tensor) ---
	atg_signbit_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_silu :: proc(out: ^Tensor, self: Tensor) ---
	atg_silu_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_silu_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor) ---
	atg_silu_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, self: Tensor) ---
	atg_silu_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_sin :: proc(out: ^Tensor, self: Tensor) ---
	atg_sin_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_sin_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_sinc :: proc(out: ^Tensor, self: Tensor) ---
	atg_sinc_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_sinc_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_sinh :: proc(out: ^Tensor, self: Tensor) ---
	atg_sinh_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_sinh_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_slice :: proc(out: ^Tensor, self: Tensor, dim: i64, start_v: i64, start_null: rawptr, end_v: i64, end_null: rawptr, step: i64) ---
	atg_slice_backward :: proc(out: ^Tensor, grad_output: Tensor, input_sizes_data: [^]i64, input_sizes_len: c.int, dim: i64, start: i64, end: i64, step: i64) ---
	atg_slice_backward_out :: proc(out: ^Tensor, grad_output: Tensor, input_sizes_data: [^]i64, input_sizes_len: c.int, dim: i64, start: i64, end: i64, step: i64) ---
	atg_slice_copy :: proc(out: ^Tensor, self: Tensor, dim: i64, start_v: i64, start_null: rawptr, end_v: i64, end_null: rawptr, step: i64) ---
	atg_slice_copy_tensor_out :: proc(out: ^Tensor, self: Tensor, dim: i64, start_v: i64, start_null: rawptr, end_v: i64, end_null: rawptr, step: i64) ---
	atg_slice_inverse :: proc(out: ^Tensor, self: Tensor, src: Tensor, dim: i64, start_v: i64, start_null: rawptr, end_v: i64, end_null: rawptr, step: i64) ---
	atg_slice_scatter :: proc(out: ^Tensor, self: Tensor, src: Tensor, dim: i64, start_v: i64, start_null: rawptr, end_v: i64, end_null: rawptr, step: i64) ---
	atg_slice_scatter_out :: proc(out: ^Tensor, self: Tensor, src: Tensor, dim: i64, start_v: i64, start_null: rawptr, end_v: i64, end_null: rawptr, step: i64) ---
	atg_slogdet :: proc(out: ^Tensor, self: Tensor) ---
	atg_slogdet_out :: proc(out: ^Tensor, sign: Tensor, logabsdet: Tensor, self: Tensor) ---
	atg_slow_conv3d :: proc(out: ^Tensor, self: Tensor, weight: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int) ---
	atg_slow_conv3d_out :: proc(out: ^Tensor, self: Tensor, weight: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int) ---
	atg_slow_conv_dilated2d :: proc(out: ^Tensor, self: Tensor, weight: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int) ---
	atg_slow_conv_dilated2d_out :: proc(out: ^Tensor, self: Tensor, weight: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int) ---
	atg_slow_conv_dilated3d :: proc(out: ^Tensor, self: Tensor, weight: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int) ---
	atg_slow_conv_dilated3d_out :: proc(out: ^Tensor, self: Tensor, weight: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int) ---
	atg_slow_conv_transpose2d :: proc(out: ^Tensor, self: Tensor, weight: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, output_padding_data: [^]i64, output_padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int) ---
	atg_slow_conv_transpose2d_out :: proc(out: ^Tensor, self: Tensor, weight: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, output_padding_data: [^]i64, output_padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int) ---
	atg_slow_conv_transpose3d :: proc(out: ^Tensor, self: Tensor, weight: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, output_padding_data: [^]i64, output_padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int) ---
	atg_slow_conv_transpose3d_out :: proc(out: ^Tensor, self: Tensor, weight: Tensor, kernel_size_data: [^]i64, kernel_size_len: c.int, bias: Tensor, stride_data: [^]i64, stride_len: c.int, padding_data: [^]i64, padding_len: c.int, output_padding_data: [^]i64, output_padding_len: c.int, dilation_data: [^]i64, dilation_len: c.int) ---
	atg_smm :: proc(out: ^Tensor, self: Tensor, mat2: Tensor) ---
	atg_smooth_l1_loss :: proc(out: ^Tensor, self: Tensor, target: Tensor, reduction: i64, beta: f64) ---
	atg_smooth_l1_loss_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, target: Tensor, reduction: i64, beta: f64) ---
	atg_smooth_l1_loss_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, self: Tensor, target: Tensor, reduction: i64, beta: f64) ---
	atg_smooth_l1_loss_out :: proc(out: ^Tensor, self: Tensor, target: Tensor, reduction: i64, beta: f64) ---
	atg_soft_margin_loss :: proc(out: ^Tensor, self: Tensor, target: Tensor, reduction: i64) ---
	atg_soft_margin_loss_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, target: Tensor, reduction: i64) ---
	atg_soft_margin_loss_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, self: Tensor, target: Tensor, reduction: i64) ---
	atg_soft_margin_loss_out :: proc(out: ^Tensor, self: Tensor, target: Tensor, reduction: i64) ---
	atg_softmax :: proc(out: ^Tensor, self: Tensor, dim: i64, dtype: c.int) ---
	atg_softmax_int_out :: proc(out: ^Tensor, self: Tensor, dim: i64, dtype: c.int) ---
	atg_softplus :: proc(out: ^Tensor, self: Tensor) ---
	atg_softplus_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, beta: Scalar, threshold: Scalar) ---
	atg_softplus_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, self: Tensor, beta: Scalar, threshold: Scalar) ---
	atg_softplus_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_softshrink :: proc(out: ^Tensor, self: Tensor) ---
	atg_softshrink_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, lambd: Scalar) ---
	atg_softshrink_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, self: Tensor, lambd: Scalar) ---
	atg_softshrink_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_sort :: proc(out: ^Tensor, self: Tensor, dim: i64, descending: c.int) ---
	atg_sort_stable :: proc(out: ^Tensor, self: Tensor, stable: c.int, dim: i64, descending: c.int) ---
	atg_sort_values :: proc(out: ^Tensor, values: Tensor, indices: Tensor, self: Tensor, dim: i64, descending: c.int) ---
	atg_sort_values_stable :: proc(out: ^Tensor, values: Tensor, indices: Tensor, self: Tensor, stable: c.int, dim: i64, descending: c.int) ---
	atg_sparse_bsc_tensor :: proc(out: ^Tensor, ccol_indices: Tensor, row_indices: Tensor, values: Tensor, options_kind: c.int, options_device: c.int) ---
	atg_sparse_bsc_tensor_ccol_row_value_size :: proc(out: ^Tensor, ccol_indices: Tensor, row_indices: Tensor, values: Tensor, size_data: [^]i64, size_len: c.int, options_kind: c.int, options_device: c.int) ---
	atg_sparse_bsr_tensor :: proc(out: ^Tensor, crow_indices: Tensor, col_indices: Tensor, values: Tensor, options_kind: c.int, options_device: c.int) ---
	atg_sparse_bsr_tensor_crow_col_value_size :: proc(out: ^Tensor, crow_indices: Tensor, col_indices: Tensor, values: Tensor, size_data: [^]i64, size_len: c.int, options_kind: c.int, options_device: c.int) ---
	atg_sparse_compressed_tensor :: proc(out: ^Tensor, compressed_indices: Tensor, plain_indices: Tensor, values: Tensor, options_kind: c.int, options_device: c.int) ---
	atg_sparse_compressed_tensor_comp_plain_value_size :: proc(out: ^Tensor, compressed_indices: Tensor, plain_indices: Tensor, values: Tensor, size_data: [^]i64, size_len: c.int, options_kind: c.int, options_device: c.int) ---
	atg_sparse_coo_tensor :: proc(out: ^Tensor, size_data: [^]i64, size_len: c.int, options_kind: c.int, options_device: c.int) ---
	atg_sparse_coo_tensor_indices :: proc(out: ^Tensor, indices: Tensor, values: Tensor, options_kind: c.int, options_device: c.int, is_coalesced: c.int) ---
	atg_sparse_coo_tensor_indices_size :: proc(out: ^Tensor, indices: Tensor, values: Tensor, size_data: [^]i64, size_len: c.int, options_kind: c.int, options_device: c.int, is_coalesced: c.int) ---
	atg_sparse_coo_tensor_size_out :: proc(out: ^Tensor, size_data: [^]i64, size_len: c.int) ---
	atg_sparse_csc_tensor :: proc(out: ^Tensor, ccol_indices: Tensor, row_indices: Tensor, values: Tensor, options_kind: c.int, options_device: c.int) ---
	atg_sparse_csc_tensor_ccol_row_value_size :: proc(out: ^Tensor, ccol_indices: Tensor, row_indices: Tensor, values: Tensor, size_data: [^]i64, size_len: c.int, options_kind: c.int, options_device: c.int) ---
	atg_sparse_csr_tensor :: proc(out: ^Tensor, crow_indices: Tensor, col_indices: Tensor, values: Tensor, options_kind: c.int, options_device: c.int) ---
	atg_sparse_csr_tensor_crow_col_value_size :: proc(out: ^Tensor, crow_indices: Tensor, col_indices: Tensor, values: Tensor, size_data: [^]i64, size_len: c.int, options_kind: c.int, options_device: c.int) ---
	atg_sparse_mask :: proc(out: ^Tensor, self: Tensor, mask: Tensor) ---
	atg_sparse_mask_out :: proc(out: ^Tensor, self: Tensor, mask: Tensor) ---
	atg_sparse_resize :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int, sparse_dim: i64, dense_dim: i64) ---
	atg_sparse_resize_ :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int, sparse_dim: i64, dense_dim: i64) ---
	atg_sparse_resize_and_clear :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int, sparse_dim: i64, dense_dim: i64) ---
	atg_sparse_resize_and_clear_ :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int, sparse_dim: i64, dense_dim: i64) ---
	atg_sparse_resize_and_clear_out :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int, sparse_dim: i64, dense_dim: i64) ---
	atg_sparse_resize_out :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int, sparse_dim: i64, dense_dim: i64) ---
	atg_sparse_sampled_addmm :: proc(out: ^Tensor, self: Tensor, mat1: Tensor, mat2: Tensor) ---
	atg_sparse_sampled_addmm_out :: proc(out: ^Tensor, self: Tensor, mat1: Tensor, mat2: Tensor) ---
	atg_special_airy_ai :: proc(out: ^Tensor, x: Tensor) ---
	atg_special_airy_ai_out :: proc(out: ^Tensor, x: Tensor) ---
	atg_special_bessel_j0 :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_bessel_j0_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_bessel_j1 :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_bessel_j1_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_bessel_y0 :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_bessel_y0_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_bessel_y1 :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_bessel_y1_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_chebyshev_polynomial_t :: proc(out: ^Tensor, x: Tensor, n: Tensor) ---
	atg_special_chebyshev_polynomial_t_n_scalar :: proc(out: ^Tensor, x: Tensor, n: Scalar) ---
	atg_special_chebyshev_polynomial_t_n_scalar_out :: proc(out: ^Tensor, x: Tensor, n: Scalar) ---
	atg_special_chebyshev_polynomial_t_out :: proc(out: ^Tensor, x: Tensor, n: Tensor) ---
	atg_special_chebyshev_polynomial_t_x_scalar :: proc(out: ^Tensor, x: Scalar, n: Tensor) ---
	atg_special_chebyshev_polynomial_t_x_scalar_out :: proc(out: ^Tensor, x: Scalar, n: Tensor) ---
	atg_special_chebyshev_polynomial_u :: proc(out: ^Tensor, x: Tensor, n: Tensor) ---
	atg_special_chebyshev_polynomial_u_n_scalar :: proc(out: ^Tensor, x: Tensor, n: Scalar) ---
	atg_special_chebyshev_polynomial_u_n_scalar_out :: proc(out: ^Tensor, x: Tensor, n: Scalar) ---
	atg_special_chebyshev_polynomial_u_out :: proc(out: ^Tensor, x: Tensor, n: Tensor) ---
	atg_special_chebyshev_polynomial_u_x_scalar :: proc(out: ^Tensor, x: Scalar, n: Tensor) ---
	atg_special_chebyshev_polynomial_u_x_scalar_out :: proc(out: ^Tensor, x: Scalar, n: Tensor) ---
	atg_special_chebyshev_polynomial_v :: proc(out: ^Tensor, x: Tensor, n: Tensor) ---
	atg_special_chebyshev_polynomial_v_n_scalar :: proc(out: ^Tensor, x: Tensor, n: Scalar) ---
	atg_special_chebyshev_polynomial_v_n_scalar_out :: proc(out: ^Tensor, x: Tensor, n: Scalar) ---
	atg_special_chebyshev_polynomial_v_out :: proc(out: ^Tensor, x: Tensor, n: Tensor) ---
	atg_special_chebyshev_polynomial_v_x_scalar :: proc(out: ^Tensor, x: Scalar, n: Tensor) ---
	atg_special_chebyshev_polynomial_v_x_scalar_out :: proc(out: ^Tensor, x: Scalar, n: Tensor) ---
	atg_special_chebyshev_polynomial_w :: proc(out: ^Tensor, x: Tensor, n: Tensor) ---
	atg_special_chebyshev_polynomial_w_n_scalar :: proc(out: ^Tensor, x: Tensor, n: Scalar) ---
	atg_special_chebyshev_polynomial_w_n_scalar_out :: proc(out: ^Tensor, x: Tensor, n: Scalar) ---
	atg_special_chebyshev_polynomial_w_out :: proc(out: ^Tensor, x: Tensor, n: Tensor) ---
	atg_special_chebyshev_polynomial_w_x_scalar :: proc(out: ^Tensor, x: Scalar, n: Tensor) ---
	atg_special_chebyshev_polynomial_w_x_scalar_out :: proc(out: ^Tensor, x: Scalar, n: Tensor) ---
	atg_special_digamma :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_digamma_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_entr :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_entr_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_erf :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_erf_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_erfc :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_erfc_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_erfcx :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_erfcx_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_erfinv :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_erfinv_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_exp2 :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_exp2_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_expit :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_expit_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_expm1 :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_expm1_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_gammainc :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_special_gammainc_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_special_gammaincc :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_special_gammaincc_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_special_gammaln :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_gammaln_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_hermite_polynomial_h :: proc(out: ^Tensor, x: Tensor, n: Tensor) ---
	atg_special_hermite_polynomial_h_n_scalar :: proc(out: ^Tensor, x: Tensor, n: Scalar) ---
	atg_special_hermite_polynomial_h_n_scalar_out :: proc(out: ^Tensor, x: Tensor, n: Scalar) ---
	atg_special_hermite_polynomial_h_out :: proc(out: ^Tensor, x: Tensor, n: Tensor) ---
	atg_special_hermite_polynomial_h_x_scalar :: proc(out: ^Tensor, x: Scalar, n: Tensor) ---
	atg_special_hermite_polynomial_h_x_scalar_out :: proc(out: ^Tensor, x: Scalar, n: Tensor) ---
	atg_special_hermite_polynomial_he :: proc(out: ^Tensor, x: Tensor, n: Tensor) ---
	atg_special_hermite_polynomial_he_n_scalar :: proc(out: ^Tensor, x: Tensor, n: Scalar) ---
	atg_special_hermite_polynomial_he_n_scalar_out :: proc(out: ^Tensor, x: Tensor, n: Scalar) ---
	atg_special_hermite_polynomial_he_out :: proc(out: ^Tensor, x: Tensor, n: Tensor) ---
	atg_special_hermite_polynomial_he_x_scalar :: proc(out: ^Tensor, x: Scalar, n: Tensor) ---
	atg_special_hermite_polynomial_he_x_scalar_out :: proc(out: ^Tensor, x: Scalar, n: Tensor) ---
	atg_special_i0 :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_i0_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_i0e :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_i0e_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_i1 :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_i1_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_i1e :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_i1e_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_laguerre_polynomial_l :: proc(out: ^Tensor, x: Tensor, n: Tensor) ---
	atg_special_laguerre_polynomial_l_n_scalar :: proc(out: ^Tensor, x: Tensor, n: Scalar) ---
	atg_special_laguerre_polynomial_l_n_scalar_out :: proc(out: ^Tensor, x: Tensor, n: Scalar) ---
	atg_special_laguerre_polynomial_l_out :: proc(out: ^Tensor, x: Tensor, n: Tensor) ---
	atg_special_laguerre_polynomial_l_x_scalar :: proc(out: ^Tensor, x: Scalar, n: Tensor) ---
	atg_special_laguerre_polynomial_l_x_scalar_out :: proc(out: ^Tensor, x: Scalar, n: Tensor) ---
	atg_special_legendre_polynomial_p :: proc(out: ^Tensor, x: Tensor, n: Tensor) ---
	atg_special_legendre_polynomial_p_n_scalar :: proc(out: ^Tensor, x: Tensor, n: Scalar) ---
	atg_special_legendre_polynomial_p_n_scalar_out :: proc(out: ^Tensor, x: Tensor, n: Scalar) ---
	atg_special_legendre_polynomial_p_out :: proc(out: ^Tensor, x: Tensor, n: Tensor) ---
	atg_special_legendre_polynomial_p_x_scalar :: proc(out: ^Tensor, x: Scalar, n: Tensor) ---
	atg_special_legendre_polynomial_p_x_scalar_out :: proc(out: ^Tensor, x: Scalar, n: Tensor) ---
	atg_special_log1p :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_log1p_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_log_ndtr :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_log_ndtr_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_log_softmax :: proc(out: ^Tensor, self: Tensor, dim: i64, dtype: c.int) ---
	atg_special_logit :: proc(out: ^Tensor, self: Tensor, eps_v: f64, eps_null: rawptr) ---
	atg_special_logit_out :: proc(out: ^Tensor, self: Tensor, eps_v: f64, eps_null: rawptr) ---
	atg_special_logsumexp :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, keepdim: c.int) ---
	atg_special_logsumexp_out :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, keepdim: c.int) ---
	atg_special_modified_bessel_i0 :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_modified_bessel_i0_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_modified_bessel_i1 :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_modified_bessel_i1_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_modified_bessel_k0 :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_modified_bessel_k0_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_modified_bessel_k1 :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_modified_bessel_k1_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_multigammaln :: proc(out: ^Tensor, self: Tensor, p: i64) ---
	atg_special_multigammaln_out :: proc(out: ^Tensor, self: Tensor, p: i64) ---
	atg_special_ndtr :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_ndtr_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_ndtri :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_ndtri_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_polygamma :: proc(out: ^Tensor, n: i64, self: Tensor) ---
	atg_special_polygamma_out :: proc(out: ^Tensor, n: i64, self: Tensor) ---
	atg_special_psi :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_psi_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_round :: proc(out: ^Tensor, self: Tensor, decimals: i64) ---
	atg_special_round_out :: proc(out: ^Tensor, self: Tensor, decimals: i64) ---
	atg_special_scaled_modified_bessel_k0 :: proc(out: ^Tensor, x: Tensor) ---
	atg_special_scaled_modified_bessel_k0_out :: proc(out: ^Tensor, x: Tensor) ---
	atg_special_scaled_modified_bessel_k1 :: proc(out: ^Tensor, x: Tensor) ---
	atg_special_scaled_modified_bessel_k1_out :: proc(out: ^Tensor, x: Tensor) ---
	atg_special_shifted_chebyshev_polynomial_t :: proc(out: ^Tensor, x: Tensor, n: Tensor) ---
	atg_special_shifted_chebyshev_polynomial_t_n_scalar :: proc(out: ^Tensor, x: Tensor, n: Scalar) ---
	atg_special_shifted_chebyshev_polynomial_t_n_scalar_out :: proc(out: ^Tensor, x: Tensor, n: Scalar) ---
	atg_special_shifted_chebyshev_polynomial_t_out :: proc(out: ^Tensor, x: Tensor, n: Tensor) ---
	atg_special_shifted_chebyshev_polynomial_t_x_scalar :: proc(out: ^Tensor, x: Scalar, n: Tensor) ---
	atg_special_shifted_chebyshev_polynomial_t_x_scalar_out :: proc(out: ^Tensor, x: Scalar, n: Tensor) ---
	atg_special_shifted_chebyshev_polynomial_u :: proc(out: ^Tensor, x: Tensor, n: Tensor) ---
	atg_special_shifted_chebyshev_polynomial_u_n_scalar :: proc(out: ^Tensor, x: Tensor, n: Scalar) ---
	atg_special_shifted_chebyshev_polynomial_u_n_scalar_out :: proc(out: ^Tensor, x: Tensor, n: Scalar) ---
	atg_special_shifted_chebyshev_polynomial_u_out :: proc(out: ^Tensor, x: Tensor, n: Tensor) ---
	atg_special_shifted_chebyshev_polynomial_u_x_scalar :: proc(out: ^Tensor, x: Scalar, n: Tensor) ---
	atg_special_shifted_chebyshev_polynomial_u_x_scalar_out :: proc(out: ^Tensor, x: Scalar, n: Tensor) ---
	atg_special_shifted_chebyshev_polynomial_v :: proc(out: ^Tensor, x: Tensor, n: Tensor) ---
	atg_special_shifted_chebyshev_polynomial_v_n_scalar :: proc(out: ^Tensor, x: Tensor, n: Scalar) ---
	atg_special_shifted_chebyshev_polynomial_v_n_scalar_out :: proc(out: ^Tensor, x: Tensor, n: Scalar) ---
	atg_special_shifted_chebyshev_polynomial_v_out :: proc(out: ^Tensor, x: Tensor, n: Tensor) ---
	atg_special_shifted_chebyshev_polynomial_v_x_scalar :: proc(out: ^Tensor, x: Scalar, n: Tensor) ---
	atg_special_shifted_chebyshev_polynomial_v_x_scalar_out :: proc(out: ^Tensor, x: Scalar, n: Tensor) ---
	atg_special_shifted_chebyshev_polynomial_w :: proc(out: ^Tensor, x: Tensor, n: Tensor) ---
	atg_special_shifted_chebyshev_polynomial_w_n_scalar :: proc(out: ^Tensor, x: Tensor, n: Scalar) ---
	atg_special_shifted_chebyshev_polynomial_w_n_scalar_out :: proc(out: ^Tensor, x: Tensor, n: Scalar) ---
	atg_special_shifted_chebyshev_polynomial_w_out :: proc(out: ^Tensor, x: Tensor, n: Tensor) ---
	atg_special_shifted_chebyshev_polynomial_w_x_scalar :: proc(out: ^Tensor, x: Scalar, n: Tensor) ---
	atg_special_shifted_chebyshev_polynomial_w_x_scalar_out :: proc(out: ^Tensor, x: Scalar, n: Tensor) ---
	atg_special_sinc :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_sinc_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_special_softmax :: proc(out: ^Tensor, self: Tensor, dim: i64, dtype: c.int) ---
	atg_special_spherical_bessel_j0 :: proc(out: ^Tensor, x: Tensor) ---
	atg_special_spherical_bessel_j0_out :: proc(out: ^Tensor, x: Tensor) ---
	atg_special_xlog1py :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_special_xlog1py_other_scalar :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_special_xlog1py_other_scalar_out :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_special_xlog1py_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_special_xlog1py_self_scalar :: proc(out: ^Tensor, self_scalar: Scalar, other: Tensor) ---
	atg_special_xlog1py_self_scalar_out :: proc(out: ^Tensor, self_scalar: Scalar, other: Tensor) ---
	atg_special_xlogy :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_special_xlogy_other_scalar :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_special_xlogy_other_scalar_out :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_special_xlogy_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_special_xlogy_self_scalar :: proc(out: ^Tensor, self_scalar: Scalar, other: Tensor) ---
	atg_special_xlogy_self_scalar_out :: proc(out: ^Tensor, self_scalar: Scalar, other: Tensor) ---
	atg_special_zeta :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_special_zeta_other_scalar :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_special_zeta_other_scalar_out :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_special_zeta_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_special_zeta_self_scalar :: proc(out: ^Tensor, self_scalar: Scalar, other: Tensor) ---
	atg_special_zeta_self_scalar_out :: proc(out: ^Tensor, self_scalar: Scalar, other: Tensor) ---
	atg_split_copy_tensor_out :: proc(out_data: ^Tensor, out_len: c.int, self: Tensor, split_size: i64, dim: i64) ---
	atg_split_with_sizes_copy_out :: proc(out_data: ^Tensor, out_len: c.int, self: Tensor, split_sizes_data: [^]i64, split_sizes_len: c.int, dim: i64) ---
	atg_sqrt :: proc(out: ^Tensor, self: Tensor) ---
	atg_sqrt_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_sqrt_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_square :: proc(out: ^Tensor, self: Tensor) ---
	atg_square_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_square_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_squeeze :: proc(out: ^Tensor, self: Tensor) ---
	atg_squeeze_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_squeeze_copy :: proc(out: ^Tensor, self: Tensor) ---
	atg_squeeze_copy_dim :: proc(out: ^Tensor, self: Tensor, dim: i64) ---
	atg_squeeze_copy_dim_out :: proc(out: ^Tensor, self: Tensor, dim: i64) ---
	atg_squeeze_copy_dims :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int) ---
	atg_squeeze_copy_dims_out :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int) ---
	atg_squeeze_copy_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_squeeze_dim :: proc(out: ^Tensor, self: Tensor, dim: i64) ---
	atg_squeeze_dim_ :: proc(out: ^Tensor, self: Tensor, dim: i64) ---
	atg_squeeze_dims :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int) ---
	atg_squeeze_dims_ :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int) ---
	atg_sspaddmm :: proc(out: ^Tensor, self: Tensor, mat1: Tensor, mat2: Tensor) ---
	atg_sspaddmm_out :: proc(out: ^Tensor, self: Tensor, mat1: Tensor, mat2: Tensor) ---
	atg_stack :: proc(out: ^Tensor, tensors_data: ^Tensor, tensors_len: c.int, dim: i64) ---
	atg_stack_out :: proc(out: ^Tensor, tensors_data: ^Tensor, tensors_len: c.int, dim: i64) ---
	atg_std :: proc(out: ^Tensor, self: Tensor, unbiased: c.int) ---
	atg_std_correction :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, correction: Scalar, keepdim: c.int) ---
	atg_std_correction_out :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, correction: Scalar, keepdim: c.int) ---
	atg_std_dim :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, unbiased: c.int, keepdim: c.int) ---
	atg_std_mean :: proc(out: ^Tensor, self: Tensor, unbiased: c.int) ---
	atg_std_mean_correction :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, correction: Scalar, keepdim: c.int) ---
	atg_std_mean_correction_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, correction: Scalar, keepdim: c.int) ---
	atg_std_mean_dim :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, unbiased: c.int, keepdim: c.int) ---
	atg_std_out :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, unbiased: c.int, keepdim: c.int) ---
	atg_stft :: proc(out: ^Tensor, self: Tensor, n_fft: i64, hop_length_v: i64, hop_length_null: rawptr, win_length_v: i64, win_length_null: rawptr, window: Tensor, normalized: c.int, onesided: c.int, return_complex: c.int, align_to_window: c.int) ---
	atg_stft_center :: proc(out: ^Tensor, self: Tensor, n_fft: i64, hop_length_v: i64, hop_length_null: rawptr, win_length_v: i64, win_length_null: rawptr, window: Tensor, center: c.int, pad_mode_ptr: cstring, pad_mode_len: c.int, normalized: c.int, onesided: c.int, return_complex: c.int, align_to_window: c.int) ---
	atg_sub :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_sub_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_sub_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_sub_scalar :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_sub_scalar_ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_sub_scalar_out :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_subtract :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_subtract_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_subtract_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_subtract_scalar :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_subtract_scalar_ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_sum :: proc(out: ^Tensor, self: Tensor, dtype: c.int) ---
	atg_sum_dim_intlist :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, keepdim: c.int, dtype: c.int) ---
	atg_sum_intlist_out :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, keepdim: c.int, dtype: c.int) ---
	atg_sum_out :: proc(out: ^Tensor, self: Tensor, dtype: c.int) ---
	atg_sum_to_size :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int) ---
	atg_svd :: proc(out: ^Tensor, self: Tensor, some: c.int, compute_uv: c.int) ---
	atg_svd_u :: proc(out: ^Tensor, U: Tensor, S: Tensor, V: Tensor, self: Tensor, some: c.int, compute_uv: c.int) ---
	atg_swapaxes :: proc(out: ^Tensor, self: Tensor, axis0: i64, axis1: i64) ---
	atg_swapaxes_ :: proc(out: ^Tensor, self: Tensor, axis0: i64, axis1: i64) ---
	atg_swapdims :: proc(out: ^Tensor, self: Tensor, dim0: i64, dim1: i64) ---
	atg_swapdims_ :: proc(out: ^Tensor, self: Tensor, dim0: i64, dim1: i64) ---
	atg_t :: proc(out: ^Tensor, self: Tensor) ---
	atg_t_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_t_copy :: proc(out: ^Tensor, self: Tensor) ---
	atg_t_copy_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_take :: proc(out: ^Tensor, self: Tensor, index: Tensor) ---
	atg_take_along_dim :: proc(out: ^Tensor, self: Tensor, indices: Tensor, dim_v: i64, dim_null: rawptr) ---
	atg_take_along_dim_out :: proc(out: ^Tensor, self: Tensor, indices: Tensor, dim_v: i64, dim_null: rawptr) ---
	atg_take_out :: proc(out: ^Tensor, self: Tensor, index: Tensor) ---
	atg_tan :: proc(out: ^Tensor, self: Tensor) ---
	atg_tan_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_tan_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_tanh :: proc(out: ^Tensor, self: Tensor) ---
	atg_tanh_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_tanh_backward :: proc(out: ^Tensor, grad_output: Tensor, output: Tensor) ---
	atg_tanh_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, output: Tensor) ---
	atg_tanh_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_tensordot :: proc(out: ^Tensor, self: Tensor, other: Tensor, dims_self_data: [^]i64, dims_self_len: c.int, dims_other_data: [^]i64, dims_other_len: c.int) ---
	atg_tensordot_out :: proc(out: ^Tensor, self: Tensor, other: Tensor, dims_self_data: [^]i64, dims_self_len: c.int, dims_other_data: [^]i64, dims_other_len: c.int) ---
	atg_threshold :: proc(out: ^Tensor, self: Tensor, threshold: Scalar, value: Scalar) ---
	atg_threshold_ :: proc(out: ^Tensor, self: Tensor, threshold: Scalar, value: Scalar) ---
	atg_threshold_backward :: proc(out: ^Tensor, grad_output: Tensor, self: Tensor, threshold: Scalar) ---
	atg_threshold_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, self: Tensor, threshold: Scalar) ---
	atg_threshold_out :: proc(out: ^Tensor, self: Tensor, threshold: Scalar, value: Scalar) ---
	atg_tile :: proc(out: ^Tensor, self: Tensor, dims_data: [^]i64, dims_len: c.int) ---
	atg_to :: proc(out: ^Tensor, self: Tensor, device: c.int) ---
	atg_to_dense :: proc(out: ^Tensor, self: Tensor, dtype: c.int, masked_grad: c.int) ---
	atg_to_dense_backward :: proc(out: ^Tensor, grad: Tensor, input: Tensor, masked_grad: c.int) ---
	atg_to_device :: proc(out: ^Tensor, self: Tensor, device: c.int, dtype: c.int, non_blocking: c.int, copy: c.int) ---
	atg_to_dtype :: proc(out: ^Tensor, self: Tensor, dtype: c.int, non_blocking: c.int, copy: c.int) ---
	atg_to_dtype_layout :: proc(out: ^Tensor, self: Tensor, options_kind: c.int, options_device: c.int, non_blocking: c.int, copy: c.int) ---
	atg_to_mkldnn :: proc(out: ^Tensor, self: Tensor, dtype: c.int) ---
	atg_to_mkldnn_backward :: proc(out: ^Tensor, grad: Tensor, input: Tensor) ---
	atg_to_mkldnn_out :: proc(out: ^Tensor, self: Tensor, dtype: c.int) ---
	atg_to_other :: proc(out: ^Tensor, self: Tensor, other: Tensor, non_blocking: c.int, copy: c.int) ---
	atg_to_padded_tensor :: proc(out: ^Tensor, self: Tensor, padding: f64, output_size_data: [^]i64, output_size_len: c.int) ---
	atg_to_padded_tensor_out :: proc(out: ^Tensor, self: Tensor, padding: f64, output_size_data: [^]i64, output_size_len: c.int) ---
	atg_to_sparse :: proc(out: ^Tensor, self: Tensor, layout: rawptr, blocksize_data: [^]i64, blocksize_len: c.int, dense_dim_v: i64, dense_dim_null: rawptr) ---
	atg_to_sparse_bsc :: proc(out: ^Tensor, self: Tensor, blocksize_data: [^]i64, blocksize_len: c.int, dense_dim_v: i64, dense_dim_null: rawptr) ---
	atg_to_sparse_bsr :: proc(out: ^Tensor, self: Tensor, blocksize_data: [^]i64, blocksize_len: c.int, dense_dim_v: i64, dense_dim_null: rawptr) ---
	atg_to_sparse_csc :: proc(out: ^Tensor, self: Tensor, dense_dim_v: i64, dense_dim_null: rawptr) ---
	atg_to_sparse_csr :: proc(out: ^Tensor, self: Tensor, dense_dim_v: i64, dense_dim_null: rawptr) ---
	atg_to_sparse_sparse_dim :: proc(out: ^Tensor, self: Tensor, sparse_dim: i64) ---
	atg_topk :: proc(out: ^Tensor, self: Tensor, k: i64, dim: i64, largest: c.int, sorted: c.int) ---
	atg_topk_values :: proc(out: ^Tensor, values: Tensor, indices: Tensor, self: Tensor, k: i64, dim: i64, largest: c.int, sorted: c.int) ---
	atg_totype :: proc(out: ^Tensor, self: Tensor, scalar_type: c.int) ---
	atg_trace :: proc(out: ^Tensor, self: Tensor) ---
	atg_trace_backward :: proc(out: ^Tensor, grad: Tensor, sizes_data: [^]i64, sizes_len: c.int) ---
	atg_trace_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_transpose :: proc(out: ^Tensor, self: Tensor, dim0: i64, dim1: i64) ---
	atg_transpose_ :: proc(out: ^Tensor, self: Tensor, dim0: i64, dim1: i64) ---
	atg_transpose_copy :: proc(out: ^Tensor, self: Tensor, dim0: i64, dim1: i64) ---
	atg_transpose_copy_int_out :: proc(out: ^Tensor, self: Tensor, dim0: i64, dim1: i64) ---
	atg_trapezoid :: proc(out: ^Tensor, y: Tensor, dim: i64) ---
	atg_trapezoid_x :: proc(out: ^Tensor, y: Tensor, x: Tensor, dim: i64) ---
	atg_trapz :: proc(out: ^Tensor, y: Tensor, x: Tensor, dim: i64) ---
	atg_trapz_dx :: proc(out: ^Tensor, y: Tensor, dx: f64, dim: i64) ---
	atg_triangular_solve :: proc(out: ^Tensor, self: Tensor, A: Tensor, upper: c.int, transpose: c.int, unitriangular: c.int) ---
	atg_triangular_solve_x :: proc(out: ^Tensor, X: Tensor, M: Tensor, self: Tensor, A: Tensor, upper: c.int, transpose: c.int, unitriangular: c.int) ---
	atg_tril :: proc(out: ^Tensor, self: Tensor, diagonal: i64) ---
	atg_tril_ :: proc(out: ^Tensor, self: Tensor, diagonal: i64) ---
	atg_tril_indices :: proc(out: ^Tensor, row: i64, col: i64, offset: i64, options_kind: c.int, options_device: c.int) ---
	atg_tril_indices_out :: proc(out: ^Tensor, row: i64, col: i64, offset: i64) ---
	atg_tril_out :: proc(out: ^Tensor, self: Tensor, diagonal: i64) ---
	atg_triplet_margin_loss :: proc(out: ^Tensor, anchor: Tensor, positive: Tensor, negative: Tensor, margin: f64, p: f64, eps: f64, swap: c.int, reduction: i64) ---
	atg_triu :: proc(out: ^Tensor, self: Tensor, diagonal: i64) ---
	atg_triu_ :: proc(out: ^Tensor, self: Tensor, diagonal: i64) ---
	atg_triu_indices :: proc(out: ^Tensor, row: i64, col: i64, offset: i64, options_kind: c.int, options_device: c.int) ---
	atg_triu_indices_out :: proc(out: ^Tensor, row: i64, col: i64, offset: i64) ---
	atg_triu_out :: proc(out: ^Tensor, self: Tensor, diagonal: i64) ---
	atg_true_divide :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_true_divide_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_true_divide_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_true_divide_scalar :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_true_divide_scalar_ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_trunc :: proc(out: ^Tensor, self: Tensor) ---
	atg_trunc_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_trunc_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_type_as :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_unbind_copy_int_out :: proc(out_data: ^Tensor, out_len: c.int, self: Tensor, dim: i64) ---
	atg_unflatten :: proc(out: ^Tensor, self: Tensor, dim: i64, sizes_data: [^]i64, sizes_len: c.int) ---
	atg_unfold :: proc(out: ^Tensor, self: Tensor, dimension: i64, size: i64, step: i64) ---
	atg_unfold_backward :: proc(out: ^Tensor, grad_in: Tensor, input_sizes_data: [^]i64, input_sizes_len: c.int, dim: i64, size: i64, step: i64) ---
	atg_unfold_backward_out :: proc(out: ^Tensor, grad_in: Tensor, input_sizes_data: [^]i64, input_sizes_len: c.int, dim: i64, size: i64, step: i64) ---
	atg_unfold_copy :: proc(out: ^Tensor, self: Tensor, dimension: i64, size: i64, step: i64) ---
	atg_unfold_copy_out :: proc(out: ^Tensor, self: Tensor, dimension: i64, size: i64, step: i64) ---
	atg_uniform :: proc(out: ^Tensor, self: Tensor, from: f64, to: f64) ---
	atg_uniform_ :: proc(out: ^Tensor, self: Tensor, from: f64, to: f64) ---
	atg_uniform_out :: proc(out: ^Tensor, self: Tensor, from: f64, to: f64) ---
	atg_unique_consecutive :: proc(out: ^Tensor, self: Tensor, return_inverse: c.int, return_counts: c.int, dim_v: i64, dim_null: rawptr) ---
	atg_unique_consecutive_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, out2: Tensor, self: Tensor, return_inverse: c.int, return_counts: c.int, dim_v: i64, dim_null: rawptr) ---
	atg_unique_dim :: proc(out: ^Tensor, self: Tensor, dim: i64, sorted: c.int, return_inverse: c.int, return_counts: c.int) ---
	atg_unique_dim_consecutive :: proc(out: ^Tensor, self: Tensor, dim: i64, return_inverse: c.int, return_counts: c.int) ---
	atg_unique_dim_consecutive_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, out2: Tensor, self: Tensor, dim: i64, return_inverse: c.int, return_counts: c.int) ---
	atg_unique_dim_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, out2: Tensor, self: Tensor, dim: i64, sorted: c.int, return_inverse: c.int, return_counts: c.int) ---
	atg_unsafe_split_tensor_out :: proc(out_data: ^Tensor, out_len: c.int, self: Tensor, split_size: i64, dim: i64) ---
	atg_unsafe_split_with_sizes_out :: proc(out_data: ^Tensor, out_len: c.int, self: Tensor, split_sizes_data: [^]i64, split_sizes_len: c.int, dim: i64) ---
	atg_unsqueeze :: proc(out: ^Tensor, self: Tensor, dim: i64) ---
	atg_unsqueeze_ :: proc(out: ^Tensor, self: Tensor, dim: i64) ---
	atg_unsqueeze_copy :: proc(out: ^Tensor, self: Tensor, dim: i64) ---
	atg_unsqueeze_copy_out :: proc(out: ^Tensor, self: Tensor, dim: i64) ---
	atg_upsample_bicubic2d :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int, align_corners: c.int, scales_h_v: f64, scales_h_null: rawptr, scales_w_v: f64, scales_w_null: rawptr) ---
	atg_upsample_bicubic2d_backward :: proc(out: ^Tensor, grad_output: Tensor, output_size_data: [^]i64, output_size_len: c.int, input_size_data: [^]i64, input_size_len: c.int, align_corners: c.int, scales_h_v: f64, scales_h_null: rawptr, scales_w_v: f64, scales_w_null: rawptr) ---
	atg_upsample_bicubic2d_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, output_size_data: [^]i64, output_size_len: c.int, input_size_data: [^]i64, input_size_len: c.int, align_corners: c.int, scales_h_v: f64, scales_h_null: rawptr, scales_w_v: f64, scales_w_null: rawptr) ---
	atg_upsample_bicubic2d_out :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int, align_corners: c.int, scales_h_v: f64, scales_h_null: rawptr, scales_w_v: f64, scales_w_null: rawptr) ---
	atg_upsample_bicubic2d_vec :: proc(out: ^Tensor, input: Tensor, output_size_data: [^]i64, output_size_len: c.int, align_corners: c.int, scale_factors_data: [^]f64, scale_factors_len: c.int) ---
	atg_upsample_bilinear2d :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int, align_corners: c.int, scales_h_v: f64, scales_h_null: rawptr, scales_w_v: f64, scales_w_null: rawptr) ---
	atg_upsample_bilinear2d_backward :: proc(out: ^Tensor, grad_output: Tensor, output_size_data: [^]i64, output_size_len: c.int, input_size_data: [^]i64, input_size_len: c.int, align_corners: c.int, scales_h_v: f64, scales_h_null: rawptr, scales_w_v: f64, scales_w_null: rawptr) ---
	atg_upsample_bilinear2d_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, output_size_data: [^]i64, output_size_len: c.int, input_size_data: [^]i64, input_size_len: c.int, align_corners: c.int, scales_h_v: f64, scales_h_null: rawptr, scales_w_v: f64, scales_w_null: rawptr) ---
	atg_upsample_bilinear2d_out :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int, align_corners: c.int, scales_h_v: f64, scales_h_null: rawptr, scales_w_v: f64, scales_w_null: rawptr) ---
	atg_upsample_bilinear2d_vec :: proc(out: ^Tensor, input: Tensor, output_size_data: [^]i64, output_size_len: c.int, align_corners: c.int, scale_factors_data: [^]f64, scale_factors_len: c.int) ---
	atg_upsample_bilinear2d_vec_out :: proc(out: ^Tensor, input: Tensor, output_size_data: [^]i64, output_size_len: c.int, align_corners: c.int, scale_factors_data: [^]f64, scale_factors_len: c.int) ---
	atg_upsample_linear1d :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int, align_corners: c.int, scales_v: f64, scales_null: rawptr) ---
	atg_upsample_linear1d_backward :: proc(out: ^Tensor, grad_output: Tensor, output_size_data: [^]i64, output_size_len: c.int, input_size_data: [^]i64, input_size_len: c.int, align_corners: c.int, scales_v: f64, scales_null: rawptr) ---
	atg_upsample_linear1d_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, output_size_data: [^]i64, output_size_len: c.int, input_size_data: [^]i64, input_size_len: c.int, align_corners: c.int, scales_v: f64, scales_null: rawptr) ---
	atg_upsample_linear1d_out :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int, align_corners: c.int, scales_v: f64, scales_null: rawptr) ---
	atg_upsample_linear1d_vec :: proc(out: ^Tensor, input: Tensor, output_size_data: [^]i64, output_size_len: c.int, align_corners: c.int, scale_factors_data: [^]f64, scale_factors_len: c.int) ---
	atg_upsample_nearest1d :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int, scales_v: f64, scales_null: rawptr) ---
	atg_upsample_nearest1d_backward :: proc(out: ^Tensor, grad_output: Tensor, output_size_data: [^]i64, output_size_len: c.int, input_size_data: [^]i64, input_size_len: c.int, scales_v: f64, scales_null: rawptr) ---
	atg_upsample_nearest1d_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, output_size_data: [^]i64, output_size_len: c.int, input_size_data: [^]i64, input_size_len: c.int, scales_v: f64, scales_null: rawptr) ---
	atg_upsample_nearest1d_out :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int, scales_v: f64, scales_null: rawptr) ---
	atg_upsample_nearest1d_vec :: proc(out: ^Tensor, input: Tensor, output_size_data: [^]i64, output_size_len: c.int, scale_factors_data: [^]f64, scale_factors_len: c.int) ---
	atg_upsample_nearest2d :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int, scales_h_v: f64, scales_h_null: rawptr, scales_w_v: f64, scales_w_null: rawptr) ---
	atg_upsample_nearest2d_backward :: proc(out: ^Tensor, grad_output: Tensor, output_size_data: [^]i64, output_size_len: c.int, input_size_data: [^]i64, input_size_len: c.int, scales_h_v: f64, scales_h_null: rawptr, scales_w_v: f64, scales_w_null: rawptr) ---
	atg_upsample_nearest2d_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, output_size_data: [^]i64, output_size_len: c.int, input_size_data: [^]i64, input_size_len: c.int, scales_h_v: f64, scales_h_null: rawptr, scales_w_v: f64, scales_w_null: rawptr) ---
	atg_upsample_nearest2d_out :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int, scales_h_v: f64, scales_h_null: rawptr, scales_w_v: f64, scales_w_null: rawptr) ---
	atg_upsample_nearest2d_vec :: proc(out: ^Tensor, input: Tensor, output_size_data: [^]i64, output_size_len: c.int, scale_factors_data: [^]f64, scale_factors_len: c.int) ---
	atg_upsample_nearest2d_vec_out :: proc(out: ^Tensor, input: Tensor, output_size_data: [^]i64, output_size_len: c.int, scale_factors_data: [^]f64, scale_factors_len: c.int) ---
	atg_upsample_nearest3d :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int, scales_d_v: f64, scales_d_null: rawptr, scales_h_v: f64, scales_h_null: rawptr, scales_w_v: f64, scales_w_null: rawptr) ---
	atg_upsample_nearest3d_backward :: proc(out: ^Tensor, grad_output: Tensor, output_size_data: [^]i64, output_size_len: c.int, input_size_data: [^]i64, input_size_len: c.int, scales_d_v: f64, scales_d_null: rawptr, scales_h_v: f64, scales_h_null: rawptr, scales_w_v: f64, scales_w_null: rawptr) ---
	atg_upsample_nearest3d_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, output_size_data: [^]i64, output_size_len: c.int, input_size_data: [^]i64, input_size_len: c.int, scales_d_v: f64, scales_d_null: rawptr, scales_h_v: f64, scales_h_null: rawptr, scales_w_v: f64, scales_w_null: rawptr) ---
	atg_upsample_nearest3d_out :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int, scales_d_v: f64, scales_d_null: rawptr, scales_h_v: f64, scales_h_null: rawptr, scales_w_v: f64, scales_w_null: rawptr) ---
	atg_upsample_nearest3d_vec :: proc(out: ^Tensor, input: Tensor, output_size_data: [^]i64, output_size_len: c.int, scale_factors_data: [^]f64, scale_factors_len: c.int) ---
	atg_upsample_trilinear3d :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int, align_corners: c.int, scales_d_v: f64, scales_d_null: rawptr, scales_h_v: f64, scales_h_null: rawptr, scales_w_v: f64, scales_w_null: rawptr) ---
	atg_upsample_trilinear3d_backward :: proc(out: ^Tensor, grad_output: Tensor, output_size_data: [^]i64, output_size_len: c.int, input_size_data: [^]i64, input_size_len: c.int, align_corners: c.int, scales_d_v: f64, scales_d_null: rawptr, scales_h_v: f64, scales_h_null: rawptr, scales_w_v: f64, scales_w_null: rawptr) ---
	atg_upsample_trilinear3d_backward_grad_input :: proc(out: ^Tensor, grad_input: Tensor, grad_output: Tensor, output_size_data: [^]i64, output_size_len: c.int, input_size_data: [^]i64, input_size_len: c.int, align_corners: c.int, scales_d_v: f64, scales_d_null: rawptr, scales_h_v: f64, scales_h_null: rawptr, scales_w_v: f64, scales_w_null: rawptr) ---
	atg_upsample_trilinear3d_out :: proc(out: ^Tensor, self: Tensor, output_size_data: [^]i64, output_size_len: c.int, align_corners: c.int, scales_d_v: f64, scales_d_null: rawptr, scales_h_v: f64, scales_h_null: rawptr, scales_w_v: f64, scales_w_null: rawptr) ---
	atg_upsample_trilinear3d_vec :: proc(out: ^Tensor, input: Tensor, output_size_data: [^]i64, output_size_len: c.int, align_corners: c.int, scale_factors_data: [^]f64, scale_factors_len: c.int) ---
	atg_value_selecting_reduction_backward :: proc(out: ^Tensor, grad: Tensor, dim: i64, indices: Tensor, sizes_data: [^]i64, sizes_len: c.int, keepdim: c.int) ---
	atg_values :: proc(out: ^Tensor, self: Tensor) ---
	atg_values_copy :: proc(out: ^Tensor, self: Tensor) ---
	atg_values_copy_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_vander :: proc(out: ^Tensor, x: Tensor, n_v: i64, n_null: rawptr, increasing: c.int) ---
	atg_var :: proc(out: ^Tensor, self: Tensor, unbiased: c.int) ---
	atg_var_correction :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, correction: Scalar, keepdim: c.int) ---
	atg_var_correction_out :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, correction: Scalar, keepdim: c.int) ---
	atg_var_dim :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, unbiased: c.int, keepdim: c.int) ---
	atg_var_mean :: proc(out: ^Tensor, self: Tensor, unbiased: c.int) ---
	atg_var_mean_correction :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, correction: Scalar, keepdim: c.int) ---
	atg_var_mean_correction_out :: proc(out: ^Tensor, out0: Tensor, out1: Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, correction: Scalar, keepdim: c.int) ---
	atg_var_mean_dim :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, unbiased: c.int, keepdim: c.int) ---
	atg_var_out :: proc(out: ^Tensor, self: Tensor, dim_data: [^]i64, dim_len: c.int, unbiased: c.int, keepdim: c.int) ---
	atg_vdot :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_vdot_out :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_view :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int) ---
	atg_view_as :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_view_as_complex :: proc(out: ^Tensor, self: Tensor) ---
	atg_view_as_complex_copy :: proc(out: ^Tensor, self: Tensor) ---
	atg_view_as_complex_copy_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_view_as_real :: proc(out: ^Tensor, self: Tensor) ---
	atg_view_as_real_copy :: proc(out: ^Tensor, self: Tensor) ---
	atg_view_as_real_copy_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_view_copy :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int) ---
	atg_view_copy_dtype :: proc(out: ^Tensor, self: Tensor, dtype: c.int) ---
	atg_view_copy_dtype_out :: proc(out: ^Tensor, self: Tensor, dtype: c.int) ---
	atg_view_copy_out :: proc(out: ^Tensor, self: Tensor, size_data: [^]i64, size_len: c.int) ---
	atg_view_dtype :: proc(out: ^Tensor, self: Tensor, dtype: c.int) ---
	atg_vstack :: proc(out: ^Tensor, tensors_data: ^Tensor, tensors_len: c.int) ---
	atg_vstack_out :: proc(out: ^Tensor, tensors_data: ^Tensor, tensors_len: c.int) ---
	atg_where_scalar :: proc(out: ^Tensor, condition: Tensor, self_scalar: Scalar, other: Scalar) ---
	atg_where_scalarother :: proc(out: ^Tensor, condition: Tensor, self: Tensor, other: Scalar) ---
	atg_where_scalarself :: proc(out: ^Tensor, condition: Tensor, self_scalar: Scalar, other: Tensor) ---
	atg_where_self :: proc(out: ^Tensor, condition: Tensor, self: Tensor, other: Tensor) ---
	atg_where_self_out :: proc(out: ^Tensor, condition: Tensor, self: Tensor, other: Tensor) ---
	atg_xlogy :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_xlogy_ :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_xlogy_outscalar_other :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_xlogy_outscalar_self :: proc(out: ^Tensor, self_scalar: Scalar, other: Tensor) ---
	atg_xlogy_outtensor :: proc(out: ^Tensor, self: Tensor, other: Tensor) ---
	atg_xlogy_scalar_other :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_xlogy_scalar_other_ :: proc(out: ^Tensor, self: Tensor, other: Scalar) ---
	atg_xlogy_scalar_self :: proc(out: ^Tensor, self_scalar: Scalar, other: Tensor) ---
	atg_zero :: proc(out: ^Tensor, self: Tensor) ---
	atg_zero_ :: proc(out: ^Tensor, self: Tensor) ---
	atg_zero_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_zeros :: proc(out: ^Tensor, size_data: [^]i64, size_len: c.int, options_kind: c.int, options_device: c.int) ---
	atg_zeros_like :: proc(out: ^Tensor, self: Tensor) ---
	atg_zeros_like_out :: proc(out: ^Tensor, self: Tensor) ---
	atg_zeros_out :: proc(out: ^Tensor, size_data: [^]i64, size_len: c.int) ---
}