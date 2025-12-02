package atg

import "core:c"

// Opaque Handles ---
Tensor    :: distinct rawptr
Scalar    :: distinct rawptr
Optimizer :: distinct rawptr
Module    :: distinct rawptr
IValue    :: distinct rawptr

// Callback Types ---
LoadCallback :: proc "c" (data: rawptr, name: cstring, t: Tensor)

when ODIN_OS == .Windows {
    #panic("Windows is not supported yet")
    foreign import lib "libtorch_wrapper.dll" // TODO: Contributions welcome!
} else when ODIN_OS == .Darwin {
    foreign import lib "libtorch_wrapper.dylib"
} else {
    foreign import lib "libtorch_wrapper.so"
}

foreign lib {

    // Error Handling ---
    get_and_reset_last_err :: proc() -> cstring ---

    // Tensor Creation & Manipulation ---
    at_manual_seed        :: proc(seed: i64) ---
    at_new_tensor         :: proc() -> Tensor ---
    at_tensor_of_blob     :: proc(data: rawptr, dims: [^]i64, ndims: c.size_t, strides: [^]i64, nstrides: c.size_t, type: c.int, device: c.int) -> Tensor ---
    at_tensor_of_data     :: proc(vs: rawptr, dims: [^]i64, ndims: c.size_t, element_size_in_bytes: c.size_t, type: c.int) -> Tensor ---
    at_copy_data          :: proc(t: Tensor, vs: rawptr, numel: c.size_t, element_size_in_bytes: c.size_t) ---
    at_shallow_clone      :: proc(t: Tensor) -> Tensor ---

    // Tensor Properties & Access ---
    at_data_ptr           :: proc(t: Tensor) -> rawptr ---
    at_defined            :: proc(t: Tensor) -> c.int ---
    at_is_mkldnn          :: proc(t: Tensor) -> c.int ---
    at_is_sparse          :: proc(t: Tensor) -> c.int ---
    at_is_contiguous      :: proc(t: Tensor) -> c.int ---
    at_device             :: proc(t: Tensor) -> c.int ---
    at_dim                :: proc(t: Tensor) -> c.size_t ---
    at_shape              :: proc(t: Tensor, dims: [^]i64) ---
    at_stride             :: proc(t: Tensor, strides: [^]i64) ---
    at_scalar_type        :: proc(t: Tensor) -> c.int ---

    // AMP (Automatic Mixed Precision) ---
    at__amp_non_finite_check_and_unscale :: proc(t: Tensor, found_inf: Tensor, inv_scale: Tensor) ---
    at_autocast_clear_cache              :: proc() ---
    at_autocast_decrement_nesting        :: proc() -> c.int ---
    at_autocast_increment_nesting        :: proc() -> c.int ---
    at_autocast_is_enabled               :: proc() -> bool ---
    at_autocast_set_enabled              :: proc(b: bool) -> bool ---

    // Autograd ---
    at_backward           :: proc(t: Tensor, keep_graph: c.int, create_graph: c.int) ---
    at_requires_grad      :: proc(t: Tensor) -> c.int ---
    at_grad_set_enabled   :: proc(enabled: c.int) -> c.int ---
    at_run_backward       :: proc(tensors: [^]Tensor, ntensors: c.int, inputs: [^]Tensor, ninputs: c.int, outputs: [^]Tensor, keep_graph: c.int, create_graph: c.int) ---

    // Operations & Indexing ---
    at_get                :: proc(t: Tensor, index: c.int) -> Tensor ---
    at_fill_double        :: proc(t: Tensor, v: f64) ---
    at_fill_int64         :: proc(t: Tensor, v: i64) ---
    
    at_double_value_at_indexes      :: proc(t: Tensor, indexes: [^]i64, indexes_len: c.int) -> f64 ---
    at_int64_value_at_indexes       :: proc(t: Tensor, indexes: [^]i64, indexes_len: c.int) -> i64 ---
    at_set_double_value_at_indexes  :: proc(t: Tensor, indexes: [^]c.int, indexes_len: c.int, v: f64) ---
    at_set_int64_value_at_indexes   :: proc(t: Tensor, indexes: [^]c.int, indexes_len: c.int, v: i64) ---
    
    at_copy_              :: proc(dst: Tensor, src: Tensor) ---

    // I/O & Serialization ---
    at_print              :: proc(t: Tensor) ---
    at_to_string          :: proc(t: Tensor, line_size: c.int) -> cstring ---
    at_save               :: proc(t: Tensor, filename: cstring) ---
    at_save_to_stream     :: proc(t: Tensor, stream_ptr: rawptr) ---
    at_load               :: proc(filename: cstring) -> Tensor ---
    at_load_from_stream   :: proc(stream_ptr: rawptr) -> Tensor ---
    
    // Image Ops (libtorch specific) ---
    at_load_image         :: proc(filename: cstring) -> Tensor ---
    at_load_image_from_memory :: proc(img_data: [^]u8, img_size: c.size_t) -> Tensor ---
    at_save_image         :: proc(t: Tensor, filename: cstring) -> c.int ---
    at_resize_image       :: proc(t: Tensor, w: c.int, h: c.int) -> Tensor ---

    // Multi-Tensor I/O ---
    at_save_multi           :: proc(tensors: [^]Tensor, names: [^]cstring, ntensors: c.int, filename: cstring) ---
    at_save_multi_to_stream :: proc(tensors: [^]Tensor, names: [^]cstring, ntensors: c.int, stream_ptr: rawptr) ---
    at_load_multi           :: proc(tensors: [^]Tensor, names: [^]cstring, ntensors: c.int, filename: cstring) ---
    at_load_multi_          :: proc(tensors: [^]Tensor, names: [^]cstring, ntensors: c.int, filename: cstring) ---

    // Callback-based Loading ---
    at_loadz_callback             :: proc(filename: cstring, data: rawptr, f: LoadCallback) ---
    at_loadz_callback_with_device :: proc(filename: cstring, data: rawptr, f: LoadCallback, device_id: c.int) ---
    at_load_callback              :: proc(filename: cstring, data: rawptr, f: LoadCallback) ---
    at_load_callback_with_device  :: proc(filename: cstring, data: rawptr, f: LoadCallback, device_id: c.int) ---
    at_load_from_stream_callback  :: proc(stream_ptr: rawptr, data: rawptr, f: LoadCallback, enable_device_id: bool, device_id: c.int) ---

    // Threading & Config ---
    at_get_num_interop_threads :: proc() -> c.int ---
    at_get_num_threads         :: proc() -> c.int ---
    at_set_num_interop_threads :: proc(n_threads: c.int) ---
    at_set_num_threads         :: proc(n_threads: c.int) ---
    at_set_qengine             :: proc(qengine: c.int) ---
    at_free                    :: proc(t: Tensor) ---

    // Optimizers ---
    ato_adam            :: proc(lr, beta1, beta2, weight_decay, eps: f64, amsgrad: bool) -> Optimizer ---
    ato_adamw           :: proc(lr, beta1, beta2, weight_decay, eps: f64, amsgrad: bool) -> Optimizer ---
    ato_rms_prop        :: proc(lr, alpha, eps, weight_decay, momentum: f64, centered: c.int) -> Optimizer ---
    ato_sgd             :: proc(lr, momentum, dampening, weight_decay: f64, nesterov: c.int) -> Optimizer ---
    
    ato_add_parameters          :: proc(opt: Optimizer, t: Tensor, group: c.size_t) ---
    ato_set_learning_rate       :: proc(opt: Optimizer, lr: f64) ---
    ato_set_momentum            :: proc(opt: Optimizer, momentum: f64) ---
    ato_set_learning_rate_group :: proc(opt: Optimizer, group: c.size_t, lr: f64) ---
    ato_set_momentum_group      :: proc(opt: Optimizer, group: c.size_t, momentum: f64) ---
    ato_set_weight_decay        :: proc(opt: Optimizer, weight_decay: f64) ---
    ato_set_weight_decay_group  :: proc(opt: Optimizer, group: c.size_t, weight_decay: f64) ---
    ato_zero_grad               :: proc(opt: Optimizer) ---
    ato_step                    :: proc(opt: Optimizer) ---
    ato_free                    :: proc(opt: Optimizer) ---

    // Scalars ---
    ats_int       :: proc(v: i64) -> Scalar ---
    ats_float     :: proc(v: f64) -> Scalar ---
    ats_to_int    :: proc(s: Scalar) -> i64 ---
    ats_to_float  :: proc(s: Scalar) -> f64 ---
    ats_to_string :: proc(s: Scalar) -> cstring ---
    ats_free      :: proc(s: Scalar) ---

    // Context / Hardware Support ---
    at_context_has_openmp    :: proc() -> bool ---
    at_context_has_mkl       :: proc() -> bool ---
    at_context_has_lapack    :: proc() -> bool ---
    at_context_has_mkldnn    :: proc() -> bool ---
    at_context_has_magma     :: proc() -> bool ---
    at_context_has_cuda      :: proc() -> bool ---
    at_context_has_cudart    :: proc() -> bool ---
    at_context_has_cudnn     :: proc() -> bool ---
    at_context_version_cudnn :: proc() -> i64 ---
    at_context_version_cudart:: proc() -> i64 ---
    at_context_has_cusolver  :: proc() -> bool ---
    at_context_has_hip       :: proc() -> bool ---
    at_context_has_ipu       :: proc() -> bool ---
    at_context_has_xla       :: proc() -> bool ---
    at_context_has_lazy      :: proc() -> bool ---
    at_context_has_mps       :: proc() -> bool ---

    // CUDA Specifics ---
    atc_cuda_device_count      :: proc() -> c.int ---
    atc_cuda_is_available      :: proc() -> c.int ---
    atc_cudnn_is_available     :: proc() -> c.int ---
    atc_manual_seed            :: proc(seed: u64) ---
    atc_manual_seed_all        :: proc(seed: u64) ---
    atc_synchronize            :: proc(device_index: i64) ---
    atc_user_enabled_cudnn     :: proc() -> c.int ---
    atc_set_user_enabled_cudnn :: proc(b: c.int) ---
    atc_set_benchmark_cudnn    :: proc(b: c.int) ---

    // Modules (TorchScript) ---
    atm_load                   :: proc(filename: cstring) -> Module ---
    atm_load_on_device         :: proc(filename: cstring, device: c.int) -> Module ---
    atm_load_str               :: proc(data: cstring, sz: c.size_t) -> Module ---
    atm_load_str_on_device     :: proc(data: cstring, sz: c.size_t, device: c.int) -> Module ---
    atm_forward                :: proc(m: Module, tensors: [^]Tensor, ntensors: c.int) -> Tensor ---
    atm_forward_               :: proc(m: Module, ivalues: [^]IValue, nivalues: c.int) -> IValue ---
    atm_method                 :: proc(m: Module, method_name: cstring, tensors: [^]Tensor, ntensors: c.int) -> Tensor ---
    atm_method_                :: proc(m: Module, method_name: cstring, ivalues: [^]IValue, nivalues: c.int) -> IValue ---
    atm_create_class_          :: proc(m: Module, clz_name: cstring, ivalues: [^]IValue, nivalues: c.int) -> IValue ---
    atm_eval                   :: proc(m: Module) ---
    atm_train                  :: proc(m: Module) ---
    atm_free                   :: proc(m: Module) ---
    atm_to                     :: proc(m: Module, device: c.int, dtype: c.int, non_blocking: bool) ---
    atm_save                   :: proc(m: Module, filename: cstring) ---
    atm_get_profiling_mode     :: proc() -> c.int ---
    atm_set_profiling_mode     :: proc(mode: c.int) ---
    atm_fuser_cuda_set_enabled :: proc(enabled: bool) ---
    atm_fuser_cuda_is_enabled  :: proc() -> bool ---
    atm_named_parameters       :: proc(m: Module, data: rawptr, f: LoadCallback) ---

    // Tracing ---
    atm_create_for_tracing     :: proc(modl_name: cstring, inputs: [^]Tensor, ninputs: c.int) -> Module ---
    atm_end_tracing            :: proc(m: Module, fn_name: cstring, outputs: [^]Tensor, noutputs: c.int) ---

    // IValues (Intermediate Values) ---
    ati_none           :: proc() -> IValue ---
    ati_tensor         :: proc(t: Tensor) -> IValue ---
    ati_int            :: proc(v: i64) -> IValue ---
    ati_double         :: proc(v: f64) -> IValue ---
    ati_bool           :: proc(v: c.int) -> IValue ---
    ati_string         :: proc(s: cstring) -> IValue ---
    ati_tuple          :: proc(is: [^]IValue, n: c.int) -> IValue ---
    ati_generic_list   :: proc(is: [^]IValue, n: c.int) -> IValue ---
    ati_generic_dict   :: proc(is: [^]IValue, n: c.int) -> IValue ---
    ati_int_list       :: proc(v: [^]i64, n: c.int) -> IValue ---
    ati_double_list    :: proc(v: [^]f64, n: c.int) -> IValue ---
    ati_bool_list      :: proc(v: cstring, n: c.int) -> IValue --- // char* in C for bool list
    ati_string_list    :: proc(v: [^]cstring, n: c.int) -> IValue ---
    ati_tensor_list    :: proc(v: [^]Tensor, n: c.int) -> IValue ---
    ati_device         :: proc(d: c.int) -> IValue ---

    ati_to_tensor      :: proc(iv: IValue) -> Tensor ---
    ati_to_int         :: proc(iv: IValue) -> i64 ---
    ati_to_double      :: proc(iv: IValue) -> f64 ---
    ati_to_string      :: proc(iv: IValue) -> cstring ---
    ati_to_bool        :: proc(iv: IValue) -> c.int ---
    ati_length         :: proc(iv: IValue) -> c.int ---
    ati_tuple_length   :: proc(iv: IValue) -> c.int ---
    
    ati_to_tuple       :: proc(iv: IValue, outputs: [^]IValue, n: c.int) ---
    ati_to_generic_list:: proc(iv: IValue, outputs: [^]IValue, n: c.int) ---
    ati_to_generic_dict:: proc(iv: IValue, outputs: [^]IValue, n: c.int) ---
    ati_to_int_list    :: proc(iv: IValue, outputs: [^]i64, n: c.int) ---
    ati_to_double_list :: proc(iv: IValue, outputs: [^]f64, n: c.int) ---
    ati_to_bool_list   :: proc(iv: IValue, outputs: cstring, n: c.int) ---
    ati_to_tensor_list :: proc(iv: IValue, outputs: [^]Tensor, n: c.int) ---

    atm_set_tensor_expr_fuser_enabled :: proc(enabled: c.int) ---
    atm_get_tensor_expr_fuser_enabled :: proc() -> bool ---

    ati_tag            :: proc(iv: IValue) -> c.int ---
    ati_object_method_ :: proc(iv: IValue, name: cstring, args: [^]IValue, nargs: c.int) -> IValue ---
    ati_object_getattr_:: proc(iv: IValue, name: cstring) -> IValue ---
    ati_clone          :: proc(iv: IValue) -> IValue ---
    ati_free           :: proc(iv: IValue) ---

    // Graph Executor ---
    at_set_graph_executor_optimize :: proc(optimize: bool) ---
}

check_error :: proc() {
    err := get_and_reset_last_err()
    if err != nil {
        // Log or panic
        // Note: 'err' is a cstring that was strdup'ed in C++, 
        // depending on your wrapper implementation you might need to free it 
        // or the wrapper handles the buffer.
        // Assuming the wrapper gives us ownership or a buffer we read immediately:
        panic(string(err))
    }
}