package otorch

import "core:c"
import "core:fmt"
import "core:mem"
import "core:strings"
import t "atg"

Tensor      :: t.Tensor
Scalar      :: t.Scalar
Optimizer   :: t.Optimizer
Module      :: t.Module
IValue      :: t.IValue

DEFAULT_DTYPE :: ScalarType.Float
DEFAULT_DEVICE :: DeviceType.CPU

ScalarType :: enum i32 {
    Byte   = 0,   // uint8
    Char   = 1,   // int8
    Short  = 2,   // int16
    Int    = 3,   // int32
    Long   = 4,   // int64
    Half   = 5,   // float16
    Float  = 6,   // float32
    Double = 7,   // float64
}

Reduction :: enum i64 {
    None = 0,
    Mean = 1,
    Sum  = 2,
}

DeviceType :: enum i32 {
    CPU  = -1,
    CUDA = 0,
    MKLDNN = 1, // Reserved for explicit MKLDNN
    OPENGL = 2,
    OPENCL = 3,
    IDEEP = 4,
    HIP = 5,
    FPGA = 6,
    MAIA = 7,
    XLA = 8,
    Vulkan = 9,
    Metal = 10,
    XPU = 11,
    MPS = 12,
    Meta = 13,
    HPU = 14,
    VE = 15,
    Lazy = 16,
    IPU = 17,
    MTIA = 18,
    PrivateUse1 = 19,
}

// POOL MECHANICS

@thread_local private_tensor_pool: [dynamic]t.Tensor
@thread_local private_scalar_pool: [dynamic]t.Scalar
@thread_local private_optimizer_pool: [dynamic]t.Optimizer
@thread_local private_module_pool: [dynamic]t.Module
@thread_local private_ivalue_pool: [dynamic]t.IValue

track :: proc{track_tensor, track_scalar, track_optimizer, track_module, track_ivalue}

@(private)
track_tensor :: proc(raw: t.Tensor) -> t.Tensor {
    append(&private_tensor_pool, raw)
    return raw
}

@(private)
track_scalar :: proc(raw: t.Scalar) -> t.Scalar {
    append(&private_scalar_pool, raw)
    return raw
}

@(private)
track_optimizer :: proc(raw: t.Optimizer) -> t.Optimizer {
    append(&private_optimizer_pool, raw)
    return raw
}

@(private)
track_module :: proc(raw: t.Module) -> t.Module {
    append(&private_module_pool, raw)
    return raw
}

@(private)
track_ivalue :: proc(raw: t.IValue) -> t.IValue {
    append(&private_ivalue_pool, raw)
    return raw
}

// Keep a value from being freed at the end of pool scope
keep :: proc{keep_tensor, keep_scalar, keep_ivalue}

@(private)
keep_tensor :: proc(wrapped: Tensor) -> Tensor {
    raw_target := t.Tensor(wrapped)
    
    // Iterate backwards
    for i := len(private_tensor_pool) - 1; i >= 0; i -= 1 {
        if private_tensor_pool[i] == raw_target {
            // Swap-remove is O(1) and safe here since order of cleanup doesn't matter
            unordered_remove(&private_tensor_pool, i) 
            return wrapped
        }
    }
    return wrapped
}

@(private)
keep_scalar :: proc(wrapped: Scalar) -> Scalar {
    for i := len(private_scalar_pool) - 1; i >= 0; i -= 1 {
        if private_scalar_pool[i] == wrapped {
            unordered_remove(&private_scalar_pool, i)
            return wrapped
        }
    }
    return wrapped
}

@(private)
keep_ivalue :: proc(wrapped: IValue) -> IValue {
    raw_target := t.IValue(wrapped)
    for i := len(private_ivalue_pool) - 1; i >= 0; i -= 1 {
        if private_ivalue_pool[i] == raw_target {
            ordered_remove(&private_ivalue_pool, i)
            return wrapped
        }
    }
    return wrapped
}

// TODO: use handle nil slices gracefully
@(private)
_slice_args :: proc(s: []i64) -> ([^]i64, i32) {
    return raw_data(s), i32(len(s))
}

// Helper for Optional values
// eg raw binding expects (value: f64, null_ptr: rawptr) then:
// If we want to pass a value: we pass (value, nil)
// If we want to pass None: we pass (0.0, &dummy)
@(private)
_opt :: proc(val: Maybe(f64)) -> (f64, rawptr) {
    if v, ok := val.?; ok {
        return v, nil
    }
    @static dummy: u8 = 0
    return 0.0, &dummy
}

PoolState :: struct {
    tensor_len: int,
    scalar_len: int,
    ivalue_len: int,
}

pool_start :: proc() -> PoolState {
    return PoolState{
        tensor_len = len(private_tensor_pool),
        scalar_len = len(private_scalar_pool),
        ivalue_len = len(private_ivalue_pool),
    }
}

pool_end :: proc(state: PoolState) {

    // Clean up Tensors
    current_t_len := len(private_tensor_pool)
    for i := current_t_len - 1; i >= state.tensor_len; i -= 1 {
        raw := private_tensor_pool[i]
        if t.at_defined(raw) != 0 {
            t.at_free(raw)
        }
    }
    resize(&private_tensor_pool, state.tensor_len)

    // Clean up Scalars
    current_s_len := len(private_scalar_pool)
    for i := current_s_len - 1; i >= state.scalar_len; i -= 1 {
        raw := private_scalar_pool[i]
        t.ats_free(raw)
    }
    resize(&private_scalar_pool, state.scalar_len)

    // Clean up IValues
    current_iv_len := len(private_ivalue_pool)
    for i := current_iv_len - 1; i >= state.ivalue_len; i -= 1 {
        raw := private_ivalue_pool[i]
        t.ati_free(raw)
    }
    resize(&private_ivalue_pool, state.ivalue_len)
}

get_and_reset_last_err :: proc() -> string {
    return string(t.get_and_reset_last_err())
}

manual_seed :: proc(seed: i64) {
    t.at_manual_seed(seed)
}

defined :: proc(self: Tensor) -> i32 {
    return t.at_defined(self)
}

new_tensor :: proc() -> Tensor {
    out: Tensor = nil 
    return out 
}

free_tensor :: proc(self: Tensor) {
    if defined(self) != 0 {
        t.at_free(self)
    }
}

add :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_add(&out, self, t.Tensor(other))
    return track(out)
}

print :: proc(self: Tensor, label := "") {
    if label != "" {
        fmt.print(label, ": ")
    }
    t.at_print(self)
}

tensor_from_data :: proc(
    data: rawptr,
    dims: []i64,
    element_size: int,
    type: ScalarType
) -> Tensor {
    out := t.at_tensor_of_data(
        data,
        raw_data(dims),
        len(dims),
        uint(element_size),
        i32(type),
    )
    return track(out)
}

tensor_from_slice :: proc(data: []$T, dims: []i64) -> Tensor {
    // 1. Determine Type
    dtype: ScalarType
    when T == f32 { dtype = .Float }
    else when T == f64 { dtype = .Double }
    else when T == i32 { dtype = .Int }
    else when T == i64 { dtype = .Long }
    else when T == u8  { dtype = .Byte }
    else { fmt.panicf("Unsupported Tensor type: %v", typeid_of(T)) }
    
    out := t.at_tensor_of_data(
        raw_data(data),
        raw_data(dims),
        len(dims),
        size_of(T),
        i32(dtype),
    )
    return track(out)
}

// Helper: Get the scalar type (float, int, etc)
get_dtype :: proc(self: Tensor) -> ScalarType {
    return ScalarType(t.at_scalar_type(self))
}

tensor_to_slice :: proc(self: Tensor, $T: typeid, allocator := context.allocator) -> []T {
    n := numel(self)
    if n == 0 {
        return make([]T, 0, allocator)
    }

    // 1. Validation: Ensure T matches the Tensor's data type
    // This prevents interpreting float bits as int bits, etc.
    dtype := get_dtype(self)
    when T == f32 { assert(dtype == .Float, "Type mismatch: Expected f32 tensor") }
    else when T == f64 { assert(dtype == .Double, "Type mismatch: Expected f64 tensor") }
    else when T == i32 { assert(dtype == .Int, "Type mismatch: Expected i32 tensor") }
    else when T == i64 { assert(dtype == .Long, "Type mismatch: Expected i64 tensor") }
    else when T == u8  { assert(dtype == .Byte, "Type mismatch: Expected u8 tensor") }
    
    // 2. Contiguity: Ensure memory is laid out sequentially (C-order)
    // atg_contiguous returns a NEW tensor. 
    contig := contiguous(self)

    // 3. Pointers
    src_ptr := data_ptr(contig)
    
    // 4. Allocation
    res := make([]T, n, allocator)
    
    // 5. Fast Copy (Memcpy)
    // raw_data(res) gives us the pointer to the start of the slice
    byte_count := n * size_of(T)
    mem.copy(raw_data(res), src_ptr, byte_count)
    
    return res
}

numel :: proc(self: Tensor) -> int {
    ndim := dim(self)
    if ndim == 0 { 
        return 1 // Scalars have 1 element
    }
    // Alloc shape array
    s := make([]i64, ndim, context.temp_allocator)
    t.at_shape(self, raw_data(s))

    total := 1
    for size in s {
        total *= int(size)
    }
    return total
}

size_of_scalar :: proc(st: ScalarType) -> int {
    switch st {
    case .Byte, .Char:   return 1
    case .Short, .Half:  return 2
    case .Int, .Float:   return 4
    case .Long, .Double: return 8
    case: return 0
    }
}

// otorch.to(f32, my_tensor)
to :: proc($T: typeid, self: Tensor, non_blocking := false) -> Tensor {
    when T == f32 {
        return _cast_float(self, non_blocking)
    } else when T == f64 {
        return _cast_double(self, non_blocking)
    } else when T == i32 || T == int { 
        // NOTE: Torch 'Int' is strictly 32-bit.
        return _cast_int(self, non_blocking)
    } else when T == i64 {
        return _cast_long(self, non_blocking)
    } else when T == i16 {
        return _cast_short(self, non_blocking)
    } else when T == i8 {
        return _cast_char(self, non_blocking)
    } else when T == u8 {
        return _cast_byte(self, non_blocking)
    } else when T == f16 {
        return _cast_half(self, non_blocking)
    } else {
        fmt.panicf("otorch.to: Unsupported type %v", typeid_of(T))
    }
}

// Public API usage -> otorch.to_type(my_tensor, .Float)
to_type :: proc(self: Tensor, type: ScalarType, non_blocking := false) -> Tensor {
    switch type {
    case .Byte:   return cast_byte(self, non_blocking)
    case .Char:   return cast_char(self, non_blocking)
    case .Short:  return cast_short(self, non_blocking)
    case .Int:    return cast_int(self, non_blocking)
    case .Long:   return cast_long(self, non_blocking)
    case .Half:   return cast_half(self, non_blocking)
    case .Float:  return cast_float(self, non_blocking)
    case .Double: return cast_double(self, non_blocking)
    case: 
        fmt.println("Warning: Unknown ScalarType in cast")
        return self
    }
}

tensor_from_blob :: proc(
    data: rawptr,
    dims: []i64,
    strides: []i64,
    type: ScalarType,
    device: DeviceType = .CPU,
) -> Tensor {
    out := t.at_tensor_of_blob(
        data,
        raw_data(dims),
        c.size_t(len(dims)),
        raw_data(strides),
        c.size_t(len(strides)),
        i32(type),
        i32(device),
    )
    return track(out)
}

tensor_from_slice_blob :: proc(
    data: []$T, 
    dims: []i64, 
    device: DeviceType = .CPU
) -> Tensor {
    // 1. Infer ScalarType from $T
    dtype: ScalarType
    when T == f32 { dtype = .Float }
    else when T == f64 { dtype = .Double }
    else when T == i32 { dtype = .Int }
    else when T == i64 { dtype = .Long }
    else when T == i16 { dtype = .Short }
    else when T == i8  { dtype = .Char }
    else when T == u8  { dtype = .Byte }
    else {
        fmt.panicf("Unsupported Tensor type: %v", typeid_of(T))
    }

    // 2. Auto-calculate strides for contiguous memory (Row Major)
    // If your data is standard C-order contiguous, we can generate strides.
    // If you already have strides, use the raw 'tensor_from_blob' instead.
    
    // NOTE: We need to allocate strides to pass them to C. 
    // using context.temp_allocator ensures it is freed at end of frame/scope.
    strides := make([]i64, len(dims), context.temp_allocator)
    
    running_stride := i64(1)
    for i := len(dims) - 1; i >= 0; i -= 1 {
        strides[i] = running_stride
        running_stride *= dims[i]
    }

    // 3. Call raw wrapper
    return tensor_from_blob(
        raw_data(data),
        dims,
        strides,
        dtype,
        device,
    )
}

/* Runs a block of code and frees all temporary tensors created inside it:

    otorch.scoped(proc() {
        a := otorch.rand(10, 10)
        b := otorch.rand(10, 10)

        otorch.print(otorch.add(a, b))
        // Everything cleans up automatically here
    })
*/
scoped :: proc(block: proc()) {
    idx := pool_start()
    defer pool_end(idx)

    block()
}

// MEMORY & COPYING

copy_data :: proc(self: Tensor, data: rawptr, num_bytes: int) {
    t.at_copy_data(self, data, c.size_t(num_bytes), 1)
}

shallow_clone :: proc(self: Tensor) -> Tensor {
    // Returns a new tensor handle, so we must track it
    return track(t.at_shallow_clone(self))
}

// PROPERTIES & ACCESS

data_ptr :: proc(self: Tensor) -> rawptr {
    return t.at_data_ptr(self)
}

is_mkldnn :: proc(self: Tensor) -> bool {
    return t.at_is_mkldnn(self) != 0
}

is_sparse :: proc(self: Tensor) -> bool {
    return t.at_is_sparse(self) != 0
}

is_contiguous :: proc(self: Tensor) -> bool {
    return t.at_is_contiguous(self) != 0
}

device :: proc(self: Tensor) -> int {
    return int(t.at_device(self))
}

dim :: proc(self: Tensor) -> int {
    return int(t.at_dim(self))
}

scalar_type :: proc(self: Tensor) -> ScalarType {
    return ScalarType(t.at_scalar_type(self))
}

// Returns shape as a dynamic array or slice using context allocator
shape :: proc(self: Tensor) -> []i64 {
    ndim := dim(self)
    if ndim == 0 { return nil }
    
    // Alloc slice using context (temp or heap)
    s := make([]i64, ndim)
    t.at_shape(self, raw_data(s))
    return s
}

// Returns strides as a slice
stride :: proc(self: Tensor) -> []i64 {
    ndim := dim(self)
    if ndim == 0 { return nil }
    
    s := make([]i64, ndim)
    t.at_stride(self, raw_data(s))
    return s
}

// AMP AUTOMATIC MIXED PRECISION

amp_non_finite_check_and_unscale :: proc(self: Tensor, found_inf: Tensor, inv_scale: Tensor) {
    t.at__amp_non_finite_check_and_unscale(
        self, 
        t.Tensor(found_inf), 
        t.Tensor(inv_scale),
    )
}

autocast_clear_cache :: proc() {
    t.at_autocast_clear_cache()
}

autocast_decrement_nesting :: proc() -> int {
    return int(t.at_autocast_decrement_nesting())
}

autocast_increment_nesting :: proc() -> int {
    return int(t.at_autocast_increment_nesting())
}

autocast_is_enabled :: proc() -> bool {
    return t.at_autocast_is_enabled()
}

autocast_set_enabled :: proc(enabled: bool) -> bool {
    return t.at_autocast_set_enabled(enabled)
}

// AUTOGRAD

backward :: proc(self: Tensor, keep_graph := false, create_graph := false) {
    t.at_backward(
        self, 
        i32(keep_graph), 
        i32(create_graph),
    )
}

requires_grad :: proc(self: Tensor) -> bool {
    return t.at_requires_grad(self) != 0
}

grad_set_enabled :: proc(enabled: bool) -> bool {
    return t.at_grad_set_enabled(i32(enabled)) != 0
}

run_backward :: proc(
    tensors: []Tensor, 
    inputs:  []Tensor, 
    outputs: []Tensor = nil, 
    keep_graph := false, 
    create_graph := false
) {
    // Cast data ptr of []Tensor slice to [^]t.Tensor
    t.at_run_backward(
        cast([^]t.Tensor)raw_data(tensors), i32(len(tensors)),
        cast([^]t.Tensor)raw_data(inputs),  i32(len(inputs)),
        cast([^]t.Tensor)raw_data(outputs),
        i32(keep_graph),
        i32(create_graph),
    )
}

// OPERATIONS & INDEXING

// Returns a new tracked tensor
get :: proc(self: Tensor, index: int) -> Tensor {
    return track(t.at_get(self, i32(index)))
}

double_value_at :: proc(self: Tensor, indices: []i64) -> f64 {
    return t.at_double_value_at_indexes(
        self, 
        raw_data(indices), 
        i32(len(indices)),
    )
}

int64_value_at :: proc(self: Tensor, indices: []i64) -> i64 {
    return t.at_int64_value_at_indexes(
        self, 
        raw_data(indices), 
        i32(len(indices)),
    )
}

set_double_value_at :: proc(self: Tensor, indices: []i32, value: f64) {
    t.at_set_double_value_at_indexes(
        self, 
        cast([^]i32)raw_data(indices), 
        i32(len(indices)), 
        value,
    )
}

// I/O & SERIALIZATION

to_string :: proc(self: Tensor, line_size := 80) -> string {
    c_str := t.at_to_string(self, i32(line_size))
    return string(c_str) 
}

save :: proc(self: Tensor, filename: string) {
    t.at_save(self, strings.clone_to_cstring(filename, context.temp_allocator))
}

load :: proc(filename: string) -> Tensor {
    return track(t.at_load(strings.clone_to_cstring(filename, context.temp_allocator)))
}

// Multi-tensor IO
save_multi :: proc(tensors: []Tensor, names: []cstring, filename: string) {
    assert(len(tensors) == len(names))
    t.at_save_multi(
        cast([^]t.Tensor)raw_data(tensors), 
        raw_data(names), 
        i32(len(tensors)), 
        strings.clone_to_cstring(filename, context.temp_allocator),
    )
}

load_multi :: proc(tensors: []Tensor, names: []cstring, filename: string) {
    assert(len(tensors) == len(names))
    t.at_load_multi(
        cast([^]t.Tensor)raw_data(tensors), 
        raw_data(names), 
        i32(len(tensors)), 
        strings.clone_to_cstring(filename, context.temp_allocator),
    )
    // Auto-track the loaded tensors
    for tensor in tensors {
        track(tensor)
    }
}

// IMAGE OPS (Libtorch Specific)

load_image :: proc(filename: string) -> Tensor {
    return track(t.at_load_image(strings.clone_to_cstring(filename, context.temp_allocator)))
}

load_image_mem :: proc(data: []u8) -> Tensor {
    return track(t.at_load_image_from_memory(
        raw_data(data), 
        c.size_t(len(data)),
    ))
}

save_image :: proc(self: Tensor, filename: string) -> bool {
    res := t.at_save_image(self, strings.clone_to_cstring(filename, context.temp_allocator))
    return res == 1
}

resize_image :: proc(self: Tensor, w, h: int) -> Tensor {
    return track(t.at_resize_image(self, i32(w), i32(h)))
}

// SCALARS

scalar :: proc{scalar_int, scalar_i64, scalar_float}

@(private)
scalar_int :: proc(v: int) -> Scalar {
    return scalar_i64(i64(v))
}

@(private)
scalar_i64 :: proc(v: i64) -> Scalar {
    return track_scalar(t.ats_int(v))
}

@(private)
scalar_float :: proc(v: f64) -> Scalar {
    return track_scalar(t.ats_float(v))
}

scalar_to_int :: proc(s: Scalar) -> i64 {
    return t.ats_to_int(s)
}

scalar_to_float :: proc(s: Scalar) -> f64 {
    return t.ats_to_float(s)
}

scalar_to_string :: proc(s: Scalar) -> string {
    c_str := t.ats_to_string(s)
    return string(c_str)
}

scalar_free :: proc(s: Scalar) {
    t.ats_free(s)
}

// CONTEXT / HARDWARE SUPPORT

has_openmp :: proc() -> bool {
    return t.at_context_has_openmp()
}

has_mkl :: proc() -> bool {
    return t.at_context_has_mkl()
}

has_lapack :: proc() -> bool {
    return t.at_context_has_lapack()
}

has_mkldnn :: proc() -> bool {
    return t.at_context_has_mkldnn()
}

has_magma :: proc() -> bool {
    return t.at_context_has_magma()
}

has_cuda :: proc() -> bool {
    return t.at_context_has_cuda()
}

has_cudart :: proc() -> bool {
    return t.at_context_has_cudart()
}

has_cudnn :: proc() -> bool {
    return t.at_context_has_cudnn()
}

version_cudnn :: proc() -> i64 {
    return t.at_context_version_cudnn()
}

version_cudart :: proc() -> i64 {
    return t.at_context_version_cudart()
}

has_cusolver :: proc() -> bool {
    return t.at_context_has_cusolver()
}

has_hip :: proc() -> bool {
    return t.at_context_has_hip()
}

has_ipu :: proc() -> bool {
    return t.at_context_has_ipu()
}

has_xla :: proc() -> bool {
    return t.at_context_has_xla()
}

has_lazy :: proc() -> bool {
    return t.at_context_has_lazy()
}

has_mps :: proc() -> bool {
    return t.at_context_has_mps()
}

// CUDA Specifics

cuda_device_count :: proc() -> int {
    return int(t.atc_cuda_device_count())
}

cuda_is_available :: proc() -> bool {
    return t.atc_cuda_is_available() != 0
}

cudnn_is_available :: proc() -> bool {
    return t.atc_cudnn_is_available() != 0
}

manual_seed_cuda :: proc(seed: u64) {
    t.atc_manual_seed(seed)
}

manual_seed_all_cuda :: proc(seed: u64) {
    t.atc_manual_seed_all(seed)
}

synchronize_cuda :: proc(device_index: i64) {
    t.atc_synchronize(device_index)
}

user_enabled_cudnn :: proc() -> bool {
    return t.atc_user_enabled_cudnn() != 0
}

set_user_enabled_cudnn :: proc(b: bool) {
    t.atc_set_user_enabled_cudnn(i32(b))
}

set_benchmark_cudnn :: proc(b: bool) {
    t.atc_set_benchmark_cudnn(i32(b))
}

// OPTIMIZERS

o_adam :: proc(lr, beta1, beta2, weight_decay, eps: f64, amsgrad: bool) -> Optimizer {
    return track(t.ato_adam(lr, beta1, beta2, weight_decay, eps, amsgrad))
}

o_adamw :: proc(lr, beta1, beta2, weight_decay, eps: f64, amsgrad: bool) -> Optimizer {
    return track(t.ato_adamw(lr, beta1, beta2, weight_decay, eps, amsgrad))
}

o_rms_prop :: proc(lr, alpha, eps, weight_decay, momentum: f64, centered: bool) -> Optimizer {
    return track(t.ato_rms_prop(
        lr, alpha, eps, weight_decay, momentum, 
        i32(centered),
    ))
}

o_sgd :: proc(lr, momentum, dampening, weight_decay: f64, nesterov: bool) -> Optimizer {
    return track(t.ato_sgd(
        lr, momentum, dampening, weight_decay, 
        i32(nesterov),
    ))
}

o_add_parameters :: proc(opt: Optimizer, tensor: Tensor, group: int) {
    t.ato_add_parameters(opt, tensor, c.size_t(group))
}

o_set_learning_rate :: proc(opt: Optimizer, lr: f64) {
    t.ato_set_learning_rate(opt, lr)
}

o_set_momentum :: proc(opt: Optimizer, momentum: f64) {
    t.ato_set_momentum(opt, momentum)
}

o_set_learning_rate_group :: proc(opt: Optimizer, group: int, lr: f64) {
    t.ato_set_learning_rate_group(opt, c.size_t(group), lr)
}

o_set_momentum_group :: proc(opt: Optimizer, group: int, momentum: f64) {
    t.ato_set_momentum_group(opt, c.size_t(group), momentum)
}

o_set_weight_decay :: proc(opt: Optimizer, weight_decay: f64) {
    t.ato_set_weight_decay(opt, weight_decay)
}

o_set_weight_decay_group :: proc(opt: Optimizer, group: int, weight_decay: f64) {
    t.ato_set_weight_decay_group(opt, c.size_t(group), weight_decay)
}

o_zero_grad :: proc(opt: Optimizer) {
    t.ato_zero_grad(opt)
}

o_step :: proc(opt: Optimizer) {
    t.ato_step(opt)
}

o_free :: proc(opt: Optimizer) {
    t.ato_free(opt)
}

// MODULES / TorchScript

m_load :: proc(filename: string) -> Module {
    c_filename := strings.clone_to_cstring(filename, context.temp_allocator)
    return track(t.atm_load(c_filename))
}

m_load_on_device :: proc(filename: string, device: DeviceType = .CPU) -> Module {
    c_filename := strings.clone_to_cstring(filename, context.temp_allocator)
    return track(t.atm_load_on_device(c_filename, i32(device)))
}

m_load_str :: proc(data: string) -> Module {
    c_data := strings.clone_to_cstring(data, context.temp_allocator)
    return track(t.atm_load_str(c_data, c.size_t(len(data))))
}

m_load_str_on_device :: proc(data: string, device: DeviceType = .CPU) -> Module {
    c_data := strings.clone_to_cstring(data, context.temp_allocator)
    return track(t.atm_load_str_on_device(c_data, c.size_t(len(data)), i32(device)))
}

m_forward :: proc(m: Module, tensors: []Tensor) -> Tensor {
    // Pass raw data of slice and cast len to i32
    return track(t.atm_forward(m, raw_data(tensors), i32(len(tensors))))
}

m_forward_ivalues :: proc(m: Module, ivalues: []IValue) -> IValue {
    // Renamed from atm_forward_ to be more clear in Odin
    return track(t.atm_forward_(m, raw_data(ivalues), i32(len(ivalues))))
}

m_method :: proc(m: Module, method_name: string, tensors: []Tensor) -> Tensor {
    c_method := strings.clone_to_cstring(method_name, context.temp_allocator)
    return track(t.atm_method(m, c_method, raw_data(tensors), i32(len(tensors))))
}

m_method_ivalues :: proc(m: Module, method_name: string, ivalues: []IValue) -> IValue {
    c_method := strings.clone_to_cstring(method_name, context.temp_allocator)
    return track(t.atm_method_(m, c_method, raw_data(ivalues), i32(len(ivalues))))
}

m_create_class :: proc(m: Module, clz_name: string, ivalues: []IValue) -> IValue {
    c_name := strings.clone_to_cstring(clz_name, context.temp_allocator)
    return track(t.atm_create_class_(m, c_name, raw_data(ivalues), i32(len(ivalues))))
}

m_eval :: proc(m: Module) {
    t.atm_eval(m)
}

m_train :: proc(m: Module) {
    t.atm_train(m)
}

m_free :: proc(m: Module) {
    t.atm_free(m)
}

m_to :: proc(m: Module, device: DeviceType = .CPU, dtype: ScalarType = .Float, non_blocking: bool = false) {
    t.atm_to(m, i32(device), i32(dtype), non_blocking)
}

m_save :: proc(m: Module, filename: string) {
    c_filename := strings.clone_to_cstring(filename, context.temp_allocator)
    t.atm_save(m, c_filename)
}

m_get_profiling_mode :: proc() -> int {
    return int(t.atm_get_profiling_mode())
}

m_set_profiling_mode :: proc(mode: int) {
    t.atm_set_profiling_mode(i32(mode))
}

m_fuser_cuda_set_enabled :: proc(enabled: bool) {
    t.atm_fuser_cuda_set_enabled(enabled)
}

m_fuser_cuda_is_enabled :: proc() -> bool {
    return t.atm_fuser_cuda_is_enabled()
}

m_named_parameters :: proc(m: Module, data: rawptr, f: t.LoadCallback) {
    t.atm_named_parameters(m, data, f)
}

// TRACING

m_create_for_tracing :: proc(modl_name: string, inputs: []Tensor) -> Module {
    c_name := strings.clone_to_cstring(modl_name, context.temp_allocator)
    return track(t.atm_create_for_tracing(c_name, raw_data(inputs), i32(len(inputs))))
}

m_end_tracing :: proc(m: Module, fn_name: string, outputs: []Tensor) {
    c_name := strings.clone_to_cstring(fn_name, context.temp_allocator)
    t.atm_end_tracing(m, c_name, raw_data(outputs), i32(len(outputs)))
}

// IValue Constructors

i_none :: proc() -> IValue {
    return track(t.ati_none())
}

i_tensor :: proc(self: Tensor) -> IValue {
    return track(t.ati_tensor(self))
}

i_int :: proc(v: int) -> IValue {
    return track(t.ati_int(i64(v)))
}

i_double :: proc(v: f64) -> IValue {
    return track(t.ati_double(v))
}

i_bool :: proc(v: bool) -> IValue {
    val: i32 = 1 if v else 0
    return track(t.ati_bool(val))
}

i_string :: proc(s: string) -> IValue {
    c_str := strings.clone_to_cstring(s, context.temp_allocator)
    return track(t.ati_string(c_str))
}

i_tuple :: proc(items: []IValue) -> IValue {
    if len(items) == 0 { return track(t.ati_tuple(nil, 0)) }
    return track(t.ati_tuple(raw_data(items), i32(len(items))))
}

i_generic_list :: proc(items: []IValue) -> IValue {
    if len(items) == 0 { return track(t.ati_generic_list(nil, 0)) }
    return track(t.ati_generic_list(raw_data(items), i32(len(items))))
}

i_generic_dict :: proc(items: []IValue) -> IValue {
    if len(items) == 0 { return track(t.ati_generic_dict(nil, 0)) }
    return track(t.ati_generic_dict(raw_data(items), i32(len(items))))
}

i_int_list :: proc(v: []i64) -> IValue {
    if len(v) == 0 { return track(t.ati_int_list(nil, 0)) }
    return track(t.ati_int_list(raw_data(v), i32(len(v))))
}

i_double_list :: proc(v: []f64) -> IValue {
    if len(v) == 0 { return track(t.ati_double_list(nil, 0)) }
    return track(t.ati_double_list(raw_data(v), i32(len(v))))
}

i_bool_list :: proc(v: []bool) -> IValue {
    if len(v) == 0 { return track(t.ati_bool_list(nil, 0)) }
    
    temp_bytes := make([]u8, len(v), context.temp_allocator)
    for b, i in v {
        temp_bytes[i] = 1 if b else 0
    }
    // cast raw_data to cstring (char*)
    return track(t.ati_bool_list(cstring(raw_data(temp_bytes)), i32(len(v))))
}

i_string_list :: proc(v: []string) -> IValue {
    if len(v) == 0 { return track(t.ati_string_list(nil, 0)) }

    // Convert []string to []cstring
    temp_cstrings := make([]cstring, len(v), context.temp_allocator)
    for s, i in v {
        temp_cstrings[i] = strings.clone_to_cstring(s, context.temp_allocator)
    }

    return track(t.ati_string_list(raw_data(temp_cstrings), i32(len(v))))
}

i_tensor_list :: proc(v: []Tensor) -> IValue {
    if len(v) == 0 { return track(t.ati_tensor_list(nil, 0)) }
    return track(t.ati_tensor_list(raw_data(v), i32(len(v))))
}

i_device :: proc(d: int) -> IValue {
    return track(t.ati_device(i32(d)))
}

// ACCESSORS IValue

i_to_tensor :: proc(iv: IValue) -> Tensor {
    return track(t.ati_to_tensor(iv))
}

i_to_int :: proc(iv: IValue) -> int {
    return int(t.ati_to_int(iv))
}

i_to_double :: proc(iv: IValue) -> f64 {
    return t.ati_to_double(iv)
}

i_to_string :: proc(iv: IValue) -> string {
    c_str := t.ati_to_string(iv)
    return string(c_str)
}

i_to_bool :: proc(iv: IValue) -> bool {
    return t.ati_to_bool(iv) != 0
}

i_length :: proc(iv: IValue) -> int {
    return int(t.ati_length(iv))
}

i_tuple_length :: proc(iv: IValue) -> int {
    return int(t.ati_tuple_length(iv))
}

// EXTRACTORS IValue

// These functions automatically allocate the result slice using the
// context.allocator (standard heap) unless specified otherwise

i_to_tuple :: proc(iv: IValue, allocator := context.allocator) -> []IValue {
    n := t.ati_tuple_length(iv)
    if n <= 0 { return nil }
    
    res := make([]IValue, n, allocator)
    t.ati_to_tuple(iv, raw_data(res), n)
    
    for val in res { track(val) }
    return res
}

i_to_generic_list :: proc(iv: IValue, allocator := context.allocator) -> []IValue {
    n := t.ati_length(iv)
    if n <= 0 { return nil }

    res := make([]IValue, n, allocator)
    t.ati_to_generic_list(iv, raw_data(res), n)

    for val in res { track(val) }
    return res
}

i_to_int_list :: proc(iv: IValue, allocator := context.allocator) -> []i64 {
    n := t.ati_length(iv)
    if n <= 0 { return nil }

    res := make([]i64, n, allocator)
    t.ati_to_int_list(iv, raw_data(res), n)
    return res
}

i_to_double_list :: proc(iv: IValue, allocator := context.allocator) -> []f64 {
    n := t.ati_length(iv)
    if n <= 0 { return nil }

    res := make([]f64, n, allocator)
    t.ati_to_double_list(iv, raw_data(res), n)
    return res
}

i_to_bool_list :: proc(iv: IValue, allocator := context.allocator) -> []bool {
    n := t.ati_length(iv)
    if n <= 0 { return nil }

    // Torch returns char* (bytes), we want []bool
    // Get bytes
    bytes := make([]u8, n, context.temp_allocator)
    t.ati_to_bool_list(iv, cstring(raw_data(bytes)), n)

    // 2. Convert to bool
    res := make([]bool, n, allocator)
    for b, i in bytes {
        res[i] = (b != 0)
    }
    return res
}

i_to_tensor_list :: proc(iv: IValue, allocator := context.allocator) -> []Tensor {
    n := t.ati_length(iv)
    if n <= 0 { return nil }

    res := make([]Tensor, n, allocator)
    t.ati_to_tensor_list(iv, raw_data(res), n)

    for tensor in res { track(tensor) }
    return res
}

// OBJECT METHODS

i_tag :: proc(iv: IValue) -> int {
    return int(t.ati_tag(iv))
}

i_object_method :: proc(iv: IValue, name: string, args: []IValue) -> IValue {
    c_name := strings.clone_to_cstring(name, context.temp_allocator)
    
    // Handle empty args
    args_ptr: [^]IValue = nil
    if len(args) > 0 {
        args_ptr = raw_data(args)
    }

    return track(t.ati_object_method_(iv, c_name, args_ptr, i32(len(args))))
}

i_object_getattr :: proc(iv: IValue, name: string) -> IValue {
    c_name := strings.clone_to_cstring(name, context.temp_allocator)
    return track(t.ati_object_getattr_(iv, c_name))
}

i_clone :: proc(iv: IValue) -> IValue {
    return track(t.ati_clone(iv))
}

// BITWISE AND (&)

bitwise_and :: proc{
    bitwise_and_tensor, 
    bitwise_and_scalar, 
    bitwise_and_scalar_tensor,
}
bitwise_and_ :: proc{
    bitwise_and_tensor_, 
    bitwise_and_scalar_,
}

@(private)
bitwise_and_tensor :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_bitwise_and_tensor(&out, self, other)
    return track(out)
}

@(private)
bitwise_and_scalar :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_bitwise_and(&out, self, other)
    return track(out)
}

@(private)
bitwise_and_scalar_tensor :: proc(self: Scalar, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_bitwise_and_scalar_tensor(&out, self, other)
    return track(out)
}

@(private)
bitwise_and_tensor_ :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_bitwise_and_tensor_(&out, self, other)
    return self
}

@(private)
bitwise_and_scalar_ :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_bitwise_and_(&out, self, other)
    return self
}

// BITWISE NOT (~)

bitwise_not :: proc{bitwise_not_tensor}
bitwise_not_ :: proc{bitwise_not_tensor_}

@(private)
bitwise_not_tensor :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_bitwise_not(&out, self)
    return track(out)
}

@(private)
bitwise_not_tensor_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_bitwise_not_(&out, self)
    return self
}

// BITWISE OR (|)

bitwise_or :: proc{
    bitwise_or_tensor, 
    bitwise_or_scalar, 
    bitwise_or_scalar_tensor,
}

@(private)
bitwise_or_tensor :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_bitwise_or_tensor(&out, self, other)
    return track(out)
}

@(private)
bitwise_or_scalar :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_bitwise_or(&out, self, other)
    return track(out)
}

@(private)
bitwise_or_scalar_tensor :: proc(self: Scalar, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_bitwise_or_scalar_tensor(&out, self, other)
    return track(out)
}

// BITWISE XOR (^)

bitwise_xor :: proc{
    bitwise_xor_tensor, 
    bitwise_xor_scalar, 
    bitwise_xor_scalar_tensor,
}
bitwise_xor_ :: proc{
    bitwise_xor_tensor_, 
    bitwise_xor_scalar_,
}

@(private)
bitwise_xor_tensor :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_bitwise_xor_tensor(&out, self, other)
    return track(out)
}

@(private)
bitwise_xor_scalar :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_bitwise_xor(&out, self, other)
    return track(out)
}

@(private)
bitwise_xor_scalar_tensor :: proc(self: Scalar, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_bitwise_xor_scalar_tensor(&out, self, other)
    return track(out)
}

@(private)
bitwise_xor_tensor_ :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_bitwise_xor_tensor_(&out, self, other)
    return self
}

@(private)
bitwise_xor_scalar_ :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_bitwise_xor_(&out, self, other)
    return self
}

// BITWISE LEFT SHIFT (<<)

bitwise_left_shift :: proc{
    bitwise_left_shift_tensor, 
    bitwise_left_shift_tensor_scalar, 
    bitwise_left_shift_scalar_tensor,
}
bitwise_left_shift_ :: proc{
    bitwise_left_shift_tensor_, 
    bitwise_left_shift_tensor_scalar_,
}

@(private)
bitwise_left_shift_tensor :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_bitwise_left_shift(&out, self, other)
    return track(out)
}

@(private)
bitwise_left_shift_tensor_scalar :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_bitwise_left_shift_tensor_scalar(&out, self, other)
    return track(out)
}

@(private)
bitwise_left_shift_scalar_tensor :: proc(self: Scalar, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_bitwise_left_shift_scalar_tensor(&out, self, other)
    return track(out)
}

@(private)
bitwise_left_shift_tensor_ :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_bitwise_left_shift_(&out, self, other)
    return self
}

@(private)
bitwise_left_shift_tensor_scalar_ :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_bitwise_left_shift_tensor_scalar_(&out, self, other)
    return self
}

// BITWISE RIGHT SHIFT (>>)

bitwise_right_shift :: proc{
    bitwise_right_shift_tensor, 
    bitwise_right_shift_tensor_scalar, 
    bitwise_right_shift_scalar_tensor,
}
bitwise_right_shift_ :: proc{
    bitwise_right_shift_tensor_, 
    bitwise_right_shift_tensor_scalar_,
}

@(private)
bitwise_right_shift_tensor :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_bitwise_right_shift(&out, self, other)
    return track(out)
}

@(private)
bitwise_right_shift_tensor_scalar :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_bitwise_right_shift_tensor_scalar(&out, self, other)
    return track(out)
}

@(private)
bitwise_right_shift_scalar_tensor :: proc(self: Scalar, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_bitwise_right_shift_scalar_tensor(&out, self, other)
    return track(out)
}

@(private)
bitwise_right_shift_tensor_ :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_bitwise_right_shift_(&out, self, other)
    return self
}

@(private)
bitwise_right_shift_tensor_scalar_ :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_bitwise_right_shift_tensor_scalar_(&out, self, other)
    return self
}

// ADAPTIVE POOLING

adaptive_avg_pool2d :: proc(self: Tensor, output_size: []i64) -> Tensor {
    out: Tensor
    t.atg__adaptive_avg_pool2d(
        &out, 
        self, 
        raw_data(output_size), 
        i32(len(output_size)),
    )
    return track(out)
}

adaptive_avg_pool3d :: proc(self: Tensor, output_size: []i64) -> Tensor {
    out: Tensor
    t.atg__adaptive_avg_pool3d(
        &out, 
        self, 
        raw_data(output_size), 
        i32(len(output_size)),
    )
    return track(out)
}

// BATCH DIMENSION UTILS

// Often used internally to unsqueeze a specific dim for batching
add_batch_dim :: proc(self: Tensor, batch_dim, level: i64) -> Tensor {
    out: Tensor
    t.atg__add_batch_dim(&out, self, batch_dim, level)
    return track(out)
}

// FUSED ADD RELU

add_relu :: proc{add_relu_tensor, add_relu_scalar}
add_relu_ :: proc{add_relu_tensor_, add_relu_scalar_}

@(private)
add_relu_tensor :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg__add_relu(&out, self, other)
    return track(out)
}

@(private)
add_relu_scalar :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg__add_relu_scalar(&out, self, other)
    return track(out)
}

@(private)
add_relu_tensor_ :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg__add_relu_(&out, self, other)
    return self
}

@(private)
add_relu_scalar_ :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg__add_relu_scalar_(&out, self, other)
    return self
}

// ADDMM ACTIVATION

// Performs: activation(beta * self + alpha * (mat1 @ mat2))
// Commonly used for fused Linear + Activation layers
addmm_activation :: proc(self, mat1, mat2: Tensor, use_gelu: bool = false) -> Tensor {
    out: Tensor
    use_gelu_int := i32(1) if use_gelu else i32(0)
    t.atg__addmm_activation(&out, self, mat1, mat2, use_gelu_int)
    return track(out)
}

// AMINMAX (Fused Min/Max)

/* NOTE: _aminmax returns a tuple (min, max).
   Because handling C-struct tuple returns can be fragile, we implement the 
   standard version by creating two tensors and calling the `_out` variant.
   This ensures memory safety and correct tracking in the pool.
*/

aminmax :: proc(self: Tensor) -> (min, max: Tensor) {
    min = new_tensor()
    max = new_tensor()
    
    dummy: Tensor
    t.atg__aminmax_out(&dummy, min, max, self)
    
    return min, max
}

aminmax_dim :: proc(self: Tensor, dim: i64, keepdim: bool = false) -> (min, max: Tensor) {
    min = new_tensor()
    max = new_tensor()
    
    keep_int := i32(1) if keepdim else i32(0)
    
    dummy: Tensor
    t.atg__aminmax_dim_out(&dummy, min, max, self, dim, keep_int)
    
    return min, max
}

@(private)
cast_byte :: proc(self: Tensor, non_blocking := false) -> Tensor {
    out: Tensor
    t.atg__cast_byte(&out, self, i32(non_blocking))
    return track(out)
}

@(private)
cast_float :: proc(self: Tensor, non_blocking := false) -> Tensor {
    out: Tensor
    t.atg__cast_float(&out, self, i32(non_blocking))
    return track(out)
}

@(private)
cast_double :: proc(self: Tensor, non_blocking := false) -> Tensor {
    out: Tensor
    t.atg__cast_double(&out, self, i32(non_blocking))
    return track(out)
}

@(private)
cast_int :: proc(self: Tensor, non_blocking := false) -> Tensor {
    out: Tensor
    t.atg__cast_int(&out, self, i32(non_blocking))
    return track(out)
}

@(private)
cast_long :: proc(self: Tensor, non_blocking := false) -> Tensor {
    out: Tensor
    t.atg__cast_long(&out, self, i32(non_blocking))
    return track(out)
}

@(private)
cast_short :: proc(self: Tensor, non_blocking := false) -> Tensor {
    out: Tensor
    t.atg__cast_short(&out, self, i32(non_blocking))
    return track(out)
}

@(private)
cast_char :: proc(self: Tensor, non_blocking := false) -> Tensor {
    out: Tensor
    t.atg__cast_char(&out, self, i32(non_blocking))
    return track(out)
}

@(private)
cast_half :: proc(self: Tensor, non_blocking := false) -> Tensor {
    out: Tensor
    t.atg__cast_half(&out, self, i32(non_blocking))
    return track(out)
}

// AMP Automatic Mixed Precision

amp_update_scale :: proc(
    self: Tensor,
    growth_tracker: Tensor,
    found_inf: Tensor,
    scale_growth_factor: f64,
    scale_backoff_factor: f64,
    growth_interval: i64,
) -> Tensor {
    out: Tensor
    t.atg__amp_update_scale(&out, self, growth_tracker, found_inf, scale_growth_factor, scale_backoff_factor, growth_interval)
    return track(out)
}

amp_update_scale_ :: proc(
    self: Tensor,
    growth_tracker: Tensor,
    found_inf: Tensor,
    scale_growth_factor: f64,
    scale_backoff_factor: f64,
    growth_interval: i64,
) -> Tensor {
    out: Tensor
    t.atg__amp_update_scale_(&out, self, growth_tracker, found_inf, scale_growth_factor, scale_backoff_factor, growth_interval)
    return self
}

autocast_to_full_precision :: proc(self: Tensor, cuda_enabled: bool, cpu_enabled: bool) -> Tensor {
    out: Tensor
    t.atg__autocast_to_full_precision(
        &out, 
        self, 
        i32(1) if cuda_enabled else i32(0), 
        i32(1) if cpu_enabled else i32(0),
    )
    return track(out)
}

autocast_to_reduced_precision :: proc(
    self: Tensor, 
    cuda_enabled, cpu_enabled: bool, 
    cuda_dtype: ScalarType, 
    cpu_dtype: ScalarType
) -> Tensor {
    out: Tensor
    t.atg__autocast_to_reduced_precision(
        &out, 
        self, 
        i32(1) if cuda_enabled else i32(0), 
        i32(1) if cpu_enabled else i32(0),
        i32(cuda_dtype),
        i32(cpu_dtype),
    )
    return track(out)
}

// ASSERTIONS

assert_scalar :: proc(self: Scalar, msg: string) {
    c_msg := strings.clone_to_cstring(msg, context.temp_allocator)
    t.atg__assert_scalar(self, c_msg, i32(len(msg)))
}

assert_tensor_metadata :: proc(
    a: Tensor, 
    size: []i64, 
    stride: []i64, 
    dtype: ScalarType,
    device: DeviceType, // TODO Map to c10::DeviceType
    layout: rawptr = nil,
) {
    t.atg__assert_tensor_metadata(
        a, 
        raw_data(size), i32(len(size)),
        raw_data(stride), i32(len(stride)),
        i32(dtype),
        i32(device),
        layout,
    )
}

// BATCH NORM

batch_norm_no_update :: proc(
    input, weight, bias, running_mean, running_var: Tensor, 
    momentum: f64, 
    eps: f64
) -> Tensor {
    out: Tensor
    t.atg__batch_norm_no_update(&out, input, weight, bias, running_mean, running_var, momentum, eps)
    return track(out)
}

batch_norm_with_update :: proc(
    input, weight, bias, running_mean, running_var: Tensor, 
    momentum: f64, 
    eps: f64
) -> Tensor {
    out: Tensor
    t.atg__batch_norm_with_update(&out, input, weight, bias, running_mean, running_var, momentum, eps)
    return track(out)
}

batch_norm_with_update_functional :: proc(
    input, weight, bias, running_mean, running_var: Tensor, 
    momentum: f64, 
    eps: f64
) -> Tensor {
    out: Tensor
    t.atg__batch_norm_with_update_functional(&out, input, weight, bias, running_mean, running_var, momentum, eps)
    return track(out)
}

// CONVOLUTION & SPATIAL

conv_depthwise2d :: proc(
    self, weight: Tensor, 
    kernel_size: []i64, 
    bias: Tensor, 
    stride: []i64, 
    padding: []i64, 
    dilation: []i64
) -> Tensor {
    out: Tensor
    t.atg__conv_depthwise2d(
        &out, self, weight, 
        raw_data(kernel_size), i32(len(kernel_size)),
        bias,
        raw_data(stride), i32(len(stride)),
        raw_data(padding), i32(len(padding)),
        raw_data(dilation), i32(len(dilation)),
    )
    return track(out)
}

// CONVOLUTION (CUDNN)
convolution :: proc(
    self, weight: Tensor, 
    bias: Tensor = Tensor{}, // Optional, pass undefined Tensor if not needed
    stride: []i64, 
    padding: []i64, 
    dilation: []i64, 
    groups: i64 = 1,
    benchmark: bool = false, 
    deterministic: bool = false,
    allow_tf32: bool = true,
) -> Tensor {
    out: Tensor
    
    t.atg_cudnn_convolution(
        &out,
        self,
        weight,
        raw_data(padding), i32(len(padding)),
        raw_data(stride), i32(len(stride)),
        raw_data(dilation), i32(len(dilation)),
        groups,
        i32(1) if benchmark else i32(0),
        i32(1) if deterministic else i32(0),
        i32(1) if allow_tf32 else i32(0),
    )
    return track(out)
}


// TODO: use "mode" string enum for padding (e.g. "zeros", "reflect")
convolution_mode :: proc(
    input, weight, bias: Tensor, 
    stride: []i64, 
    padding: string, 
    dilation: []i64, 
    groups: i64
) -> Tensor {
    out: Tensor
    c_padding := strings.clone_to_cstring(padding, context.temp_allocator)
    t.atg__convolution_mode(
        &out, input, weight, bias,
        raw_data(stride), i32(len(stride)),
        c_padding, i32(len(padding)),
        raw_data(dilation), i32(len(dilation)),
        groups,
    )
    return track(out)
}

// CUDNN RNN

cudnn_rnn :: proc(
    input: Tensor,
    weights: []Tensor,
    weight_buf: Tensor,
    hx: Tensor,
    cx: Tensor,
    mode: i64,
    hidden_size: i64,
    proj_size: i64,
    num_layers: i64,
    batch_first: bool,
    dropout: f64,
    train: bool,
    bidirectional: bool,
    batch_sizes: []i64, // Variable length sequences
    dropout_state: Tensor,
) -> Tensor {
    out: Tensor
    
    // weights is []Tensor, we need to pass pointer to data and length
    // We also need the stride, usually 1 for contiguous arrays
    weight_stride := i64(1) 

    t.atg__cudnn_rnn(
        &out, input,
        raw_data(weights), i32(len(weights)), weight_stride,
        weight_buf, hx, cx,
        mode, hidden_size, proj_size, num_layers,
        i32(1) if batch_first else i32(0),
        dropout,
        i32(1) if train else i32(0),
        i32(1) if bidirectional else i32(0),
        raw_data(batch_sizes), i32(len(batch_sizes)),
        dropout_state,
    )
    return track(out)
}

// TENSOR OPS & MATH

chunk_cat :: proc(tensors: []Tensor, dim: i64, num_chunks: i64) -> Tensor {
    out: Tensor
    t.atg__chunk_cat(&out, raw_data(tensors), i32(len(tensors)), dim, num_chunks)
    return track(out)
}

coalesce :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg__coalesce(&out, self)
    return track(out)
}

coalesced :: proc(self: Tensor, coalesced: bool) -> Tensor {
    out: Tensor
    t.atg__coalesced(&out, self, i32(1) if coalesced else i32(0))
    return track(out)
}

compute_linear_combination :: proc(input, coefficients: Tensor) -> Tensor {
    out: Tensor
    t.atg__compute_linear_combination(&out, input, coefficients)
    return track(out)
}

conj :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg__conj(&out, self)
    return track(out)
}

conj_physical :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg__conj_physical(&out, self)
    return track(out)
}

cdist_backward :: proc(grad, x1, x2: Tensor, p: f64, cdist: Tensor) -> Tensor {
    out: Tensor
    t.atg__cdist_backward(&out, grad, x1, x2, p, cdist)
    return track(out)
}

cholesky_solve_helper :: proc(self, A: Tensor, upper: bool) -> Tensor {
    out: Tensor
    t.atg__cholesky_solve_helper(&out, self, A, i32(1) if upper else i32(0))
    return track(out)
}

convert_indices_from_coo_to_csr :: proc(self: Tensor, size: i64, out_int32: bool) -> Tensor {
    out: Tensor
    t.atg__convert_indices_from_coo_to_csr(&out, self, size, i32(1) if out_int32 else i32(0))
    return track(out)
}

convert_indices_from_csr_to_coo :: proc(crow_indices, col_indices: Tensor, out_int32: bool, transpose: bool) -> Tensor {
    out: Tensor
    t.atg__convert_indices_from_csr_to_coo(
        &out, crow_indices, col_indices, 
        i32(1) if out_int32 else i32(0),
        i32(1) if transpose else i32(0),
    )
    return track(out)
}

copy_from :: proc(self, dst: Tensor, non_blocking: bool = false) -> Tensor {
    out: Tensor
    t.atg__copy_from(&out, self, dst, i32(1) if non_blocking else i32(0))
    return track(out)
}

copy_from_and_resize :: proc(self, dst: Tensor) -> Tensor {
    out: Tensor
    t.atg__copy_from_and_resize(&out, self, dst)
    return track(out)
}

dim_arange :: proc(like: Tensor, dim: i64) -> Tensor {
    out: Tensor
    t.atg__dim_arange(&out, like, dim)
    return track(out)
}

// EMBEDDINGS

embedding_bag :: proc(
    weight: Tensor,
    indices: Tensor,
    offsets: Tensor,
    scale_grad_by_freq: bool = false,
    mode: i64 = 0, // 0: sum, 1: mean, 2: max
    sparse: bool = false,
    per_sample_weights: Tensor = {},
    include_last_offset: bool = false,
    padding_idx: i64 = -1,
) -> Tensor {
    out: Tensor
    
    // Handle optional empty tensor for per_sample_weights
    psw := per_sample_weights
    if defined(psw) == 0 {
        // Create an undefined tensor handle if passed empty
        psw = t.Tensor{} 
    }
    t.atg__embedding_bag(
        &out,
        weight,
        indices,
        offsets,
        i32(1) if scale_grad_by_freq else i32(0),
        mode,
        i32(1) if sparse else i32(0),
        psw,
        i32(1) if include_last_offset else i32(0),
        padding_idx,
    )
    return track(out)
}

embedding_renorm :: proc(self, indices: Tensor, max_norm: f64, norm_type: f64) -> Tensor {
    out: Tensor
    t.atg_embedding_renorm(&out, self, indices, max_norm, norm_type)
    return track(out)
}

// Handles specific padding_idx logic where nullability is explicit in C
embedding_bag_padding_idx :: proc(
    weight, indices, offsets: Tensor, 
    scale_grad_by_freq: bool = false, 
    mode: i64 = 0, 
    sparse: bool = false, 
    per_sample_weights: Tensor = Tensor{}, 
    include_last_offset: bool = false,
    padding_idx: i64 = -1,
) -> Tensor {
    out: Tensor
    // If padding_idx is -1 (or user defined "null"), we pass nil to the rawptr arg
    // otherwise we pass a dummy pointer to indicate the value is valid
    
    ptr_val: rawptr = nil
    if padding_idx != -1 {
        ptr_val = rawptr(uintptr(1)) // Non-nil to indicate presence
    }

    t.atg_embedding_bag_padding_idx(
        &out, 
        weight, indices, offsets, 
        i32(1) if scale_grad_by_freq else i32(0),
        mode,
        i32(1) if sparse else i32(0),
        per_sample_weights,
        i32(1) if include_last_offset else i32(0),
        padding_idx,
        ptr_val,
    )
    return track(out)
}

// LINEAR ALGEBRA

linalg_det :: proc(A: Tensor) -> Tensor {
    out: Tensor
    t.atg__linalg_det(&out, A)
    return track(out)
}

// Returns the sign and natural logarithm of the absolute value of the determinant
linalg_slogdet :: proc(A: Tensor) -> (sign, logabsdet: Tensor) {    
    out: Tensor
    t.atg_linalg_slogdet(&out, A)
    return track(out), track(out) // Warning: Verify the C tuple struct layout
}

linalg_solve :: proc(A: Tensor, B: Tensor, left: bool = true) -> Tensor {
    out: Tensor
    // check_errors = 1 (true) by default
    t.atg__linalg_solve_ex(&out, A, B, i32(1) if left else i32(0), i32(1))
    return track(out)
}

// FFT Fast Fourier Transform

fft_c2c :: proc(self: Tensor, dim: []i64, normalization: i64 = 1, forward: bool = true) -> Tensor {
    out: Tensor
    t.atg__fft_c2c(
        &out,
        self,
        raw_data(dim),
        i32(len(dim)),
        normalization,
        i32(1) if forward else i32(0),
    )
    return track(out)
}

fft_r2c :: proc(self: Tensor, dim: []i64, normalization: i64 = 1, onesided: bool = true) -> Tensor {
    out: Tensor
    t.atg__fft_r2c(
        &out,
        self,
        raw_data(dim),
        i32(len(dim)),
        normalization,
        i32(1) if onesided else i32(0),
    )
    return track(out)
}

fft_c2r :: proc(self: Tensor, dim: []i64, normalization: i64 = 1, last_dim_size: i64 = -1) -> Tensor {
    out: Tensor
    // TODO: If last_dim_size is not provided (standard pytorch usage), usually it's inferred
    t.atg__fft_c2r(
        &out,
        self,
        raw_data(dim),
        i32(len(dim)),
        normalization,
        last_dim_size,
    )
    return track(out)
}

// ACTIVATIONS / LAYERS

fused_rms_norm :: proc(input: Tensor, normalized_shape: []i64, weight: Tensor, eps: f64) -> Tensor {
    out: Tensor
    // eps_null: rawptr - if we pass nil, it implies we are providing the value in eps_v
    t.atg__fused_rms_norm(
        &out, 
        input, 
        raw_data(normalized_shape), 
        i32(len(normalized_shape)), 
        weight, 
        eps, 
        nil,
    )
    return track(out)
}

fused_dropout :: proc(self: Tensor, p: f64) -> Tensor {
    out: Tensor
    t.atg__fused_dropout(&out, self, p)
    return track(out)
}

// OPS

euclidean_dist :: proc(x1, x2: Tensor) -> Tensor {
    out: Tensor
    t.atg__euclidean_dist(&out, x1, x2)
    return track(out)
}

masked_scale :: proc(self: Tensor, mask: Tensor, scale: f64) -> Tensor {
    out: Tensor
    t.atg__masked_scale(&out, self, mask, scale)
    return track(out)
}

// Returns boolean tensor
is_any_true :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg__is_any_true(&out, self)
    return track(out)
}

// Returns boolean tensor
is_all_true :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg__is_all_true(&out, self)
    return track(out)
}

neg_view :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg__neg_view(&out, self)
    return track(out)
}

// NESTED TENSORS

nested_from_padded :: proc(padded: Tensor, nested_shape_example: Tensor, fuse_transform: bool = false) -> Tensor {
    out: Tensor
    t.atg__nested_from_padded(
        &out, 
        padded, 
        nested_shape_example, 
        i32(1) if fuse_transform else i32(0),
    )
    return track(out)
}

nested_get_values :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg__nested_get_values(&out, self)
    return track(out)
}

nested_get_offsets :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg__nested_get_offsets(&out, self)
    return track(out)
}

nested_get_lengths :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg__nested_get_lengths(&out, self)
    return track(out)
}

// Used to assert async safety in JIT/Graphs, effectively no-op in eager mode
functional_assert_async :: proc(self: Tensor, msg: string, dep_token: Tensor) -> Tensor {
    out: Tensor
    c_msg := strings.clone_to_cstring(msg, context.temp_allocator)
    t.atg__functional_assert_async(&out, self, c_msg, i32(len(msg)), dep_token)
    return track(out)
}

grid_sampler_2d_cpu_fallback :: proc(
    input: Tensor, 
    grid: Tensor, 
    interpolation_mode: i64, 
    padding_mode: i64, 
    align_corners: bool,
) -> Tensor {
    out: Tensor
    t.atg__grid_sampler_2d_cpu_fallback(
        &out, 
        input, 
        grid, 
        interpolation_mode, 
        padding_mode, 
        i32(1) if align_corners else i32(0),
    )
    return track(out)
}

mkldnn_transpose_ :: proc(self: Tensor, dim0, dim1: i64) -> Tensor {
    out: Tensor
    t.atg__mkldnn_transpose_(&out, self, dim0, dim1)
    // For in-place, we return the input `self`
    return self
}

fill_mem_eff_dropout_mask_ :: proc(self: Tensor, dropout_p: f64, seed: i64, offset: i64) -> Tensor {
    out: Tensor
    t.atg__fill_mem_eff_dropout_mask_(&out, self, dropout_p, seed, offset)
    return self
}

dirichlet_grad :: proc(x, alpha, total: Tensor) -> Tensor {
    out: Tensor
    t.atg__dirichlet_grad(&out, x, alpha, total)
    return track(out)
}

histgramdd_from_bin_cts :: proc(self, weight: Tensor, bins: []i64, range: []f64, density: bool = false) -> Tensor {
    out: Tensor
    t.atg__histogramdd_from_bin_cts(&out, self, raw_data(bins), i32(len(bins)), raw_data(range), i32(len(range)), weight, i32(density))
    return track(out)
}

// Example of TensorList input
histgramdd_from_bin_tensors :: proc(self, weight: Tensor, bins: []Tensor, density: bool = false) -> Tensor {
    out: Tensor
    // We assume []Tensor matches the memory layout of ^Tensor (C-array of pointers)
    t.atg__histogramdd_from_bin_tensors(&out, self, raw_data(bins), i32(len(bins)), weight, i32(density))
    return track(out)
}

linalg_det_result :: proc(A: Tensor) -> (result, LU, pivots: Tensor) {
    result = new_tensor()
    LU = new_tensor()
    pivots = new_tensor()
    dummy: Tensor // Return value of the C function usually void or status, ignoring
    t.atg__linalg_det_result(&dummy, result, LU, pivots, A)
    return
}

// Returns eigenvalues, eigenvectors
linalg_eigh_eigenvalues :: proc(A: Tensor, uplo: string = "L", compute_v: bool = true) -> (evals, evecs: Tensor) {
    evals = new_tensor()
    evecs = new_tensor()
    c_uplo := strings.clone_to_cstring(uplo, context.temp_allocator)
    dummy: Tensor
    t.atg__linalg_eigh_eigenvalues(&dummy, evals, evecs, A, c_uplo, i32(len(uplo)), i32(compute_v))
    return
}

// Returns sign, logabsdet, LU, pivots
linalg_slogdet_sign :: proc(A: Tensor) -> (signed, logabsdet, LU, pivots: Tensor) {
    signed = new_tensor()
    logabsdet = new_tensor()
    LU = new_tensor()
    pivots = new_tensor()
    dummy: Tensor
    t.atg__linalg_slogdet_sign(&dummy, signed, logabsdet, LU, pivots, A)
    return
}

// Returns result, LU, pivots, info
linalg_solve_ex_result :: proc(A, B: Tensor, left: bool = true, check_errors: bool = true) -> (result, LU, pivots, info: Tensor) {
    result = new_tensor()
    LU = new_tensor()
    pivots = new_tensor()
    info = new_tensor()
    dummy: Tensor
    t.atg__linalg_solve_ex_result(&dummy, result, LU, pivots, info, A, B, i32(left), i32(check_errors))
    return
}

linalg_svd_u :: proc(A: Tensor, full_matrices: bool = true, compute_uv: bool = true) -> (U, S, Vh: Tensor) {
    U = new_tensor()
    S = new_tensor()
    Vh = new_tensor()
    c_driver := cstring(nil)
    dummy: Tensor
    t.atg__linalg_svd_u(&dummy, U, S, Vh, A, i32(full_matrices), i32(compute_uv), c_driver, 0)
    return
}

lu_with_info :: proc(self: Tensor, pivot: bool = true, check_errors: bool = true) -> Tensor {
    out: Tensor
    t.atg__lu_with_info(&out, self, i32(pivot), i32(check_errors))
    return track(out)
}

//  QUANTIZATION

fake_quantize_learnable_per_tensor_affine :: proc(self, scale, zero_point: Tensor, quant_min, quant_max: i64, grad_factor: f64 = 1.0) -> Tensor {
    out: Tensor
    t.atg__fake_quantize_learnable_per_tensor_affine(&out, self, scale, zero_point, quant_min, quant_max, grad_factor)
    return track(out)
}

fake_quantize_learnable_per_channel_affine :: proc(self, scale, zero_point: Tensor, axis, quant_min, quant_max: i64, grad_factor: f64 = 1.0) -> Tensor {
    out: Tensor
    t.atg__fake_quantize_learnable_per_channel_affine(&out, self, scale, zero_point, axis, quant_min, quant_max, grad_factor)
    return track(out)
}

fake_quantize_per_tensor_affine_tensor_qparams :: proc(
    self, scale, zero_point: Tensor, 
    quant_min, quant_max: i64
) -> Tensor {
    out: Tensor
    t.atg_fake_quantize_per_tensor_affine_tensor_qparams(
        &out, self, scale, zero_point, quant_min, quant_max,
    )
    return track(out)
}

fake_quantize_per_channel_affine_cachemask_backward :: proc(grad, mask: Tensor) -> Tensor {
    out: Tensor
    t.atg_fake_quantize_per_channel_affine_cachemask_backward(&out, grad, mask)
    return track(out)
}

fake_quantize_per_tensor_affine_cachemask_backward :: proc(grad, mask: Tensor) -> Tensor {
    out: Tensor
    t.atg_fake_quantize_per_tensor_affine_cachemask_backward(&out, grad, mask)
    return track(out)
}

make_per_tensor_quantized_tensor :: proc(self: Tensor, scale: f64, zero_point: i64) -> Tensor {
    out: Tensor
    t.atg__make_per_tensor_quantized_tensor(&out, self, scale, zero_point)
    return track(out)
}

make_per_channel_quantized_tensor :: proc(self, scale, zero_point: Tensor, axis: i64) -> Tensor {
    out: Tensor
    t.atg__make_per_channel_quantized_tensor(&out, self, scale, zero_point, axis)
    return track(out)
}

// Useful for manual Attention implementation
flash_attention_backward :: proc(grad_out, query, key, value, logsumexp, cum_seq_q, cum_seq_k: Tensor, max_q, max_k: i64, dropout_p: f64, is_causal: bool) -> Tensor {
    out: Tensor
    // TODO: scales and window sizes
    t.atg__flash_attention_backward(&out, grad_out, query, key, value, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, dropout_p, i32(is_causal), Tensor{}, Tensor{}, 1.0, nil, 0, nil, 0, nil)
    return track(out)
}

// LSTM

lstm_mps :: proc(input: Tensor, hx: []Tensor, params: []Tensor, has_biases: bool, num_layers: i64, dropout: f64, train: bool, bidirectional: bool, batch_first: bool) -> Tensor {
    out: Tensor
    t.atg__lstm_mps(&out, input, raw_data(hx), i32(len(hx)), raw_data(params), i32(len(params)), i32(has_biases), num_layers, dropout, i32(train), i32(bidirectional), i32(batch_first))
    return track(out)
}

mps_convolution :: proc(self, weight, bias: Tensor, padding, stride, dilation: []i64, groups: i64) -> Tensor {
    out: Tensor
    t.atg__mps_convolution(&out, self, weight, bias, raw_data(padding), i32(len(padding)), raw_data(stride), i32(len(stride)), raw_data(dilation), i32(len(dilation)), groups)
    return track(out)
}

gather_sparse_backward :: proc(self: Tensor, dim: i64, index, grad: Tensor) -> Tensor {
    out: Tensor
    t.atg__gather_sparse_backward(&out, self, dim, index, grad)
    return track(out)
}

mkldnn_reshape :: proc(self: Tensor, shape: []i64) -> Tensor {
    out: Tensor
    t.atg__mkldnn_reshape(&out, self, raw_data(shape), i32(len(shape)))
    return track(out)
}

make_dep_token :: proc(device: DeviceType = .CPU) -> Tensor {
    out: Tensor
    t.atg__make_dep_token(&out, 0, i32(device)) 
    return track(out)
}

// NESTED TENSORS

nested_select_backward :: proc(grad_output: Tensor, self: Tensor, dim: i64, index: i64) -> Tensor {
    out: Tensor
    t.atg__nested_select_backward(&out, grad_output, self, dim, index)
    return track(out)
}

nested_sum_backward :: proc(grad: Tensor, self: Tensor, dims: []i64, keepdim: bool = false) -> Tensor {
    out: Tensor
    t.atg__nested_sum_backward(
        &out, 
        grad, 
        self, 
        raw_data(dims), 
        i32(len(dims)), 
        i32(keepdim),
    )
    return track(out)
}

nested_view_from_buffer :: proc(self, nested_size, nested_strides, offsets: Tensor) -> Tensor {
    out: Tensor
    t.atg__nested_view_from_buffer(&out, self, nested_size, nested_strides, offsets)
    return track(out)
}

// PADDING & PACKING (RNNs/Sequences)

pack_padded_sequence :: proc(input: Tensor, lengths: Tensor, batch_first: bool = false) -> Tensor {
    out: Tensor
    t.atg__pack_padded_sequence(&out, input, lengths, i32(batch_first))
    return track(out)
}

pad_circular :: proc(self: Tensor, padding: []i64) -> Tensor {
    out: Tensor
    t.atg__pad_circular(&out, self, raw_data(padding), i32(len(padding)))
    return track(out)
}

// pad handles constant padding with specific mode and value
// TODO enum for mode (e.g. "constant", "reflect", "replicate", "circular")
pad :: proc(self: Tensor, padding: []i64, mode: i64, value: f64 = 0.0) -> Tensor {
    out: Tensor
    // value_null determines if the optional value is present. 
    // Since we pass a default or specific float, we pass the pointer to indicate presence.
    val := value 
    t.atg__pad_enum(
        &out, 
        self, 
        raw_data(padding), 
        i32(len(padding)), 
        mode, 
        val, 
        &val, // Non-nil pointer implies value is set
    )
    return track(out)
}

pad_sequence :: proc(sequences: []Tensor, batch_first: bool = false, padding_value: f64 = 0, padding_side: string = "right") -> Tensor {
    out: Tensor
    side_cstr := strings.clone_to_cstring(padding_side, context.temp_allocator)
    batch_first_int := i32(1) if batch_first else i32(0)
    
    t.atg_pad_sequence(
        &out, 
        raw_data(sequences), 
        i32(len(sequences)), 
        batch_first_int, 
        padding_value, 
        side_cstr, 
        i32(len(padding_side)),
    )
    return track(out)
}

// RESHAPING & VIEWS

reshape_alias :: proc(self: Tensor, size: []i64, stride: []i64) -> Tensor {
    out: Tensor
    t.atg__reshape_alias(
        &out, 
        self, 
        raw_data(size), 
        i32(len(size)), 
        raw_data(stride), 
        i32(len(stride)),
    )
    return track(out)
}

// ATTENTION (SDPA)

scaled_dot_product_efficient_attention :: proc(
    query, key, value: Tensor,
    attn_bias: Tensor,
    compute_log_sumexp: bool,
    dropout_p: f64,
    is_causal: bool,
    scale: Maybe(f64) = nil,
) -> Tensor {
    out: Tensor
    s_val: f64 = 0.0
    s_ptr: rawptr = nil
    if v, ok := scale.?; ok { s_val = v; s_ptr = &s_val }

    t.atg__scaled_dot_product_efficient_attention(
        &out, query, key, value, attn_bias,
        i32(compute_log_sumexp),
        dropout_p,
        i32(is_causal),
        s_val, s_ptr,
    )
    return track(out)
}

// SPARSE TENSORS

sparse_mm :: proc(sparse_mat, dense_mat: Tensor) -> Tensor {
    out: Tensor
    t.atg__sparse_mm(&out, sparse_mat, dense_mat)
    return track(out)
}

sparse_addmm :: proc(self, mat1, mat2: Tensor) -> Tensor {
    out: Tensor
    t.atg__sparse_addmm(&out, self, mat1, mat2)
    return track(out)
}

sparse_sum :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg__sparse_sum(&out, self)
    return track(out)
}

sparse_sum_dim :: proc(self: Tensor, dims: []i64) -> Tensor {
    out: Tensor
    t.atg__sparse_sum_dim(&out, self, raw_data(dims), i32(len(dims)))
    return track(out)
}

abs :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_abs(&out, self)
    return track(out)
}

abs_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_abs_(&out, self)
    return self
}

acos :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_acos(&out, self)
    return track(out)
}

acos_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_acos_(&out, self)
    return self
}

acosh :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_acosh(&out, self)
    return track(out)
}

acosh_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_acosh_(&out, self)
    return self
}

add_scalar :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_add_scalar(&out, self, other)
    return track(out)
}

add_scalar_ :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_add_scalar_(&out, self, other)
    return self
}

// LINEAR ALGEBRA & MATRIX OPS

// addmm: beta * self + alpha * (mat1 @ mat2)
addmm :: proc(self, mat1, mat2: Tensor) -> Tensor {
    out: Tensor
    t.atg_addmm(&out, self, mat1, mat2)
    return track(out)
}

addmm_ :: proc(self, mat1, mat2: Tensor) -> Tensor {
    out: Tensor
    t.atg_addmm_(&out, self, mat1, mat2)
    return self
}

// addbmm: beta * self + alpha * (batch1 @ batch2).sum(0)
addbmm :: proc(self, batch1, batch2: Tensor) -> Tensor {
    out: Tensor
    t.atg_addbmm(&out, self, batch1, batch2)
    return track(out)
}

addbmm_ :: proc(self, batch1, batch2: Tensor) -> Tensor {
    out: Tensor
    t.atg_addbmm_(&out, self, batch1, batch2)
    return self
}

// addmv: beta * self + alpha * (mat @ vec)
addmv :: proc(self, mat, vec: Tensor) -> Tensor {
    out: Tensor
    t.atg_addmv(&out, self, mat, vec)
    return track(out)
}

addmv_ :: proc(self, mat, vec: Tensor) -> Tensor {
    out: Tensor
    t.atg_addmv_(&out, self, mat, vec)
    return self
}

// addr: beta * self + alpha * (vec1 @ vec2.T) (Outer product)
addr :: proc(self, vec1, vec2: Tensor) -> Tensor {
    out: Tensor
    t.atg_addr(&out, self, vec1, vec2)
    return track(out)
}

addr_ :: proc(self, vec1, vec2: Tensor) -> Tensor {
    out: Tensor
    t.atg_addr_(&out, self, vec1, vec2)
    return self
}

adjoint :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_adjoint(&out, self)
    return track(out)
}

// ADAPTIVE POOLING

adaptive_avg_pool1d :: proc(self: Tensor, output_size: []i64) -> Tensor {
    out: Tensor
    t.atg_adaptive_avg_pool1d(
        &out, 
        self, 
        raw_data(output_size), 
        i32(len(output_size)),
    )
    return track(out)
}

adaptive_max_pool2d :: proc(self: Tensor, output_size: []i64) -> Tensor {
    // NOTE: In C++ this usually returns a tuple (Output, Indices).
    // The provided signature only has `out`
    // usually implies it returns the values tensor.
    out: Tensor
    t.atg_adaptive_max_pool2d(
        &out, 
        self, 
        raw_data(output_size), 
        i32(len(output_size)),
    )
    return track(out)
}

// TRANSFORMER & ATTENTION

// Multi-Head Attention (Triton optimized)
multi_head_attention :: proc(
    query, key, value: Tensor,
    embed_dim, num_head: i64,
    qkv_weight, qkv_bias: Tensor,
    proj_weight, proj_bias: Tensor,
    mask: Tensor = {},
) -> Tensor {
    out: Tensor
    // NOTE: mask can be undefined (empty tensor) if not used
    t.atg__triton_multi_head_attention(
        &out,
        query, key, value,
        embed_dim, num_head,
        qkv_weight, qkv_bias,
        proj_weight, proj_bias,
        mask,
    )
    return track(out)
}

transformer_encoder_layer :: proc(
    src: Tensor,
    embed_dim, num_heads: i64,
    qkv_weight, qkv_bias: Tensor,
    proj_weight, proj_bias: Tensor,
    norm_weight_1, norm_bias_1: Tensor,
    norm_weight_2, norm_bias_2: Tensor,
    ffn_weight_1, ffn_bias_1: Tensor,
    ffn_weight_2, ffn_bias_2: Tensor,
    use_gelu: bool = true,
    norm_first: bool = false,
    eps: f64 = 1e-5,
    mask: Tensor = {}, 
) -> Tensor {
    out: Tensor
    
    use_gelu_int := i32(1) if use_gelu else i32(0)
    norm_first_int := i32(1) if norm_first else i32(0)
    
    // Handle optional Mask Type
    // If mask is defined, we assume default type behavior (passed as null ptr)
    // TODO: `mask_type` int64 pointer needs to be parsed as enum
    
    t.atg__transformer_encoder_layer_fwd(
        &out,
        src,
        embed_dim,
        num_heads,
        qkv_weight, qkv_bias,
        proj_weight, proj_bias,
        use_gelu_int,
        norm_first_int,
        eps,
        norm_weight_1, norm_bias_1,
        norm_weight_2, norm_bias_2,
        ffn_weight_1, ffn_bias_1,
        ffn_weight_2, ffn_bias_2,
        mask,
        0,   // mask_type_v (ignored if null ptr)
        nil, // mask_type_null (nil = not provided)
    )
    
    return track(out)
}

//  GENERATORS & CREATION ---

affine_grid_generator :: proc(theta: Tensor, size: []i64, align_corners: bool) -> Tensor {
    out: Tensor
    align_c := i32(1) if align_corners else i32(0)
    t.atg_affine_grid_generator(&out, theta, raw_data(size), i32(len(size)), align_c)
    return track(out)
}

arange_end :: proc(end: Scalar, dtype: ScalarType = .Float, device: DeviceType = .CPU) -> Tensor {
    out: Tensor
    t.atg_arange(&out, end, i32(dtype), i32(device))
    return track(out)
}

arange_start :: proc(start, end: Scalar, dtype: ScalarType = .Float, device: DeviceType = .CPU) -> Tensor {
    out: Tensor
    t.atg_arange_start(&out, start, end, i32(dtype), i32(device))
    return track(out)
}

arange_step :: proc(start, end, step: Scalar, dtype: ScalarType = .Float, device: DeviceType = .CPU) -> Tensor {
    out: Tensor
    t.atg_arange_start_step(&out, start, end, step, i32(dtype), i32(device))
    return track(out)
}

bartlett_window :: proc(window_length: i64, periodic: bool = true, dtype: ScalarType = .Float, device: DeviceType = .CPU) -> Tensor {
    out: Tensor
    p_int := i32(1) if periodic else i32(0)
    t.atg_bartlett_window_periodic(&out, window_length, p_int, i32(dtype), i32(device))
    return track(out)
}

//  MEMORY / VIEWS

alias :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_alias(&out, self)
    return track(out)
}

alias_copy :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_alias_copy(&out, self)
    return track(out)
}

as_strided :: proc(self: Tensor, size: []i64, stride: []i64, storage_offset: Maybe(i64) = nil) -> Tensor {
    out: Tensor
    offset_val: i64 = 0
    offset_ptr: rawptr = nil
    
    if v, ok := storage_offset.?; ok {
        offset_val = v
        offset_ptr = &offset_val // Not strictly used by binding logic if null is passed
    }
    // NOTE: The raw binding requires `storage_offset_v` AND `storage_offset_null` (as a ptr check).
    t.atg_as_strided(&out, self, raw_data(size), i32(len(size)), raw_data(stride), i32(len(stride)), offset_val, offset_ptr)
    return track(out)
}

//  REDUCTION & LOGIC

all :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_all(&out, self)
    return track(out)
}

all_dim :: proc(self: Tensor, dim: i64, keepdim: bool = false) -> Tensor {
    out: Tensor
    t.atg_all_dim(&out, self, dim, i32(keepdim))
    return track(out)
}

any_tensor :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_any(&out, self)
    return track(out)
}

any_dim :: proc(self: Tensor, dim: i64, keepdim: bool = false) -> Tensor {
    out: Tensor
    t.atg_any_dim(&out, self, dim, i32(keepdim))
    return track(out)
}

amax :: proc(self: Tensor, dim: []i64, keepdim: bool = false) -> Tensor {
    out: Tensor
    t.atg_amax(&out, self, raw_data(dim), i32(len(dim)), i32(keepdim))
    return track(out)
}

amin :: proc(self: Tensor, dim: []i64, keepdim: bool = false) -> Tensor {
    out: Tensor
    t.atg_amin(&out, self, raw_data(dim), i32(len(dim)), i32(keepdim))
    return track(out)
}

argmax :: proc(self: Tensor, dim: Maybe(i64) = nil, keepdim: bool = false) -> Tensor {
    out: Tensor
    dim_val: i64 = 0
    dim_ptr: rawptr = nil
    if v, ok := dim.?; ok {
        dim_val = v
        dim_ptr = &dim_val
    }
    t.atg_argmax(&out, self, dim_val, dim_ptr, i32(keepdim))
    return track(out)
}

argmin :: proc(self: Tensor, dim: Maybe(i64) = nil, keepdim: bool = false) -> Tensor {
    out: Tensor
    dim_val: i64 = 0
    dim_ptr: rawptr = nil
    if v, ok := dim.?; ok {
        dim_val = v
        dim_ptr = &dim_val
    }
    t.atg_argmin(&out, self, dim_val, dim_ptr, i32(keepdim))
    return track(out)
}

argsort :: proc(self: Tensor, dim: i64 = -1, descending: bool = false, stable: bool = false) -> Tensor {
    out: Tensor
    if stable {
        t.atg_argsort_stable(&out, self, i32(1), dim, i32(descending))
    } else {
        t.atg_argsort(&out, self, dim, i32(descending))
    }
    return track(out)
}

//  TRIGONOMETRY

arccos :: proc(self: Tensor) -> Tensor { out: Tensor; t.atg_arccos(&out, self); return track(out) }
arccos_ :: proc(self: Tensor) -> Tensor { 
    out: Tensor
    t.atg_arccos_(&out, self)
    return self 
}

arcsin :: proc(self: Tensor) -> Tensor { out: Tensor; t.atg_arcsin(&out, self); return track(out) }
arcsin_ :: proc(self: Tensor) -> Tensor { 
    out: Tensor
    t.atg_arcsin_(&out, self)
    return self 
}

arctan :: proc(self: Tensor) -> Tensor { out: Tensor; t.atg_arctan(&out, self); return track(out) }
arctan_ :: proc(self: Tensor) -> Tensor { 
    out: Tensor
    t.atg_arctan_(&out, self)
    return self 
}

arctan2 :: proc(self, other: Tensor) -> Tensor { out: Tensor; t.atg_arctan2(&out, self, other); return track(out) }
arctan2_ :: proc(self, other: Tensor) -> Tensor { 
    out: Tensor
    t.atg_arctan2_(&out, self, other)
    return self 
}

angle :: proc(self: Tensor) -> Tensor { out: Tensor; t.atg_angle(&out, self); return track(out) }

//  POOLING

avg_pool2d :: proc(self: Tensor, kernel_size: []i64, stride: []i64, padding: []i64, ceil_mode: bool = false, count_include_pad: bool = true, divisor_override: Maybe(i64) = nil) -> Tensor {
    out: Tensor
    
    div_val: i64 = 0
    div_ptr: rawptr = nil
    if v, ok := divisor_override.?; ok {
        div_val = v
        div_ptr = &div_val
    }

    t.atg_avg_pool2d(
        &out, self,
        raw_data(kernel_size), i32(len(kernel_size)),
        raw_data(stride), i32(len(stride)),
        raw_data(padding), i32(len(padding)),
        i32(ceil_mode), i32(count_include_pad),
        div_val, div_ptr,
    )
    return track(out)
}

avg_pool3d :: proc(self: Tensor, kernel_size: []i64, stride: []i64, padding: []i64, ceil_mode: bool = false, count_include_pad: bool = true, divisor_override: Maybe(i64) = nil) -> Tensor {
    out: Tensor
    
    div_val: i64 = 0
    div_ptr: rawptr = nil
    if v, ok := divisor_override.?; ok {
        div_val = v
        div_ptr = &div_val
    }

    t.atg_avg_pool3d(
        &out, self,
        raw_data(kernel_size), i32(len(kernel_size)),
        raw_data(stride), i32(len(stride)),
        raw_data(padding), i32(len(padding)),
        i32(ceil_mode), i32(count_include_pad),
        div_val, div_ptr,
    )
    return track(out)
}

//  NORMALIZATION / DROPOUT

alpha_dropout :: proc(self: Tensor, p: f64, train: bool) -> Tensor {
    out: Tensor
    t.atg_alpha_dropout(&out, self, p, i32(train))
    return track(out)
}

alpha_dropout_ :: proc(self: Tensor, p: f64, train: bool) -> Tensor {
    out: Tensor
    t.atg_alpha_dropout_(&out, self, p, i32(train))
    return self
}

batch_norm :: proc(input, weight, bias, running_mean, running_var: Tensor, training: bool, momentum: f64, eps: f64, cudnn_enabled: bool) -> Tensor {
    out: Tensor
    t.atg_batch_norm(
        &out, input, weight, bias, running_mean, running_var, 
        i32(training), momentum, eps, i32(cudnn_enabled),
    )
    return track(out)
}

//  PROBABILITY

bernoulli :: proc(self: Tensor, p: Tensor) -> Tensor {
    out: Tensor
    t.atg_bernoulli_tensor(&out, self, p)
    return track(out)
}

bernoulli_p :: proc(self: Tensor, p: f64) -> Tensor {
    out: Tensor
    t.atg_bernoulli_p(&out, self, p)
    return track(out)
}

bernoulli_:: proc(self: Tensor, p: Tensor) -> Tensor {
    out: Tensor
    t.atg_bernoulli_(&out, self, p)
    return self
}

bernoulli_float_ :: proc(self: Tensor, p: f64) -> Tensor {
    out: Tensor
    t.atg_bernoulli_float_(&out, self, p)
    return self
}

bincount :: proc(self: Tensor, weights: Tensor = {}, minlength: i64 = 0) -> Tensor {
    out: Tensor
    t.atg_bincount(&out, self, weights, minlength)
    return track(out)
}

//  LINEAR ALGEBRA / MATRIX OPS

baddbmm :: proc(self, batch1, batch2: Tensor, beta: Scalar, alpha: Scalar) -> Tensor {
    out: Tensor
    t.atg_baddbmm(&out, self, batch1, batch2, beta, alpha)
    return track(out)
}

bilinear :: proc(input1, input2, weight, bias: Tensor) -> Tensor {
    out: Tensor
    t.atg_bilinear(&out, input1, input2, weight, bias)
    return track(out)
}

blackman_window :: proc(window_length: i64, periodic: bool = true, dtype: ScalarType = .Float, device: DeviceType = .CPU) -> Tensor {
    out: Tensor
    if periodic {
        t.atg_blackman_window_periodic(&out, window_length, 1, i32(dtype), i32(device))
    } else {
        // Standard call if not periodic specific, periodic=1 is usually default in torch
        t.atg_blackman_window(&out, window_length, i32(dtype), i32(device))
    }
    return track(out)
}

bmm :: proc(self, mat2: Tensor) -> Tensor {
    out: Tensor
    t.atg_bmm(&out, self, mat2)
    return track(out)
}

block_diag :: proc(tensors: []Tensor) -> Tensor {
    out: Tensor
    t.atg_block_diag(&out, raw_data(tensors), i32(len(tensors)))
    return track(out)
}

chain_matmul :: proc(matrices: []Tensor) -> Tensor {
    out: Tensor
    t.atg_chain_matmul(&out, raw_data(matrices), i32(len(matrices)))
    return track(out)
}

cholesky :: proc(self: Tensor, upper: bool = false) -> Tensor {
    out: Tensor
    t.atg_cholesky(&out, self, i32(upper))
    return track(out)
}

cholesky_inverse :: proc(self: Tensor, upper: bool = false) -> Tensor {
    out: Tensor
    t.atg_cholesky_inverse(&out, self, i32(upper))
    return track(out)
}

cholesky_solve :: proc(self, input2: Tensor, upper: bool = false) -> Tensor {
    out: Tensor
    t.atg_cholesky_solve(&out, self, input2, i32(upper))
    return track(out)
}

// cdist: compute_mode is optional (use -1 or 0 if not specified, or handle via null)
cdist :: proc(x1, x2: Tensor, p: f64 = 2.0, compute_mode: Maybe(i64) = nil) -> Tensor {
    out: Tensor
    
    mode_val: i64 = 0
    mode_ptr: rawptr = nil

    if m, ok := compute_mode.?; ok {
        mode_val = m
        mode_ptr = &mode_val // Actually pointing to stack var might be unsafe if C stores it, but here it reads immediately.
        t.atg_cdist(&out, x1, x2, p, mode_val, nil) 
    } else {
        // Pass a dummy value and a marker
        t.atg_cdist(&out, x1, x2, p, 0, rawptr(&mode_val)) // sending non-null to indicate "default"? 
    }
    return track(out)
}

corrcoef :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_corrcoef(&out, self)
    return track(out)
}

cov :: proc(self: Tensor, correction: i64 = 1, fweights: Tensor = {}, aweights: Tensor = {}) -> Tensor {
    out: Tensor
    t.atg_cov(&out, self, correction, fweights, aweights)
    return track(out)
}

cross :: proc(self, other: Tensor, dim: Maybe(i64) = nil) -> Tensor {
    out: Tensor
    if d, ok := dim.?; ok {
        t.atg_cross(&out, self, other, d, nil)
    } else {
        dummy: i64 = 0
        t.atg_cross(&out, self, other, 0, &dummy)
    }
    return track(out)
}

// TENSOR SHAPE AND MANIPULATION

broadcast_to :: proc(self: Tensor, size: []i64) -> Tensor {
    out: Tensor
    t.atg_broadcast_to(&out, self, raw_data(size), i32(len(size)))
    return track(out)
}

bucketize :: proc{bucketize_tensor, bucketize_scalar}

@(private)
bucketize_tensor :: proc(self: Tensor, boundaries: Tensor, out_int32: bool = false, right: bool = false) -> Tensor {
    out: Tensor
    t.atg_bucketize(&out, self, boundaries, i32(out_int32), i32(right))
    return track(out)
}

@(private)
bucketize_scalar :: proc(self: Scalar, boundaries: Tensor, out_int32: bool = false, right: bool = false) -> Tensor {
    out: Tensor
    t.atg_bucketize_scalar(&out, self, boundaries, i32(out_int32), i32(right))
    return track(out)
}

cartesian_prod :: proc(tensors: []Tensor) -> Tensor {
    out: Tensor
    t.atg_cartesian_prod(&out, raw_data(tensors), i32(len(tensors)))
    return track(out)
}

cat :: proc(tensors: []Tensor, dim: i64 = 0) -> Tensor {
    out: Tensor
    t.atg_cat(&out, raw_data(tensors), i32(len(tensors)), dim)
    return track(out)
}

concat :: cat
concatenate :: cat

column_stack :: proc(tensors: []Tensor) -> Tensor {
    out: Tensor
    t.atg_column_stack(&out, raw_data(tensors), i32(len(tensors)))
    return track(out)
}

combinations :: proc(self: Tensor, r: i64 = 2, with_replacement: bool = false) -> Tensor {
    out: Tensor
    t.atg_combinations(&out, self, r, i32(with_replacement))
    return track(out)
}

clone :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_clone(&out, self)
    return track(out)
}

contiguous :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_contiguous(&out, self)
    return track(out)
}

col2im :: proc(self: Tensor, output_size, kernel_size, dilation, padding, stride: []i64) -> Tensor {
    out: Tensor
    t.atg_col2im(&out, self, 
        raw_data(output_size), i32(len(output_size)),
        raw_data(kernel_size), i32(len(kernel_size)),
        raw_data(dilation), i32(len(dilation)),
        raw_data(padding), i32(len(padding)),
        raw_data(stride), i32(len(stride)),
    )
    return track(out)
}

clamp :: proc{
    clamp_scalar, 
    clamp_tensor,
}

clamp_ :: proc{
    clamp_scalar_, 
    clamp_tensor_,
}

// Separate groups for single-bound clamping to avoid ambiguity
clamp_min :: proc{ clamp_min_scalar, clamp_min_tensor }
clamp_min_ :: proc{ clamp_min_scalar_, clamp_min_tensor_ }

clamp_max :: proc{ clamp_max_scalar, clamp_max_tensor }
clamp_max_ :: proc{ clamp_max_scalar_, clamp_max_tensor_ }

// Alias clip to clamp (PyTorch convention)
clip :: clamp
clip_ :: clamp_
clip_min :: clamp_min
clip_max :: clamp_max

@(private)
clamp_scalar :: proc(self: Tensor, min, max: Scalar) -> Tensor {
    out: Tensor
    t.atg_clamp(&out, self, min, max)
    return track(out)
}

@(private)
clamp_scalar_ :: proc(self: Tensor, min, max: Scalar) -> Tensor {
    out: Tensor
    t.atg_clamp_(&out, self, min, max)
    return self
}

@(private)
clamp_tensor :: proc(self: Tensor, min, max: Tensor) -> Tensor {
    out: Tensor
    t.atg_clamp_tensor(&out, self, min, max)
    return track(out)
}

@(private)
clamp_tensor_ :: proc(self: Tensor, min, max: Tensor) -> Tensor {
    out: Tensor
    t.atg_clamp_tensor_(&out, self, min, max)
    return self
}

@(private)
clamp_min_scalar :: proc(self: Tensor, min: Scalar) -> Tensor {
    out: Tensor
    t.atg_clamp_min(&out, self, min)
    return track(out)
}

@(private)
clamp_min_scalar_ :: proc(self: Tensor, min: Scalar) -> Tensor {
    out: Tensor
    t.atg_clamp_min_(&out, self, min)
    return self
}

@(private)
clamp_min_tensor :: proc(self: Tensor, min: Tensor) -> Tensor {
    out: Tensor
    t.atg_clamp_min_tensor(&out, self, min)
    return track(out)
}

@(private)
clamp_min_tensor_ :: proc(self: Tensor, min: Tensor) -> Tensor {
    out: Tensor
    t.atg_clamp_min_tensor_(&out, self, min)
    return self
}

@(private)
clamp_max_scalar :: proc(self: Tensor, max: Scalar) -> Tensor {
    out: Tensor
    t.atg_clamp_max(&out, self, max)
    return track(out)
}

@(private)
clamp_max_scalar_ :: proc(self: Tensor, max: Scalar) -> Tensor {
    out: Tensor
    t.atg_clamp_max_(&out, self, max)
    return self
}

@(private)
clamp_max_tensor :: proc(self: Tensor, max: Tensor) -> Tensor {
    out: Tensor
    t.atg_clamp_max_tensor(&out, self, max)
    return track(out)
}

@(private)
clamp_max_tensor_ :: proc(self: Tensor, max: Tensor) -> Tensor {
    out: Tensor
    t.atg_clamp_max_tensor_(&out, self, max)
    return self
}

// DIVISION

div :: proc{
    div_tensor, 
    div_scalar, 
}

div_ :: proc{
    div_tensor_, 
    div_scalar_,
}

div_mode :: proc{
    div_tensor_mode, 
    div_scalar_mode,
}

div_mode_ :: proc{
    div_tensor_mode_, 
    div_scalar_mode_,
}

@(private)
div_tensor :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_div(&out, self, other)
    return track(out)
}

@(private)
div_scalar :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_div_scalar(&out, self, other)
    return track(out)
}

// TODO: enum "trunc" or "floor"
@(private)
div_tensor_mode :: proc(self, other: Tensor, rounding_mode: string) -> Tensor {
    out: Tensor
    mode_cstr := strings.clone_to_cstring(rounding_mode, context.temp_allocator)
    t.atg_div_tensor_mode(&out, self, other, mode_cstr, i32(len(rounding_mode)))
    return track(out)
}

// TODO: enum "trunc" or "floor"
@(private)
div_scalar_mode :: proc(self: Tensor, other: Scalar, rounding_mode: string) -> Tensor {
    out: Tensor
    mode_cstr := strings.clone_to_cstring(rounding_mode, context.temp_allocator)
    t.atg_div_scalar_mode(&out, self, other, mode_cstr, i32(len(rounding_mode)))
    return track(out)
}

@(private)
div_tensor_ :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_div_(&out, self, other)
    return self
}

@(private)
div_scalar_ :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_div_scalar_(&out, self, other)
    return self
}

// TODO: enum "trunc" or "floor"
@(private)
div_tensor_mode_ :: proc(self, other: Tensor, rounding_mode: string) -> Tensor {
    out: Tensor
    mode_cstr := strings.clone_to_cstring(rounding_mode, context.temp_allocator)
    t.atg_div_tensor_mode_(&out, self, other, mode_cstr, i32(len(rounding_mode)))
    return self
}

// TODO: enum "trunc" or "floor"
@(private)
div_scalar_mode_ :: proc(self: Tensor, other: Scalar, rounding_mode: string) -> Tensor {
    out: Tensor
    mode_cstr := strings.clone_to_cstring(rounding_mode, context.temp_allocator)
    t.atg_div_scalar_mode_(&out, self, other, mode_cstr, i32(len(rounding_mode)))
    return self
}

// EQUALITY
eq :: proc{eq_scalar, eq_tensor}
eq_ :: proc{eq_scalar_, eq_tensor_}

@(private)
eq_tensor :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_eq_tensor(&out, self, other)
    return track(out)
}

@(private)
eq_scalar :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_eq(&out, self, other)
    return track(out)
}

@(private)
eq_tensor_ :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_eq_tensor_(&out, self, other)
    return self
}

@(private)
eq_scalar_ :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_eq_(&out, self, other)
    return self
}

// UNARY MATH
exp :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_exp(&out, self)
    return track(out)
}

exp_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_exp_(&out, self)
    return self
}

erf :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_erf(&out, self)
    return track(out)
}

digamma :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_digamma(&out, self)
    return track(out)
}

// Distance
dist :: proc(self, other: Tensor, p: f64 = 2.0) -> Tensor {
    out: Tensor
    t.atg_dist(&out, self, other)
    return track(out)
}

dot :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_dot(&out, self, other)
    return track(out)
}

det :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_det(&out, self)
    return track(out)
}

// Einstein Summation
einsum :: proc(equation: string, tensors: []Tensor) -> Tensor {
    out: Tensor
    eq_cstr := strings.clone_to_cstring(equation, context.temp_allocator)
    
    // TODO: path_data is optional optimization
    t.atg_einsum(
        &out, 
        eq_cstr, 
        i32(len(equation)), 
        raw_data(tensors), 
        i32(len(tensors)), 
        nil, 
        0,
    )
    return track(out)
}

diag :: proc(self: Tensor, diagonal: i64 = 0) -> Tensor {
    out: Tensor
    t.atg_diag(&out, self, diagonal)
    return track(out)
}

diagonal :: proc(self: Tensor, offset: i64 = 0, dim1: i64 = 0, dim2: i64 = 1) -> Tensor {
    out: Tensor
    t.atg_diagonal(&out, self, offset, dim1, dim2)
    return track(out)
}

// DROPOUT
dropout :: proc(input: Tensor, p: f64, train: bool) -> Tensor {
    out: Tensor
    t.atg_dropout(&out, input, p, i32(1) if train else i32(0))
    return track(out)
}

dropout_ :: proc(self: Tensor, p: f64, train: bool) -> Tensor {
    out: Tensor
    t.atg_dropout_(&out, self, p, i32(1) if train else i32(0))
    return self
}

// EMBEDDING
embedding :: proc(
    weight, indices: Tensor, 
    padding_idx: i64 = -1, 
    scale_grad_by_freq: bool = false, 
    sparse: bool = false,
) -> Tensor {
    out: Tensor
    t.atg_embedding(
        &out, 
        weight, 
        indices, 
        padding_idx, 
        i32(1) if scale_grad_by_freq else i32(0), 
        i32(1) if sparse else i32(0),
    )
    return track(out)
}

// CTC LOSS
ctc_loss_arrays :: proc(
    log_probs, targets: Tensor, 
    input_lengths: []i64, 
    target_lengths: []i64, 
    blank: i64 = 0, 
    reduction: i64 = 1, 
    zero_infinity: bool = false,
) -> Tensor {
    out: Tensor
    t.atg_ctc_loss(
        &out, 
        log_probs, 
        targets, 
        raw_data(input_lengths), i32(len(input_lengths)),
        raw_data(target_lengths), i32(len(target_lengths)),
        blank,
        reduction,
        i32(1) if zero_infinity else i32(0),
    )
    return track(out)
}

ctc_loss_tensors :: proc(
    log_probs, targets: Tensor, 
    input_lengths: Tensor, 
    target_lengths: Tensor, 
    blank: i64 = 0, 
    reduction: i64 = 1, 
    zero_infinity: bool = false,
) -> Tensor {
    out: Tensor
    t.atg_ctc_loss_tensor(
        &out, 
        log_probs, 
        targets, 
        input_lengths,
        target_lengths,
        blank,
        reduction,
        i32(1) if zero_infinity else i32(0),
    )
    return track(out)
}

ctc_loss :: proc(
    log_probs, targets: Tensor, 
    input_lengths, target_lengths: []i64, 
    blank: i64, 
    zero_infinity: bool
) -> Tensor {
    out: Tensor
    t.atg__ctc_loss(
        &out, log_probs, targets,
        raw_data(input_lengths), i32(len(input_lengths)),
        raw_data(target_lengths), i32(len(target_lengths)),
        blank,
        i32(1) if zero_infinity else i32(0),
    )
    return track(out)
}

// FACTORY METHODS
empty :: proc(
    size: []i64, 
    kind: ScalarType = .Float, 
    device: DeviceType = .CPU
) -> Tensor {
    out: Tensor
    t.atg_empty(
        &out, 
        raw_data(size), i32(len(size)), 
        i32(kind), 
        i32(device),
    )
    return track(out)
}

eye :: proc(n: i64, kind: ScalarType = .Float, device: DeviceType = .CPU) -> Tensor {
    out: Tensor
    t.atg_eye(&out, n, i32(kind), i32(device))
    return track(out)
}

eye_m :: proc(n, m: i64, kind: ScalarType = .Float, device: DeviceType = .CPU) -> Tensor {
    out: Tensor
    t.atg_eye_m(&out, n, m, i32(kind), i32(device))
    return track(out)
}

empty_like :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_empty_like(&out, self)
    return track(out)
}

// MANIPULATION
detach :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_detach(&out, self)
    return track(out)
}

detach_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_detach_(&out, self)
    return self
}

dstack :: proc(tensors: []Tensor) -> Tensor {
    out: Tensor
    t.atg_dstack(&out, raw_data(tensors), i32(len(tensors)))
    return track(out)
}

expand :: proc(self: Tensor, size: []i64, implicit: bool = false) -> Tensor {
    out: Tensor
    t.atg_expand(
        &out, 
        self, 
        raw_data(size), 
        i32(len(size)), 
        i32(1) if implicit else i32(0),
    )
    return track(out)
}

cummax :: proc(self: Tensor, dim: i64) -> Tensor {
    // NOTE: PyTorch cummax usually returns (values, indices).
    // TODO: wrapper assumes the binding returns the value tensor.
    out: Tensor
    t.atg_cummax(&out, self, dim)
    return track(out)
}

cummin :: proc(self: Tensor, dim: i64) -> Tensor {
    out: Tensor
    t.atg_cummin(&out, self, dim)
    return track(out)
}

cumprod :: proc(self: Tensor, dim: i64, dtype: ScalarType = .Float) -> Tensor {
    out: Tensor
    t.atg_cumprod(&out, self, dim, i32(dtype))
    return track(out)
}

cumsum :: proc(self: Tensor, dim: i64, dtype: ScalarType = .Float) -> Tensor {
    out: Tensor
    t.atg_cumsum(&out, self, dim, i32(dtype))
    return track(out)
}

cumprod_ :: proc(self: Tensor, dim: i64, dtype: ScalarType = .Float) -> Tensor {
    out: Tensor
    t.atg_cumprod_(&out, self, dim, i32(dtype))
    return self
}

cumsum_ :: proc(self: Tensor, dim: i64, dtype: ScalarType = .Float) -> Tensor {
    out: Tensor
    t.atg_cumsum_(&out, self, dim, i32(dtype))
    return self
}

cummaxmin_backward :: proc(grad, input, indices: Tensor, dim: i64) -> Tensor {
    out: Tensor
    t.atg_cummaxmin_backward(&out, grad, input, indices, dim)
    return track(out)
}

cumprod_backward :: proc(grad, input: Tensor, dim: i64, output: Tensor) -> Tensor {
    out: Tensor
    t.atg_cumprod_backward(&out, grad, input, dim, output)
    return track(out)
}

elu_backward_grad_input :: proc(
    grad_input, grad_output: Tensor, 
    alpha, scale, input_scale: Scalar, 
    is_result: bool, 
    self_or_result: Tensor,
) -> Tensor {
    out: Tensor
    t.atg_elu_backward_grad_input(
        &out, 
        grad_input, grad_output, 
        alpha, scale, input_scale, 
        i32(1) if is_result else i32(0), 
        self_or_result,
    )
    return track(out)
}

embedding_backward :: proc(
    grad, indices: Tensor, 
    num_weights: i64, 
    padding_idx: i64, 
    scale_grad_by_freq: bool, 
    sparse: bool,
) -> Tensor {
    out: Tensor
    t.atg_embedding_backward(
        &out, grad, indices, num_weights, padding_idx, 
        i32(1) if scale_grad_by_freq else i32(0), 
        i32(1) if sparse else i32(0),
    )
    return track(out)
}

embedding_dense_backward :: proc(
    grad_output, indices: Tensor, 
    num_weights: i64, 
    padding_idx: i64, 
    scale_grad_by_freq: bool,
) -> Tensor {
    out: Tensor
    t.atg_embedding_dense_backward(
        &out, grad_output, indices, num_weights, padding_idx, 
        i32(1) if scale_grad_by_freq else i32(0),
    )
    return track(out)
}

embedding_sparse_backward :: proc(
    grad, indices: Tensor, 
    num_weights: i64, 
    padding_idx: i64, 
    scale_grad_by_freq: bool,
) -> Tensor {
    out: Tensor
    t.atg_embedding_sparse_backward(
        &out, grad, indices, num_weights, padding_idx, 
        i32(1) if scale_grad_by_freq else i32(0),
    )
    return track(out)
}

digamma_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_digamma_(&out, self)
    return self
}

// Activations (ELU)

elu :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_elu(&out, self)
    return track(out)
}

elu_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_elu_(&out, self)
    return self
}

// Feature Dropout (often distinct from standard dropout in how it handles channels)
feature_dropout :: proc(input: Tensor, p: f64, train: bool) -> Tensor {
    out: Tensor
    t.atg_feature_dropout(&out, input, p, i32(1) if train else i32(0))
    return track(out)
}

feature_dropout_ :: proc(self: Tensor, p: f64, train: bool) -> Tensor {
    out: Tensor
    t.atg_feature_dropout_(&out, self, p, i32(1) if train else i32(0))
    return self
}

feature_alpha_dropout :: proc(input: Tensor, p: f64, train: bool) -> Tensor {
    out: Tensor
    t.atg_feature_alpha_dropout(&out, input, p, i32(1) if train else i32(0))
    return track(out)
}

feature_alpha_dropout_ :: proc(self: Tensor, p: f64, train: bool) -> Tensor {
    out: Tensor
    t.atg_feature_alpha_dropout_(&out, self, p, i32(1) if train else i32(0))
    return self
}

// Batch Normalization (CUDNN)

cudnn_batch_norm :: proc(
    input, weight, bias: Tensor, 
    running_mean, running_var: Tensor, 
    training: bool, 
    exponential_average_factor: f64, 
    epsilon: f64
) -> Tensor {
    out: Tensor
    t.atg_cudnn_batch_norm(
        &out, 
        input, weight, bias, 
        running_mean, running_var, 
        i32(1) if training else i32(0),
        exponential_average_factor, 
        epsilon,
    )
    return track(out)
}

// Useful for constraining embedding norms during training
embedding_renorm_ :: proc(self, indices: Tensor, max_norm: f64, norm_type: f64) -> Tensor {
    out: Tensor
    t.atg_embedding_renorm_(&out, self, indices, max_norm, norm_type)
    return self
}

deg2rad :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_deg2rad(&out, self)
    return track(out)
}

deg2rad_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_deg2rad_(&out, self)
    return self
}

diff :: proc(
    self: Tensor, 
    n: i64 = 1, 
    dim: i64 = -1, 
    prepend: Tensor = Tensor{}, 
    append: Tensor = Tensor{}
) -> Tensor {
    out: Tensor
    t.atg_diff(&out, self, n, dim, prepend, append)
    return track(out)
}

cumulative_trapezoid :: proc(y: Tensor, x: Tensor = Tensor{}, dim: i64 = -1) -> Tensor {
    out: Tensor
    // If x is defined, use the _x variant, otherwise standard one
    if defined(x) != 0 {
        t.atg_cumulative_trapezoid_x(&out, y, x, dim)
    } else {
        t.atg_cumulative_trapezoid(&out, y, dim)
    }
    return track(out)
}

// Quantization used in QAT - Quantization Aware Training
dequantize :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_dequantize(&out, self)
    return track(out)
}

fake_quantize_per_tensor_affine :: proc(
    self: Tensor, 
    scale: f64, 
    zero_point: i64, 
    quant_min: i64, 
    quant_max: i64
) -> Tensor {
    out: Tensor
    t.atg_fake_quantize_per_tensor_affine(&out, self, scale, zero_point, quant_min, quant_max)
    return track(out)
}

fake_quantize_per_channel_affine :: proc(
    self, scale, zero_point: Tensor, 
    axis: i64, 
    quant_min, quant_max: i64
) -> Tensor {
    out: Tensor
    t.atg_fake_quantize_per_channel_affine(&out, self, scale, zero_point, axis, quant_min, quant_max)
    return track(out)
}

// Empty Tensor Variants

empty_strided :: proc(
    size: []i64, 
    stride: []i64, 
    kind: ScalarType = .Float, 
    device: DeviceType = .CPU
) -> Tensor {
    out: Tensor
    t.atg_empty_strided(
        &out, 
        raw_data(size), i32(len(size)), 
        raw_data(stride), i32(len(stride)), 
        i32(kind), i32(device),
    )
    return track(out)
}

empty_permuted :: proc(
    size: []i64, 
    physical_layout: []i64, 
    kind: ScalarType = .Float, 
    device: DeviceType = .CPU
) -> Tensor {
    out: Tensor
    t.atg_empty_permuted(
        &out, 
        raw_data(size), i32(len(size)), 
        raw_data(physical_layout), i32(len(physical_layout)), 
        i32(kind), i32(device),
    )
    return track(out)
}

// CUDNN Convolution Variants

convolution_relu :: proc(
    self, weight, bias: Tensor,
    stride: []i64,
    padding: []i64,
    dilation: []i64,
    groups: i64 = 1,
) -> Tensor {
    out: Tensor
    t.atg_cudnn_convolution_relu(
        &out,
        self, weight, bias,
        raw_data(stride), i32(len(stride)),
        raw_data(padding), i32(len(padding)),
        raw_data(dilation), i32(len(dilation)),
        groups,
    )
    return track(out)
}

convolution_add_relu :: proc(
    self, weight, z: Tensor,
    alpha: Scalar,
    bias: Tensor,
    stride: []i64,
    padding: []i64,
    dilation: []i64,
    groups: i64 = 1,
) -> Tensor {
    out: Tensor
    t.atg_cudnn_convolution_add_relu(
        &out,
        self, weight, z,
        alpha,
        bias,
        raw_data(stride), i32(len(stride)),
        raw_data(padding), i32(len(padding)),
        raw_data(dilation), i32(len(dilation)),
        groups,
    )
    return track(out)
}

convolution_transpose :: proc(
    self, weight: Tensor,
    padding: []i64,
    output_padding: []i64,
    stride: []i64,
    dilation: []i64,
    groups: i64 = 1,
    benchmark: bool = false,
    deterministic: bool = false,
    allow_tf32: bool = true,
) -> Tensor {
    out: Tensor
    t.atg_cudnn_convolution_transpose(
        &out,
        self, weight,
        raw_data(padding), i32(len(padding)),
        raw_data(output_padding), i32(len(output_padding)),
        raw_data(stride), i32(len(stride)),
        raw_data(dilation), i32(len(dilation)),
        groups,
        i32(1) if benchmark else i32(0),
        i32(1) if deterministic else i32(0),
        i32(1) if allow_tf32 else i32(0),
    )
    return track(out)
}

// Matrix & Diagonal Ops

diag_embed :: proc(self: Tensor, offset: i64 = 0, dim1: i64 = -2, dim2: i64 = -1) -> Tensor {
    out: Tensor
    t.atg_diag_embed(&out, self, offset, dim1, dim2)
    return track(out)
}

diagflat :: proc(self: Tensor, offset: i64 = 0) -> Tensor {
    out: Tensor
    t.atg_diagflat(&out, self, offset)
    return track(out)
}

diagonal_copy :: proc(self: Tensor, offset: i64 = 0, dim1: i64 = 0, dim2: i64 = 1) -> Tensor {
    out: Tensor
    t.atg_diagonal_copy(&out, self, offset, dim1, dim2)
    return track(out)
}

diagonal_scatter :: proc(self, src: Tensor, offset: i64 = 0, dim1: i64 = 0, dim2: i64 = 1) -> Tensor {
    out: Tensor
    t.atg_diagonal_scatter(&out, self, src, offset, dim1, dim2)
    return track(out)
}

// Exponentials

exp2 :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_exp2(&out, self)
    return track(out)
}

exp2_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_exp2_(&out, self)
    return self
}

expm1 :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_expm1(&out, self)
    return track(out)
}

expm1_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_expm1_(&out, self)
    return self
}

erf_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_erf_(&out, self)
    return self
}

erfc :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_erfc(&out, self)
    return track(out)
}

erfc_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_erfc_(&out, self)
    return self
}

erfinv :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_erfinv(&out, self)
    return track(out)
}

erfinv_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_erfinv_(&out, self)
    return self
}

// Random Sampling

exponential :: proc(self: Tensor, lambd: f64 = 1.0) -> Tensor {
    out: Tensor
    t.atg_exponential(&out, self, lambd)
    return track(out)
}

exponential_ :: proc(self: Tensor, lambd: f64 = 1.0) -> Tensor {
    out: Tensor
    t.atg_exponential_(&out, self, lambd)
    return self
}

// DIVISION

divide :: proc{
    divide_tensor, 
    divide_scalar, 
}

divide_ :: proc{
    divide_tensor_, 
    divide_scalar_, 
}

divide_mode :: proc{
    divide_tensor_mode, 
    divide_scalar_mode,
}

divide_mode_ :: proc{
    divide_tensor_mode_, 
    divide_scalar_mode_,
}

@private
divide_tensor :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_divide(&out, self, other)
    return track(out)
}

@private
divide_scalar :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_divide_scalar(&out, self, other)
    return track(out)
}

@private
divide_tensor_mode :: proc(self, other: Tensor, rounding_mode: string) -> Tensor {
    out: Tensor
    mode_cstr := strings.clone_to_cstring(rounding_mode, context.temp_allocator)
    t.atg_divide_tensor_mode(&out, self, other, mode_cstr, i32(len(rounding_mode)))
    return track(out)
}

@private
divide_scalar_mode :: proc(self: Tensor, other: Scalar, rounding_mode: string) -> Tensor {
    out: Tensor
    mode_cstr := strings.clone_to_cstring(rounding_mode, context.temp_allocator)
    t.atg_divide_scalar_mode(&out, self, other, mode_cstr, i32(len(rounding_mode)))
    return track(out)
}

@private
divide_tensor_ :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_divide_(&out, self, other)
    return self
}

@private
divide_scalar_ :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_divide_scalar_(&out, self, other)
    return self
}

@private
divide_tensor_mode_ :: proc(self, other: Tensor, rounding_mode: string) -> Tensor {
    out: Tensor
    mode_cstr := strings.clone_to_cstring(rounding_mode, context.temp_allocator)
    t.atg_divide_tensor_mode_(&out, self, other, mode_cstr, i32(len(rounding_mode)))
    return self
}

@private
divide_scalar_mode_ :: proc(self: Tensor, other: Scalar, rounding_mode: string) -> Tensor {
    out: Tensor
    mode_cstr := strings.clone_to_cstring(rounding_mode, context.temp_allocator)
    t.atg_divide_scalar_mode_(&out, self, other, mode_cstr, i32(len(rounding_mode)))
    return self
}

// Returns the underlying data tensor
tensor_data :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_data(&out, self)
    return track(out)
}

empty_quantized :: proc(
    size: []i64, 
    qtensor: Tensor, 
    kind: ScalarType = .Float, 
    device: DeviceType = .CPU,
) -> Tensor {
    out: Tensor
    t.atg_empty_quantized(
        &out, 
        raw_data(size), i32(len(size)), 
        qtensor,
        i32(kind), 
        i32(device),
    )
    return track(out)
}

// Backward Functions (Manual Autograd)

cudnn_affine_grid_generator_backward :: proc(grad: Tensor, n, c_dim, h, w: i64) -> Tensor {
    out: Tensor
    t.atg_cudnn_affine_grid_generator_backward(&out, grad, n, c_dim, h, w)
    return track(out)
}

cudnn_batch_norm_backward :: proc(
    input, grad_output, weight, 
    running_mean, running_var, 
    save_mean, save_var: Tensor, 
    epsilon: f64, 
    reserveSpace: Tensor
) -> Tensor {
    out: Tensor
    t.atg_cudnn_batch_norm_backward(
        &out, 
        input, grad_output, weight, 
        running_mean, running_var, 
        save_mean, save_var, 
        epsilon, 
        reserveSpace,
    )
    return track(out)
}

cudnn_grid_sampler_backward :: proc(self, grid, grad_output: Tensor) -> Tensor {
    out: Tensor
    t.atg_cudnn_grid_sampler_backward(&out, self, grid, grad_output)
    return track(out)
}

// NOTE: elu_backward takes "is_result" (boolean as int) indicating if self_or_result is the output or input
elu_backward :: proc(
    grad_output: Tensor, 
    alpha, scale, input_scale: Scalar, 
    is_result: bool, 
    self_or_result: Tensor
) -> Tensor {
    out: Tensor
    t.atg_elu_backward(
        &out, 
        grad_output, 
        alpha, scale, input_scale, 
        i32(1) if is_result else i32(0), 
        self_or_result,
    )
    return track(out)
}

diagonal_backward :: proc(
    grad_output: Tensor, 
    input_sizes: []i64, 
    offset: i64, 
    dim1: i64, 
    dim2: i64
) -> Tensor {
    out: Tensor
    t.atg_diagonal_backward(
        &out, 
        grad_output, 
        raw_data(input_sizes), i32(len(input_sizes)), 
        offset, dim1, dim2,
    )
    return track(out)
}

// Copy & Expand

detach_copy :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_detach_copy(&out, self)
    return track(out)
}

expand_as :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_expand_as(&out, self, other)
    return track(out)
}

expand_copy :: proc(self: Tensor, size: []i64, implicit: bool = false) -> Tensor {
    out: Tensor
    t.atg_expand_copy(
        &out, 
        self, 
        raw_data(size), i32(len(size)), 
        i32(1) if implicit else i32(0),
    )
    return track(out)
}

// FFT Functions (Fast Fourier Transforms)

// Helper to handle optional integers for C bindings
// If `n` is set, we pass the value and a nil pointer.
// If `n` is logically "None", we pass a dummy pointer to signal nullability to the C wrapper.
@(private)
_fft_n_args :: proc(n: Maybe(i64)) -> (val: i64, null_ptr: rawptr) {
    if value, ok := n.?; ok {
        return value, nil
    }
    // Static dummy to use as address for "is null" signal
    @static dummy: i64 = 0 
    return 0, &dummy
}

// TODO: handle string normalization args ("forward", "backward", "ortho")
@(private)
_fft_norm_args :: proc(norm: string) -> (cstring, i32) {
    if len(norm) == 0 { return nil, 0 }
    // Unsafe is okay here because the C call is synchronous and copies if needed
    return strings.unsafe_string_to_cstring(norm), i32(len(norm))
}

//  1D FFT

fft :: proc(self: Tensor, n: Maybe(i64) = nil, dim: i64 = -1, norm: string = "") -> Tensor {
    out: Tensor
    n_val, n_ptr := _fft_n_args(n)
    norm_ptr, norm_len := _fft_norm_args(norm)
    
    t.atg_fft_fft(&out, self, n_val, n_ptr, dim, norm_ptr, norm_len)
    return track(out)
}

ifft :: proc(self: Tensor, n: Maybe(i64) = nil, dim: i64 = -1, norm: string = "") -> Tensor {
    out: Tensor
    n_val, n_ptr := _fft_n_args(n)
    norm_ptr, norm_len := _fft_norm_args(norm)

    t.atg_fft_ifft(&out, self, n_val, n_ptr, dim, norm_ptr, norm_len)
    return track(out)
}

rfft :: proc(self: Tensor, n: Maybe(i64) = nil, dim: i64 = -1, norm: string = "") -> Tensor {
    out: Tensor
    n_val, n_ptr := _fft_n_args(n)
    norm_ptr, norm_len := _fft_norm_args(norm)

    t.atg_fft_rfft(&out, self, n_val, n_ptr, dim, norm_ptr, norm_len)
    return track(out)
}

irfft :: proc(self: Tensor, n: Maybe(i64) = nil, dim: i64 = -1, norm: string = "") -> Tensor {
    out: Tensor
    n_val, n_ptr := _fft_n_args(n)
    norm_ptr, norm_len := _fft_norm_args(norm)

    t.atg_fft_irfft(&out, self, n_val, n_ptr, dim, norm_ptr, norm_len)
    return track(out)
}

hfft :: proc(self: Tensor, n: Maybe(i64) = nil, dim: i64 = -1, norm: string = "") -> Tensor {
    out: Tensor
    n_val, n_ptr := _fft_n_args(n)
    norm_ptr, norm_len := _fft_norm_args(norm)

    t.atg_fft_hfft(&out, self, n_val, n_ptr, dim, norm_ptr, norm_len)
    return track(out)
}

ihfft :: proc(self: Tensor, n: Maybe(i64) = nil, dim: i64 = -1, norm: string = "") -> Tensor {
    out: Tensor
    n_val, n_ptr := _fft_n_args(n)
    norm_ptr, norm_len := _fft_norm_args(norm)

    t.atg_fft_ihfft(&out, self, n_val, n_ptr, dim, norm_ptr, norm_len)
    return track(out)
}

//  2D FFT

fft2 :: proc(self: Tensor, s: []i64 = nil, dim: []i64 = nil, norm: string = "") -> Tensor {
    out: Tensor
    norm_ptr, norm_len := _fft_norm_args(norm)
    
    // Default handles for nil slices
    s_ptr := raw_data(s)
    s_len := i32(len(s))
    dim_ptr := raw_data(dim)
    dim_len := i32(len(dim))

    t.atg_fft_fft2(&out, self, s_ptr, s_len, dim_ptr, dim_len, norm_ptr, norm_len)
    return track(out)
}

ifft2 :: proc(self: Tensor, s: []i64 = nil, dim: []i64 = nil, norm: string = "") -> Tensor {
    out: Tensor
    norm_ptr, norm_len := _fft_norm_args(norm)
    
    t.atg_fft_ifft2(&out, self, raw_data(s), i32(len(s)), raw_data(dim), i32(len(dim)), norm_ptr, norm_len)
    return track(out)
}

rfft2 :: proc(self: Tensor, s: []i64 = nil, dim: []i64 = nil, norm: string = "") -> Tensor {
    out: Tensor
    norm_ptr, norm_len := _fft_norm_args(norm)

    t.atg_fft_rfft2(&out, self, raw_data(s), i32(len(s)), raw_data(dim), i32(len(dim)), norm_ptr, norm_len)
    return track(out)
}

irfft2 :: proc(self: Tensor, s: []i64 = nil, dim: []i64 = nil, norm: string = "") -> Tensor {
    out: Tensor
    norm_ptr, norm_len := _fft_norm_args(norm)

    t.atg_fft_irfft2(&out, self, raw_data(s), i32(len(s)), raw_data(dim), i32(len(dim)), norm_ptr, norm_len)
    return track(out)
}

hfft2 :: proc(self: Tensor, s: []i64 = nil, dim: []i64 = nil, norm: string = "") -> Tensor {
    out: Tensor
    norm_ptr, norm_len := _fft_norm_args(norm)

    t.atg_fft_hfft2(&out, self, raw_data(s), i32(len(s)), raw_data(dim), i32(len(dim)), norm_ptr, norm_len)
    return track(out)
}

ihfft2 :: proc(self: Tensor, s: []i64 = nil, dim: []i64 = nil, norm: string = "") -> Tensor {
    out: Tensor
    norm_ptr, norm_len := _fft_norm_args(norm)

    t.atg_fft_ihfft2(&out, self, raw_data(s), i32(len(s)), raw_data(dim), i32(len(dim)), norm_ptr, norm_len)
    return track(out)
}

//  N-Dimensional FFT

fftn :: proc(self: Tensor, s: []i64 = nil, dim: []i64 = nil, norm: string = "") -> Tensor {
    out: Tensor
    norm_ptr, norm_len := _fft_norm_args(norm)

    t.atg_fft_fftn(&out, self, raw_data(s), i32(len(s)), raw_data(dim), i32(len(dim)), norm_ptr, norm_len)
    return track(out)
}

ifftn :: proc(self: Tensor, s: []i64 = nil, dim: []i64 = nil, norm: string = "") -> Tensor {
    out: Tensor
    norm_ptr, norm_len := _fft_norm_args(norm)

    t.atg_fft_ifftn(&out, self, raw_data(s), i32(len(s)), raw_data(dim), i32(len(dim)), norm_ptr, norm_len)
    return track(out)
}

rfftn :: proc(self: Tensor, s: []i64 = nil, dim: []i64 = nil, norm: string = "") -> Tensor {
    out: Tensor
    norm_ptr, norm_len := _fft_norm_args(norm)

    t.atg_fft_rfftn(&out, self, raw_data(s), i32(len(s)), raw_data(dim), i32(len(dim)), norm_ptr, norm_len)
    return track(out)
}

irfftn :: proc(self: Tensor, s: []i64 = nil, dim: []i64 = nil, norm: string = "") -> Tensor {
    out: Tensor
    norm_ptr, norm_len := _fft_norm_args(norm)

    t.atg_fft_irfftn(&out, self, raw_data(s), i32(len(s)), raw_data(dim), i32(len(dim)), norm_ptr, norm_len)
    return track(out)
}

hfftn :: proc(self: Tensor, s: []i64 = nil, dim: []i64 = nil, norm: string = "") -> Tensor {
    out: Tensor
    norm_ptr, norm_len := _fft_norm_args(norm)

    t.atg_fft_hfftn(&out, self, raw_data(s), i32(len(s)), raw_data(dim), i32(len(dim)), norm_ptr, norm_len)
    return track(out)
}

ihfftn :: proc(self: Tensor, s: []i64 = nil, dim: []i64 = nil, norm: string = "") -> Tensor {
    out: Tensor
    norm_ptr, norm_len := _fft_norm_args(norm)

    t.atg_fft_ihfftn(&out, self, raw_data(s), i32(len(s)), raw_data(dim), i32(len(dim)), norm_ptr, norm_len)
    return track(out)
}

fftshift :: proc(self: Tensor, dim: []i64 = nil) -> Tensor {
    out: Tensor
    t.atg_fft_fftshift(&out, self, raw_data(dim), i32(len(dim)))
    return track(out)
}

ifftshift :: proc(self: Tensor, dim: []i64 = nil) -> Tensor {
    out: Tensor
    t.atg_fft_ifftshift(&out, self, raw_data(dim), i32(len(dim)))
    return track(out)
}

fftfreq :: proc(n: i64, d: f64 = 1.0, dtype: ScalarType = .Float, device: DeviceType = .CPU) -> Tensor {
    out: Tensor
    t.atg_fft_fftfreq(&out, n, d, i32(dtype), i32(device))
    return track(out)
}

rfftfreq :: proc(n: i64, d: f64 = 1.0, dtype: ScalarType = .Float, device: DeviceType = .CPU) -> Tensor {
    out: Tensor
    t.atg_fft_rfftfreq(&out, n, d, i32(dtype), i32(device))
    return track(out)
}

// Fill & Fix

fill :: proc{fill_scalar, fill_tensor, fill_int64, fill_double}
fill_ :: proc{fill_scalar_, fill_tensor_}

fill_int64 :: proc(self: Tensor, value: i64) {
    t.at_fill_int64(self, value)
}

fill_double :: proc(self: Tensor, value: f64) {
    t.at_fill_double(self, value)
}

@private
fill_scalar :: proc(self: Tensor, value: Scalar) -> Tensor {
    out: Tensor
    t.atg_fill(&out, self, value)
    return track(out)
}

@private
fill_tensor :: proc(self: Tensor, value: Tensor) -> Tensor {
    out: Tensor
    t.atg_fill_tensor(&out, self, value)
    return track(out)
}

@private
fill_scalar_ :: proc(self: Tensor, value: Scalar) -> Tensor {
    out: Tensor
    t.atg_fill_(&out, self, value)
    return self
}

@private
fill_tensor_ :: proc(self: Tensor, value: Tensor) -> Tensor {
    out: Tensor
    t.atg_fill_tensor_(&out, self, value)
    return self
}

fill_diagonal_ :: proc(self: Tensor, fill_value: Scalar, wrap: bool = false) -> Tensor {
    out: Tensor
    wrap_int := i32(1) if wrap else i32(0)
    t.atg_fill_diagonal_(&out, self, fill_value, wrap_int)
    return self
}

fix :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_fix(&out, self)
    return track(out)
}

fix_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_fix_(&out, self)
    return self
}

// Flatten & Flip

flatten :: proc(self: Tensor, start_dim: i64 = 0, end_dim: i64 = -1) -> Tensor {
    out: Tensor
    t.atg_flatten(&out, self, start_dim, end_dim)
    return track(out)
}

flatten_dense_tensors :: proc(tensors: []Tensor) -> Tensor {
    out: Tensor
    t.atg_flatten_dense_tensors(
        &out, 
        raw_data(tensors), 
        i32(len(tensors)),
    )
    return track(out)
}

flip :: proc(self: Tensor, dims: []i64) -> Tensor {
    out: Tensor
    t.atg_flip(&out, self, raw_data(dims), i32(len(dims)))
    return track(out)
}

fliplr :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_fliplr(&out, self)
    return track(out)
}

flipud :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_flipud(&out, self)
    return track(out)
}

// POWER

float_power :: proc{
    float_power_tensor, 
    float_power_scalar_base, // scalar ^ tensor
    float_power_tensor_exp_scalar // tensor ^ scalar
}

float_power_ :: proc{
    float_power_tensor_,
    float_power_tensor_exp_scalar_
}

float_power_tensor :: proc(self: Tensor, exponent: Tensor) -> Tensor {
    out: Tensor
    t.atg_float_power(&out, self, exponent)
    return track(out)
}

// self = base ^ exponent where base is scalar
float_power_scalar_base :: proc(base: Scalar, exponent: Tensor) -> Tensor {
    out: Tensor
    t.atg_float_power_scalar(&out, base, exponent)
    return track(out)
}

// self = self ^ exponent where exponent is scalar
float_power_tensor_exp_scalar :: proc(self: Tensor, exponent: Scalar) -> Tensor {
    out: Tensor
    t.atg_float_power_tensor_scalar(&out, self, exponent)
    return track(out)
}


float_power_tensor_ :: proc(self: Tensor, exponent: Tensor) -> Tensor {
    out: Tensor
    t.atg_float_power_tensor_(&out, self, exponent)
    return self
}

float_power_tensor_exp_scalar_ :: proc(self: Tensor, exponent: Scalar) -> Tensor {
    out: Tensor
    t.atg_float_power_(&out, self, exponent)
    return self
}

// FLOOR

floor :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_floor(&out, self)
    return track(out)
}

floor_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_floor_(&out, self)
    return self
}

floor_divide :: proc{floor_divide_tensor, floor_divide_scalar}
floor_divide_ :: proc{floor_divide_tensor_, floor_divide_scalar_}

@private
floor_divide_tensor :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_floor_divide(&out, self, other)
    return track(out)
}

@private
floor_divide_scalar :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_floor_divide_scalar(&out, self, other)
    return track(out)
}

@private
floor_divide_tensor_ :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_floor_divide_(&out, self, other)
    return self
}

@private
floor_divide_scalar_ :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_floor_divide_scalar_(&out, self, other)
    return self
}

fmax :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_fmax(&out, self, other)
    return track(out)
}

fmin :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_fmin(&out, self, other)
    return track(out)
}

fmod :: proc{fmod_tensor, fmod_scalar}
fmod_ :: proc{fmod_tensor_, fmod_scalar_}

@private
fmod_tensor :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_fmod_tensor(&out, self, other)
    return track(out)
}

@private
fmod_scalar :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_fmod(&out, self, other)
    return track(out)
}

@private
fmod_tensor_ :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_fmod_tensor_(&out, self, other)
    return self
}

@private
fmod_scalar_ :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_fmod_(&out, self, other)
    return self
}

frac :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_frac(&out, self)
    return track(out)
}

frac_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_frac_(&out, self)
    return self
}

// FRACTIONAL MAX POOL 2D

fractional_max_pool2d :: proc(
    self: Tensor, 
    kernel_size: []i64, 
    output_size: []i64, 
    random_samples: Tensor
) -> (output: Tensor, indices: Tensor) {
    // Pre-allocate tracked tensors to hold results
    output = new_tensor()
    indices = new_tensor()
    
    dummy: Tensor
    t.atg_fractional_max_pool2d_output(
        &dummy,
        output,
        indices,
        self,
        raw_data(kernel_size), i32(len(kernel_size)),
        raw_data(output_size), i32(len(output_size)),
        random_samples,
    )
    return output, indices
}

fractional_max_pool2d_backward :: proc(
    grad_output: Tensor,
    self: Tensor,
    kernel_size: []i64,
    output_size: []i64,
    indices: Tensor,
) -> Tensor {
    // Use named grad_input version for clarity
    grad_input := new_tensor()
    
    dummy: Tensor
    t.atg_fractional_max_pool2d_backward_grad_input(
        &dummy,
        grad_input,
        grad_output,
        self,
        raw_data(kernel_size), i32(len(kernel_size)),
        raw_data(output_size), i32(len(output_size)),
        indices,
    )
    return grad_input
}

// FRACTIONAL MAX POOL 3D

fractional_max_pool3d :: proc(
    self: Tensor, 
    kernel_size: []i64, 
    output_size: []i64, 
    random_samples: Tensor
) -> (output: Tensor, indices: Tensor) {
    output = new_tensor()
    indices = new_tensor()
    
    dummy: Tensor
    t.atg_fractional_max_pool3d_output(
        &dummy,
        output,
        indices,
        self,
        raw_data(kernel_size), i32(len(kernel_size)),
        raw_data(output_size), i32(len(output_size)),
        random_samples,
    )
    return output, indices
}

fractional_max_pool3d_backward :: proc(
    grad_output: Tensor,
    self: Tensor,
    kernel_size: []i64,
    output_size: []i64,
    indices: Tensor,
) -> Tensor {
    grad_input := new_tensor()
    
    dummy: Tensor
    t.atg_fractional_max_pool3d_backward_grad_input(
        &dummy,
        grad_input,
        grad_output,
        self,
        raw_data(kernel_size), i32(len(kernel_size)),
        raw_data(output_size), i32(len(output_size)),
        indices,
    )
    return grad_input
}

// MATH & ARITHMETIC

frexp :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_frexp(&out, self)
    return track(out)
}

gcd_tensor :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_gcd(&out, self, other)
    return track(out)
}

gcd_tensor_ :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_gcd_(&out, self, other)
    return self
}

geometric:: proc(self: Tensor, p: f64) -> Tensor {
    out: Tensor
    t.atg_geometric(&out, self, p)
    return track(out)
}

geometric_ :: proc(self: Tensor, p: f64) -> Tensor {
    out: Tensor
    t.atg_geometric_(&out, self, p)
    return self
}

// LINEAR ALGEBRA

frobenius_norm :: proc(self: Tensor, dim: []i64 = nil, keepdim: bool = false) -> Tensor {
    out: Tensor
    keep_int := i32(1) if keepdim else i32(0)
    
    // Handle nil slice safely
    dim_ptr := raw_data(dim)
    dim_len := i32(len(dim))
    
    t.atg_frobenius_norm(&out, self, dim_ptr, dim_len, keep_int)
    return track(out)
}

// Returns Q, Tau (Wrapped in single handle based on provided signature)
geqrf :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_geqrf(&out, self)
    return track(out)
}

// Out-param version using specific a/tau buffers
geqrf_out :: proc(self: Tensor, a_out: Tensor, tau_out: Tensor) -> (Tensor, Tensor) {
    dummy: Tensor
    t.atg_geqrf_a(&dummy, a_out, tau_out, self)
    return a_out, tau_out
}

ger :: proc(self: Tensor, vec2: Tensor) -> Tensor {
    out: Tensor
    t.atg_ger(&out, self, vec2)
    return track(out)
}

// CREATION & IO

from_file :: proc(
    filename: string, 
    shared: bool = false, 
    size: i64 = 0, 
    kind: ScalarType = .Float, 
    device: DeviceType = .CPU,
) -> Tensor {
    out: Tensor
    c_filename := strings.clone_to_cstring(filename, context.temp_allocator)
    shared_int := i32(1) if shared else i32(0)
    
    t.atg_from_file(
        &out, 
        c_filename, 
        i32(len(filename)), 
        shared_int, 
        size, 
        nil, 
        i32(kind), 
        i32(device),
    )
    return track(out)
}

full :: proc(size: []i64, fill_value: Scalar, kind: ScalarType = .Float, device: DeviceType = .CPU) -> Tensor {
    out: Tensor
    t.atg_full(
        &out, 
        raw_data(size), 
        i32(len(size)), 
        fill_value, 
        i32(kind), 
        i32(device),
    )
    return track(out)
}

full_like :: proc(self: Tensor, fill_value: Scalar) -> Tensor {
    out: Tensor
    t.atg_full_like(&out, self, fill_value)
    return track(out)
}

// INDEXING & GATHERING

gather :: proc(self: Tensor, dim: i64, index: Tensor, sparse_grad: bool = false) -> Tensor {
    out: Tensor
    sparse_int := i32(1) if sparse_grad else i32(0)
    t.atg_gather(&out, self, dim, index, sparse_int)
    return track(out)
}

gather_backward :: proc(grad: Tensor, self: Tensor, dim: i64, index: Tensor, sparse_grad: bool = false) -> Tensor {
    out: Tensor
    sparse_int := i32(1) if sparse_grad else i32(0)
    t.atg_gather_backward(&out, grad, self, dim, index, sparse_int)
    return track(out)
}

// COMPARISON (Greater or Equal)

ge :: proc{ge_scalar, ge_tensor}
ge_ :: proc{ge_scalar_, ge_tensor_}

@private
ge_scalar :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_ge(&out, self, other)
    return track(out)
}

@private
ge_scalar_ :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_ge_(&out, self, other)
    return self
}

@private
ge_tensor :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_ge_tensor(&out, self, other)
    return track(out)
}

@private
ge_tensor_ :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_ge_tensor_(&out, self, other)
    return self
}

// ACTIVATIONS

// approximate: "none" or "tanh"

gelu:: proc(self: Tensor, approximate: string = "none") -> Tensor {
    out: Tensor
    c_approx := strings.clone_to_cstring(approximate, context.temp_allocator)
    t.atg_gelu(&out, self, c_approx, i32(len(approximate)))
    return track(out)
}

gelu_ :: proc(self: Tensor, approximate: string = "none") -> Tensor {
    out: Tensor
    c_approx := strings.clone_to_cstring(approximate, context.temp_allocator)
    t.atg_gelu_(&out, self, c_approx, i32(len(approximate)))
    return self
}

gelu_backward :: proc(grad_output: Tensor, self: Tensor, approximate: string = "none") -> Tensor {
    out: Tensor
    c_approx := strings.clone_to_cstring(approximate, context.temp_allocator)
    t.atg_gelu_backward(&out, grad_output, self, c_approx, i32(len(approximate)))
    return track(out)
}

gelu_backward_grad_input :: proc(grad_input: Tensor, grad_output: Tensor, self: Tensor, approximate: string = "none") -> Tensor {
    out: Tensor
    c_approx := strings.clone_to_cstring(approximate, context.temp_allocator)
    t.atg_gelu_backward_grad_input(&out, grad_input, grad_output, self, c_approx, i32(len(approximate)))
    return track(out)
}

glu :: proc(self: Tensor, dim: i64 = -1) -> Tensor {
    out: Tensor
    t.atg_glu(&out, self, dim)
    return track(out)
}

glu_backward :: proc(grad_output: Tensor, self: Tensor, dim: i64 = -1) -> Tensor {
    out: Tensor
    t.atg_glu_backward(&out, grad_output, self, dim)
    return track(out)
}

glu_backward_grad_input :: proc(grad_input: Tensor, grad_output: Tensor, self: Tensor, dim: i64 = -1) -> Tensor {
    out: Tensor
    t.atg_glu_backward_grad_input(&out, grad_input, grad_output, self, dim)
    return track(out)
}

glu_jvp :: proc(glu_out: Tensor, x: Tensor, dx: Tensor, dim: i64 = -1) -> Tensor {
    out: Tensor
    t.atg_glu_jvp(&out, glu_out, x, dx, dim)
    return track(out)
}

glu_backward_jvp :: proc(
    grad_x: Tensor, 
    grad_glu: Tensor, 
    x: Tensor, 
    dgrad_glu: Tensor, 
    dx: Tensor, 
    dim: i64 = -1
) -> Tensor {
    out: Tensor
    t.atg_glu_backward_jvp(&out, grad_x, grad_glu, x, dgrad_glu, dx, dim)
    return track(out)
}

fused_moving_avg_obs_fake_quant :: proc(
    self: Tensor,
    observer_on: Tensor,
    fake_quant_on: Tensor,
    running_min: Tensor,
    running_max: Tensor,
    scale: Tensor,
    zero_point: Tensor,
    averaging_const: f64,
    quant_min: i64,
    quant_max: i64,
    ch_axis: i64,
    per_row_fake_quant: bool = false,
    symmetric_quant: bool = false
) -> Tensor {
    out: Tensor
    per_row_int := i32(1) if per_row_fake_quant else i32(0)
    sym_int := i32(1) if symmetric_quant else i32(0)

    t.atg_fused_moving_avg_obs_fake_quant(
        &out,
        self,
        observer_on,
        fake_quant_on,
        running_min,
        running_max,
        scale,
        zero_point,
        averaging_const,
        quant_min,
        quant_max,
        ch_axis,
        per_row_int,
        sym_int,
    )
    return track(out)
}

grad :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_grad(&out, self)
    return track(out)
}

// HARDSHRINK

// Forward: The binding provided does not accept a lambda argument
hardshrink :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_hardshrink(&out, self)
    return track(out)
}

// Backward: The binding requires 'lambd'
hardshrink_backward :: proc(grad_out: Tensor, self: Tensor, lambd: Scalar) -> Tensor {
    out: Tensor
    t.atg_hardshrink_backward(&out, grad_out, self, lambd)
    return track(out)
}

// Backward with explicit gradient input buffer
hardshrink_backward_grad_input :: proc(grad_input: Tensor, grad_out: Tensor, self: Tensor, lambd: Scalar) -> Tensor {
    out: Tensor
    t.atg_hardshrink_backward_grad_input(&out, grad_input, grad_out, self, lambd)
    return track(out)
}

// HARDSIGMOID

hardsigmoid :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_hardsigmoid(&out, self)
    return track(out)
}

hardsigmoid_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_hardsigmoid_(&out, self)
    return self
}

hardsigmoid_backward :: proc(grad_output: Tensor, self: Tensor) -> Tensor {
    out: Tensor
    t.atg_hardsigmoid_backward(&out, grad_output, self)
    return track(out)
}

// HARDSWISH

hardswish :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_hardswish(&out, self)
    return track(out)
}

hardswish_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_hardswish_(&out, self)
    return self
}

hardswish_backward :: proc(grad_output: Tensor, self: Tensor) -> Tensor {
    out: Tensor
    t.atg_hardswish_backward(&out, grad_output, self)
    return track(out)
}

// HARDTANH

hardtanh :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_hardtanh(&out, self)
    return track(out)
}

hardtanh_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_hardtanh_(&out, self)
    return self
}

hardtanh_backward :: proc(grad_output: Tensor, self: Tensor, min_val: Scalar, max_val: Scalar) -> Tensor {
    out: Tensor
    t.atg_hardtanh_backward(&out, grad_output, self, min_val, max_val)
    return track(out)
}

// HEAVISIDE

heaviside :: proc(self: Tensor, values: Tensor) -> Tensor {
    out: Tensor
    t.atg_heaviside(&out, self, values)
    return track(out)
}

heaviside_ :: proc(self: Tensor, values: Tensor) -> Tensor {
    out: Tensor
    t.atg_heaviside_(&out, self, values)
    return self
}

// HASHING & HISTOGRAMS

hash_tensor :: proc(self: Tensor, dim: []i64, keepdim: bool, mode: i64) -> Tensor {
    out: Tensor
    keep_int := i32(1) if keepdim else i32(0)
    t.atg_hash_tensor(
        &out, 
        self, 
        raw_data(dim), 
        i32(len(dim)), 
        keep_int, 
        mode,
    )
    return track(out)
}

histc :: proc(self: Tensor, bins: i64) -> Tensor {
    out: Tensor
    t.atg_histc(&out, self, bins)
    return track(out)
}

histogram :: proc(self: Tensor, bins: Tensor, weight: Tensor = {}, density: bool = false) -> Tensor {
    out: Tensor
    dens_int := i32(1) if density else i32(0)
    t.atg_histogram(&out, self, bins, weight, dens_int)
    return track(out)
}

// Wrapper for atg_histogram_bin_ct

histogram_range :: proc(self: Tensor, bins: i64, range: []f64, weight: Tensor = {}, density: bool = false) -> Tensor {
    out: Tensor
    dens_int := i32(1) if density else i32(0)
    t.atg_histogram_bin_ct(
        &out, 
        self, 
        bins, 
        raw_data(range), 
        i32(len(range)), 
        weight, 
        dens_int,
    )
    return track(out)
}

// LOSS FUNCTIONS

hinge_embedding_loss :: proc(self: Tensor, target: Tensor, margin: f64, reduction: i64) -> Tensor {
    out: Tensor
    t.atg_hinge_embedding_loss(&out, self, target, margin, reduction)
    return track(out)
}

huber_loss :: proc(self: Tensor, target: Tensor, reduction: i64, delta: f64) -> Tensor {
    out: Tensor
    t.atg_huber_loss(&out, self, target, reduction, delta)
    return track(out)
}

huber_loss_backward :: proc(grad_output: Tensor, self: Tensor, target: Tensor, reduction: i64, delta: f64) -> Tensor {
    out: Tensor
    t.atg_huber_loss_backward(&out, grad_output, self, target, reduction, delta)
    return track(out)
}

// MATH & LINEAR ALGEBRA

hspmm :: proc(mat1: Tensor, mat2: Tensor) -> Tensor {
    out: Tensor
    t.atg_hspmm(&out, mat1, mat2)
    return track(out)
}

hstack :: proc(tensors: []Tensor) -> Tensor {
    out: Tensor
    t.atg_hstack(&out, raw_data(tensors), i32(len(tensors)))
    return track(out)
}

hypot :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_hypot(&out, self, other)
    return track(out)
}

hypot_ :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_hypot_(&out, self, other)
    return self
}

// i0 (Modified Bessel function of order 0)
i0 :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_i0(&out, self)
    return track(out)
}

i0_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_i0_(&out, self)
    return self
}

// igamma (Regularized lower incomplete gamma function)
igamma :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_igamma(&out, self, other)
    return track(out)
}

igamma_ :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_igamma_(&out, self, other)
    return self
}

// igammac (Regularized upper incomplete gamma function)
igammac :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_igammac(&out, self, other)
    return track(out)
}

igammac_ :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_igammac_(&out, self, other)
    return self
}

// imag (Imaginary part of a complex tensor)
imag :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_imag(&out, self)
    return track(out)
}

// Image Manipulation

// im2col - Image to Column
im2col :: proc(
    self: Tensor, 
    kernel_size: []i64, 
    dilation: []i64, 
    padding: []i64, 
    stride: []i64,
) -> Tensor {
    out: Tensor
    t.atg_im2col(
        &out, 
        self, 
        raw_data(kernel_size), i32(len(kernel_size)),
        raw_data(dilation),    i32(len(dilation)),
        raw_data(padding),     i32(len(padding)),
        raw_data(stride),      i32(len(stride)),
    )
    return track(out)
}

// Indexing & Slicing
// Indexing - Equivalent to tensor[indices]
index :: proc(self: Tensor, indices: []Tensor) -> Tensor {
    out: Tensor
    t.atg_index(&out, self, raw_data(indices), i32(len(indices)))
    return track(out)
}

// Index Add
index_add :: proc(self: Tensor, dim: i64, idx: Tensor, source: Tensor) -> Tensor {
    out: Tensor
    t.atg_index_add(&out, self, dim, idx, source)
    return track(out)
}

index_add_ :: proc(self: Tensor, dim: i64, idx: Tensor, source: Tensor) -> Tensor {
    out: Tensor
    t.atg_index_add_(&out, self, dim, idx, source)
    return self
}

// Index Copy
index_copy :: proc(self: Tensor, dim: i64, idx: Tensor, source: Tensor) -> Tensor {
    out: Tensor
    t.atg_index_copy(&out, self, dim, idx, source)
    return track(out)
}

index_copy_ :: proc(self: Tensor, dim: i64, idx: Tensor, source: Tensor) -> Tensor {
    out: Tensor
    t.atg_index_copy_(&out, self, dim, idx, source)
    return self
}

// Index Fill - Polyform for Scalar and Tensor values
index_fill :: proc{index_fill_scalar, index_fill_tensor}
index_fill_ :: proc{index_fill_scalar_, index_fill_tensor_}

@private
index_fill_scalar :: proc(self: Tensor, dim: i64, idx: Tensor, value: Scalar) -> Tensor {
    out: Tensor
    t.atg_index_fill(&out, self, dim, idx, value)
    return track(out)
}

@private
index_fill_scalar_ :: proc(self: Tensor, dim: i64, idx: Tensor, value: Scalar) -> Tensor {
    out: Tensor
    t.atg_index_fill_(&out, self, dim, idx, value)
    return self
}

@private
index_fill_tensor :: proc(self: Tensor, dim: i64, idx: Tensor, value: Tensor) -> Tensor {
    out: Tensor
    t.atg_index_fill_int_tensor(&out, self, dim, idx, value)
    return track(out)
}

@private
index_fill_tensor_ :: proc(self: Tensor, dim: i64, idx: Tensor, value: Tensor) -> Tensor {
    out: Tensor
    t.atg_index_fill_int_tensor_(&out, self, dim, idx, value)
    return self
}

// Index Put - Equivalent to tensor[indices] = values
index_put :: proc(self: Tensor, indices: []Tensor, values: Tensor, accumulate: bool = false) -> Tensor {
    out: Tensor
    acc_int := i32(1) if accumulate else i32(0)
    t.atg_index_put(&out, self, raw_data(indices), i32(len(indices)), values, acc_int)
    return track(out)
}

index_put_ :: proc(self: Tensor, indices: []Tensor, values: Tensor, accumulate: bool = false) -> Tensor {
    out: Tensor
    acc_int := i32(1) if accumulate else i32(0)
    t.atg_index_put_(&out, self, raw_data(indices), i32(len(indices)), values, acc_int)
    return self
}

// TODO: Index Reduce (eg "mean", "prod", "amin", "amax")
index_reduce :: proc(
    self: Tensor, 
    dim: i64, 
    idx: Tensor, 
    source: Tensor, 
    reduce: string, 
    include_self: bool = false,
) -> Tensor {
    out: Tensor
    incl_int := i32(1) if include_self else i32(0)
    
    c_reduce := strings.clone_to_cstring(reduce, context.temp_allocator)
    
    t.atg_index_reduce(&out, self, dim, idx, source, c_reduce, i32(len(reduce)), incl_int)
    return track(out)
}

index_reduce_ :: proc(
    self: Tensor, 
    dim: i64, 
    idx: Tensor, 
    source: Tensor, 
    reduce: string, 
    include_self: bool = false,
) -> Tensor {
    out: Tensor
    incl_int := i32(1) if include_self else i32(0)
    c_reduce := strings.clone_to_cstring(reduce, context.temp_allocator)
    
    t.atg_index_reduce_(&out, self, dim, idx, source, c_reduce, i32(len(reduce)), incl_int)
    return self
}

// Index Select
index_select :: proc(self: Tensor, dim: i64, idx: Tensor) -> Tensor {
    out: Tensor
    t.atg_index_select(&out, self, dim, idx)
    return track(out)
}

// Index Select Backward (Gradient calculation)
index_select_backward :: proc(grad: Tensor, self_sizes: []i64, dim: i64, idx: Tensor) -> Tensor {
    out: Tensor
    t.atg_index_select_backward(
        &out, 
        grad, 
        raw_data(self_sizes), i32(len(self_sizes)), 
        dim, 
        idx,
    )
    return track(out)
}

indices :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg__indices(&out, self)
    return track(out)
}

indices_copy :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_indices_copy(&out, self)
    return track(out)
}

// GELU Backward
infinitely_differentiable_gelu_backward :: proc(grad: Tensor, self: Tensor) -> Tensor {
    out: Tensor
    t.atg_infinitely_differentiable_gelu_backward(&out, grad, self)
    return track(out)
}

// Instance Norm
instance_norm :: proc(
    input, weight, bias, running_mean, running_var: Tensor,
    use_input_stats: bool = true,
    momentum: f64 = 0.1,
    eps: f64 = 1e-5,
    cudnn_enabled: bool = true,
) -> Tensor {
    out: Tensor
    t.atg_instance_norm(
        &out, 
        input, weight, bias, running_mean, running_var, 
        i32(use_input_stats), 
        momentum, 
        eps, 
        i32(cudnn_enabled),
    )
    return track(out)
}

// Layer Norm
layer_norm :: proc(
    input: Tensor, 
    normalized_shape: []i64, 
    weight: Tensor = {}, 
    bias: Tensor = {}, 
    eps: f64 = 1e-5, 
    cudnn_enable: bool = true,
) -> Tensor {
    out: Tensor
    t.atg_layer_norm(
        &out, 
        input, 
        raw_data(normalized_shape), 
        i32(len(normalized_shape)), 
        weight, 
        bias, 
        eps, 
        i32(cudnn_enable),
    )
    return track(out)
}

// KL Divergence
kl_div :: proc(self, target: Tensor, reduction: i64 = 1, log_target: bool = false) -> Tensor {
    out: Tensor
    t.atg_kl_div(&out, self, target, reduction, i32(log_target))
    return track(out)
}

// L1 Loss
l1_loss :: proc(self, target: Tensor, reduction: i64 = 1) -> Tensor {
    out: Tensor
    t.atg_l1_loss(&out, self, target, reduction)
    return track(out)
}

// Inner Product
inner :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_inner(&out, self, other)
    return track(out)
}

// Inverse
inverse :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_inverse(&out, self)
    return track(out)
}

// Kronecker Product
kron :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_kron(&out, self, other)
    return track(out)
}

// Inverse Short-Time Fourier Transform
istft :: proc(
    self: Tensor, 
    n_fft: i64, 
    hop_length: i64, 
    win_length: i64, 
    window: Tensor, 
    center: bool = true, 
    normalized: bool = false, 
    onesided: bool = true, 
    length: i64 = -1, // -1 implies None
    return_complex: bool = false,
) -> Tensor {
    out: Tensor
    // NOTE: pass nil to the _null pointers, implying we rely on the _v values provided
    t.atg_istft(
        &out, self, 
        n_fft, 
        hop_length, nil, 
        win_length, nil, 
        window, 
        i32(center), i32(normalized), i32(onesided), 
        length, nil, 
        i32(return_complex),
    )
    return track(out)
}

// Close Check
isclose :: proc(
    self, other: Tensor, 
    rtol: f64 = 1e-05, 
    atol: f64 = 1e-08, 
    equal_nan: bool = false
) -> Tensor {
    out: Tensor
    t.atg_isclose(&out, self, other, rtol, atol, i32(equal_nan))
    return track(out)
}

// Boolean State Checks
isfinite :: proc(self: Tensor) -> Tensor { out: Tensor; t.atg_isfinite(&out, self); return track(out) }
isinf    :: proc(self: Tensor) -> Tensor { out: Tensor; t.atg_isinf(&out, self);    return track(out) }
isnan    :: proc(self: Tensor) -> Tensor { out: Tensor; t.atg_isnan(&out, self);    return track(out) }
isneginf :: proc(self: Tensor) -> Tensor { out: Tensor; t.atg_isneginf(&out, self); return track(out) }
isposinf :: proc(self: Tensor) -> Tensor { out: Tensor; t.atg_isposinf(&out, self); return track(out) }
isreal   :: proc(self: Tensor) -> Tensor { out: Tensor; t.atg_isreal(&out, self);   return track(out) }

// ISIN Group
isin :: proc{isin_tensor_tensor, isin_scalar_tensor, isin_tensor_scalar}

@private
isin_tensor_tensor :: proc(elements, test_elements: Tensor, assume_unique: bool = false, invert: bool = false) -> Tensor {
    out: Tensor
    t.atg_isin(&out, elements, test_elements, i32(assume_unique), i32(invert))
    return track(out)
}

@private
isin_scalar_tensor :: proc(element: Scalar, test_elements: Tensor, assume_unique: bool = false, invert: bool = false) -> Tensor {
    out: Tensor
    t.atg_isin_scalar_tensor(&out, element, test_elements, i32(assume_unique), i32(invert))
    return track(out)
}

@private
isin_tensor_scalar :: proc(elements: Tensor, test_element: Scalar, assume_unique: bool = false, invert: bool = false) -> Tensor {
    out: Tensor
    t.atg_isin_tensor_scalar(&out, elements, test_element, i32(assume_unique), i32(invert))
    return track(out)
}

// Less Equal (LE) Group
le :: proc{le_scalar, le_tensor}
le_ :: proc{le_scalar_, le_tensor_}

@private
le_scalar :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_le(&out, self, other)
    return track(out)
}

@private
le_tensor :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_le_tensor(&out, self, other)
    return track(out)
}

@private
le_scalar_ :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_le_(&out, self, other)
    return self
}

@private
le_tensor_ :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_le_tensor_(&out, self, other)
    return self
}

// Kthvalue
// Returns (values, indices) where the k-th element is taken along the dim.
kthvalue :: proc(self: Tensor, k: i64, dim: i64 = -1, keepdim: bool = false) -> (values, indices: Tensor) {
    values = new_tensor()
    indices = new_tensor()
    
    dummy: Tensor
    t.atg_kthvalue_values(&dummy, values, indices, self, k, dim, i32(keepdim))
    
    return values, indices
}

// Int Representation
int_repr :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_int_repr(&out, self)
    return track(out)
}

// LCM Group

lcm :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_lcm(&out, self, other)
    return track(out)
}

lcm_ :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_lcm_(&out, self, other)
    return self
}

// LDEXP Group - Load Exponent
ldexp :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_ldexp(&out, self, other)
    return track(out)
}

ldexp_ :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_ldexp_(&out, self, other)
    return self
}

kaiser_window_simple :: proc(window_length: i64, device: DeviceType = .CPU, dtype: ScalarType = .Float) -> Tensor {
    out: Tensor
    t.atg_kaiser_window(&out, window_length, i32(dtype), i32(device))
    return track(out)
}

kaiser_window_beta :: proc(window_length: i64, periodic: bool, beta: f64, device: DeviceType = .CPU, dtype: ScalarType = .Float) -> Tensor {
    out: Tensor
    t.atg_kaiser_window_beta(&out, window_length, i32(periodic), beta, i32(dtype), i32(device))
    return track(out)
}

kaiser_window_periodic :: proc(window_length: i64, periodic: bool, device: DeviceType = .CPU, dtype: ScalarType = .Float) -> Tensor {
    out: Tensor
    t.atg_kaiser_window_periodic(&out, window_length, i32(periodic), i32(dtype), i32(device))
    return track(out)
}

// LEAKY RELU

leaky_relu :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_leaky_relu(&out, self)
    return track(out)
}

leaky_relu_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_leaky_relu_(&out, self)
    return self
}

// Gradients are rarely called manually, but included for completeness
leaky_relu_backward :: proc(
    grad_output: Tensor, 
    self: Tensor, 
    negative_slope: Scalar, 
    self_is_result: bool
) -> Tensor {
    out: Tensor
    is_res := i32(1) if self_is_result else i32(0)
    t.atg_leaky_relu_backward(&out, grad_output, self, negative_slope, is_res)
    return track(out)
}

leaky_relu_backward_grad_input :: proc(
    grad_input: Tensor, 
    grad_output: Tensor, 
    self: Tensor, 
    negative_slope: Scalar, 
    self_is_result: bool
) -> Tensor {
    out: Tensor
    is_res := i32(1) if self_is_result else i32(0)
    t.atg_leaky_relu_backward_grad_input(&out, grad_input, grad_output, self, negative_slope, is_res)
    return track(out)
}

// LERP (Linear Interpolation)

lerp :: proc{lerp_scalar, lerp_tensor}
lerp_ :: proc{lerp_scalar_, lerp_tensor_}

@private
lerp_scalar :: proc(self, end: Tensor, weight: Scalar) -> Tensor {
    out: Tensor
    t.atg_lerp(&out, self, end, weight)
    return track(out)
}

@private
lerp_tensor :: proc(self, end, weight: Tensor) -> Tensor {
    out: Tensor
    t.atg_lerp_tensor(&out, self, end, weight)
    return track(out)
}

@private
lerp_scalar_ :: proc(self, end: Tensor, weight: Scalar) -> Tensor {
    out: Tensor
    t.atg_lerp_(&out, self, end, weight)
    return self
}

@private
lerp_tensor_ :: proc(self, end, weight: Tensor) -> Tensor {
    out: Tensor
    t.atg_lerp_tensor_(&out, self, end, weight)
    return self
}

// COMPARISON (Less / Less Equal)

less :: proc{less_scalar, less_tensor}
less_ :: proc{less_scalar_, less_tensor_}

@private
less_scalar :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_less(&out, self, other)
    return track(out)
}

@private
less_tensor :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_less_tensor(&out, self, other)
    return track(out)
}

@private
less_scalar_ :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_less_(&out, self, other)
    return self
}

@private
less_tensor_ :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_less_tensor_(&out, self, other)
    return self
}

less_equal :: proc{less_equal_scalar, less_equal_tensor}
less_equal_ :: proc{less_equal_scalar_, less_equal_tensor_}

@private
less_equal_scalar :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_less_equal(&out, self, other)
    return track(out)
}

@private
less_equal_tensor :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_less_equal_tensor(&out, self, other)
    return track(out)
}

@private
less_equal_scalar_ :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_less_equal_(&out, self, other)
    return self
}

@private
less_equal_tensor_ :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_less_equal_tensor_(&out, self, other)
    return self
}

// MATH / LGamma

lgamma :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_lgamma(&out, self)
    return track(out)
}

lgamma_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_lgamma_(&out, self)
    return self
}

// INTERNAL / LIFTS
// These manipulate the lifecycle or view status of tensors, 
// usually for JIT/Graph operations.

lift :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_lift(&out, self)
    return track(out)
}

lift_fresh :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_lift_fresh(&out, self)
    return track(out)
}

lift_fresh_copy :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_lift_fresh_copy(&out, self)
    return track(out)
}

//  Cholesky Decomposition

linalg_cholesky :: proc(self: Tensor, upper: bool = false) -> Tensor {
    out: Tensor
    t.atg_linalg_cholesky(&out, self, i32(1) if upper else i32(0))
    return track(out)
}

// Returns (L, info) tuple-wrapped in a Tensor, or use explicit variant below
linalg_cholesky_ex :: proc(self: Tensor, upper: bool = false, check_errors: bool = false) -> Tensor {
    out: Tensor
    check_int := i32(1) if check_errors else i32(0)
    upper_int := i32(1) if upper else i32(0)
    t.atg_linalg_cholesky_ex(&out, self, upper_int, check_int)
    return track(out)
}

// Explicit output version returning (L, info) separately
linalg_cholesky_ex_out :: proc(self: Tensor, upper: bool = false, check_errors: bool = false) -> (L, info: Tensor) {
    L = new_tensor()
    info = new_tensor()
    
    check_int := i32(1) if check_errors else i32(0)
    upper_int := i32(1) if upper else i32(0)
    
    dummy: Tensor
    t.atg_linalg_cholesky_ex_l(&dummy, L, info, self, upper_int, check_int)
    
    return L, info
}

//  Condition Number / Determinant / Norm

linalg_cond :: proc(self: Tensor, p: Scalar) -> Tensor {
    out: Tensor
    t.atg_linalg_cond(&out, self, p)
    return track(out)
}

linalg_cond_str :: proc(self: Tensor, p: string) -> Tensor {
    out: Tensor
    p_cstr := strings.clone_to_cstring(p, context.temp_allocator)
    t.atg_linalg_cond_p_str(&out, self, p_cstr, i32(len(p)))
    return track(out)
}

linalg_norm :: proc(self: Tensor, ord: Scalar, dim: []i64 = nil, keepdim: bool = false, dtype: ScalarType = .Float) -> Tensor {
    out: Tensor
    keep_int := i32(1) if keepdim else i32(0)
    
    // Handle empty slice/nil
    dim_ptr: ^i64 = nil
    dim_len: i32 = 0
    if len(dim) > 0 {
        dim_ptr = raw_data(dim)
        dim_len = i32(len(dim))
    }

    t.atg_linalg_norm(&out, self, ord, dim_ptr, dim_len, keep_int, i32(dtype))
    return track(out)
}

linalg_norm_str :: proc(self: Tensor, ord: string, dim: []i64 = nil, keepdim: bool = false, dtype: ScalarType = .Float) -> Tensor {
    out: Tensor
    ord_cstr := strings.clone_to_cstring(ord, context.temp_allocator)
    keep_int := i32(1) if keepdim else i32(0)
    
    dim_ptr: ^i64 = nil
    dim_len: i32 = 0
    if len(dim) > 0 {
        dim_ptr = raw_data(dim)
        dim_len = i32(len(dim))
    }

    t.atg_linalg_norm_ord_str(&out, self, ord_cstr, i32(len(ord)), dim_ptr, dim_len, keep_int, i32(dtype))
    return track(out)
}

//  Cross / Diagonal

linalg_cross :: proc(self, other: Tensor, dim: i64 = -1) -> Tensor {
    out: Tensor
    t.atg_linalg_cross(&out, self, other, dim)
    return track(out)
}

linalg_diagonal :: proc(A: Tensor, offset: i64 = 0, dim1: i64 = -2, dim2: i64 = -1) -> Tensor {
    out: Tensor
    t.atg_linalg_diagonal(&out, A, offset, dim1, dim2)
    return track(out)
}

linalg_matmul :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_linalg_matmul(&out, self, other)
    return track(out)
}

linalg_multi_dot :: proc(tensors: []Tensor) -> Tensor {
    out: Tensor
    t.atg_linalg_multi_dot(&out, raw_data(tensors), i32(len(tensors)))
    return track(out)
}

linalg_householder_product :: proc(input, tau: Tensor) -> Tensor {
    out: Tensor
    t.atg_linalg_householder_product(&out, input, tau)
    return track(out)
}

linalg_matrix_power :: proc(self: Tensor, n: i64) -> Tensor {
    out: Tensor
    t.atg_linalg_matrix_power(&out, self, n)
    return track(out)
}

linalg_matrix_exp :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_linalg_matrix_exp(&out, self)
    return track(out)
}

linalg_vander :: proc(x: Tensor, n: i64 = 0, n_is_none: bool = true) -> Tensor {
    out: Tensor
    // NOTE: optional 'n' if n_is_none is true, we pass a dummy non-null pointer to 'n_null' 
    dummy_val: i32 = 1
    n_null_ptr: rawptr = nil
    if n_is_none {
        n_null_ptr = &dummy_val
    }
    t.atg_linalg_vander(&out, x, n, n_null_ptr)
    return track(out)
}

linalg_vecdot :: proc(x, y: Tensor, dim: i64 = -1) -> Tensor {
    out: Tensor
    t.atg_linalg_vecdot(&out, x, y, dim)
    return track(out)
}

//  Eigenvalues

// Returns Tuple(eigenvalues, eigenvectors)
linalg_eig :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_linalg_eig(&out, self)
    return track(out)
}

linalg_eigvals :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_linalg_eigvals(&out, self)
    return track(out)
}

// Returns Tuple(eigenvalues, eigenvectors)
linalg_eigh :: proc(self: Tensor, UPLO: string = "L") -> Tensor {
    out: Tensor
    uplo_cstr := strings.clone_to_cstring(UPLO, context.temp_allocator)
    t.atg_linalg_eigh(&out, self, uplo_cstr, i32(len(UPLO)))
    return track(out)
}

linalg_eigvalsh :: proc(self: Tensor, UPLO: string = "L") -> Tensor {
    out: Tensor
    uplo_cstr := strings.clone_to_cstring(UPLO, context.temp_allocator)
    t.atg_linalg_eigvalsh(&out, self, uplo_cstr, i32(len(UPLO)))
    return track(out)
}

// Explicit output for eigh
linalg_eigh_out :: proc(self: Tensor, UPLO: string = "L") -> (eigvals, eigvecs: Tensor) {
    eigvals = new_tensor()
    eigvecs = new_tensor()
    
    uplo_cstr := strings.clone_to_cstring(UPLO, context.temp_allocator)
    dummy: Tensor
    t.atg_linalg_eigh_eigvals(&dummy, eigvals, eigvecs, self, uplo_cstr, i32(len(UPLO)))
    
    return eigvals, eigvecs
}

//  Inverses / Solving

linalg_inv :: proc(A: Tensor) -> Tensor {
    out: Tensor
    t.atg_linalg_inv(&out, A)
    return track(out)
}

linalg_inv_ex :: proc(A: Tensor, check_errors: bool = false) -> Tensor {
    // Returns Tuple(inverse, info)
    out: Tensor
    check_int := i32(1) if check_errors else i32(0)
    t.atg_linalg_inv_ex(&out, A, check_int)
    return track(out)
}

linalg_inv_ex_out :: proc(A: Tensor, check_errors: bool = false) -> (inverse, info: Tensor) {
    inverse = new_tensor()
    info = new_tensor()
    check_int := i32(1) if check_errors else i32(0)
    
    dummy: Tensor
    t.atg_linalg_inv_ex_inverse(&dummy, inverse, info, A, check_int)
    return inverse, info
}

linalg_tensorinv :: proc(self: Tensor, ind: i64 = 2) -> Tensor {
    out: Tensor
    t.atg_linalg_tensorinv(&out, self, ind)
    return track(out)
}

linalg_tensorsolve :: proc(self, other: Tensor, dims: []i64 = nil) -> Tensor {
    out: Tensor
    
    dim_ptr: ^i64 = nil
    dim_len: i32 = 0
    if len(dims) > 0 {
        dim_ptr = raw_data(dims)
        dim_len = i32(len(dims))
    }

    t.atg_linalg_tensorsolve(&out, self, other, dim_ptr, dim_len)
    return track(out)
}

linalg_solve_ex :: proc(A, B: Tensor, left: bool = true, check_errors: bool = false) -> Tensor {
    out: Tensor
    left_int := i32(1) if left else i32(0)
    check_int := i32(1) if check_errors else i32(0)
    t.atg_linalg_solve_ex(&out, A, B, left_int, check_int)
    return track(out)
}

linalg_solve_triangular :: proc(self, B: Tensor, upper: bool, left: bool = true, unitriangular: bool = false) -> Tensor {
    out: Tensor
    t.atg_linalg_solve_triangular(&out, self, B, 
        i32(1) if upper else i32(0), 
        i32(1) if left else i32(0), 
        i32(1) if unitriangular else i32(0))
    return track(out)
}

//  Matrix Rank

linalg_matrix_rank :: proc(self: Tensor, tol: f64, hermitian: bool = false) -> Tensor {
    out: Tensor
    herm_int := i32(1) if hermitian else i32(0)
    t.atg_linalg_matrix_rank(&out, self, tol, herm_int)
    return track(out)
}

// Default (tol=None)
linalg_matrix_rank_tensor_tol :: proc(input: Tensor, tol: Tensor, hermitian: bool = false) -> Tensor {
    out: Tensor
    herm_int := i32(1) if hermitian else i32(0)
    // There are several variants in your list, using the explicit tensor tol one here
    t.atg_linalg_matrix_rank_tol_tensor(&out, input, tol, herm_int)
    return track(out)
}

//  Decompositions (QR, SVD, LU, LDL)

// Returns Tuple(Q, R)
linalg_qr :: proc(A: Tensor, mode: string = "reduced") -> Tensor {
    out: Tensor
    mode_cstr := strings.clone_to_cstring(mode, context.temp_allocator)
    t.atg_linalg_qr(&out, A, mode_cstr, i32(len(mode)))
    return track(out)
}

// Computes the singular value decomposition. 
// Returns Tuple(U, S, Vh)
linalg_svd :: proc(A: Tensor, full_matrices: bool = true, driver: string = "") -> Tensor {
    out: Tensor
    full_int := i32(1) if full_matrices else i32(0)
    
    // Driver can be nil/empty
    driver_ptr: cstring = nil
    driver_len: i32 = 0
    if len(driver) > 0 {
        driver_ptr = strings.clone_to_cstring(driver, context.temp_allocator)
        driver_len = i32(len(driver))
    }
    t.atg_linalg_svd(&out, A, full_int, driver_ptr, driver_len)
    return track(out)
}

// Explicit output SVD
linalg_svd_out :: proc(A: Tensor, full_matrices: bool = true, driver: string = "") -> (U, S, Vh: Tensor) {
    U = new_tensor()
    S = new_tensor()
    Vh = new_tensor()
    
    full_int := i32(1) if full_matrices else i32(0)
    
    driver_ptr: cstring = nil
    driver_len: i32 = 0
    if len(driver) > 0 {
        driver_ptr = strings.clone_to_cstring(driver, context.temp_allocator)
        driver_len = i32(len(driver))
    }

    dummy: Tensor
    t.atg_linalg_svd_u(&dummy, U, S, Vh, A, full_int, driver_ptr, driver_len)
    return U, S, Vh
}

linalg_svdvals :: proc(A: Tensor, driver: string = "") -> Tensor {
    out: Tensor
    
    driver_ptr: cstring = nil
    driver_len: i32 = 0
    if len(driver) > 0 {
        driver_ptr = strings.clone_to_cstring(driver, context.temp_allocator)
        driver_len = i32(len(driver))
    }

    t.atg_linalg_svdvals(&out, A, driver_ptr, driver_len)
    return track(out)
}

// Returns Tuple(P, L, U) or (LU, Pivots) depending on pivot arg, 
// wrapped in single Tensor handle
linalg_lu :: proc(A: Tensor, pivot: bool = true) -> Tensor {
    out: Tensor
    t.atg_linalg_lu(&out, A, i32(1) if pivot else i32(0))
    return track(out)
}

// Returns Tuple(LU, pivots)
linalg_lu_factor :: proc(A: Tensor, pivot: bool = true) -> Tensor {
    out: Tensor
    t.atg_linalg_lu_factor(&out, A, i32(1) if pivot else i32(0))
    return track(out)
}

linalg_lu_solve :: proc(LU, pivots, B: Tensor, left: bool = true, adjoint: bool = false) -> Tensor {
    out: Tensor
    t.atg_linalg_lu_solve(&out, LU, pivots, B, 
        i32(1) if left else i32(0), 
        i32(1) if adjoint else i32(0))
    return track(out)
}

// Returns Tuple(LD, pivots)
linalg_ldl_factor :: proc(self: Tensor, hermitian: bool = false) -> Tensor {
    out: Tensor
    t.atg_linalg_ldl_factor(&out, self, i32(1) if hermitian else i32(0))
    return track(out)
}

linalg_ldl_solve :: proc(LD, pivots, B: Tensor, hermitian: bool = false) -> Tensor {
    out: Tensor
    t.atg_linalg_ldl_solve(&out, LD, pivots, B, i32(1) if hermitian else i32(0))
    return track(out)
}

//  Least Squares

linalg_lstsq :: proc(self, b: Tensor, rcond: f64 = 0, driver: string = "") -> Tensor {
    out: Tensor
    
    driver_ptr: cstring = nil
    driver_len: i32 = 0
    if len(driver) > 0 {
        driver_ptr = strings.clone_to_cstring(driver, context.temp_allocator)
        driver_len = i32(len(driver))
    }

    // rcond_null pointer handling:
    // Usually in C-shims, if we want to pass "None", we pass a non-null pointer to 'rcond_null'.
    // Here we assume if the user provides 0.0 (and didn't specify a specific small epsilon),
    // they might mean default. 
    // *Implementation NOTE*: For stricter control, we assume 'rcond' is used if not default.
    // TODO: add a bool flag for None
    // 'rcond_null' is a flag pointer: nil = use value, !nil = use default/none.
    
    t.atg_linalg_lstsq(&out, self, b, rcond, nil, driver_ptr, driver_len)
    return track(out)
}

linalg_pinv :: proc(self: Tensor, rcond: f64, hermitian: bool = false) -> Tensor {
    out: Tensor
    t.atg_linalg_pinv(&out, self, rcond, i32(1) if hermitian else i32(0))
    return track(out)
}

// Applies a linear transformation to the incoming data: y = xA^T + b
linear :: proc(input: Tensor, weight: Tensor, bias: Tensor = Tensor{}) -> Tensor {
    out: Tensor
    t.atg_linear(&out, input, weight, bias)
    return track(out)
}

// LINSPACE
// Generates a one-dimensional tensor of size `steps` whose values are evenly spaced from `start` to `end` inclusive

linspace :: proc{
    linspace_scalar_scalar,
    linspace_scalar_tensor,
    linspace_tensor_scalar,
    linspace_tensor_tensor,
}

@private
linspace_scalar_scalar :: proc(
    start: Scalar, 
    end: Scalar, 
    steps: i64, 
    dtype: ScalarType = .Float, 
    device: DeviceType = .CPU
) -> Tensor {
    out: Tensor
    t.atg_linspace(&out, start, end, steps, i32(dtype), i32(device))
    return track(out)
}

@private
linspace_scalar_tensor :: proc(
    start: Scalar, 
    end: Tensor, 
    steps: i64, 
    dtype: ScalarType = .Float, 
    device: DeviceType = .CPU
) -> Tensor {
    out: Tensor
    t.atg_linspace_scalar_tensor(&out, start, end, steps, i32(dtype), i32(device))
    return track(out)
}

@private
linspace_tensor_scalar :: proc(
    start: Tensor, 
    end: Scalar, 
    steps: i64, 
    dtype: ScalarType = .Float, 
    device: DeviceType = .CPU
) -> Tensor {
    out: Tensor
    t.atg_linspace_tensor_scalar(&out, start, end, steps, i32(dtype), i32(device))
    return track(out)
}

@private
linspace_tensor_tensor :: proc(
    start: Tensor, 
    end: Tensor, 
    steps: i64, 
    dtype: ScalarType = .Float, 
    device: DeviceType = .CPU
) -> Tensor {
    out: Tensor
    t.atg_linspace_tensor_tensor(&out, start, end, steps, i32(dtype), i32(device))
    return track(out)
}

// LOGARITHMS / EXPONENTIALS

// Basic Log (Natural Logarithm)
log:: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_log(&out, self)
    return track(out)
}

log_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_log_(&out, self)
    return self
}

// Log10 (Base 10)
log10:: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_log10(&out, self)
    return track(out)
}

log10_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_log10_(&out, self)
    return self
}

// Log1p (Natural Log of 1 + input)
log1p:: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_log1p(&out, self)
    return track(out)
}

log1p_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_log1p_(&out, self)
    return self
}

// Log2 (Base 2)
log2:: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_log2(&out, self)
    return track(out)
}

log2_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_log2_(&out, self)
    return self
}

// Log Normal
// Returns a tensor of random numbers drawn from a log-normal distribution.
log_normal:: proc(self: Tensor, mean: f64 = 1.0, std: f64 = 2.0) -> Tensor {
    out: Tensor
    t.atg_log_normal(&out, self, mean, std)
    return track(out)
}

log_normal_ :: proc(self: Tensor, mean: f64 = 1.0, std: f64 = 2.0) -> Tensor {
    out: Tensor
    t.atg_log_normal_(&out, self, mean, std)
    return self
}

// Log Sigmoid
log_sigmoid :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_log_sigmoid(&out, self)
    return track(out)
}

// Log Softmax
log_softmax :: proc(self: Tensor, dim: i64, dtype: ScalarType) -> Tensor {
    out: Tensor
    t.atg_log_softmax(&out, self, dim, i32(dtype))
    return track(out)
}

// Log Add Exp
// logarithm of the sum of exponentiations: log(exp(x) + exp(y))
logaddexp :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_logaddexp(&out, self, other)
    return track(out)
}

// Log Add Exp 2
// logarithm of the sum of exponentiations in base 2: log2(2^x + 2^y)
logaddexp2 :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_logaddexp2(&out, self, other)
    return track(out)
}

// Log CumSum Exp
// logarithm of the cumulative sum of exponentiations
logcumsumexp :: proc(self: Tensor, dim: i64) -> Tensor {
    out: Tensor
    t.atg_logcumsumexp(&out, self, dim)
    return track(out)
}

// Log Determinant
logdet :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_logdet(&out, self)
    return track(out)
}

// LOW LEVEL BACKWARD OPS

// Usually not called directly unless implementing custom autograd functions
log_sigmoid_backward :: proc(grad_output: Tensor, self: Tensor, buffer: Tensor) -> Tensor {
    out: Tensor
    t.atg_log_sigmoid_backward(&out, grad_output, self, buffer)
    return track(out)
}

log_sigmoid_backward_grad_input :: proc(grad_input: Tensor, grad_output: Tensor, self: Tensor, buffer: Tensor) -> Tensor {
    out: Tensor
    t.atg_log_sigmoid_backward_grad_input(&out, grad_input, grad_output, self, buffer)
    return track(out)
}

// LOGICAL OPERATIONS

// Logical And
logical_and_tensor :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_logical_and(&out, self, other)
    return track(out)
}

logical_and_tensor_ :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_logical_and_(&out, self, other)
    return self
}

// Logical Or
logical_or_tensor :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_logical_or(&out, self, other)
    return track(out)
}

logical_or_tensor_ :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_logical_or_(&out, self, other)
    return self
}

// Logical Xor
logical_xor_tensor :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_logical_xor(&out, self, other)
    return track(out)
}

logical_xor_tensor_ :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_logical_xor_(&out, self, other)
    return self
}

// Logical Not (Unary)
logical_not_tensor :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_logical_not(&out, self)
    return track(out)
}

logical_not_tensor_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_logical_not_(&out, self)
    return self
}

// LOGIT
// Inverse of sigmoid. eps is used to clamp probabilities.
// eps_null can typically be nil if you want default behavior

logit:: proc(self: Tensor, eps: f64 = -1.0) -> Tensor {
    out: Tensor
    // We assume the C-shim handles nil to mean "None" or to check the eps value
    t.atg_logit(&out, self, eps, nil)
    return track(out)
}

logit_:: proc(self: Tensor, eps: f64 = -1.0) -> Tensor {
    out: Tensor
    t.atg_logit_(&out, self, eps, nil)
    return self
}

logit_backward :: proc(grad_output: Tensor, self: Tensor, eps: f64 = -1.0) -> Tensor {
    out: Tensor
    t.atg_logit_backward(&out, grad_output, self, eps, nil)
    return track(out)
}

// LOGSPACE - Generates a 1D tensor of logarithmically spaced values
logspace :: proc{
    logspace_scalar_scalar,
    logspace_scalar_tensor,
    logspace_tensor_scalar,
    logspace_tensor_tensor,
}

@private
logspace_scalar_scalar :: proc(start, end: Scalar, steps: i64, base: f64 = 10.0, dtype: ScalarType = .Float, device: DeviceType = .CPU) -> Tensor {
    out: Tensor
    t.atg_logspace(&out, start, end, steps, base, i32(dtype), i32(device))
    return track(out)
}

@private
logspace_scalar_tensor :: proc(start: Scalar, end: Tensor, steps: i64, base: f64 = 10.0, dtype: ScalarType = .Float, device: DeviceType = .CPU) -> Tensor {
    out: Tensor
    t.atg_logspace_scalar_tensor(&out, start, end, steps, base, i32(dtype), i32(device))
    return track(out)
}

@private
logspace_tensor_scalar :: proc(start: Tensor, end: Scalar, steps: i64, base: f64 = 10.0, dtype: ScalarType = .Float, device: DeviceType = .CPU) -> Tensor {
    out: Tensor
    t.atg_logspace_tensor_scalar(&out, start, end, steps, base, i32(dtype), i32(device))
    return track(out)
}

@private
logspace_tensor_tensor :: proc(start, end: Tensor, steps: i64, base: f64 = 10.0, dtype: ScalarType = .Float, device: DeviceType = .CPU) -> Tensor {
    out: Tensor
    t.atg_logspace_tensor_tensor(&out, start, end, steps, base, i32(dtype), i32(device))
    return track(out)
}

// LOGSUMEXP - Returns the log of the sum of the exponentials of the elements

logsumexp :: proc(self: Tensor, dim: []i64, keepdim: bool = false) -> Tensor {
    out: Tensor
    keep_int := i32(1) if keepdim else i32(0)
    
    t.atg_logsumexp(
        &out, 
        self, 
        raw_data(dim), 
        i32(len(dim)), 
        keep_int,
    )
    return track(out)
}

// LESS THAN (LT) - Computes self < other

lt :: proc{lt_scalar, lt_tensor}
lt_ :: proc{lt_scalar_, lt_tensor_}

@private
lt_scalar :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_lt(&out, self, other)
    return track(out)
}

@private
lt_tensor :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_lt_tensor(&out, self, other)
    return track(out)
}

@private
lt_scalar_ :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_lt_(&out, self, other)
    return self
}

@private
lt_tensor_ :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_lt_tensor_(&out, self, other)
    return self
}

// LU SOLVE - Returns the LU solve of the linear system Ax = b using LU factorization

lu_solve :: proc(self: Tensor, LU_data: Tensor, LU_pivots: Tensor) -> Tensor {
    out: Tensor
    t.atg_lu_solve(&out, self, LU_data, LU_pivots)
    return track(out)
}

//  LOSS FUNCTIONS

margin_ranking_loss :: proc(
    input1, input2, target: Tensor, 
    margin: f64 = 0.0, 
    reduction: i64 = 1 // TODO: enum 1=Mean, 0=None, 2=Sum
) -> Tensor {
    out: Tensor
    t.atg_margin_ranking_loss(&out, input1, input2, target, margin, reduction)
    return track(out)
}

//  MASKED OPERATIONS

// Masked Fill Scalar
masked_fill :: proc{masked_fill_scalar, masked_fill_tensor}
masked_fill_ :: proc{masked_fill_scalar_, masked_fill_tensor_}

@private
masked_fill_scalar :: proc(self: Tensor, mask: Tensor, value: Scalar) -> Tensor {
    out: Tensor
    t.atg_masked_fill(&out, self, mask, value)
    return track(out)
}

@private
masked_fill_scalar_ :: proc(self: Tensor, mask: Tensor, value: Scalar) -> Tensor {
    out: Tensor
    t.atg_masked_fill_(&out, self, mask, value)
    return self
}

@private
masked_fill_tensor :: proc(self: Tensor, mask: Tensor, value: Tensor) -> Tensor {
    out: Tensor
    t.atg_masked_fill_tensor(&out, self, mask, value)
    return track(out)
}

@private
masked_fill_tensor_ :: proc(self: Tensor, mask: Tensor, value: Tensor) -> Tensor {
    out: Tensor
    t.atg_masked_fill_tensor_(&out, self, mask, value)
    return self
}

// Masked Scatter
masked_scatter :: proc(self: Tensor, mask: Tensor, source: Tensor) -> Tensor {
    out: Tensor
    t.atg_masked_scatter(&out, self, mask, source)
    return track(out)
}

masked_scatter_ :: proc(self: Tensor, mask: Tensor, source: Tensor) -> Tensor {
    out: Tensor
    t.atg_masked_scatter_(&out, self, mask, source)
    return self
}

masked_scatter_backward :: proc(grad_output: Tensor, mask: Tensor, sizes: []i64) -> Tensor {
    out: Tensor
    t.atg_masked_scatter_backward(
        &out, 
        grad_output, 
        mask, 
        raw_data(sizes), 
        i32(len(sizes)),
    )
    return track(out)
}

// Masked Select
masked_select :: proc(self: Tensor, mask: Tensor) -> Tensor {
    out: Tensor
    t.atg_masked_select(&out, self, mask)
    return track(out)
}

masked_select_backward :: proc(grad: Tensor, input: Tensor, mask: Tensor) -> Tensor {
    out: Tensor
    t.atg_masked_select_backward(&out, grad, input, mask)
    return track(out)
}

//  MATRIX OPS

matmul :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_matmul(&out, self, other)
    return track(out)
}

matrix_exp :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_matrix_exp(&out, self)
    return track(out)
}

matrix_exp_backward :: proc(self: Tensor, grad: Tensor) -> Tensor {
    out: Tensor
    t.atg_matrix_exp_backward(&out, self, grad)
    return track(out)
}

// Matrix conjugate transpose (Hermitian)
matrix_h :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_matrix_h(&out, self)
    return track(out)
}

matrix_power :: proc(self: Tensor, n: i64) -> Tensor {
    out: Tensor
    t.atg_matrix_power(&out, self, n)
    return track(out)
}

//  REDUCTIONS & MAX

// Element-wise max
maximum :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_maximum(&out, self, other)
    return track(out)
}

max :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_max(&out, self)
    return track(out)
}

// Max Dim returns tuple (values, indices)
max_dim :: proc(self: Tensor, dim: i64, keepdim: bool = false) -> (values, indices: Tensor) {
    values = new_tensor()
    indices = new_tensor()
    
    keep_int := i32(1) if keepdim else i32(0)
    
    // Using the explicit out variant to fill our tracked tensors
    dummy: Tensor
    t.atg_max_dim_max(&dummy, values, indices, self, dim, keep_int)
    
    return values, indices
}

mean :: proc(self: Tensor, dtype: ScalarType) -> Tensor {
    out: Tensor
    t.atg_mean(&out, self, i32(dtype))
    return track(out)
}

mean_dim :: proc(
    self: Tensor, 
    dim: []i64, 
    keepdim: bool = false, 
    dtype: ScalarType = ScalarType.Float
) -> Tensor {
    out: Tensor
    keep_int := i32(1) if keepdim else i32(0)
    
    t.atg_mean_dim(
        &out, 
        self, 
        raw_data(dim), 
        i32(len(dim)), 
        keep_int, 
        i32(dtype),
    )
    return track(out)
}

// Median
median :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_median(&out, self)
    return track(out)
}

// Median Dim returns tuple (values, indices)
median_dim :: proc(self: Tensor, dim: i64, keepdim: bool = false) -> (values, indices: Tensor) {
    values = new_tensor()
    indices = new_tensor()
    
    keep_int := i32(1) if keepdim else i32(0)
    
    dummy: Tensor
    t.atg_median_dim_values(&dummy, values, indices, self, dim, keep_int)
    
    return values, indices
}

//  POOLING LAYERS

// Max Pool 1D
max_pool1d :: proc(
    self: Tensor, 
    kernel_size, stride, padding, dilation: []i64, 
    ceil_mode: bool = false
) -> Tensor {
    out: Tensor
    c_ceil := i32(1) if ceil_mode else i32(0)
    
    t.atg_max_pool1d(
        &out, self,
        raw_data(kernel_size), i32(len(kernel_size)),
        raw_data(stride),      i32(len(stride)),
        raw_data(padding),     i32(len(padding)),
        raw_data(dilation),    i32(len(dilation)),
        c_ceil,
    )
    return track(out)
}

max_pool1d_with_indices :: proc(
    self: Tensor, 
    kernel_size, stride, padding, dilation: []i64, 
    ceil_mode: bool = false
) -> (output, indices: Tensor) {
    // TODO: placeholder assuming 'out' is a tuple-container tensor
    out_tuple: Tensor
    c_ceil := i32(1) if ceil_mode else i32(0)
    
    t.atg_max_pool1d_with_indices(
        &out_tuple, self,
        raw_data(kernel_size), i32(len(kernel_size)),
        raw_data(stride),      i32(len(stride)),
        raw_data(padding),     i32(len(padding)),
        raw_data(dilation),    i32(len(dilation)),
        c_ceil,
    )
    // TODO: "tuple_get" func
    return track(out_tuple), new_tensor() // Placeholder for indices
}

// Max Pool 2D
max_pool2d :: proc(
    self: Tensor, 
    kernel_size, stride, padding, dilation: []i64, 
    ceil_mode: bool = false
) -> Tensor {
    out: Tensor
    c_ceil := i32(1) if ceil_mode else i32(0)
    
    t.atg_max_pool2d(
        &out, self,
        raw_data(kernel_size), i32(len(kernel_size)),
        raw_data(stride),      i32(len(stride)),
        raw_data(padding),     i32(len(padding)),
        raw_data(dilation),    i32(len(dilation)),
        c_ceil,
    )
    return track(out)
}

max_pool2d_backward :: proc(
    grad_output, self: Tensor, 
    kernel_size, stride, padding, dilation: []i64, 
    ceil_mode: bool = false
) -> Tensor {
    out: Tensor
    c_ceil := i32(1) if ceil_mode else i32(0)

    t.atg_max_pool2d_backward(
        &out, grad_output, self,
        raw_data(kernel_size), i32(len(kernel_size)),
        raw_data(stride),      i32(len(stride)),
        raw_data(padding),     i32(len(padding)),
        raw_data(dilation),    i32(len(dilation)),
        c_ceil,
    )
    return track(out)
}

// Max Pool 3D
max_pool3d :: proc(
    self: Tensor, 
    kernel_size, stride, padding, dilation: []i64, 
    ceil_mode: bool = false
) -> Tensor {
    out: Tensor
    c_ceil := i32(1) if ceil_mode else i32(0)

    t.atg_max_pool3d(
        &out, self,
        raw_data(kernel_size), i32(len(kernel_size)),
        raw_data(stride),      i32(len(stride)),
        raw_data(padding),     i32(len(padding)),
        raw_data(dilation),    i32(len(dilation)),
        c_ceil,
    )
    return track(out)
}

// Max Unpool
max_unpool2d :: proc(
    self: Tensor, 
    indices: Tensor, 
    output_size: []i64
) -> Tensor {
    out: Tensor
    t.atg_max_unpool2d(
        &out, self, indices, 
        raw_data(output_size), i32(len(output_size)),
    )
    return track(out)
}

max_unpool3d :: proc(
    self: Tensor, 
    indices: Tensor, 
    output_size, stride, padding: []i64
) -> Tensor {
    out: Tensor
    t.atg_max_unpool3d(
        &out, self, indices,
        raw_data(output_size), i32(len(output_size)),
        raw_data(stride),      i32(len(stride)),
        raw_data(padding),     i32(len(padding)),
    )
    return track(out)
}

// MIN / MINIMUM

// Computes the min of all elements in the input tensor.
min :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_min(&out, self)
    return track(out)
}

// Computes the min along a dimension. Returns (values, indices).
min_dim :: proc(self: Tensor, dim: i64, keepdim: bool = false) -> (values, indices: Tensor) {
    values = new_tensor()
    indices = new_tensor()
    
    keep_int := i32(1) if keepdim else i32(0)
    
    // We use a dummy for the function return value, as the actual results 
    // are written into the pre-allocated tensors 'values' and 'indices'.
    dummy: Tensor
    t.atg_min_dim_min(&dummy, values, indices, self, dim, keep_int)
    
    return values, indices
}

// Alias for element-wise minimum
minimum :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_minimum(&out, self, other)
    return track(out)
}

// MH (Multi-Head / Mahalanobis / Misc generic handle depending on context)
// Based on signature, it is a unary op.
mh :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_mh(&out, self)
    return track(out)
}

// MISH ACTIVATION

mish :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_mish(&out, self)
    return track(out)
}

mish_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_mish_(&out, self)
    return self
}

mish_backward :: proc(grad_output: Tensor, self: Tensor) -> Tensor {
    out: Tensor
    t.atg_mish_backward(&out, grad_output, self)
    return track(out)
}

// MIOPEN (Low-level backend functions for AMD GPUs)

miopen_batch_norm :: proc(
    input, weight, bias, running_mean, running_var: Tensor, 
    training: bool, 
    exponential_average_factor: f64, 
    epsilon: f64
) -> Tensor {
    out: Tensor
    is_training := i32(1) if training else i32(0)
    
    t.atg_miopen_batch_norm(
        &out, 
        input, weight, bias, running_mean, running_var, 
        is_training, 
        exponential_average_factor, 
        epsilon,
    )
    return track(out)
}

miopen_batch_norm_backward :: proc(
    input, grad_output, weight, running_mean, running_var, save_mean, save_var: Tensor, 
    epsilon: f64
) -> Tensor {
    out: Tensor
    t.atg_miopen_batch_norm_backward(
        &out, 
        input, grad_output, weight, running_mean, running_var, save_mean, save_var, 
        epsilon,
    )
    return track(out)
}

miopen_convolution :: proc(
    self, weight, bias: Tensor,
    padding:  []i64,
    stride:   []i64,
    dilation: []i64,
    groups:   i64,
    benchmark: bool = false,
    deterministic: bool = false,
) -> Tensor {
    out: Tensor
    
    bench_int := i32(1) if benchmark else i32(0)
    det_int   := i32(1) if deterministic else i32(0)
    
    t.atg_miopen_convolution(
        &out,
        self, weight, bias,
        raw_data(padding), i32(len(padding)),
        raw_data(stride), i32(len(stride)),
        raw_data(dilation), i32(len(dilation)),
        groups,
        bench_int,
        det_int,
    )
    return track(out)
}

miopen_convolution_add_relu :: proc(
    self, weight, z: Tensor,
    alpha: Scalar,
    bias: Tensor,
    stride:   []i64,
    padding:  []i64,
    dilation: []i64,
    groups: i64,
) -> Tensor {
    out: Tensor
    
    t.atg_miopen_convolution_add_relu(
        &out,
        self, weight, z,
        alpha,
        bias,
        raw_data(stride), i32(len(stride)),
        raw_data(padding), i32(len(padding)),
        raw_data(dilation), i32(len(dilation)),
        groups,
    )
    return track(out)
}

miopen_convolution_relu :: proc(
    self, weight, bias: Tensor,
    stride:   []i64,
    padding:  []i64,
    dilation: []i64,
    groups: i64,
) -> Tensor {
    out: Tensor
    
    t.atg_miopen_convolution_relu(
        &out,
        self, weight, bias,
        raw_data(stride), i32(len(stride)),
        raw_data(padding), i32(len(padding)),
        raw_data(dilation), i32(len(dilation)),
        groups,
    )
    return track(out)
}

miopen_convolution_transpose :: proc(
    self, weight, bias: Tensor,
    padding: []i64,
    output_padding: []i64,
    stride: []i64,
    dilation: []i64,
    groups: i64,
    benchmark: bool = false,
    deterministic: bool = false,
) -> Tensor {
    out: Tensor
    
    bench_int := i32(1) if benchmark else i32(0)
    det_int   := i32(1) if deterministic else i32(0)

    t.atg_miopen_convolution_transpose(
        &out,
        self, weight, bias,
        raw_data(padding), i32(len(padding)),
        raw_data(output_padding), i32(len(output_padding)),
        raw_data(stride), i32(len(stride)),
        raw_data(dilation), i32(len(dilation)),
        groups,
        bench_int,
        det_int,
    )
    return track(out)
}

miopen_depthwise_convolution :: proc(
    self, weight, bias: Tensor,
    padding: []i64,
    stride: []i64,
    dilation: []i64,
    groups: i64,
    benchmark: bool = false,
    deterministic: bool = false,
) -> Tensor {
    out: Tensor

    bench_int := i32(1) if benchmark else i32(0)
    det_int   := i32(1) if deterministic else i32(0)

    t.atg_miopen_depthwise_convolution(
        &out,
        self, weight, bias,
        raw_data(padding), i32(len(padding)),
        raw_data(stride), i32(len(stride)),
        raw_data(dilation), i32(len(dilation)),
        groups,
        bench_int,
        det_int,
    )
    return track(out)
}

miopen_rnn :: proc(
    input: Tensor,
    weights: []Tensor, // Passed as slice of tensors
    stride0: i64,
    hx, cx: Tensor,
    mode: i64,
    hidden_size: i64,
    num_layers: i64,
    batch_first: bool,
    dropout: f64,
    train: bool,
    bidirectional: bool,
    batch_sizes: []i64,
    dropout_state: Tensor,
) -> Tensor {
    out: Tensor
    
    bf_int    := i32(1) if batch_first else i32(0)
    train_int := i32(1) if train else i32(0)
    bi_int    := i32(1) if bidirectional else i32(0)
    
    // get pointer to first tensor in slice
    weight_ptr := raw_data(weights) if len(weights) > 0 else nil

    t.atg_miopen_rnn(
        &out,
        input,
        weight_ptr, i32(len(weights)), stride0,
        hx, cx,
        mode,
        hidden_size,
        num_layers,
        bf_int,
        dropout,
        train_int,
        bi_int,
        raw_data(batch_sizes), i32(len(batch_sizes)),
        dropout_state,
    )
    return track(out)
}

// MKLDNN LAYERS

mkldnn_adaptive_avg_pool2d :: proc(self: Tensor, output_size: []i64) -> Tensor {
    out: Tensor
    t.atg_mkldnn_adaptive_avg_pool2d(
        &out, 
        self, 
        raw_data(output_size), 
        i32(len(output_size)),
    )
    return track(out)
}

mkldnn_adaptive_avg_pool2d_backward :: proc(grad_output: Tensor, self: Tensor) -> Tensor {
    out: Tensor
    t.atg_mkldnn_adaptive_avg_pool2d_backward(&out, grad_output, self)
    return track(out)
}

mkldnn_convolution :: proc(
    self, weight, bias: Tensor, 
    padding, stride, dilation: []i64, 
    groups: i64,
) -> Tensor {
    out: Tensor
    t.atg_mkldnn_convolution(
        &out, 
        self, weight, bias,
        raw_data(padding), i32(len(padding)),
        raw_data(stride), i32(len(stride)),
        raw_data(dilation), i32(len(dilation)),
        groups,
    )
    return track(out)
}

mkldnn_linear :: proc(self, weight, bias: Tensor) -> Tensor {
    out: Tensor
    t.atg_mkldnn_linear(&out, self, weight, bias)
    return track(out)
}

mkldnn_linear_backward_input :: proc(
    input_size: []i64, 
    grad_output: Tensor, 
    weight: Tensor,
) -> Tensor {
    out: Tensor
    t.atg_mkldnn_linear_backward_input(
        &out, 
        raw_data(input_size), 
        i32(len(input_size)), 
        grad_output, 
        weight,
    )
    return track(out)
}

mkldnn_linear_backward_weights :: proc(
    grad_output: Tensor, 
    input: Tensor, 
    weight: Tensor, 
    bias_defined: bool,
) -> Tensor {
    out: Tensor
    b_def := i32(1) if bias_defined else i32(0)
    t.atg_mkldnn_linear_backward_weights(&out, grad_output, input, weight, b_def)
    return track(out)
}

mkldnn_max_pool2d :: proc(
    self: Tensor, 
    kernel_size, stride, padding, dilation: []i64, 
    ceil_mode: bool = false,
) -> Tensor {
    out: Tensor
    ceil_int := i32(1) if ceil_mode else i32(0)
    t.atg_mkldnn_max_pool2d(
        &out, 
        self,
        raw_data(kernel_size), i32(len(kernel_size)),
        raw_data(stride), i32(len(stride)),
        raw_data(padding), i32(len(padding)),
        raw_data(dilation), i32(len(dilation)),
        ceil_int,
    )
    return track(out)
}

mkldnn_max_pool2d_backward :: proc(
    grad_output, output, input: Tensor,
    kernel_size, stride, padding, dilation: []i64,
    ceil_mode: bool = false,
) -> Tensor {
    out: Tensor
    ceil_int := i32(1) if ceil_mode else i32(0)
    t.atg_mkldnn_max_pool2d_backward(
        &out, 
        grad_output, output, input,
        raw_data(kernel_size), i32(len(kernel_size)),
        raw_data(stride), i32(len(stride)),
        raw_data(padding), i32(len(padding)),
        raw_data(dilation), i32(len(dilation)),
        ceil_int,
    )
    return track(out)
}

mkldnn_max_pool3d :: proc(
    self: Tensor, 
    kernel_size, stride, padding, dilation: []i64, 
    ceil_mode: bool = false,
) -> Tensor {
    out: Tensor
    ceil_int := i32(1) if ceil_mode else i32(0)
    t.atg_mkldnn_max_pool3d(
        &out, 
        self,
        raw_data(kernel_size), i32(len(kernel_size)),
        raw_data(stride), i32(len(stride)),
        raw_data(padding), i32(len(padding)),
        raw_data(dilation), i32(len(dilation)),
        ceil_int,
    )
    return track(out)
}

mkldnn_max_pool3d_backward :: proc(
    grad_output, output, input: Tensor,
    kernel_size, stride, padding, dilation: []i64,
    ceil_mode: bool = false,
) -> Tensor {
    out: Tensor
    ceil_int := i32(1) if ceil_mode else i32(0)
    t.atg_mkldnn_max_pool3d_backward(
        &out, 
        grad_output, output, input,
        raw_data(kernel_size), i32(len(kernel_size)),
        raw_data(stride), i32(len(stride)),
        raw_data(padding), i32(len(padding)),
        raw_data(dilation), i32(len(dilation)),
        ceil_int,
    )
    return track(out)
}

mkldnn_reorder_conv2d_weight :: proc(
    self: Tensor, 
    padding, stride, dilation: []i64, 
    groups: i64,
    input_size: []i64,
) -> Tensor {
    out: Tensor
    t.atg_mkldnn_reorder_conv2d_weight(
        &out, 
        self,
        raw_data(padding), i32(len(padding)),
        raw_data(stride), i32(len(stride)),
        raw_data(dilation), i32(len(dilation)),
        groups,
        raw_data(input_size), i32(len(input_size)),
    )
    return track(out)
}

mkldnn_reorder_conv3d_weight :: proc(
    self: Tensor, 
    padding, stride, dilation: []i64, 
    groups: i64,
    input_size: []i64,
) -> Tensor {
    out: Tensor
    t.atg_mkldnn_reorder_conv3d_weight(
        &out, 
        self,
        raw_data(padding), i32(len(padding)),
        raw_data(stride), i32(len(stride)),
        raw_data(dilation), i32(len(dilation)),
        groups,
        raw_data(input_size), i32(len(input_size)),
    )
    return track(out)
}

mkldnn_rnn_layer :: proc(
    input: Tensor,
    w0, w1, w2, w3: Tensor,
    hx, cx: Tensor,
    reverse: bool,
    batch_sizes: []i64,
    mode: i64,
    hidden_size: i64,
    num_layers: i64,
    has_biases: bool,
    bidirectional: bool,
    batch_first: bool,
    train: bool,
) -> Tensor {
    out: Tensor
    t.atg_mkldnn_rnn_layer(
        &out, 
        input, w0, w1, w2, w3, hx, cx,
        i32(1) if reverse else i32(0),
        raw_data(batch_sizes), i32(len(batch_sizes)),
        mode,
        hidden_size,
        num_layers,
        i32(1) if has_biases else i32(0),
        i32(1) if bidirectional else i32(0),
        i32(1) if batch_first else i32(0),
        i32(1) if train else i32(0),
    )
    return track(out)
}

mkldnn_rnn_layer_backward :: proc(
    input: Tensor,
    w1, w2, w3, w4: Tensor,
    hx, cx_tmp, output, hy, cy: Tensor,
    grad_output, grad_hy, grad_cy: Tensor,
    reverse: bool,
    mode: i64,
    hidden_size: i64,
    num_layers: i64,
    has_biases: bool,
    train: bool,
    bidirectional: bool,
    batch_sizes: []i64,
    batch_first: bool,
    workspace: Tensor,
) -> Tensor {
    out: Tensor
    t.atg_mkldnn_rnn_layer_backward(
        &out,
        input, w1, w2, w3, w4,
        hx, cx_tmp, output, hy, cy,
        grad_output, grad_hy, grad_cy,
        i32(1) if reverse else i32(0),
        mode,
        hidden_size,
        num_layers,
        i32(1) if has_biases else i32(0),
        i32(1) if train else i32(0),
        i32(1) if bidirectional else i32(0),
        raw_data(batch_sizes), i32(len(batch_sizes)),
        i32(1) if batch_first else i32(0),
        workspace,
    )
    return track(out)
}

// MATRIX MULTIPLICATION

mm :: proc(self, mat2: Tensor) -> Tensor {
    out: Tensor
    t.atg_mm(&out, self, mat2)
    return track(out)
}

mm_dtype :: proc(self, mat2: Tensor, dtype: ScalarType) -> Tensor {
    out: Tensor
    t.atg_mm_dtype(&out, self, mat2, i32(dtype))
    return track(out)
}

// MANIPULATION & SORTING

mode :: proc(self: Tensor, dim: i64, keepdim: bool = false) -> (values, indices: Tensor) {
    values = new_tensor()
    indices = new_tensor()
    
    keep_int := i32(1) if keepdim else i32(0)
    
    // We use the explicit output version to safely fill our tracked tensors
    dummy: Tensor
    t.atg_mode_values(&dummy, values, indices, self, dim, keep_int)
    
    return values, indices
}

moveaxis :: proc{moveaxis_slice, moveaxis_int}
movedim  :: proc{movedim_slice, movedim_int}

@private
moveaxis_slice :: proc(self: Tensor, source, destination: []i64) -> Tensor {
    out: Tensor
    t.atg_moveaxis(
        &out, 
        self, 
        raw_data(source), i32(len(source)), 
        raw_data(destination), i32(len(destination)),
    )
    return track(out)
}

@private
moveaxis_int :: proc(self: Tensor, source, destination: i64) -> Tensor {
    out: Tensor
    t.atg_moveaxis_int(&out, self, source, destination)
    return track(out)
}

@private
movedim_slice :: proc(self: Tensor, source, destination: []i64) -> Tensor {
    out: Tensor
    t.atg_movedim(
        &out, 
        self, 
        raw_data(source), i32(len(source)), 
        raw_data(destination), i32(len(destination)),
    )
    return track(out)
}

@private
movedim_int :: proc(self: Tensor, source, destination: i64) -> Tensor {
    out: Tensor
    t.atg_movedim_int(&out, self, source, destination)
    return track(out)
}

msort :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_msort(&out, self)
    return track(out)
}

mt :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_mt(&out, self)
    return track(out)
}

// LOSS FUNCTIONS

mse_loss :: proc(self, target: Tensor, reduction: i64 = 1) -> Tensor {
    // TODO: map to enum reduction: 0=none, 1=mean, 2=sum...
    out: Tensor
    t.atg_mse_loss(&out, self, target, reduction)
    return track(out)
}

mse_loss_backward :: proc(grad_output, self, target: Tensor, reduction: i64 = 1) -> Tensor {
    out: Tensor
    t.atg_mse_loss_backward(&out, grad_output, self, target, reduction)
    return track(out)
}

mse_loss_backward_grad_input :: proc(
    grad_input, grad_output, self, target: Tensor, 
    reduction: i64 = 1,
) -> Tensor {
    out: Tensor
    t.atg_mse_loss_backward_grad_input(
        &out, 
        grad_input, grad_output, self, target, reduction,
    )
    return track(out)
}

// MULTIPLICATION

mul :: proc{mul_tensor, mul_scalar}
mul_ :: proc{mul_tensor_, mul_scalar_}

@private
mul_tensor :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_mul(&out, self, other)
    return track(out)
}

@private
mul_scalar :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_mul_scalar(&out, self, other)
    return track(out)
}

@private
mul_tensor_ :: proc(self, other: Tensor) -> Tensor {
    out := new_tensor()
    t.atg_mul_(&out, self, other)
    return self
}

@private
mul_scalar_ :: proc(self: Tensor, other: Scalar) -> Tensor {
    out := new_tensor()
    t.atg_mul_scalar_(&out, self, other)
    return self
}

// MATRIX VECTOR
mv :: proc(self, vec: Tensor) -> Tensor {
    out: Tensor
    t.atg_mv(&out, self, vec)
    return track(out)
}

// MULTINOMIAL
multinomial :: proc(self: Tensor, num_samples: i64, replacement: bool = false) -> Tensor {
    out: Tensor
    rep_int := i32(1) if replacement else i32(0)
    t.atg_multinomial(&out, self, num_samples, rep_int)
    return track(out)
}

// LOG GAMMA
mvlgamma :: proc(self: Tensor, p: i64) -> Tensor {
    out: Tensor
    t.atg_mvlgamma(&out, self, p)
    return track(out)
}

mvlgamma_ :: proc(self: Tensor, p: i64) -> Tensor {
    out: Tensor
    t.atg_mvlgamma_(&out, self, p)
    return self
}

// NAN OPS

// If val is nil, we pass 0.0 and a nil pointer. 
// If val is set, we pass the value and a pointer to a dummy to indicate "presence"
@(private)
_opt_f64 :: proc(val: Maybe(f64)) -> (f64, rawptr) {
    if v, ok := val.?; ok {
        // cast dummy address to rawptr for the flag
        return v, rawptr(uintptr(1))
    }
    return 0.0, nil
}

nan_to_num :: proc(
    self: Tensor, 
    nan: Maybe(f64) = nil, 
    posinf: Maybe(f64) = nil, 
    neginf: Maybe(f64) = nil
) -> Tensor {
    out: Tensor
    n_v, n_p := _opt_f64(nan)
    p_v, p_p := _opt_f64(posinf)
    ni_v, ni_p := _opt_f64(neginf)

    t.atg_nan_to_num(&out, self, n_v, n_p, p_v, p_p, ni_v, ni_p)
    return track(out)
}

nan_to_num_ :: proc(
    self: Tensor, 
    nan: Maybe(f64) = nil, 
    posinf: Maybe(f64) = nil, 
    neginf: Maybe(f64) = nil
) -> Tensor {
    out: Tensor
    n_v, n_p := _opt_f64(nan)
    p_v, p_p := _opt_f64(posinf)
    ni_v, ni_p := _opt_f64(neginf)

    t.atg_nan_to_num_(&out, self, n_v, n_p, p_v, p_p, ni_v, ni_p)
    return self
}

nanmean :: proc(
    self: Tensor, 
    dim: []i64 = nil, 
    keepdim: bool = false, 
    dtype: Maybe(ScalarType) = nil
) -> Tensor {
    out: Tensor
    kd_int := i32(1) if keepdim else i32(0)
    
    dt_int := i32(dtype.?) if dtype != nil else i32(6) 
    actual_dtype := i32(dtype.?) if dtype != nil else i32(-1) 

    t.atg_nanmean(
        &out, 
        self, 
        raw_data(dim), 
        i32(len(dim)), 
        kd_int, 
        actual_dtype
    )
    return track(out)
}

nansum :: proc(
    self: Tensor, 
    dim: []i64 = nil, 
    keepdim: bool = false, 
    dtype: Maybe(ScalarType) = nil
) -> Tensor {
    out: Tensor
    kd_int := i32(1) if keepdim else i32(0)
    actual_dtype := i32(dtype.?) if dtype != nil else i32(-1)

    t.atg_nansum(
        &out, 
        self, 
        raw_data(dim), 
        i32(len(dim)), 
        kd_int, 
        actual_dtype
    )
    return track(out)
}

// NANMEDIAN

nanmedian :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_nanmedian(&out, self)
    return track(out)
}

// Nanmedian with dim returns (values, indices)
nanmedian_dim :: proc(self: Tensor, dim: i64, keepdim: bool = false) -> (values, indices: Tensor) {
    values = new_tensor()
    indices = new_tensor()
    kd_int := i32(1) if keepdim else i32(0)
    
    dummy: Tensor
    t.atg_nanmedian_dim_values(&dummy, values, indices, self, dim, kd_int)
    
    return values, indices
}

// NANQUANTILE

nanquantile :: proc{nanquantile_tensor, nanquantile_scalar}

@private
nanquantile_tensor :: proc(
    self: Tensor, 
    q: Tensor, 
    dim: Maybe(i64) = nil, 
    keepdim: bool = false, 
    interpolation: string = "linear"
) -> Tensor {
    out: Tensor
    
    // Handle Dim Optionality
    dim_v: i64 = 0
    dim_p: rawptr = nil
    if d, ok := dim.?; ok {
        dim_v = d
        dim_p = rawptr(uintptr(1))
    }

    kd_int := i32(1) if keepdim else i32(0)

    // Handle String
    c_interp := strings.clone_to_cstring(interpolation, context.temp_allocator)
    
    t.atg_nanquantile(
        &out, 
        self, 
        q, 
        dim_v, 
        dim_p, 
        kd_int, 
        c_interp, 
        i32(len(interpolation))
    )
    return track(out)
}

@private
nanquantile_scalar :: proc(
    self: Tensor, 
    q: f64, 
    dim: Maybe(i64) = nil, 
    keepdim: bool = false, 
    interpolation: string = "linear"
) -> Tensor {
    out: Tensor
    
    dim_v: i64 = 0
    dim_p: rawptr = nil
    if d, ok := dim.?; ok {
        dim_v = d
        dim_p = rawptr(uintptr(1))
    }

    kd_int := i32(1) if keepdim else i32(0)
    c_interp := strings.clone_to_cstring(interpolation, context.temp_allocator)
    
    t.atg_nanquantile_scalar(
        &out, 
        self, 
        q, 
        dim_v, 
        dim_p, 
        kd_int, 
        c_interp, 
        i32(len(interpolation))
    )
    return track(out)
}

// MARGIN LOSSES

multi_margin_loss_backward :: proc(
    grad_output, self, target: Tensor,
    p: Scalar, margin: Scalar, weight: Tensor,
    reduction: Reduction
) -> Tensor {
    out: Tensor // grad_input
    t.atg_multi_margin_loss_backward(
        &out, 
        grad_output, 
        self, 
        target, 
        p, 
        margin, 
        weight, 
        i64(reduction)
    )
    return track(out)
}

multilabel_margin_loss :: proc(
    self, target: Tensor, 
    reduction: Reduction = .Mean
) -> Tensor {
    out: Tensor
    t.atg_multilabel_margin_loss(&out, self, target, i64(reduction))
    return track(out)
}

multilabel_margin_loss_backward :: proc(
    grad_output, self, target: Tensor,
    reduction: Reduction,
    is_target: Tensor
) -> Tensor {
    out: Tensor
    t.atg_multilabel_margin_loss_backward(
        &out, 
        grad_output, 
        self, 
        target, 
        i64(reduction), 
        is_target
    )
    return track(out)
}

// NARROW / SLICING

narrow :: proc{narrow_int, narrow_tensor}

@private
narrow_int :: proc(self: Tensor, dim: i64, start: i64, length: i64) -> Tensor {
    out: Tensor
    t.atg_narrow(&out, self, dim, start, length)
    return track(out)
}

@private
narrow_tensor :: proc(self: Tensor, dim: i64, start: Tensor, length: i64) -> Tensor {
    out: Tensor
    t.atg_narrow_tensor(&out, self, dim, start, length)
    return track(out)
}

narrow_copy :: proc(self: Tensor, dim: i64, start: i64, length: i64) -> Tensor {
    out: Tensor
    t.atg_narrow_copy(&out, self, dim, start, length)
    return track(out)
}

// NORMALIZATION & DROPOUT

native_batch_norm :: proc(
    input, weight, bias, running_mean, running_var: Tensor, 
    training: bool, 
    momentum: f64, 
    eps: f64,
) -> Tensor {
    out: Tensor
    train_int := i32(1) if training else i32(0)
    t.atg_native_batch_norm(&out, input, weight, bias, running_mean, running_var, train_int, momentum, eps)
    return track(out)
}

native_channel_shuffle :: proc(self: Tensor, groups: i64) -> Tensor {
    out: Tensor
    t.atg_native_channel_shuffle(&out, self, groups)
    return track(out)
}

native_dropout :: proc(input: Tensor, p: f64, train: bool) -> Tensor {
    out: Tensor
    train_int := i32(1) if train else i32(0)
    t.atg_native_dropout(&out, input, p, train_int)
    return track(out)
}

native_dropout_backward :: proc(grad_output: Tensor, mask: Tensor, scale: f64) -> Tensor {
    out: Tensor
    t.atg_native_dropout_backward(&out, grad_output, mask, scale)
    return track(out)
}

native_group_norm :: proc(
    input, weight, bias: Tensor, 
    n: i64, 
    c_channels: i64, 
    HxW: i64, 
    group: i64, 
    eps: f64,
) -> Tensor {
    out: Tensor
    t.atg_native_group_norm(&out, input, weight, bias, n, c_channels, HxW, group, eps)
    return track(out)
}

native_layer_norm :: proc(
    input: Tensor, 
    normalized_shape: []i64, 
    weight, bias: Tensor, 
    eps: f64,
) -> Tensor {
    out: Tensor
    t.atg_native_layer_norm(
        &out, 
        input, 
        raw_data(normalized_shape), 
        i32(len(normalized_shape)), 
        weight, 
        bias, 
        eps,
    )
    return track(out)
}

// NORM
native_norm_simple :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_native_norm(&out, self)
    return track(out)
}

native_norm_dims :: proc(
    self: Tensor, 
    p: Scalar, 
    dim: []i64, 
    keepdim: bool, 
    dtype: ScalarType,
) -> Tensor {
    out: Tensor
    keep_int := i32(1) if keepdim else i32(0)
    t.atg_native_norm_scalaropt_dim_dtype(
        &out, 
        self, 
        p, 
        raw_data(dim), 
        i32(len(dim)), 
        keep_int, 
        i32(dtype),
    )
    return track(out)
}

// Compare Not Equal

ne :: proc{ne_scalar, ne_tensor}
ne_ :: proc{ne_scalar_, ne_tensor_}

@private
ne_scalar :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_ne(&out, self, other)
    return track(out)
}

@private
ne_tensor :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_ne_tensor(&out, self, other)
    return track(out)
}

@private
ne_scalar_ :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_ne_(&out, self, other)
    return self
}

@private
ne_tensor_ :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_ne_tensor_(&out, self, other)
    return self
}

// NEGATION

neg:: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_neg(&out, self)
    return track(out)
}

neg_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_neg_(&out, self)
    return self
}

negative:: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_negative(&out, self)
    return track(out)
}

negative_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_negative_(&out, self)
    return self
}

nextafter :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_nextafter(&out, self, other)
    return track(out)
}

nextafter_ :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_nextafter_(&out, self, other)
    return self
}

// CONVERSION

nested_to_padded_tensor :: proc(self: Tensor, padding: f64, output_size: []i64) -> Tensor {
    out: Tensor
    t.atg_nested_to_padded_tensor(
        &out, 
        self, 
        padding, 
        raw_data(output_size), 
        i32(len(output_size)),
    )
    return track(out)
}

new_empty :: proc(
    self: Tensor, 
    size: []i64, 
    kind: ScalarType = .Float, 
    device: DeviceType = .CPU,
) -> Tensor {
    out: Tensor
    t.atg_new_empty(
        &out, 
        self, 
        raw_data(size), 
        i32(len(size)), 
        i32(kind), 
        i32(device),
    )
    return track(out)
}

new_empty_strided :: proc(
    self: Tensor, 
    size: []i64, 
    stride: []i64, 
    kind: ScalarType = .Float, 
    device: DeviceType = .CPU,
) -> Tensor {
    out: Tensor
    t.atg_new_empty_strided(
        &out, 
        self, 
        raw_data(size), 
        i32(len(size)), 
        raw_data(stride), 
        i32(len(stride)), 
        i32(kind), 
        i32(device),
    )
    return track(out)
}

new_full :: proc(
    self: Tensor, 
    size: []i64, 
    fill_value: Scalar, 
    kind: ScalarType = .Float, 
    device: DeviceType = .CPU,
) -> Tensor {
    out: Tensor
    t.atg_new_full(
        &out, 
        self, 
        raw_data(size), 
        i32(len(size)), 
        fill_value, 
        i32(kind), 
        i32(device),
    )
    return track(out)
}

new_ones :: proc(
    self: Tensor, 
    size: []i64, 
    kind: ScalarType = .Float, 
    device: DeviceType = .CPU,
) -> Tensor {
    out: Tensor
    t.atg_new_ones(
        &out, 
        self, 
        raw_data(size), 
        i32(len(size)), 
        i32(kind), 
        i32(device),
    )
    return track(out)
}

new_zeros :: proc(
    self: Tensor, 
    size: []i64, 
    kind: ScalarType = .Float, 
    device: DeviceType = .CPU,
) -> Tensor {
    out: Tensor
    t.atg_new_zeros(
        &out, 
        self, 
        raw_data(size), 
        i32(len(size)), 
        i32(kind), 
        i32(device),
    )
    return track(out)
}

// NLL LOSS

nll_loss :: proc(self, target: Tensor, weight: Tensor, reduction: i64, ignore_index: i64) -> Tensor {
    out: Tensor
    t.atg_nll_loss(&out, self, target, weight, reduction, ignore_index)
    return track(out)
}

nll_loss2d :: proc(self, target: Tensor, weight: Tensor, reduction: i64, ignore_index: i64) -> Tensor {
    out: Tensor
    t.atg_nll_loss2d(&out, self, target, weight, reduction, ignore_index)
    return track(out)
}

// NLL LOSS BACKWARD Gradient functions

nll_loss2d_backward :: proc(grad_output, self, target, weight: Tensor, reduction, ignore_index: i64, total_weight: Tensor) -> Tensor {
    out: Tensor
    t.atg_nll_loss2d_backward(&out, grad_output, self, target, weight, reduction, ignore_index, total_weight)
    return track(out)
}

nll_loss2d_backward_grad_input :: proc(grad_input, grad_output, self, target, weight: Tensor, reduction, ignore_index: i64, total_weight: Tensor) -> Tensor {
    out: Tensor
    t.atg_nll_loss2d_backward_grad_input(&out, grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight)
    return track(out)
}

nll_loss_backward :: proc(grad_output, self, target, weight: Tensor, reduction, ignore_index: i64, total_weight: Tensor) -> Tensor {
    out: Tensor
    t.atg_nll_loss_backward(&out, grad_output, self, target, weight, reduction, ignore_index, total_weight)
    return track(out)
}

nll_loss_backward_grad_input :: proc(grad_input, grad_output, self, target, weight: Tensor, reduction, ignore_index: i64, total_weight: Tensor) -> Tensor {
    out: Tensor
    t.atg_nll_loss_backward_grad_input(&out, grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight)
    return track(out)
}

nll_loss_nd :: proc(self, target: Tensor, weight: Tensor, reduction: i64, ignore_index: i64) -> Tensor {
    out: Tensor
    t.atg_nll_loss_nd(&out, self, target, weight, reduction, ignore_index)
    return track(out)
}

// NONZERO

nonzero :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_nonzero(&out, self)
    return track(out)
}

nonzero_static :: proc(self: Tensor, size: i64, fill_value: i64) -> Tensor {
    out: Tensor
    t.atg_nonzero_static(&out, self, size, fill_value)
    return track(out)
}

// NORM

norm_frobenius :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_norm(&out, self)
    return track(out)
}

norm_except_dim :: proc(v: Tensor, pow: i64, dim: i64) -> Tensor {
    out: Tensor
    t.atg_norm_except_dim(&out, v, pow, dim)
    return track(out)
}

norm_p :: proc(self: Tensor, p: Scalar, dim: []i64, keepdim: bool = false) -> Tensor {
    out: Tensor
    keep_int := i32(1) if keepdim else i32(0)
    t.atg_norm_scalaropt_dim(&out, self, p, raw_data(dim), i32(len(dim)), keep_int)
    return track(out)
}

norm_p_dtype :: proc(self: Tensor, p: Scalar, dim: []i64, keepdim: bool, dtype: ScalarType) -> Tensor {
    out: Tensor
    keep_int := i32(1) if keepdim else i32(0)
    t.atg_norm_scalaropt_dim_dtype(&out, self, p, raw_data(dim), i32(len(dim)), keep_int, i32(dtype))
    return track(out)
}

norm_dtype :: proc(self: Tensor, p: Scalar, dtype: ScalarType) -> Tensor {
    out: Tensor
    t.atg_norm_scalaropt_dtype(&out, self, p, i32(dtype))
    return track(out)
}

// NORMAL Distribution

normal_ :: proc(self: Tensor, mean: f64, std: f64) -> Tensor {
    out: Tensor
    t.atg_normal_(&out, self, mean, std)
    return self
}

normal :: proc(self: Tensor, mean: f64, std: f64) -> Tensor {
    out: Tensor
    t.atg_normal_functional(&out, self, mean, std)
    return track(out)
}

// NOT EQUAL !=

not_equal :: proc{not_equal_scalar, not_equal_tensor}
not_equal_ :: proc{not_equal_scalar_, not_equal_tensor_}

@private
not_equal_scalar :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_not_equal(&out, self, other)
    return track(out)
}

@private
not_equal_scalar_ :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_not_equal_(&out, self, other)
    return self
}

@private
not_equal_tensor :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_not_equal_tensor(&out, self, other)
    return track(out)
}

@private
not_equal_tensor_ :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_not_equal_tensor_(&out, self, other)
    return self
}

// NUCLEAR NORM

nuclear_norm :: proc(self: Tensor, keepdim: bool = false) -> Tensor {
    out: Tensor
    keep_int := i32(1) if keepdim else i32(0)
    t.atg_nuclear_norm(&out, self, keep_int)
    return track(out)
}

nuclear_norm_dim :: proc(self: Tensor, dim: []i64, keepdim: bool = false) -> Tensor {
    out: Tensor
    keep_int := i32(1) if keepdim else i32(0)
    t.atg_nuclear_norm_dim(&out, self, raw_data(dim), i32(len(dim)), keep_int)
    return track(out)
}

// NUMPY COMPAT

// Returns self.T (Tensor Transpose)
numpy_T :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_numpy_t(&out, self)
    return track(out)
}

// ONE HOT

one_hot :: proc(self: Tensor, num_classes: i64) -> Tensor {
    out: Tensor
    t.atg_one_hot(&out, self, num_classes)
    return track(out)
}

// ONES

ones :: proc(size: []i64, kind: ScalarType = .Float, device: DeviceType = .CPU) -> Tensor {
    out: Tensor
    // NOTE: device mappings depend on c10::DeviceType. 0 is usually CPU.
    t.atg_ones(&out, raw_data(size), i32(len(size)), i32(kind), i32(device))
    return track(out)
}

ones_like :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_ones_like(&out, self)
    return track(out)
}

// LINEAR ALGEBRA

// Computes the orthogonal matrix Q of a QR factorization
orgqr :: proc(self: Tensor, input2: Tensor) -> Tensor {
    out: Tensor
    t.atg_orgqr(&out, self, input2)
    return track(out)
}

// Multiplies matmul(Q, input3) where Q is from QR factorization
ormqr :: proc(self: Tensor, input2, input3: Tensor, left: bool = true, transpose: bool = false) -> Tensor {
    out: Tensor
    left_int := i32(1) if left else i32(0)
    trans_int := i32(1) if transpose else i32(0)
    t.atg_ormqr(&out, self, input2, input3, left_int, trans_int)
    return track(out)
}

// OUTER PRODUCT
outer :: proc(self: Tensor, vec2: Tensor) -> Tensor {
    out: Tensor
    t.atg_outer(&out, self, vec2)
    return track(out)
}

permute :: proc(self: Tensor, dims: []i64) -> Tensor {
    out: Tensor
    t.atg_permute(&out, self, raw_data(dims), i32(len(dims)))
    return track(out)
}

permute_copy :: proc(self: Tensor, dims: []i64) -> Tensor {
    out: Tensor
    t.atg_permute_copy(&out, self, raw_data(dims), i32(len(dims)))
    return track(out)
}

pairwise_distance :: proc(x1, x2: Tensor, p: f64 = 2.0, eps: f64 = 1e-6, keepdim: bool = false) -> Tensor {
    out: Tensor
    keep_int := i32(1) if keepdim else i32(0)
    t.atg_pairwise_distance(&out, x1, x2, p, eps, keep_int)
    return track(out)
}

pdist :: proc(self: Tensor, p: f64 = 2.0) -> Tensor {
    out: Tensor
    t.atg_pdist(&out, self, p)
    return track(out)
}

// TODO: map reduction
poisson_nll_loss :: proc(input, target: Tensor, log_input: bool, full: bool, eps: f64, reduction: i64) -> Tensor {
    out: Tensor
    log_int := i32(1) if log_input else i32(0)
    full_int := i32(1) if full else i32(0)
    t.atg_poisson_nll_loss(&out, input, target, log_int, full_int, eps, reduction)
    return track(out)
}

pin_memory :: proc(self: Tensor, device: DeviceType = .CPU) -> Tensor {
    out: Tensor
    t.atg_pin_memory(&out, self, i32(device))
    return track(out)
}

pinverse :: proc(self: Tensor, rcond: f64 = 1e-15) -> Tensor {
    out: Tensor
    t.atg_pinverse(&out, self, rcond)
    return track(out)
}

pixel_shuffle :: proc(self: Tensor, upscale_factor: i64) -> Tensor {
    out: Tensor
    t.atg_pixel_shuffle(&out, self, upscale_factor)
    return track(out)
}

pixel_unshuffle :: proc(self: Tensor, downscale_factor: i64) -> Tensor {
    out: Tensor
    t.atg_pixel_unshuffle(&out, self, downscale_factor)
    return track(out)
}

poisson :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_poisson(&out, self)
    return track(out)
}

polar :: proc(abs, angle: Tensor) -> Tensor {
    out: Tensor
    t.atg_polar(&out, abs, angle)
    return track(out)
}

positive :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_positive(&out, self)
    return track(out)
}

prelu :: proc(self, weight: Tensor) -> Tensor {
    out: Tensor
    t.atg_prelu(&out, self, weight)
    return track(out)
}

// Polygamma
polygamma :: proc(n: i64, self: Tensor) -> Tensor {
    out: Tensor
    t.atg_polygamma(&out, n, self)
    return track(out)
}

polygamma_ :: proc(self: Tensor, n: i64) -> Tensor {
    out: Tensor
    t.atg_polygamma_(&out, self, n)
    return self
}

pow :: proc{pow_tensor, pow_tensor_scalar, pow_scalar_tensor}
pow_ :: proc{pow_tensor_, pow_scalar_}

@private
pow_tensor :: proc(self, exponent: Tensor) -> Tensor {
    out: Tensor
    t.atg_pow(&out, self, exponent)
    return track(out)
}

@private
pow_tensor_scalar :: proc(self: Tensor, exponent: Scalar) -> Tensor {
    out: Tensor
    t.atg_pow_tensor_scalar(&out, self, exponent)
    return track(out)
}

@private
pow_scalar_tensor :: proc(self: Scalar, exponent: Tensor) -> Tensor {
    out: Tensor
    t.atg_pow_scalar(&out, self, exponent)
    return track(out)
}

@private
pow_tensor_ :: proc(self, exponent: Tensor) -> Tensor {
    out: Tensor
    t.atg_pow_tensor_(&out, self, exponent)
    return self
}

@private
pow_scalar_ :: proc(self: Tensor, exponent: Scalar) -> Tensor {
    out: Tensor
    t.atg_pow_(&out, self, exponent)
    return self
}

prod :: proc{prod_all, prod_dim}

@private
prod_all :: proc(self: Tensor, dtype: ScalarType = .Float) -> Tensor {
    out: Tensor
    // We cast the ScalarType enum to i32 as expected by the binding
    t.atg_prod(&out, self, i32(dtype))
    return track(out)
}

@private
prod_dim :: proc(self: Tensor, dim: i64, keepdim: bool = false, dtype: ScalarType = .Float) -> Tensor {
    out: Tensor
    keep_int := i32(1) if keepdim else i32(0)
    t.atg_prod_dim_int(&out, self, dim, keep_int, i32(dtype))
    return track(out)
}

put :: proc(self, index, source: Tensor, accumulate: bool = false) -> Tensor {
    out: Tensor
    acc_int := i32(1) if accumulate else i32(0)
    t.atg_put(&out, self, index, source, acc_int)
    return track(out)
}

put_ :: proc(self, index, source: Tensor, accumulate: bool = false) -> Tensor {
    out: Tensor
    acc_int := i32(1) if accumulate else i32(0)
    t.atg_put_(&out, self, index, source, acc_int)
    return self
}

// QUANTIZATION PROPERTIES

q_per_channel_scales :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_q_per_channel_scales(&out, self)
    return track(out)
}

q_per_channel_zero_points :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_q_per_channel_zero_points(&out, self)
    return track(out)
}

// QR DECOMPOSITION

/* qr computes the QR decomposition.
   Returns a tuple (Q, R).
*/
qr :: proc(self: Tensor, some: bool = true) -> Tensor {
    out: Tensor
    some_int := i32(1) if some else i32(0)
    t.atg_qr(&out, self, some_int)
    return track(out)
}

/* qr_out writes Q and R into provided tensors. 
   Useful for memory reuse or avoiding Tuple unpacking.
   Returns Q, R for chaining.
*/
qr_out :: proc(self: Tensor, Q, R: Tensor, some: bool = true) -> (Tensor, Tensor) {
    out: Tensor
    some_int := i32(1) if some else i32(0)
    t.atg_qr_q(&out, Q, R, self, some_int)
    return Q, R
}

// QUANTILE

/* dim: Optional dimension. Pass nil to flatten.
   interpolation: "linear", "lower", "higher", "midpoint", "nearest"
*/
quantile :: proc(
    self: Tensor, 
    q: Tensor, 
    dim: Maybe(i64) = nil, 
    keepdim: bool = false, 
    interpolation: string = "linear" // TODO: map to enum
) -> Tensor {
    out: Tensor
    keep_int := i32(1) if keepdim else i32(0)
    
    // Handle Optional Dimension
    dim_val: i64 = 0
    dim_ptr: rawptr = nil
    if d, ok := dim.?; ok {
        dim_val = d
        dim_ptr = &dim_val 
    }

    // Handle String
    interp_cstr := cstring(raw_data(interpolation))
    interp_len := i32(len(interpolation))

    t.atg_quantile(&out, self, q, dim_val, dim_ptr, keep_int, interp_cstr, interp_len)
    return track(out)
}

quantile_scalar :: proc(
    self: Tensor, 
    q: f64, 
    dim: Maybe(i64) = nil, 
    keepdim: bool = false, 
    interpolation: string = "linear"
) -> Tensor {
    out: Tensor
    keep_int := i32(1) if keepdim else i32(0)
    
    dim_val: i64 = 0
    dim_ptr: rawptr = nil
    if d, ok := dim.?; ok {
        dim_val = d
        dim_ptr = &dim_val 
    }

    interp_cstr := cstring(raw_data(interpolation))
    interp_len := i32(len(interpolation))

    t.atg_quantile_scalar(&out, self, q, dim_val, dim_ptr, keep_int, interp_cstr, interp_len)
    return track(out)
}

// QUANTIZE

quantize_per_channel :: proc(self, scales, zero_points: Tensor, axis: i64, dtype: ScalarType) -> Tensor {
    out: Tensor
    t.atg_quantize_per_channel(&out, self, scales, zero_points, axis, i32(dtype))
    return track(out)
}

quantize_per_tensor :: proc(self: Tensor, scale: f64, zero_point: i64, dtype: ScalarType) -> Tensor {
    out: Tensor
    t.atg_quantize_per_tensor(&out, self, scale, zero_point, i32(dtype))
    return track(out)
}

quantize_per_tensor_dynamic :: proc(self: Tensor, dtype: ScalarType, reduce_range: bool = false) -> Tensor {
    out: Tensor
    reduce_int := i32(1) if reduce_range else i32(0)
    t.atg_quantize_per_tensor_dynamic(&out, self, i32(dtype), reduce_int)
    return track(out)
}

quantize_per_tensor_tensor_qparams :: proc(self, scale, zero_point: Tensor, dtype: ScalarType) -> Tensor {
    out: Tensor
    t.atg_quantize_per_tensor_tensor_qparams(&out, self, scale, zero_point, i32(dtype))
    return track(out)
}

quantized_batch_norm :: proc(
    input, weight, bias, mean, var: Tensor, 
    eps: f64, 
    output_scale: f64, 
    output_zero_point: i64
) -> Tensor {
    out: Tensor
    t.atg_quantized_batch_norm(&out, input, weight, bias, mean, var, eps, output_scale, output_zero_point)
    return track(out)
}

// QUANTIZED RNN CELLS

quantized_gru_cell :: proc(
    input, hx, 
    w_ih, w_hh, b_ih, b_hh, 
    packed_ih, packed_hh, 
    col_offsets_ih, col_offsets_hh: Tensor, 
    scale_ih, scale_hh, zero_point_ih, zero_point_hh: Scalar
) -> Tensor {
    out: Tensor
    t.atg_quantized_gru_cell(
        &out, input, hx, 
        w_ih, w_hh, b_ih, b_hh, 
        packed_ih, packed_hh, 
        col_offsets_ih, col_offsets_hh, 
        scale_ih, scale_hh, zero_point_ih, zero_point_hh,
    )
    return track(out)
}

quantized_lstm_cell :: proc(
    input: Tensor, 
    hx: []Tensor, // Pass {h_0, c_0}
    w_ih, w_hh, b_ih, b_hh, 
    packed_ih, packed_hh, 
    col_offsets_ih, col_offsets_hh: Tensor, 
    scale_ih, scale_hh, zero_point_ih, zero_point_hh: Scalar
) -> Tensor {
    out: Tensor
    t.atg_quantized_lstm_cell(
        &out, input, 
        raw_data(hx), i32(len(hx)), 
        w_ih, w_hh, b_ih, b_hh, 
        packed_ih, packed_hh, 
        col_offsets_ih, col_offsets_hh, 
        scale_ih, scale_hh, zero_point_ih, zero_point_hh,
    )
    return track(out)
}

quantized_rnn_relu_cell :: proc(
    input, hx, 
    w_ih, w_hh, b_ih, b_hh, 
    packed_ih, packed_hh, 
    col_offsets_ih, col_offsets_hh: Tensor, 
    scale_ih, scale_hh, zero_point_ih, zero_point_hh: Scalar
) -> Tensor {
    out: Tensor
    t.atg_quantized_rnn_relu_cell(
        &out, input, hx, 
        w_ih, w_hh, b_ih, b_hh, 
        packed_ih, packed_hh, 
        col_offsets_ih, col_offsets_hh, 
        scale_ih, scale_hh, zero_point_ih, zero_point_hh,
    )
    return track(out)
}

quantized_rnn_tanh_cell :: proc(
    input, hx, 
    w_ih, w_hh, b_ih, b_hh, 
    packed_ih, packed_hh, 
    col_offsets_ih, col_offsets_hh: Tensor, 
    scale_ih, scale_hh, zero_point_ih, zero_point_hh: Scalar
) -> Tensor {
    out: Tensor
    t.atg_quantized_rnn_tanh_cell(
        &out, input, hx, 
        w_ih, w_hh, b_ih, b_hh, 
        packed_ih, packed_hh, 
        col_offsets_ih, col_offsets_hh, 
        scale_ih, scale_hh, zero_point_ih, zero_point_hh,
    )
    return track(out)
}

// QUANTIZED POOLING

quantized_max_pool1d :: proc(
    self: Tensor, 
    kernel_size, stride, padding, dilation: []i64, 
    ceil_mode: bool = false
) -> Tensor {
    out: Tensor
    c_ceil := i32(1) if ceil_mode else i32(0)
    t.atg_quantized_max_pool1d(
        &out, self, 
        raw_data(kernel_size), i32(len(kernel_size)),
        raw_data(stride),      i32(len(stride)),
        raw_data(padding),     i32(len(padding)),
        raw_data(dilation),    i32(len(dilation)),
        c_ceil,
    )
    return track(out)
}

quantized_max_pool2d :: proc(
    self: Tensor, 
    kernel_size, stride, padding, dilation: []i64, 
    ceil_mode: bool = false
) -> Tensor {
    out: Tensor
    c_ceil := i32(1) if ceil_mode else i32(0)
    t.atg_quantized_max_pool2d(
        &out, self, 
        raw_data(kernel_size), i32(len(kernel_size)),
        raw_data(stride),      i32(len(stride)),
        raw_data(padding),     i32(len(padding)),
        raw_data(dilation),    i32(len(dilation)),
        c_ceil,
    )
    return track(out)
}

quantized_max_pool3d :: proc(
    self: Tensor, 
    kernel_size, stride, padding, dilation: []i64, 
    ceil_mode: bool = false
) -> Tensor {
    out: Tensor
    c_ceil := i32(1) if ceil_mode else i32(0)
    t.atg_quantized_max_pool3d(
        &out, self, 
        raw_data(kernel_size), i32(len(kernel_size)),
        raw_data(stride),      i32(len(stride)),
        raw_data(padding),     i32(len(padding)),
        raw_data(dilation),    i32(len(dilation)),
        c_ceil,
    )
    return track(out)
}

// TRIGONOMETRY

rad2deg :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_rad2deg(&out, self)
    return track(out)
}

rad2deg_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_rad2deg_(&out, self)
    return self
}

real :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_real(&out, self)
    return track(out)
}

ravel :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_ravel(&out, self)
    return track(out)
}

// FACTORY & RANDOM TENSORS
rand :: proc(
    size: []i64, 
    dtype := DEFAULT_DTYPE, 
    device: DeviceType = DEFAULT_DEVICE
) -> Tensor {
    out: Tensor
    t.atg_rand(
        &out, 
        raw_data(size), 
        i32(len(size)), 
        i32(dtype), 
        i32(device),
    )
    return track(out)
}

rand_like :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_rand_like(&out, self)
    return track(out)
}

randint :: proc(
    high: i64, 
    size: []i64, 
    dtype := DEFAULT_DTYPE, 
    device: DeviceType = DEFAULT_DEVICE
) -> Tensor {
    out: Tensor
    t.atg_randint(
        &out, 
        high, 
        raw_data(size), 
        i32(len(size)), 
        i32(dtype), 
        i32(device),
    )
    return track(out)
}

randint_low :: proc(
    low: i64, 
    high: i64, 
    size: []i64, 
    dtype := DEFAULT_DTYPE, 
    device: DeviceType = DEFAULT_DEVICE
) -> Tensor {
    out: Tensor
    t.atg_randint_low(
        &out, 
        low, 
        high, 
        raw_data(size), 
        i32(len(size)), 
        i32(dtype), 
        i32(device),
    )
    return track(out)
}

randint_like_high :: proc(self: Tensor, high: i64) -> Tensor {
    out: Tensor
    t.atg_randint_like(&out, self, high)
    return track(out)
}

randint_like_low :: proc(self: Tensor, low: i64, high: i64) -> Tensor {
    out: Tensor
    t.atg_randint_like_low_dtype(&out, self, low, high)
    return track(out)
}

randint_like_tensor :: proc(self: Tensor, high: Tensor) -> Tensor {
    out: Tensor
    t.atg_randint_like_tensor(&out, self, high)
    return track(out)
}

randn :: proc(
    size: []i64, 
    dtype := DEFAULT_DTYPE, 
    device: DeviceType = DEFAULT_DEVICE
) -> Tensor {
    out: Tensor
    t.atg_randn(
        &out,
        raw_data(size), 
        i32(len(size)), 
        i32(dtype), 
        i32(device),
    )
    return track(out)
}

randn_like :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_randn_like(&out, self)
    return track(out)
}

randperm :: proc(
    n: i64, 
    dtype := ScalarType.Long, 
    device: DeviceType = DEFAULT_DEVICE
) -> Tensor {
    out: Tensor
    t.atg_randperm(
        &out, 
        n, 
        i32(dtype), 
        i32(device),
    )
    return track(out)
}

// RANGES

range :: proc(
    start: Scalar, 
    end: Scalar, 
    dtype := DEFAULT_DTYPE, 
    device: DeviceType = DEFAULT_DEVICE
) -> Tensor {
    out: Tensor
    t.atg_range(&out, start, end, i32(dtype), i32(device))
    return track(out)
}

range_step :: proc(
    start: Scalar, 
    end: Scalar, 
    dtype := DEFAULT_DTYPE, 
    device: DeviceType = DEFAULT_DEVICE
) -> Tensor {
    out: Tensor
    t.atg_range_step(&out, start, end, i32(dtype), i32(device))
    return track(out)
}

random :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_random(&out, self)
    return track(out)
}

random_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_random_(&out, self)
    return self
}

random_from :: proc(self: Tensor, from: i64, to: i64) -> Tensor {
    out: Tensor
    // TODO: to_null pointer defines optional Generator, passed nil here
    t.atg_random_from(&out, self, from, to, nil)
    return track(out)
}

random_from_ :: proc(self: Tensor, from: i64, to: i64) -> Tensor {
    out: Tensor
    t.atg_random_from_(&out, self, from, to, nil)
    return self
}

random_to :: proc(self: Tensor, to: i64) -> Tensor {
    out: Tensor
    t.atg_random_to(&out, self, to)
    return track(out)
}

random_to_ :: proc(self: Tensor, to: i64) -> Tensor {
    out: Tensor
    t.atg_random_to_(&out, self, to)
    return self
}

// RECIPROCAL

reciprocal :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_reciprocal(&out, self)
    return track(out)
}

reciprocal_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_reciprocal_(&out, self)
    return self
}

// REFLECTION PAD

/* NOTE: Padding must be supplied as a slice of ints (eg {1, 1} for 1D, {1, 1, 2, 2} for 2D).
   Torch expects padding pairs to be [left, right, top, bottom, front, back].
*/

reflection_pad1d :: proc(self: Tensor, padding: []i64) -> Tensor {
    out: Tensor
    t.atg_reflection_pad1d(&out, self, raw_data(padding), i32(len(padding)))
    return track(out)
}

reflection_pad1d_backward :: proc(grad_output: Tensor, self: Tensor, padding: []i64) -> Tensor {
    out: Tensor
    t.atg_reflection_pad1d_backward(&out, grad_output, self, raw_data(padding), i32(len(padding)))
    return track(out)
}

reflection_pad2d :: proc(self: Tensor, padding: []i64) -> Tensor {
    out: Tensor
    t.atg_reflection_pad2d(&out, self, raw_data(padding), i32(len(padding)))
    return track(out)
}

reflection_pad2d_backward :: proc(grad_output: Tensor, self: Tensor, padding: []i64) -> Tensor {
    out: Tensor
    t.atg_reflection_pad2d_backward(&out, grad_output, self, raw_data(padding), i32(len(padding)))
    return track(out)
}

reflection_pad3d :: proc(self: Tensor, padding: []i64) -> Tensor {
    out: Tensor
    t.atg_reflection_pad3d(&out, self, raw_data(padding), i32(len(padding)))
    return track(out)
}

reflection_pad3d_backward :: proc(grad_output: Tensor, self: Tensor, padding: []i64) -> Tensor {
    out: Tensor
    t.atg_reflection_pad3d_backward(&out, grad_output, self, raw_data(padding), i32(len(padding)))
    return track(out)
}

// RELU & RELU6

relu :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_relu(&out, self)
    return track(out)
}

relu_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_relu_(&out, self)
    return self
}

relu6 :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_relu6(&out, self)
    return track(out)
}

relu6_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_relu6_(&out, self)
    return self
}

// REMAINDER

remainder :: proc{
    remainder_tensor_scalar, 
    remainder_scalar_tensor, 
    remainder_tensor_tensor,
}

remainder_ :: proc{
    remainder_tensor_scalar_,
    remainder_tensor_tensor_,
}

@private
remainder_tensor_scalar :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_remainder(&out, self, other)
    return track(out)
}

@private
remainder_scalar_tensor :: proc(self: Scalar, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_remainder_scalar_tensor(&out, self, other)
    return track(out)
}

@private
remainder_tensor_tensor :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_remainder_tensor(&out, self, other)
    return track(out)
}

@private
remainder_tensor_scalar_ :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_remainder_(&out, self, other)
    return self
}

@private
remainder_tensor_tensor_ :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_remainder_tensor_(&out, self, other)
    return self
}

// RENORM

renorm :: proc(self: Tensor, p: Scalar, dim: i64, maxnorm: Scalar) -> Tensor {
    out: Tensor
    t.atg_renorm(&out, self, p, dim, maxnorm)
    return track(out)
}

renorm_ :: proc(self: Tensor, p: Scalar, dim: i64, maxnorm: Scalar) -> Tensor {
    out: Tensor
    t.atg_renorm_(&out, self, p, dim, maxnorm)
    return self
}

// REPEAT

repeat :: proc(self: Tensor, sizes: []i64) -> Tensor {
    out: Tensor
    t.atg_repeat(&out, self, raw_data(sizes), i32(len(sizes)))
    return track(out)
}

// REPEAT INTERLEAVE

// Generates tensor from repeats
repeat_interleave :: proc(repeats: Tensor) -> Tensor {
    out: Tensor
    t.atg_repeat_interleave(&out, repeats, 0, nil)
    return track(out)
}

repeat_interleave_self :: proc{
    repeat_interleave_self_int,
    repeat_interleave_self_tensor,
}

// Equivalent to torch.repeat_interleave(input, repeats, dim=dim)
// Uses '0' for output_size null pointer to let Torch compute output size automatically.
@private
repeat_interleave_self_int :: proc(self: Tensor, repeats: i64, dim: i64) -> Tensor {
    out: Tensor
    // Pass dim_null as non-null (address of dim) to indicate dim is present
    d := dim
    t.atg_repeat_interleave_self_int(&out, self, repeats, d, &d, 0, nil)
    return track(out)
}

// Equivalent to torch.repeat_interleave(input, repeats, dim=dim) where repeats is a Tensor (eg per-element repetition)
@private
repeat_interleave_self_tensor :: proc(self: Tensor, repeats: Tensor, dim: i64) -> Tensor {
    out: Tensor
    d := dim
    t.atg_repeat_interleave_self_tensor(&out, self, repeats, d, &d, 0, nil)
    return track(out)
}

// Flattens input and repeats for when you want (self, repeats) but NO dim (flattened)
repeat_interleave_flat :: proc(self: Tensor, repeats: i64) -> Tensor {
    out: Tensor
    // Pass nil to dim_null to indicate flattening behavior
    t.atg_repeat_interleave_self_int(&out, self, repeats, 0, nil, 0, nil)
    return track(out)
}

// REPLICATION PAD

replication_pad1d :: proc(self: Tensor, padding: []i64) -> Tensor {
    out: Tensor
    t.atg_replication_pad1d(&out, self, raw_data(padding), i32(len(padding)))
    return track(out)
}

replication_pad1d_backward :: proc(grad_output: Tensor, self: Tensor, padding: []i64) -> Tensor {
    out: Tensor
    t.atg_replication_pad1d_backward(&out, grad_output, self, raw_data(padding), i32(len(padding)))
    return track(out)
}

replication_pad1d_backward_grad_input :: proc(grad_input: Tensor, grad_output: Tensor, self: Tensor, padding: []i64) -> Tensor {
    out: Tensor
    t.atg_replication_pad1d_backward_grad_input(&out, grad_input, grad_output, self, raw_data(padding), i32(len(padding)))
    return track(out)
}

replication_pad2d :: proc(self: Tensor, padding: []i64) -> Tensor {
    out: Tensor
    t.atg_replication_pad2d(&out, self, raw_data(padding), i32(len(padding)))
    return track(out)
}

replication_pad2d_backward :: proc(grad_output: Tensor, self: Tensor, padding: []i64) -> Tensor {
    out: Tensor
    t.atg_replication_pad2d_backward(&out, grad_output, self, raw_data(padding), i32(len(padding)))
    return track(out)
}

replication_pad2d_backward_grad_input :: proc(grad_input: Tensor, grad_output: Tensor, self: Tensor, padding: []i64) -> Tensor {
    out: Tensor
    t.atg_replication_pad2d_backward_grad_input(&out, grad_input, grad_output, self, raw_data(padding), i32(len(padding)))
    return track(out)
}

replication_pad3d :: proc(self: Tensor, padding: []i64) -> Tensor {
    out: Tensor
    t.atg_replication_pad3d(&out, self, raw_data(padding), i32(len(padding)))
    return track(out)
}

replication_pad3d_backward :: proc(grad_output: Tensor, self: Tensor, padding: []i64) -> Tensor {
    out: Tensor
    t.atg_replication_pad3d_backward(&out, grad_output, self, raw_data(padding), i32(len(padding)))
    return track(out)
}

replication_pad3d_backward_grad_input :: proc(grad_input: Tensor, grad_output: Tensor, self: Tensor, padding: []i64) -> Tensor {
    out: Tensor
    t.atg_replication_pad3d_backward_grad_input(&out, grad_input, grad_output, self, raw_data(padding), i32(len(padding)))
    return track(out)
}

// REQUIRES GRAD

requires_grad_ :: proc(self: Tensor, requires_grad: bool = true) -> Tensor {
    out: Tensor
    req_int := i32(1) if requires_grad else i32(0)
    t.atg_requires_grad_(&out, self, req_int)
    return self
}

// RESHAPE
reshape :: proc(self: Tensor, shape: []i64) -> Tensor {
    out: Tensor
    t.atg_reshape(&out, self, raw_data(shape), i32(len(shape)))
    return track(out)
}

reshape_as :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_reshape_as(&out, self, other)
    return track(out)
}

// RESIZE

resize_tensor :: proc(self: Tensor, size: []i64) -> Tensor {
    out: Tensor
    t.atg_resize(&out, self, raw_data(size), i32(len(size)))
    return track(out)
}

resize_tensor_ :: proc(self: Tensor, size: []i64) -> Tensor {
    out: Tensor
    t.atg_resize_(&out, self, raw_data(size), i32(len(size)))
    return self
}

resize_as :: proc(self: Tensor, the_template: Tensor) -> Tensor {
    out: Tensor
    t.atg_resize_as(&out, self, the_template)
    return track(out)
}

resize_as_ :: proc(self: Tensor, the_template: Tensor) -> Tensor {
    out: Tensor
    t.atg_resize_as_(&out, self, the_template)
    return self
}

resize_as_sparse :: proc(self: Tensor, the_template: Tensor) -> Tensor {
    out: Tensor
    t.atg_resize_as_sparse(&out, self, the_template)
    return track(out)
}

resize_as_sparse_ :: proc(self: Tensor, the_template: Tensor) -> Tensor {
    out: Tensor
    t.atg_resize_as_sparse_(&out, self, the_template)
    return self
}

// RESOLVE

resolve_conj :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_resolve_conj(&out, self)
    return track(out)
}

resolve_neg :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_resolve_neg(&out, self)
    return track(out)
}

// NORMALIZATION

rms_norm :: proc(
    input: Tensor, 
    normalized_shape: []i64, 
    weight: Tensor, 
    eps: f64 = 1e-6
) -> Tensor {
    out: Tensor
    // We pass nil for eps_null, assuming standard behavior where the value eps_v is used.
    t.atg_rms_norm(
        &out, 
        input, 
        raw_data(normalized_shape), 
        i32(len(normalized_shape)), 
        weight, 
        eps, 
        nil
    )
    return track(out)
}

// RNN (Recurrent Neural Networks)

rnn_relu :: proc(
    input, hx: Tensor, 
    params: []Tensor, 
    has_biases: bool, 
    num_layers: i64, 
    dropout: f64, 
    train: bool, 
    bidirectional: bool, 
    batch_first: bool
) -> Tensor {
    out: Tensor
    t.atg_rnn_relu(
        &out, 
        input, 
        hx, 
        raw_data(params), 
        i32(len(params)), 
        i32(1) if has_biases else i32(0),
        num_layers,
        dropout,
        i32(1) if train else i32(0),
        i32(1) if bidirectional else i32(0),
        i32(1) if batch_first else i32(0)
    )
    return track(out)
}

rnn_tanh :: proc(
    input, hx: Tensor, 
    params: []Tensor, 
    has_biases: bool, 
    num_layers: i64, 
    dropout: f64, 
    train: bool, 
    bidirectional: bool, 
    batch_first: bool
) -> Tensor {
    out: Tensor
    t.atg_rnn_tanh(
        &out, 
        input, 
        hx, 
        raw_data(params), 
        i32(len(params)), 
        i32(1) if has_biases else i32(0),
        num_layers,
        dropout,
        i32(1) if train else i32(0),
        i32(1) if bidirectional else i32(0),
        i32(1) if batch_first else i32(0)
    )
    return track(out)
}

rnn_relu_cell :: proc(
    input, hx: Tensor, 
    w_ih, w_hh: Tensor, 
    b_ih, b_hh: Tensor
) -> Tensor {
    out: Tensor
    t.atg_rnn_relu_cell(&out, input, hx, w_ih, w_hh, b_ih, b_hh)
    return track(out)
}

rnn_tanh_cell :: proc(
    input, hx: Tensor, 
    w_ih, w_hh: Tensor, 
    b_ih, b_hh: Tensor
) -> Tensor {
    out: Tensor
    t.atg_rnn_tanh_cell(&out, input, hx, w_ih, w_hh, b_ih, b_hh)
    return track(out)
}

// PackedSequence
rnn_relu_data :: proc(
    data, batch_sizes, hx: Tensor, 
    params: []Tensor, 
    has_biases: bool, 
    num_layers: i64, 
    dropout: f64, 
    train: bool, 
    bidirectional: bool
) -> Tensor {
    out: Tensor
    t.atg_rnn_relu_data(
        &out, data, batch_sizes, hx, 
        raw_data(params), i32(len(params)),
        i32(1) if has_biases else i32(0),
        num_layers, dropout, 
        i32(1) if train else i32(0), 
        i32(1) if bidirectional else i32(0)
    )
    return track(out)
}

rnn_tanh_data :: proc(
    data, batch_sizes, hx: Tensor, 
    params: []Tensor, 
    has_biases: bool, 
    num_layers: i64, 
    dropout: f64, 
    train: bool, 
    bidirectional: bool
) -> Tensor {
    out: Tensor
    t.atg_rnn_tanh_data(
        &out, data, batch_sizes, hx, 
        raw_data(params), i32(len(params)),
        i32(1) if has_biases else i32(0),
        num_layers, dropout, 
        i32(1) if train else i32(0), 
        i32(1) if bidirectional else i32(0)
    )
    return track(out)
}

// MATH & TRANSFORMS

roll :: proc(self: Tensor, shifts: []i64, dims: []i64 = nil) -> Tensor {
    out: Tensor
    // Handle dims being nil/empty by passing 0 length
    d_ptr := raw_data(dims)
    d_len := i32(len(dims))
    
    t.atg_roll(
        &out, 
        self, 
        raw_data(shifts), 
        i32(len(shifts)), 
        d_ptr, 
        d_len
    )
    return track(out)
}

rot90 :: proc(self: Tensor, k: i64, dims: []i64) -> Tensor {
    out: Tensor
    t.atg_rot90(
        &out, 
        self, 
        k, 
        raw_data(dims), 
        i32(len(dims))
    )
    return track(out)
}

round :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_round(&out, self)
    return track(out)
}

round_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_round_(&out, self)
    return self
}

round_decimals :: proc(self: Tensor, decimals: i64) -> Tensor {
    out: Tensor
    t.atg_round_decimals(&out, self, decimals)
    return track(out)
}

round_decimals_ :: proc(self: Tensor, decimals: i64) -> Tensor {
    out: Tensor
    t.atg_round_decimals_(&out, self, decimals)
    return self
}

row_indices :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_row_indices(&out, self)
    return track(out)
}

row_indices_copy :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_row_indices_copy(&out, self)
    return track(out)
}

// Stacks a list of tensors along dimension 0
row_stack :: proc(tensors: []Tensor) -> Tensor {
    out: Tensor
    t.atg_row_stack(&out, raw_data(tensors), i32(len(tensors)))
    return track(out)
}

// ACTIVATIONS

rrelu :: proc(self: Tensor, training: bool = false) -> Tensor {
    out: Tensor
    t.atg_rrelu(&out, self, i32(1) if training else i32(0))
    return track(out)
}

rrelu_ :: proc(self: Tensor, training: bool = false) -> Tensor {
    out: Tensor
    t.atg_rrelu_(&out, self, i32(1) if training else i32(0))
    return self
}

rrelu_with_noise :: proc(self: Tensor, noise: Tensor, training: bool = false) -> Tensor {
    out: Tensor
    t.atg_rrelu_with_noise(&out, self, noise, i32(1) if training else i32(0))
    return track(out)
}

rrelu_with_noise_ :: proc(self: Tensor, noise: Tensor, training: bool = false) -> Tensor {
    out: Tensor
    t.atg_rrelu_with_noise_(&out, self, noise, i32(1) if training else i32(0))
    return self
}

// Used for backprop
rrelu_with_noise_backward :: proc(
    grad_output: Tensor, 
    self: Tensor, 
    noise: Tensor, 
    lower: Scalar, 
    upper: Scalar, 
    training: bool, 
    self_is_result: bool
) -> Tensor {
    out: Tensor
    t.atg_rrelu_with_noise_backward(
        &out, 
        grad_output, 
        self, 
        noise, 
        lower, 
        upper, 
        i32(1) if training else i32(0),
        i32(1) if self_is_result else i32(0)
    )
    return track(out)
}

// Reciprocal Square Root
rsqrt :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_rsqrt(&out, self)
    return track(out)
}

rsqrt_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_rsqrt_(&out, self)
    return self
}

// RSUB - Reverse Subtraction: other - self

rsub :: proc{rsub_tensor, rsub_scalar}

@private
rsub_tensor :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_rsub(&out, self, other)
    return track(out)
}

@private
rsub_scalar :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_rsub_scalar(&out, self, other)
    return track(out)
}

// SCALAR CREATION & ATTENTION

// Create a 0-dim tensor from a scalar
scalar_tensor :: proc(
    s: Scalar, 
    kind: ScalarType = .Float,
    device: DeviceType = .CPU
) -> Tensor {
    out: Tensor
    t.atg_scalar_tensor(&out, s, i32(kind), i32(device))
    return track(out)
}

/* Usage 

// LibTorch calculates scale automatically
attn := otorch.scaled_dot_product_attention(q, k, v)

// Explicit scale
attn_scaled := otorch.scaled_dot_product_attention(q, k, v, scale = 0.5)

// With Mask and Causal
attn_masked := otorch.scaled_dot_product_attention(
    q, k, v, 
    is_causal = true, 
    dropout_p = 0.1
)
*/
scaled_dot_product_attention :: proc(
    query, key, value: Tensor,
    attn_mask: Tensor = {}, // Optional: Default is undefined tensor
    dropout_p: f64 = 0.0,
    is_causal: bool = false,
    scale: Maybe(f64) = nil, // If nil, defaults internally to 1/sqrt(head_dim)
    enable_gqa: bool = false
) -> Tensor {
    out: Tensor
    
    // Logic for optional scale
    // We need a pointer to a value if it exists, or nil if it doesn't.
    scale_val: f64 = 0.0
    scale_ptr: rawptr = nil

    // We unwrap the Maybe. If it exists, we point to the local variable.
    // This is safe because the C call is synchronous.
    if v, ok := scale.?; ok {
        scale_val = v
        scale_ptr = &scale_val 
    }
    t.atg_scaled_dot_product_attention(
        &out,
        query, key, value,
        attn_mask,
        dropout_p,
        i32(1) if is_causal else i32(0),
        scale_val,
        scale_ptr,
        i32(1) if enable_gqa else i32(0)
    )
    return track(out)
}

// SCATTER FAMILY

scatter :: proc{scatter_tensor, scatter_value}
scatter_ :: proc{scatter_tensor_, scatter_value_}

@private
scatter_tensor :: proc(self: Tensor, dim: i64, index: Tensor, src: Tensor) -> Tensor {
    out: Tensor
    t.atg_scatter(&out, self, dim, index, src)
    return track(out)
}

@private
scatter_tensor_ :: proc(self: Tensor, dim: i64, index: Tensor, src: Tensor) -> Tensor {
    out: Tensor
    t.atg_scatter_(&out, self, dim, index, src)
    return self
}

@private
scatter_value :: proc(self: Tensor, dim: i64, index: Tensor, value: Scalar) -> Tensor {
    out: Tensor
    t.atg_scatter_value(&out, self, dim, index, value)
    return track(out)
}

@private
scatter_value_ :: proc(self: Tensor, dim: i64, index: Tensor, value: Scalar) -> Tensor {
    out: Tensor
    t.atg_scatter_value_(&out, self, dim, index, value)
    return self
}

scatter_add :: proc(self: Tensor, dim: i64, index: Tensor, src: Tensor) -> Tensor {
    out: Tensor
    t.atg_scatter_add(&out, self, dim, index, src)
    return track(out)
}

scatter_add_ :: proc(self: Tensor, dim: i64, index: Tensor, src: Tensor) -> Tensor {
    out: Tensor
    t.atg_scatter_add_(&out, self, dim, index, src)
    return self
}

// TODO: map reduce to enum: "sum", "prod", "mean", "amax", "amin"
scatter_reduce :: proc{scatter_reduce_tensor, scatter_reduce_value}
scatter_reduce_ :: proc{scatter_reduce_tensor_, scatter_reduce_value_}

@private
scatter_reduce_tensor :: proc(self: Tensor, dim: i64, index: Tensor, src: Tensor, reduce: string) -> Tensor {
    out: Tensor
    t.atg_scatter_reduce(&out, self, dim, index, src, cstring(raw_data(reduce)), i32(len(reduce)))
    return track(out)
}

@private
scatter_reduce_tensor_ :: proc(self: Tensor, dim: i64, index: Tensor, src: Tensor, reduce: string) -> Tensor {
    out: Tensor
    t.atg_scatter_reduce_(&out, self, dim, index, src, cstring(raw_data(reduce)), i32(len(reduce)))
    return self
}

@private
scatter_reduce_value :: proc(self: Tensor, dim: i64, index: Tensor, value: Scalar, reduce: string) -> Tensor {
    out: Tensor
    t.atg_scatter_value_reduce(&out, self, dim, index, value, cstring(raw_data(reduce)), i32(len(reduce)))
    return track(out)
}

@private
scatter_reduce_value_ :: proc(self: Tensor, dim: i64, index: Tensor, value: Scalar, reduce: string) -> Tensor {
    out: Tensor
    t.atg_scatter_value_reduce_(&out, self, dim, index, value, cstring(raw_data(reduce)), i32(len(reduce)))
    return self
}

// SEARCH SORTED

searchsorted :: proc{searchsorted_tensor, searchsorted_scalar}

@private
searchsorted_tensor :: proc(
    sorted_sequence: Tensor, 
    self: Tensor, 
    out_int32: bool = false, 
    right: bool = false, 
    side: string = "", 
    sorter: Tensor = Tensor{} // Optional
) -> Tensor {
    out: Tensor
    out_int32_i := i32(1) if out_int32 else i32(0)
    right_i := i32(1) if right else i32(0)
    
    // Side "left" or "right" if provided
    // TODO: map to enum
    side_ptr := cstring(nil)
    side_len := i32(0)
    if len(side) > 0 {
        side_ptr = cstring(raw_data(side))
        side_len = i32(len(side))
    }

    t.atg_searchsorted(&out, sorted_sequence, self, out_int32_i, right_i, side_ptr, side_len, sorter)
    return track(out)
}

@private
searchsorted_scalar :: proc(
    sorted_sequence: Tensor, 
    self: Scalar, 
    out_int32: bool = false, 
    right: bool = false, 
    side: string = "", 
    sorter: Tensor = Tensor{} 
) -> Tensor {
    out: Tensor
    out_int32_i := i32(1) if out_int32 else i32(0)
    right_i := i32(1) if right else i32(0)
    
    side_ptr := cstring(nil)
    side_len := i32(0)
    if len(side) > 0 {
        side_ptr = cstring(raw_data(side))
        side_len = i32(len(side))
    }

    t.atg_searchsorted_scalar(&out, sorted_sequence, self, out_int32_i, right_i, side_ptr, side_len, sorter)
    return track(out)
}

// SEGMENT REDUCE

segment_reduce :: proc(
    data: Tensor, 
    reduce: string, 
    lengths: Tensor = Tensor{}, // Optional
    indices: Tensor = Tensor{}, // Optional
    offsets: Tensor = Tensor{}, // Optional
    axis: i64 = 0,
    unsafe: bool = false,
    initial: Scalar = Scalar{} // Optional
) -> Tensor {
    out: Tensor
    unsafe_i := i32(1) if unsafe else i32(0)
    t.atg_segment_reduce(
        &out, 
        data, 
        cstring(raw_data(reduce)), 
        i32(len(reduce)), 
        lengths, 
        indices, 
        offsets, 
        axis, 
        unsafe_i, 
        initial,
    )
    return track(out)
}

select :: proc(self: Tensor, dim: i64, index: i64) -> Tensor {
    out: Tensor
    t.atg_select(&out, self, dim, index)
    return track(out)
}

select_copy :: proc(self: Tensor, dim: i64, index: i64) -> Tensor {
    out: Tensor
    t.atg_select_copy(&out, self, dim, index)
    return track(out)
}

select_backward :: proc(grad_output: Tensor, input_sizes: []i64, dim: i64, index: i64) -> Tensor {
    out: Tensor
    t.atg_select_backward(
        &out, 
        grad_output, 
        raw_data(input_sizes), 
        i32(len(input_sizes)), 
        dim, 
        index,
    )
    return track(out)
}

select_scatter :: proc(self: Tensor, src: Tensor, dim: i64, index: i64) -> Tensor {
    out: Tensor
    t.atg_select_scatter(&out, self, src, dim, index)
    return track(out)
}

// SELU

selu :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_selu(&out, self)
    return track(out)
}

selu_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_selu_(&out, self)
    return self
}

// Sets self to be a shallow copy of source
set_from_source :: proc(self: Tensor, source: Tensor) -> Tensor {
    out: Tensor
    t.atg_set(&out, self)
    return track(out)
}

set_from_source_ :: proc(self: Tensor, source: Tensor) -> Tensor {
    out: Tensor
    t.atg_set_source_tensor_(&out, self, source)
    return self
}

// Updates the storage of the tensor
set_data :: proc(self: Tensor, new_data: Tensor) {
    t.atg_set_data(self, new_data)
}

set_requires_grad :: proc(self: Tensor, requires_grad: bool) -> Tensor {
    out: Tensor
    r := i32(1) if requires_grad else i32(0)
    t.atg_set_requires_grad(&out, self, r)
    return track(out)
}

set_source_tensor :: proc(self: Tensor, source: Tensor) -> Tensor {
    out: Tensor
    t.atg_set_source_tensor(&out, self, source)
    return track(out)
}

// Low level storage manipulation
set_storage_offset :: proc(
    self: Tensor, 
    source: Tensor, 
    storage_offset: i64, 
    sizes: []i64, 
    strides: []i64
) -> Tensor {
    out: Tensor
    t.atg_set_source_tensor_storage_offset_(
        &out, 
        self, 
        source, 
        storage_offset, 
        raw_data(sizes), 
        i32(len(sizes)), 
        raw_data(strides), 
        i32(len(strides)),
    )
    return self
}

// SGN: Returns a tensor with the same size as input, containing the sign of the elements of input
sgn :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_sgn(&out, self)
    return track(out)
}

sgn_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_sgn_(&out, self)
    return self
}

// SIGN: Returns a new tensor with the sign of the elements of input
sign :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_sign(&out, self)
    return track(out)
}

sign_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_sign_(&out, self)
    return self
}

// SIGNBIT: Tests if each element of input has its sign bit set
signbit :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_signbit(&out, self)
    return track(out)
}

// SIN
sin :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_sin(&out, self)
    return track(out)
}

sin_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_sin_(&out, self)
    return self
}

// SINC - Computes the normalized sinc of input
sinc :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_sinc(&out, self)
    return track(out)
}

sinc_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_sinc_(&out, self)
    return self
}

// SINH
sinh :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_sinh(&out, self)
    return track(out)
}

sinh_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_sinh_(&out, self)
    return self
}

// SLOGDET: Calculates the sign and log absolute value of the determinant.
// Returns tuple (sign, logabsdet)
slogdet :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_slogdet(&out, self)
    return track(out)
}

// SIGMOID
sigmoid :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_sigmoid(&out, self)
    return track(out)
}

sigmoid_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_sigmoid_(&out, self)
    return self
}

// SIGMOID BACKWARD
sigmoid_backward :: proc(grad_output: Tensor, output: Tensor) -> Tensor {
    out: Tensor
    t.atg_sigmoid_backward(&out, grad_output, output)
    return track(out)
}

// SIGMOID BACKWARD GRAD INPUT
sigmoid_backward_grad_input :: proc(grad_input: Tensor, grad_output: Tensor, output: Tensor) -> Tensor {
    out: Tensor
    t.atg_sigmoid_backward_grad_input(&out, grad_input, grad_output, output)
    return track(out)
}

// SILU (Sigmoid Linear Unit)
silu :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_silu(&out, self)
    return track(out)
}

silu_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_silu_(&out, self)
    return self
}

// SILU BACKWARD
silu_backward :: proc(grad_output: Tensor, self: Tensor) -> Tensor {
    out: Tensor
    t.atg_silu_backward(&out, grad_output, self)
    return track(out)
}

// SILU BACKWARD GRAD INPUT
silu_backward_grad_input :: proc(grad_input: Tensor, grad_output: Tensor, self: Tensor) -> Tensor {
    out: Tensor
    t.atg_silu_backward_grad_input(&out, grad_input, grad_output, self)
    return track(out)
}

// SLICE
// Python eq: tensor[start:end:step]
slice :: proc(self: Tensor, dim: i64, start: i64, end: i64, step: i64 = 1) -> Tensor {
    out: Tensor
    // Passing nil to start_null/end_null implies "use the integer values provided"
    t.atg_slice(&out, self, dim, start, nil, end, nil, step)
    return track(out)
}

// SLICE COPY: Returns a copy of the slice (not a view)
slice_copy :: proc(self: Tensor, dim: i64, start: i64, end: i64, step: i64 = 1) -> Tensor {
    out: Tensor
    t.atg_slice_copy(&out, self, dim, start, nil, end, nil, step)
    return track(out)
}

// SLICE BACKWARD
slice_backward :: proc(
    grad_output: Tensor, 
    input_sizes: []i64, 
    dim: i64, 
    start: i64, 
    end: i64, 
    step: i64
) -> Tensor {
    out: Tensor
    t.atg_slice_backward(
        &out, 
        grad_output, 
        raw_data(input_sizes), 
        i32(len(input_sizes)), 
        dim, 
        start, 
        end, 
        step,
    )
    return track(out)
}

// SLICE INVERSE
slice_inverse :: proc(self: Tensor, src: Tensor, dim: i64, start: i64, end: i64, step: i64 = 1) -> Tensor {
    out: Tensor
    t.atg_slice_inverse(&out, self, src, dim, start, nil, end, nil, step)
    return track(out)
}

// SLICE SCATTER: Embeds values of `src` into `self` at the given slice indices
slice_scatter :: proc(self: Tensor, src: Tensor, dim: i64, start: i64, end: i64, step: i64 = 1) -> Tensor {
    out: Tensor
    t.atg_slice_scatter(&out, self, src, dim, start, nil, end, nil, step)
    return track(out)
}

//  CONVOLUTION OPERATIONS

slow_conv3d :: proc(
    self, weight: Tensor, 
    kernel_size: []i64, 
    bias: Tensor = {},
    stride: []i64, 
    padding: []i64
) -> Tensor {
    out: Tensor
    t.atg_slow_conv3d(
        &out, 
        self, 
        weight, 
        raw_data(kernel_size), i32(len(kernel_size)),
        bias, 
        raw_data(stride), i32(len(stride)),
        raw_data(padding), i32(len(padding)),
    )
    return track(out)
}

slow_conv_dilated2d :: proc(
    self, weight: Tensor, 
    kernel_size: []i64, 
    bias: Tensor = {},
    stride: []i64, 
    padding: []i64,
    dilation: []i64
) -> Tensor {
    out: Tensor
    t.atg_slow_conv_dilated2d(
        &out, 
        self, 
        weight, 
        raw_data(kernel_size), i32(len(kernel_size)),
        bias, 
        raw_data(stride), i32(len(stride)),
        raw_data(padding), i32(len(padding)),
        raw_data(dilation), i32(len(dilation)),
    )
    return track(out)
}

slow_conv_dilated3d :: proc(
    self, weight: Tensor, 
    kernel_size: []i64, 
    bias: Tensor = {},
    stride: []i64, 
    padding: []i64,
    dilation: []i64
) -> Tensor {
    out: Tensor
    t.atg_slow_conv_dilated3d(
        &out, 
        self, 
        weight, 
        raw_data(kernel_size), i32(len(kernel_size)),
        bias, 
        raw_data(stride), i32(len(stride)),
        raw_data(padding), i32(len(padding)),
        raw_data(dilation), i32(len(dilation)),
    )
    return track(out)
}

slow_conv_transpose2d :: proc(
    self, weight: Tensor, 
    kernel_size: []i64, 
    bias: Tensor = {},
    stride: []i64, 
    padding: []i64,
    output_padding: []i64,
    dilation: []i64
) -> Tensor {
    out: Tensor
    t.atg_slow_conv_transpose2d(
        &out, 
        self, 
        weight, 
        raw_data(kernel_size), i32(len(kernel_size)),
        bias, 
        raw_data(stride), i32(len(stride)),
        raw_data(padding), i32(len(padding)),
        raw_data(output_padding), i32(len(output_padding)),
        raw_data(dilation), i32(len(dilation)),
    )
    return track(out)
}

slow_conv_transpose3d :: proc(
    self, weight: Tensor, 
    kernel_size: []i64, 
    bias: Tensor = {},
    stride: []i64, 
    padding: []i64,
    output_padding: []i64,
    dilation: []i64
) -> Tensor {
    out: Tensor
    t.atg_slow_conv_transpose3d(
        &out, 
        self, 
        weight, 
        raw_data(kernel_size), i32(len(kernel_size)),
        bias, 
        raw_data(stride), i32(len(stride)),
        raw_data(padding), i32(len(padding)),
        raw_data(output_padding), i32(len(output_padding)),
        raw_data(dilation), i32(len(dilation)),
    )
    return track(out)
}

// LINEAR ALGEBRA

smm :: proc(self, mat2: Tensor) -> Tensor {
    out: Tensor
    t.atg_smm(&out, self, mat2)
    return track(out)
}

// LOSS FUNCTIONS

// Smooth L1 Loss

smooth_l1_loss :: proc(self, target: Tensor, reduction: i64, beta: f64) -> Tensor {
    out: Tensor
    t.atg_smooth_l1_loss(&out, self, target, reduction, beta)
    return track(out)
}

smooth_l1_loss_backward :: proc(grad_output, self, target: Tensor, reduction: i64, beta: f64) -> Tensor {
    out: Tensor
    t.atg_smooth_l1_loss_backward(&out, grad_output, self, target, reduction, beta)
    return track(out)
}

// Calculates gradient w.r.t input, populating a new tensor
smooth_l1_loss_backward_grad_input :: proc(grad_output, self, target: Tensor, reduction: i64, beta: f64) -> Tensor {
    grad_input := new_tensor()
    dummy: Tensor
    t.atg_smooth_l1_loss_backward_grad_input(&dummy, grad_input, grad_output, self, target, reduction, beta)
    return grad_input
}

// Soft Margin Loss

soft_margin_loss :: proc(self, target: Tensor, reduction: i64) -> Tensor {
    out: Tensor
    t.atg_soft_margin_loss(&out, self, target, reduction)
    return track(out)
}

soft_margin_loss_backward :: proc(grad_output, self, target: Tensor, reduction: i64) -> Tensor {
    out: Tensor
    t.atg_soft_margin_loss_backward(&out, grad_output, self, target, reduction)
    return track(out)
}

soft_margin_loss_backward_grad_input :: proc(grad_output, self, target: Tensor, reduction: i64) -> Tensor {
    grad_input := new_tensor()
    dummy: Tensor
    t.atg_soft_margin_loss_backward_grad_input(&dummy, grad_input, grad_output, self, target, reduction)
    return grad_input
}

//  ACTIVATIONS

softmax :: proc(self: Tensor, dim: i64, dtype: ScalarType) -> Tensor {
    out: Tensor
    t.atg_softmax(&out, self, dim, i32(dtype))
    return track(out)
}

softplus :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_softplus(&out, self)
    return track(out)
}

softplus_backward :: proc(grad_output, self: Tensor, beta, threshold: Scalar) -> Tensor {
    out: Tensor
    t.atg_softplus_backward(&out, grad_output, self, beta, threshold)
    return track(out)
}

softplus_backward_grad_input :: proc(grad_output, self: Tensor, beta, threshold: Scalar) -> Tensor {
    grad_input := new_tensor()
    dummy: Tensor
    t.atg_softplus_backward_grad_input(&dummy, grad_input, grad_output, self, beta, threshold)
    return grad_input
}

softshrink :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_softshrink(&out, self)
    return track(out)
}

softshrink_backward :: proc(grad_output, self: Tensor, lambd: Scalar) -> Tensor {
    out: Tensor
    t.atg_softshrink_backward(&out, grad_output, self, lambd)
    return track(out)
}

softshrink_backward_grad_input :: proc(grad_output, self: Tensor, lambd: Scalar) -> Tensor {
    grad_input := new_tensor()
    dummy: Tensor
    t.atg_softshrink_backward_grad_input(&dummy, grad_input, grad_output, self, lambd)
    return grad_input
}

//  SORTING

sort :: proc(self: Tensor, dim: i64, descending: bool = false) -> Tensor {
    out: Tensor
    desc_int := i32(1) if descending else i32(0)
    t.atg_sort(&out, self, dim, desc_int)
    return track(out)
}

sort_stable :: proc(self: Tensor, stable: bool, dim: i64, descending: bool = false) -> Tensor {
    out: Tensor
    stable_int := i32(1) if stable else i32(0)
    desc_int := i32(1) if descending else i32(0)
    t.atg_sort_stable(&out, self, stable_int, dim, desc_int)
    return track(out)
}

// Returns both values and indices.
// Allocates two new tensors, passes them as out-buffers to the C function
sort_values :: proc(self: Tensor, dim: i64, descending: bool = false) -> (values, indices: Tensor) {
    values = new_tensor()
    indices = new_tensor()
    
    desc_int := i32(1) if descending else i32(0)
    
    dummy: Tensor
    t.atg_sort_values(&dummy, values, indices, self, dim, desc_int)
    
    return values, indices
}

sort_values_stable :: proc(self: Tensor, stable: bool, dim: i64, descending: bool = false) -> (values, indices: Tensor) {
    values = new_tensor()
    indices = new_tensor()
    
    stable_int := i32(1) if stable else i32(0)
    desc_int := i32(1) if descending else i32(0)
    
    dummy: Tensor
    t.atg_sort_values_stable(&dummy, values, indices, self, stable_int, dim, desc_int)
    
    return values, indices
}

// SPARSE BSC (Block Compressed Sparse Column)

sparse_bsc_tensor :: proc(
    ccol_indices: Tensor,
    row_indices: Tensor,
    values: Tensor,
    kind: ScalarType = .Float,
    device: DeviceType = .CPU,
) -> Tensor {
    out: Tensor
    t.atg_sparse_bsc_tensor(
        &out,
        ccol_indices,
        row_indices,
        values,
        i32(kind),
        i32(device),
    )
    return track(out)
}

sparse_bsc_tensor_size :: proc(
    ccol_indices: Tensor,
    row_indices: Tensor,
    values: Tensor,
    size: []i64,
    kind: ScalarType = .Float,
    device: DeviceType = .CPU,
) -> Tensor {
    out: Tensor
    t.atg_sparse_bsc_tensor_ccol_row_value_size(
        &out,
        ccol_indices,
        row_indices,
        values,
        raw_data(size),
        i32(len(size)),
        i32(kind),
        i32(device),
    )
    return track(out)
}

// SPARSE BSR - Block Compressed Sparse Row

sparse_bsr_tensor :: proc(
    crow_indices: Tensor,
    col_indices: Tensor,
    values: Tensor,
    kind: ScalarType = .Float,
    device: DeviceType = .CPU,
) -> Tensor {
    out: Tensor
    t.atg_sparse_bsr_tensor(
        &out,
        crow_indices,
        col_indices,
        values,
        i32(kind),
        i32(device),
    )
    return track(out)
}

sparse_bsr_tensor_size :: proc(
    crow_indices: Tensor,
    col_indices: Tensor,
    values: Tensor,
    size: []i64,
    kind: ScalarType = .Float,
    device: DeviceType = .CPU,
) -> Tensor {
    out: Tensor
    t.atg_sparse_bsr_tensor_crow_col_value_size(
        &out,
        crow_indices,
        col_indices,
        values,
        raw_data(size),
        i32(len(size)),
        i32(kind),
        i32(device),
    )
    return track(out)
}

// SPARSE COMPRESSED Generic

sparse_compressed_tensor :: proc(
    compressed_indices: Tensor,
    plain_indices: Tensor,
    values: Tensor,
    kind: ScalarType = .Float,
    device: DeviceType = .CPU,
) -> Tensor {
    out: Tensor
    t.atg_sparse_compressed_tensor(
        &out,
        compressed_indices,
        plain_indices,
        values,
        i32(kind),
        i32(device),
    )
    return track(out)
}

sparse_compressed_tensor_size :: proc(
    compressed_indices: Tensor,
    plain_indices: Tensor,
    values: Tensor,
    size: []i64,
    kind: ScalarType = .Float,
    device: DeviceType = .CPU,
) -> Tensor {
    out: Tensor
    t.atg_sparse_compressed_tensor_comp_plain_value_size(
        &out,
        compressed_indices,
        plain_indices,
        values,
        raw_data(size),
        i32(len(size)),
        i32(kind),
        i32(device),
    )
    return track(out)
}

// SPARSE CSC - Compressed Sparse Column

sparse_csc_tensor :: proc(
    ccol_indices: Tensor,
    row_indices: Tensor,
    values: Tensor,
    kind: ScalarType = .Float,
    device: DeviceType = .CPU,
) -> Tensor {
    out: Tensor
    t.atg_sparse_csc_tensor(
        &out,
        ccol_indices,
        row_indices,
        values,
        i32(kind),
        i32(device),
    )
    return track(out)
}

sparse_csc_tensor_size :: proc(
    ccol_indices: Tensor,
    row_indices: Tensor,
    values: Tensor,
    size: []i64,
    kind: ScalarType = .Float,
    device: DeviceType = .CPU,
) -> Tensor {
    out: Tensor
    t.atg_sparse_csc_tensor_ccol_row_value_size(
        &out,
        ccol_indices,
        row_indices,
        values,
        raw_data(size),
        i32(len(size)),
        i32(kind),
        i32(device),
    )
    return track(out)
}

// SPARSE CSR - Compressed Sparse Row

sparse_csr_tensor :: proc(
    crow_indices: Tensor,
    col_indices: Tensor,
    values: Tensor,
    kind: ScalarType = .Float,
    device: DeviceType = .CPU,
) -> Tensor {
    out: Tensor
    t.atg_sparse_csr_tensor(
        &out,
        crow_indices,
        col_indices,
        values,
        i32(kind),
        i32(device),
    )
    return track(out)
}

sparse_csr_tensor_size :: proc(
    crow_indices: Tensor,
    col_indices: Tensor,
    values: Tensor,
    size: []i64,
    kind: ScalarType = .Float,
    device: DeviceType = .CPU,
) -> Tensor {
    out: Tensor
    t.atg_sparse_csr_tensor_crow_col_value_size(
        &out,
        crow_indices,
        col_indices,
        values,
        raw_data(size),
        i32(len(size)),
        i32(kind),
        i32(device),
    )
    return track(out)
}

// Creates an empty sparse tensor of a given size
sparse_coo_tensor :: proc(
    size: []i64,
    kind: ScalarType = .Float,
    device: DeviceType = .CPU,
) -> Tensor {
    out: Tensor
    t.atg_sparse_coo_tensor(
        &out,
        raw_data(size),
        i32(len(size)),
        i32(kind),
        i32(device),
    )
    return track(out)
}

// Creates a sparse tensor from indices and values (deducing size)
sparse_coo_tensor_indices :: proc(
    indices: Tensor,
    values: Tensor,
    kind: ScalarType = .Float,
    device: DeviceType = .CPU,
    is_coalesced: bool = false,
) -> Tensor {
    out: Tensor
    is_coalesced_int := i32(1) if is_coalesced else i32(0)
    t.atg_sparse_coo_tensor_indices(
        &out,
        indices,
        values,
        i32(kind),
        i32(device),
        is_coalesced_int,
    )
    return track(out)
}

// Creates a sparse tensor from indices and values with explicit size
sparse_coo_tensor_indices_size :: proc(
    indices: Tensor,
    values: Tensor,
    size: []i64,
    kind: ScalarType = .Float,
    device: DeviceType = .CPU,
    is_coalesced: bool = false,
) -> Tensor {
    out: Tensor
    is_coalesced_int := i32(1) if is_coalesced else i32(0)
    t.atg_sparse_coo_tensor_indices_size(
        &out,
        indices,
        values,
        raw_data(size),
        i32(len(size)),
        i32(kind),
        i32(device),
        is_coalesced_int,
    )
    return track(out)
}

// OPERATIONS

// Returns a new sparse tensor with values from `self` only at elements present in mask
sparse_mask :: proc(self: Tensor, mask: Tensor) -> Tensor {
    out: Tensor
    t.atg_sparse_mask(&out, self, mask)
    return track(out)
}

// Performs sampled matrix multiplication: (self.to_dense() + (mat1 @ mat2)) * mask
sparse_sampled_addmm :: proc(self, mat1, mat2: Tensor) -> Tensor {
    out: Tensor
    t.atg_sparse_sampled_addmm(&out, self, mat1, mat2)
    return track(out)
}

// RESIZING (Mutation)

sparse_resize :: proc(
    self: Tensor,
    size: []i64,
    sparse_dim: i64,
    dense_dim: i64,
) -> Tensor {
    out: Tensor
    t.atg_sparse_resize(
        &out,
        self,
        raw_data(size),
        i32(len(size)),
        sparse_dim,
        dense_dim,
    )
    return track(out)
}

sparse_resize_ :: proc(
    self: Tensor,
    size: []i64,
    sparse_dim: i64,
    dense_dim: i64,
) -> Tensor {
    out: Tensor
    t.atg_sparse_resize_(
        &out,
        self,
        raw_data(size),
        i32(len(size)),
        sparse_dim,
        dense_dim,
    )
    return self
}

// Resize and Clear
sparse_resize_and_clear :: proc(
    self: Tensor,
    size: []i64,
    sparse_dim: i64,
    dense_dim: i64,
) -> Tensor {
    out: Tensor
    t.atg_sparse_resize_and_clear(
        &out,
        self,
        raw_data(size),
        i32(len(size)),
        sparse_dim,
        dense_dim,
    )
    return track(out)
}

sparse_resize_and_clear_ :: proc(
    self: Tensor,
    size: []i64,
    sparse_dim: i64,
    dense_dim: i64,
) -> Tensor {
    out: Tensor
    t.atg_sparse_resize_and_clear_(
        &out,
        self,
        raw_data(size),
        i32(len(size)),
        sparse_dim,
        dense_dim,
    )
    return self
}

// SPECIAL FUNCTIONS: AIRY & BESSEL

special_airy_ai :: proc(x: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_airy_ai(&out, x)
    return track(out)
}

special_bessel_j0 :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_bessel_j0(&out, self)
    return track(out)
}

special_bessel_j1 :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_bessel_j1(&out, self)
    return track(out)
}

special_bessel_y0 :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_bessel_y0(&out, self)
    return track(out)
}

special_bessel_y1 :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_bessel_y1(&out, self)
    return track(out)
}

// SPECIAL FUNCTIONS: CHEBYSHEV POLYNOMIALS

special_chebyshev_polynomial_t :: proc{
    special_chebyshev_polynomial_t_tt,
    special_chebyshev_polynomial_t_ts,
    special_chebyshev_polynomial_t_st,
}

special_chebyshev_polynomial_u :: proc{
    special_chebyshev_polynomial_u_tt,
    special_chebyshev_polynomial_u_ts,
    special_chebyshev_polynomial_u_st,
}

special_chebyshev_polynomial_v :: proc{
    special_chebyshev_polynomial_v_tt,
    special_chebyshev_polynomial_v_ts,
    special_chebyshev_polynomial_v_st,
}

special_chebyshev_polynomial_w :: proc{
    special_chebyshev_polynomial_w_tt,
    special_chebyshev_polynomial_w_ts,
    special_chebyshev_polynomial_w_st,
}

// Chebyshev T

@private
special_chebyshev_polynomial_t_tt :: proc(x: Tensor, n: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_chebyshev_polynomial_t(&out, x, n)
    return track(out)
}

@private
special_chebyshev_polynomial_t_ts :: proc(x: Tensor, n: Scalar) -> Tensor {
    out: Tensor
    t.atg_special_chebyshev_polynomial_t_n_scalar(&out, x, n)
    return track(out)
}

@private
special_chebyshev_polynomial_t_st :: proc(x: Scalar, n: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_chebyshev_polynomial_t_x_scalar(&out, x, n)
    return track(out)
}

// Chebyshev U
@private
special_chebyshev_polynomial_u_tt :: proc(x: Tensor, n: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_chebyshev_polynomial_u(&out, x, n)
    return track(out)
}

@private
special_chebyshev_polynomial_u_ts :: proc(x: Tensor, n: Scalar) -> Tensor {
    out: Tensor
    t.atg_special_chebyshev_polynomial_u_n_scalar(&out, x, n)
    return track(out)
}

@private
special_chebyshev_polynomial_u_st :: proc(x: Scalar, n: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_chebyshev_polynomial_u_x_scalar(&out, x, n)
    return track(out)
}

// Chebyshev V

@private
special_chebyshev_polynomial_v_tt :: proc(x: Tensor, n: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_chebyshev_polynomial_v(&out, x, n)
    return track(out)
}

@private
special_chebyshev_polynomial_v_ts :: proc(x: Tensor, n: Scalar) -> Tensor {
    out: Tensor
    t.atg_special_chebyshev_polynomial_v_n_scalar(&out, x, n)
    return track(out)
}

@private
special_chebyshev_polynomial_v_st :: proc(x: Scalar, n: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_chebyshev_polynomial_v_x_scalar(&out, x, n)
    return track(out)
}

// Chebyshev W
@private
special_chebyshev_polynomial_w_tt :: proc(x: Tensor, n: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_chebyshev_polynomial_w(&out, x, n)
    return track(out)
}

@private
special_chebyshev_polynomial_w_ts :: proc(x: Tensor, n: Scalar) -> Tensor {
    out: Tensor
    t.atg_special_chebyshev_polynomial_w_n_scalar(&out, x, n)
    return track(out)
}

@private
special_chebyshev_polynomial_w_st :: proc(x: Scalar, n: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_chebyshev_polynomial_w_x_scalar(&out, x, n)
    return track(out)
}

special_digamma :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_digamma(&out, self)
    return track(out)
}

special_entr :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_entr(&out, self)
    return track(out)
}

special_erf :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_erf(&out, self)
    return track(out)
}

special_erfc :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_erfc(&out, self)
    return track(out)
}

special_erfcx :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_erfcx(&out, self)
    return track(out)
}

special_erfinv :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_erfinv(&out, self)
    return track(out)
}

special_exp2 :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_exp2(&out, self)
    return track(out)
}

special_expit :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_expit(&out, self)
    return track(out)
}

special_expm1 :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_expm1(&out, self)
    return track(out)
}

// Gamma Functions
special_gammainc :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_gammainc(&out, self, other)
    return track(out)
}

special_gammaincc :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_gammaincc(&out, self, other)
    return track(out)
}

special_gammaln :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_gammaln(&out, self)
    return track(out)
}

special_multigammaln :: proc(self: Tensor, p: i64) -> Tensor {
    out: Tensor
    t.atg_special_multigammaln(&out, self, p)
    return track(out)
}

// Hermite Polynomial H
special_hermite_polynomial_h :: proc{
    special_hermite_polynomial_h_tt,
    special_hermite_polynomial_h_ts,
    special_hermite_polynomial_h_st,
}

@private
special_hermite_polynomial_h_tt :: proc(x, n: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_hermite_polynomial_h(&out, x, n)
    return track(out)
}

@private
special_hermite_polynomial_h_ts :: proc(x: Tensor, n: Scalar) -> Tensor {
    out: Tensor
    t.atg_special_hermite_polynomial_h_n_scalar(&out, x, n)
    return track(out)
}

@private
special_hermite_polynomial_h_st :: proc(x: Scalar, n: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_hermite_polynomial_h_x_scalar(&out, x, n)
    return track(out)
}

// Hermite Polynomial He
special_hermite_polynomial_he :: proc{
    special_hermite_polynomial_he_tt,
    special_hermite_polynomial_he_ts,
    special_hermite_polynomial_he_st,
}

@private
special_hermite_polynomial_he_tt :: proc(x, n: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_hermite_polynomial_he(&out, x, n)
    return track(out)
}

@private
special_hermite_polynomial_he_ts :: proc(x: Tensor, n: Scalar) -> Tensor {
    out: Tensor
    t.atg_special_hermite_polynomial_he_n_scalar(&out, x, n)
    return track(out)
}

@private
special_hermite_polynomial_he_st :: proc(x: Scalar, n: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_hermite_polynomial_he_x_scalar(&out, x, n)
    return track(out)
}

// Bessel Functions
special_i0 :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_i0(&out, self)
    return track(out)
}

special_i0e :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_i0e(&out, self)
    return track(out)
}

special_i1 :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_i1(&out, self)
    return track(out)
}

special_i1e :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_i1e(&out, self)
    return track(out)
}

// Laguerre Polynomial L
special_laguerre_polynomial_l :: proc{
    special_laguerre_polynomial_l_tt,
    special_laguerre_polynomial_l_ts,
    special_laguerre_polynomial_l_st,
}

@private
special_laguerre_polynomial_l_tt :: proc(x, n: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_laguerre_polynomial_l(&out, x, n)
    return track(out)
}

@private
special_laguerre_polynomial_l_ts :: proc(x: Tensor, n: Scalar) -> Tensor {
    out: Tensor
    t.atg_special_laguerre_polynomial_l_n_scalar(&out, x, n)
    return track(out)
}

@private
special_laguerre_polynomial_l_st :: proc(x: Scalar, n: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_laguerre_polynomial_l_x_scalar(&out, x, n)
    return track(out)
}

special_legendre_polynomial_p :: proc{
    special_legendre_polynomial_p_tt,
    special_legendre_polynomial_p_ts,
    special_legendre_polynomial_p_st,
}

@private
special_legendre_polynomial_p_tt :: proc(x, n: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_legendre_polynomial_p(&out, x, n)
    return track(out)
}

@private
special_legendre_polynomial_p_ts :: proc(x: Tensor, n: Scalar) -> Tensor {
    out: Tensor
    t.atg_special_legendre_polynomial_p_n_scalar(&out, x, n)
    return track(out)
}

@private
special_legendre_polynomial_p_st :: proc(x: Scalar, n: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_legendre_polynomial_p_x_scalar(&out, x, n)
    return track(out)
}

special_log1p :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_log1p(&out, self)
    return track(out)
}

special_log_ndtr :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_log_ndtr(&out, self)
    return track(out)
}

special_log_softmax :: proc(self: Tensor, dim: i64, dtype: ScalarType) -> Tensor {
    out: Tensor
    t.atg_special_log_softmax(&out, self, dim, i32(dtype))
    return track(out)
}

special_logit :: proc(self: Tensor, eps: f64) -> Tensor {
    out: Tensor
    // eps_null as nil implies the value in eps_v is valid
    t.atg_special_logit(&out, self, eps, nil)
    return track(out)
}

special_logsumexp :: proc(self: Tensor, dim: []i64, keepdim: bool = false) -> Tensor {
    out: Tensor
    keep_int := i32(1) if keepdim else i32(0)
    t.atg_special_logsumexp(&out, self, raw_data(dim), i32(len(dim)), keep_int)
    return track(out)
}

// Modified Bessel Functions
special_modified_bessel_i0 :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_modified_bessel_i0(&out, self)
    return track(out)
}

special_modified_bessel_i1 :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_modified_bessel_i1(&out, self)
    return track(out)
}

special_modified_bessel_k0 :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_modified_bessel_k0(&out, self)
    return track(out)
}

special_modified_bessel_k1 :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_modified_bessel_k1(&out, self)
    return track(out)
}

special_ndtr :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_ndtr(&out, self)
    return track(out)
}

special_ndtri :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_ndtri(&out, self)
    return track(out)
}

special_polygamma :: proc(n: i64, self: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_polygamma(&out, n, self)
    return track(out)
}

special_psi :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_psi(&out, self)
    return track(out)
}

special_round :: proc(self: Tensor, decimals: i64) -> Tensor {
    out: Tensor
    t.atg_special_round(&out, self, decimals)
    return track(out)
}

special_scaled_modified_bessel_k0 :: proc(x: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_scaled_modified_bessel_k0(&out, x)
    return track(out)
}

special_scaled_modified_bessel_k1 :: proc(x: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_scaled_modified_bessel_k1(&out, x)
    return track(out)
}

special_sinc :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_sinc(&out, self)
    return track(out)
}

special_softmax :: proc(self: Tensor, dim: i64, dtype: ScalarType) -> Tensor {
    out: Tensor
    t.atg_special_softmax(&out, self, dim, i32(dtype))
    return track(out)
}

special_spherical_bessel_j0 :: proc(x: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_spherical_bessel_j0(&out, x)
    return track(out)
}

// Shifted Chebyshev Polynomials (T, U, V, W)

special_shifted_chebyshev_polynomial_t :: proc{
    special_shifted_chebyshev_polynomial_t_tensor,
    special_shifted_chebyshev_polynomial_t_n_scalar,
    special_shifted_chebyshev_polynomial_t_x_scalar,
}

@private
special_shifted_chebyshev_polynomial_t_tensor :: proc(x: Tensor, n: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_shifted_chebyshev_polynomial_t(&out, x, n)
    return track(out)
}

@private
special_shifted_chebyshev_polynomial_t_n_scalar :: proc(x: Tensor, n: Scalar) -> Tensor {
    out: Tensor
    t.atg_special_shifted_chebyshev_polynomial_t_n_scalar(&out, x, n)
    return track(out)
}

@private
special_shifted_chebyshev_polynomial_t_x_scalar :: proc(x: Scalar, n: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_shifted_chebyshev_polynomial_t_x_scalar(&out, x, n)
    return track(out)
}

special_shifted_chebyshev_polynomial_u :: proc{
    special_shifted_chebyshev_polynomial_u_tensor,
    special_shifted_chebyshev_polynomial_u_n_scalar,
    special_shifted_chebyshev_polynomial_u_x_scalar,
}

@private
special_shifted_chebyshev_polynomial_u_tensor :: proc(x: Tensor, n: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_shifted_chebyshev_polynomial_u(&out, x, n)
    return track(out)
}

@private
special_shifted_chebyshev_polynomial_u_n_scalar :: proc(x: Tensor, n: Scalar) -> Tensor {
    out: Tensor
    t.atg_special_shifted_chebyshev_polynomial_u_n_scalar(&out, x, n)
    return track(out)
}

@private
special_shifted_chebyshev_polynomial_u_x_scalar :: proc(x: Scalar, n: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_shifted_chebyshev_polynomial_u_x_scalar(&out, x, n)
    return track(out)
}

special_shifted_chebyshev_polynomial_v :: proc{
    special_shifted_chebyshev_polynomial_v_tensor,
    special_shifted_chebyshev_polynomial_v_n_scalar,
    special_shifted_chebyshev_polynomial_v_x_scalar,
}

@private
special_shifted_chebyshev_polynomial_v_tensor :: proc(x: Tensor, n: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_shifted_chebyshev_polynomial_v(&out, x, n)
    return track(out)
}

@private
special_shifted_chebyshev_polynomial_v_n_scalar :: proc(x: Tensor, n: Scalar) -> Tensor {
    out: Tensor
    t.atg_special_shifted_chebyshev_polynomial_v_n_scalar(&out, x, n)
    return track(out)
}

@private
special_shifted_chebyshev_polynomial_v_x_scalar :: proc(x: Scalar, n: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_shifted_chebyshev_polynomial_v_x_scalar(&out, x, n)
    return track(out)
}

special_shifted_chebyshev_polynomial_w :: proc{
    special_shifted_chebyshev_polynomial_w_tensor,
    special_shifted_chebyshev_polynomial_w_n_scalar,
    special_shifted_chebyshev_polynomial_w_x_scalar,
}

@private
special_shifted_chebyshev_polynomial_w_tensor :: proc(x: Tensor, n: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_shifted_chebyshev_polynomial_w(&out, x, n)
    return track(out)
}

@private
special_shifted_chebyshev_polynomial_w_n_scalar :: proc(x: Tensor, n: Scalar) -> Tensor {
    out: Tensor
    t.atg_special_shifted_chebyshev_polynomial_w_n_scalar(&out, x, n)
    return track(out)
}

@private
special_shifted_chebyshev_polynomial_w_x_scalar :: proc(x: Scalar, n: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_shifted_chebyshev_polynomial_w_x_scalar(&out, x, n)
    return track(out)
}

// XLog1py

special_xlog1py :: proc{
    special_xlog1py_tensor,
    special_xlog1py_other_scalar,
    special_xlog1py_self_scalar,
}

@private
special_xlog1py_tensor :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_xlog1py(&out, self, other)
    return track(out)
}


special_xlog1py_other_scalar :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_special_xlog1py_other_scalar(&out, self, other)
    return track(out)
}

@private
special_xlog1py_self_scalar :: proc(self: Scalar, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_xlog1py_self_scalar(&out, self, other)
    return track(out)
}

// XLogy

special_xlogy :: proc{
    special_xlogy_tensor,
    special_xlogy_other_scalar,
    special_xlogy_self_scalar,
}

@private
special_xlogy_tensor :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_xlogy(&out, self, other)
    return track(out)
}

@private
special_xlogy_other_scalar :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_special_xlogy_other_scalar(&out, self, other)
    return track(out)
}

@private
special_xlogy_self_scalar :: proc(self: Scalar, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_xlogy_self_scalar(&out, self, other)
    return track(out)
}

// Zeta

special_zeta :: proc{
    special_zeta_tensor,
    special_zeta_other_scalar,
    special_zeta_self_scalar,
}

@private
special_zeta_tensor :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_zeta(&out, self, other)
    return track(out)
}

@private
special_zeta_other_scalar :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_special_zeta_other_scalar(&out, self, other)
    return track(out)
}

@private
special_zeta_self_scalar :: proc(self: Scalar, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_special_zeta_self_scalar(&out, self, other)
    return track(out)
}

sqrt :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_sqrt(&out, self)
    return track(out)
}

sqrt_ :: proc(self: Tensor) -> Tensor {
    dummy: Tensor
    t.atg_sqrt_(&dummy, self)
    return self
}

square :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_square(&out, self)
    return track(out)
}

square_ :: proc(self: Tensor) -> Tensor {
    dummy: Tensor
    t.atg_square_(&dummy, self)
    return self
}

// SQUEEZE OPERATIONS

squeeze :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_squeeze(&out, self)
    return track(out)
}

squeeze_dim :: proc(self: Tensor, dim: i64) -> Tensor {
    out: Tensor
    t.atg_squeeze_dim(&out, self, dim)
    return track(out)
}

squeeze_dims :: proc(self: Tensor, dims: []i64) -> Tensor {
    out: Tensor
    t.atg_squeeze_dims(&out, self, raw_data(dims), i32(len(dims)))
    return track(out)
}

squeeze_ :: proc(self: Tensor) -> Tensor {
    dummy: Tensor
    t.atg_squeeze_(&dummy, self)
    return self
}

squeeze_dim_ :: proc(self: Tensor, dim: i64) -> Tensor {
    dummy: Tensor
    t.atg_squeeze_dim_(&dummy, self, dim)
    return self
}

squeeze_dims_ :: proc(self: Tensor, dims: []i64) -> Tensor {
    dummy: Tensor
    t.atg_squeeze_dims_(&dummy, self, raw_data(dims), i32(len(dims)))
    return self
}

// Squeeze Copy Variants return a copy, with new memory alloc
squeeze_copy :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_squeeze_copy(&out, self)
    return track(out)
}

squeeze_copy_dim :: proc(self: Tensor, dim: i64) -> Tensor {
    out: Tensor
    t.atg_squeeze_copy_dim(&out, self, dim)
    return track(out)
}

squeeze_copy_dims :: proc(self: Tensor, dims: []i64) -> Tensor {
    out: Tensor
    t.atg_squeeze_copy_dims(&out, self, raw_data(dims), i32(len(dims)))
    return track(out)
}

// SPARSE ADDMM

sspaddmm :: proc(self, mat1, mat2: Tensor) -> Tensor {
    out: Tensor
    t.atg_sspaddmm(&out, self, mat1, mat2)
    return track(out)
}

// STACK
// Concatenates a sequence of tensors along a new dimension.

stack :: proc(tensors: []Tensor, dim: i64 = 0) -> Tensor {
    out: Tensor
    t.atg_stack(&out, raw_data(tensors), i32(len(tensors)), dim)
    return track(out)
}

// STANDARD DEVIATION & MEAN

std :: proc(self: Tensor, unbiased: bool = true) -> Tensor {
    out: Tensor
    u_int := i32(1) if unbiased else i32(0)
    t.atg_std(&out, self, u_int)
    return track(out)
}

std_dim :: proc(self: Tensor, dims: []i64, unbiased: bool = true, keepdim: bool = false) -> Tensor {
    out: Tensor
    u_int := i32(1) if unbiased else i32(0)
    k_int := i32(1) if keepdim else i32(0)
    
    t.atg_std_dim(&out, self, raw_data(dims), i32(len(dims)), u_int, k_int)
    return track(out)
}

std_correction :: proc(self: Tensor, dims: []i64, correction: Scalar, keepdim: bool = false) -> Tensor {
    out: Tensor
    k_int := i32(1) if keepdim else i32(0)
    
    t.atg_std_correction(&out, self, raw_data(dims), i32(len(dims)), correction, k_int)
    return track(out)
}

std_mean :: proc(self: Tensor, unbiased: bool = true) -> Tensor {
    out: Tensor
    u_int := i32(1) if unbiased else i32(0)
    t.atg_std_mean(&out, self, u_int)
    return track(out)
}

std_mean_dim :: proc(self: Tensor, dims: []i64, unbiased: bool = true, keepdim: bool = false) -> Tensor {
    out: Tensor
    u_int := i32(1) if unbiased else i32(0)
    k_int := i32(1) if keepdim else i32(0)
    
    t.atg_std_mean_dim(&out, self, raw_data(dims), i32(len(dims)), u_int, k_int)
    return track(out)
}

std_mean_correction :: proc(self: Tensor, dims: []i64, correction: Scalar, keepdim: bool = false) -> Tensor {
    out: Tensor
    k_int := i32(1) if keepdim else i32(0)
    
    t.atg_std_mean_correction(&out, self, raw_data(dims), i32(len(dims)), correction, k_int)
    return track(out)
}

// SPECTRAL OPERATIONS (STFT)

stft :: proc(
    self: Tensor, 
    n_fft: i64, 
    hop_length: i64, 
    win_length: i64, 
    window: Tensor, 
    normalized: bool = false, 
    onesided: bool = true, 
    return_complex: bool = false, // often nil in python, default logic applies
    align_to_window: bool = false
) -> Tensor {
    out: Tensor
    
    norm_int := i32(1) if normalized else i32(0)
    one_int  := i32(1) if onesided else i32(0)
    cplx_int := i32(1) if return_complex else i32(0)
    align_int := i32(1) if align_to_window else i32(0)

    t.atg_stft(
        &out, 
        self, 
        n_fft, 
        hop_length, nil, // Passing nil to 'null' ptr, using explicit int
        win_length, nil, 
        window, 
        norm_int, 
        one_int, 
        cplx_int, 
        align_int,
    )
    return track(out)
}

stft_center :: proc(
    self: Tensor, 
    n_fft: i64, 
    hop_length: i64, 
    win_length: i64, 
    window: Tensor, 
    center: bool = true,
    pad_mode: string = "reflect",
    normalized: bool = false, 
    onesided: bool = true, 
    return_complex: bool = false,
    align_to_window: bool = false
) -> Tensor {
    out: Tensor
    
    center_int := i32(1) if center else i32(0)
    norm_int := i32(1) if normalized else i32(0)
    one_int  := i32(1) if onesided else i32(0)
    cplx_int := i32(1) if return_complex else i32(0)
    align_int := i32(1) if align_to_window else i32(0)

    // Convert Odin string to C string using temp allocator
    pad_cstr := strings.clone_to_cstring(pad_mode, context.temp_allocator)

    t.atg_stft_center(
        &out, 
        self, 
        n_fft, 
        hop_length, nil, 
        win_length, nil, 
        window, 
        center_int,
        pad_cstr,
        i32(len(pad_mode)),
        norm_int, 
        one_int, 
        cplx_int, 
        align_int,
    )
    return track(out)
}

// SUBTRACTION

sub :: proc{sub_tensor, sub_scalar}
sub_ :: proc{sub_tensor_, sub_scalar_}

@private
sub_tensor :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_sub(&out, self, other)
    return track(out)
}

@private
sub_scalar :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_sub_scalar(&out, self, other)
    return track(out)
}

@private
sub_tensor_ :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_sub_(&out, self, other)
    return self
}

@private
sub_scalar_ :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_sub_scalar_(&out, self, other)
    return self
}

subtract :: proc{subtract_tensor, subtract_scalar}
subtract_ :: proc{subtract_tensor_, subtract_scalar_}

@private
subtract_tensor :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_subtract(&out, self, other)
    return track(out)
}

@private
subtract_scalar :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_subtract_scalar(&out, self, other)
    return track(out)
}

@private
subtract_tensor_ :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_subtract_(&out, self, other)
    return self
}

@private
subtract_scalar_ :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_subtract_scalar_(&out, self, other)
    return self
}

// SUMMATION

// Sum of all elements
sum :: proc(self: Tensor, dtype: ScalarType = ScalarType.Float) -> Tensor {
    out: Tensor
    t.atg_sum(&out, self, i32(dtype))
    return track(out)
}

// Sum over specific dimensions
sum_dim :: proc(
    self: Tensor, 
    dims: []i64, 
    keepdim: bool = false, 
    dtype: ScalarType = ScalarType.Float
) -> Tensor {
    out: Tensor
    keep_int := i32(1) if keepdim else i32(0)
    
    t.atg_sum_dim_intlist(
        &out, 
        self, 
        raw_data(dims), 
        i32(len(dims)), 
        keep_int, 
        i32(dtype)
    )
    return track(out)
}

// Sum to a specific shape
sum_to_size :: proc(self: Tensor, size: []i64) -> Tensor {
    out: Tensor
    t.atg_sum_to_size(&out, self, raw_data(size), i32(len(size)))
    return track(out)
}

// SINGULAR VALUE DECOMPOSITION (SVD)

svd :: proc(self: Tensor, some: bool = true, compute_uv: bool = true) -> (u, s, v: Tensor) {
    u = new_tensor()
    s = new_tensor()
    v = new_tensor()
    
    dummy: Tensor 
    
    t.atg_svd_u(
        &dummy, 
        u, 
        s, 
        v, 
        self, 
        i32(1) if some else i32(0), 
        i32(1) if compute_uv else i32(0)
    )
    return u, s, v
}

// DIMENSION MANIPULATION

swapaxes :: proc(self: Tensor, axis0: i64, axis1: i64) -> Tensor {
    out: Tensor
    t.atg_swapaxes(&out, self, axis0, axis1)
    return track(out)
}

swapaxes_ :: proc(self: Tensor, axis0: i64, axis1: i64) -> Tensor {
    out: Tensor
    t.atg_swapaxes_(&out, self, axis0, axis1)
    return self
}

swapdims :: proc(self: Tensor, dim0: i64, dim1: i64) -> Tensor {
    out: Tensor
    t.atg_swapdims(&out, self, dim0, dim1)
    return track(out)
}

swapdims_ :: proc(self: Tensor, dim0: i64, dim1: i64) -> Tensor {
    out: Tensor
    t.atg_swapdims_(&out, self, dim0, dim1)
    return self
}

// INDEXING / TAKE

take :: proc(self: Tensor, index: Tensor) -> Tensor {
    out: Tensor
    t.atg_take(&out, self, index)
    return track(out)
}

take_along_dim :: proc(self: Tensor, indices: Tensor, dim: i64) -> Tensor {
    out: Tensor
    // dim_null is explicitly nil here as we are passing a concrete dim
    t.atg_take_along_dim(&out, self, indices, dim, nil)
    return track(out)
}

// TRIGONOMETRY

tan :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_tan(&out, self)
    return track(out)
}

tan_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_tan_(&out, self)
    return self
}

tanh :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_tanh(&out, self)
    return track(out)
}

tanh_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_tanh_(&out, self)
    return self
}

// BACKWARD OPS (Gradients)

tanh_backward :: proc(grad_output: Tensor, output: Tensor) -> Tensor {
    out: Tensor
    t.atg_tanh_backward(&out, grad_output, output)
    return track(out)
}

tanh_backward_grad_input :: proc(grad_input: Tensor, grad_output: Tensor, output: Tensor) -> Tensor {
    out: Tensor
    t.atg_tanh_backward_grad_input(&out, grad_input, grad_output, output)
    return track(out)
}

tensordot :: proc(self, other: Tensor, dims_self: []i64, dims_other: []i64) -> Tensor {
    out: Tensor
    t.atg_tensordot(
        &out, 
        self, 
        other, 
        raw_data(dims_self), 
        i32(len(dims_self)), 
        raw_data(dims_other), 
        i32(len(dims_other)),
    )
    return track(out)
}

// THRESHOLD

threshold :: proc(self: Tensor, threshold_val: Scalar, value: Scalar) -> Tensor {
    out: Tensor
    t.atg_threshold(&out, self, threshold_val, value)
    return track(out)
}

threshold_ :: proc(self: Tensor, threshold_val: Scalar, value: Scalar) -> Tensor {
    out: Tensor
    t.atg_threshold_(&out, self, threshold_val, value)
    return self
}

threshold_backward :: proc(grad_output: Tensor, self: Tensor, threshold_val: Scalar) -> Tensor {
    out: Tensor
    t.atg_threshold_backward(&out, grad_output, self, threshold_val)
    return track(out)
}

// TILE

tile :: proc(self: Tensor, dims: []i64) -> Tensor {
    out: Tensor
    t.atg_tile(&out, self, raw_data(dims), i32(len(dims)))
    return track(out)
}

// TO Conversion

// Moves tensor to a specific device index (eg 0 for CUDA:0, -1 for CPU)
to_device :: proc(self: Tensor, device: DeviceType = .CPU, dtype: ScalarType, non_blocking: bool = false, copy: bool = false) -> Tensor {
    out: Tensor
    nb := i32(1) if non_blocking else i32(0)
    cp := i32(1) if copy else i32(0)
    
    t.atg_to_device(&out, self, i32(device), i32(dtype), nb, cp)
    return track(out)
}

// Simple alias for standard .to(device_index)
to_index :: proc(self: Tensor, device: DeviceType = .CPU) -> Tensor {
    out: Tensor
    t.atg_to(&out, self, i32(device))
    return track(out)
}

to_dtype :: proc(self: Tensor, dtype: ScalarType, non_blocking: bool = false, copy: bool = false) -> Tensor {
    out: Tensor
    nb := i32(1) if non_blocking else i32(0)
    cp := i32(1) if copy else i32(0)

    t.atg_to_dtype(&out, self, i32(dtype), nb, cp)
    return track(out)
}

to_other :: proc(self, other: Tensor, non_blocking: bool = false, copy: bool = false) -> Tensor {
    out: Tensor
    nb := i32(1) if non_blocking else i32(0)
    cp := i32(1) if copy else i32(0)

    t.atg_to_other(&out, self, other, nb, cp)
    return track(out)
}

// Dense/Sparse Conversions

to_dense :: proc(self: Tensor, dtype: ScalarType, masked_grad: bool = false) -> Tensor {
    out: Tensor
    masked_grad_int := i32(1) if masked_grad else i32(0)
    
    t.atg__to_dense(&out, self, i32(dtype), masked_grad_int)
    return track(out)
}

to_mkldnn :: proc(self: Tensor, dtype: ScalarType) -> Tensor {
    out: Tensor
    t.atg_to_mkldnn(&out, self, i32(dtype))
    return track(out)
}

to_padded_tensor :: proc(self: Tensor, padding: f64, output_size: []i64) -> Tensor {
    out: Tensor
    t.atg_to_padded_tensor(&out, self, padding, raw_data(output_size), i32(len(output_size)))
    return track(out)
}

// SPARSE CONVERSIONS

to_sparse :: proc(self: Tensor, sparse_dim: i64) -> Tensor {
    out: Tensor
    t.atg_to_sparse_sparse_dim(&out, self, sparse_dim)
    return track(out)
}

// Converts a dense tensor to Sparse CSR
to_sparse_csr :: proc(self: Tensor, dense_dim: Maybe(i64) = nil) -> Tensor {
    out: Tensor
    
    dd_val: i64 = 0
    dd_ptr: rawptr = nil // default (null) behavior
    
    // If user provided a specific dimension:
    if val, ok := dense_dim.?; ok {
        dd_val = val
        dd_ptr = nil 
    } else {
        dd_ptr = rawptr(uintptr(1)) // Signal Null
    }

    t.atg_to_sparse_csr(&out, self, dd_val, dd_ptr)
    return track(out)
}

to_sparse_csc :: proc(self: Tensor, dense_dim: Maybe(i64) = nil) -> Tensor {
    out: Tensor
    dd_val: i64 = 0
    dd_ptr: rawptr = nil
    
    if val, ok := dense_dim.?; ok {
        dd_val = val
    } else {
        dd_ptr = rawptr(uintptr(1))
    }

    t.atg_to_sparse_csc(&out, self, dd_val, dd_ptr)
    return track(out)
}

to_sparse_bsr :: proc(self: Tensor, blocksize: []i64, dense_dim: Maybe(i64) = nil) -> Tensor {
    out: Tensor
    dd_val: i64 = 0
    dd_ptr: rawptr = nil 
    if val, ok := dense_dim.?; ok {
        dd_val = val
    } else {
        dd_ptr = rawptr(uintptr(1))
    }

    t.atg_to_sparse_bsr(&out, self, raw_data(blocksize), i32(len(blocksize)), dd_val, dd_ptr)
    return track(out)
}

to_sparse_bsc :: proc(self: Tensor, blocksize: []i64, dense_dim: Maybe(i64) = nil) -> Tensor {
    out: Tensor
    dd_val: i64 = 0
    dd_ptr: rawptr = nil 
    if val, ok := dense_dim.?; ok {
        dd_val = val
    } else {
        dd_ptr = rawptr(uintptr(1))
    }

    t.atg_to_sparse_bsc(&out, self, raw_data(blocksize), i32(len(blocksize)), dd_val, dd_ptr)
    return track(out)
}

// TOPK

topk :: proc(self: Tensor, k: i64, dim: i64 = -1, largest: bool = true, sorted: bool = true) -> (values, indices: Tensor) {
    values = new_tensor()
    indices = new_tensor()
    
    l_int := i32(1) if largest else i32(0)
    s_int := i32(1) if sorted else i32(0)
    
    dummy: Tensor
    t.atg_topk_values(&dummy, values, indices, self, k, dim, l_int, s_int)
    
    return values, indices
}

// TOTYPE & TRACE

totype :: proc(self: Tensor, scalar_type: ScalarType) -> Tensor {
    out: Tensor
    t.atg_totype(&out, self, i32(scalar_type))
    return track(out)
}

trace :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_trace(&out, self)
    return track(out)
}

// TRACE BACKWARD

trace_backward :: proc(grad: Tensor, sizes: []i64) -> Tensor {
    out: Tensor
    t.atg_trace_backward(
        &out, 
        grad, 
        raw_data(sizes), 
        i32(len(sizes)),
    )
    return track(out)
}

// TRANSPOSE

transpose :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_t(&out, self)
    return track(out)
}

// Swaps specific dimensions
transpose_dims :: proc(self: Tensor, dim0, dim1: i64) -> Tensor {
    out: Tensor
    t.atg_transpose(&out, self, dim0, dim1)
    return track(out)
}

transpose_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_t_(&out, self)
    return self
}

transpose_dims_ :: proc(self: Tensor, dim0, dim1: i64) -> Tensor {
    out: Tensor
    t.atg_transpose_(&out, self, dim0, dim1)
    return self
}

transpose_copy :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_t_copy(&out, self)
    return track(out)
}

transpose_dims_copy :: proc(self: Tensor, dim0, dim1: i64) -> Tensor {
    out: Tensor
    t.atg_transpose_copy(&out, self, dim0, dim1)
    return track(out)
}

// TRAPEZOID

trapezoid :: proc{trapezoid_y, trapezoid_xy}

trapezoid_y :: proc(y: Tensor, dim: i64) -> Tensor {
    out: Tensor
    t.atg_trapezoid(&out, y, dim)
    return track(out)
}

trapezoid_xy :: proc(y: Tensor, x: Tensor, dim: i64) -> Tensor {
    out: Tensor
    t.atg_trapezoid_x(&out, y, x, dim)
    return track(out)
}

// TRAPZ Integration

trapz :: proc(y: Tensor, x: Tensor, dim: i64) -> Tensor {
    out: Tensor
    t.atg_trapz(&out, y, x, dim)
    return track(out)
}

trapz_dx :: proc(y: Tensor, dx: f64, dim: i64) -> Tensor {
    out: Tensor
    t.atg_trapz_dx(&out, y, dx, dim)
    return track(out)
}

// TRIANGULAR SOLVE

// Returns the solution tensor.
// NOTE: Returns a tuple (solution, cloned_A)
triangular_solve :: proc(
    self: Tensor, 
    A: Tensor, 
    upper: bool = true, 
    transpose: bool = false, 
    unitriangular: bool = false,
) -> Tensor {
    out: Tensor
    t.atg_triangular_solve(
        &out, 
        self, 
        A, 
        i32(upper), 
        i32(transpose), 
        i32(unitriangular),
    )
    return track(out)
}

// Variant allowing explicit buffers for X (solution) and M (cloned A)
triangular_solve_buffers :: proc(
    X, M, self, A: Tensor, 
    upper: bool = true, 
    transpose: bool = false, 
    unitriangular: bool = false,
) -> Tensor {
    out: Tensor
    t.atg_triangular_solve_x(
        &out, 
        X, 
        M, 
        self, 
        A, 
        i32(upper), 
        i32(transpose), 
        i32(unitriangular),
    )
    return track(out)
}

// TRIL (Triangle Lower)

tril:: proc(self: Tensor, diagonal: i64 = 0) -> Tensor {
    out: Tensor
    t.atg_tril(&out, self, diagonal)
    return track(out)
}

tril_ :: proc(self: Tensor, diagonal: i64 = 0) -> Tensor {
    out: Tensor
    t.atg_tril_(&out, self, diagonal)
    return self
}

tril_indices :: proc(
    row: i64, 
    col: i64, 
    offset: i64 = 0, 
    kind: ScalarType = .Float,
    device: DeviceType = DEFAULT_DEVICE,
) -> Tensor {
    out: Tensor
    t.atg_tril_indices(
        &out, 
        row, 
        col, 
        offset, 
        i32(kind), 
        i32(device),
    )
    return track(out)
}

// TRIU (Triangle Upper)

triu:: proc(self: Tensor, diagonal: i64 = 0) -> Tensor {
    out: Tensor
    t.atg_triu(&out, self, diagonal)
    return track(out)
}

triu_ :: proc(self: Tensor, diagonal: i64 = 0) -> Tensor {
    out: Tensor
    t.atg_triu_(&out, self, diagonal)
    return self
}

triu_indices :: proc(
    row: i64, 
    col: i64, 
    offset: i64 = 0, 
    kind: ScalarType = .Float,
    device: DeviceType = DEFAULT_DEVICE,
) -> Tensor {
    out: Tensor
    t.atg_triu_indices(
        &out, 
        row, 
        col, 
        offset, 
        i32(kind), 
        i32(device),
    )
    return track(out)
}

// TRIPLET MARGIN LOSS

triplet_margin_loss :: proc(
    anchor, positive, negative: Tensor, 
    margin: f64 = 1.0, 
    p: f64 = 2.0, 
    eps: f64 = 1e-6, 
    swap: bool = false, 
    reduction: i64 = 1, // Mean
) -> Tensor {
    out: Tensor
    t.atg_triplet_margin_loss(
        &out, 
        anchor, 
        positive, 
        negative, 
        margin, 
        p, 
        eps, 
        i32(swap), 
        reduction,
    )
    return track(out)
}

// TRUE DIVIDE

true_divide :: proc{true_divide_tensor, true_divide_scalar}
true_divide_ :: proc{true_divide_tensor_, true_divide_scalar_}

@private
true_divide_tensor :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_true_divide(&out, self, other)
    return track(out)
}

@private
true_divide_scalar :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_true_divide_scalar(&out, self, other)
    return track(out)
}

@private
true_divide_tensor_ :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_true_divide_(&out, self, other)
    return self
}

@private
true_divide_scalar_ :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_true_divide_scalar_(&out, self, other)
    return self
}

// TRUNC

trunc:: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_trunc(&out, self)
    return track(out)
}

trunc_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_trunc_(&out, self)
    return self
}

// TYPE AS

type_as :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_type_as(&out, self, other)
    return track(out)
}

// UNFLATTEN

unflatten :: proc(self: Tensor, dim: i64, sizes: []i64) -> Tensor {
    out: Tensor
    t.atg_unflatten(
        &out, 
        self, 
        dim, 
        raw_data(sizes), 
        i32(len(sizes)),
    )
    return track(out)
}

// UNFOLD

unfold:: proc(self: Tensor, dimension: i64, size: i64, step: i64) -> Tensor {
    out: Tensor
    t.atg_unfold(&out, self, dimension, size, step)
    return track(out)
}

unfold_copy :: proc(self: Tensor, dimension: i64, size: i64, step: i64) -> Tensor {
    out: Tensor
    t.atg_unfold_copy(&out, self, dimension, size, step)
    return track(out)
}

unfold_backward :: proc(
    grad_in: Tensor, 
    input_sizes: []i64, 
    dim: i64, 
    size: i64, 
    step: i64,
) -> Tensor {
    out: Tensor
    t.atg_unfold_backward(
        &out, 
        grad_in, 
        raw_data(input_sizes), 
        i32(len(input_sizes)),
        dim,
        size,
        step,
    )
    return track(out)
}

// UNIFORM

uniform:: proc(self: Tensor, from: f64 = 0.0, to: f64 = 1.0) -> Tensor {
    out: Tensor
    t.atg_uniform(&out, self, from, to)
    return track(out)
}

uniform_ :: proc(self: Tensor, from: f64 = 0.0, to: f64 = 1.0) -> Tensor {
    out: Tensor
    t.atg_uniform_(&out, self, from, to)
    return self
}

// UNIQUE

unique_consecutive :: proc(
    self: Tensor, 
    return_inverse: bool = false, 
    return_counts: bool = false, 
    dim: Maybe(i64) = nil, // Use Maybe to handle "None" vs "0" vs "-1"
) -> Tensor {
    out: Tensor
    
    dim_val: i64 = 0
    dim_ptr: rawptr = nil

    // Unwrap the Maybe logic
    if d, ok := dim.?; ok {
        dim_val = d
        dim_ptr = &dim_val // take address of the LOCAL var
    }

    t.atg_unique_consecutive(
        &out, 
        self, 
        i32(return_inverse), 
        i32(return_counts), 
        dim_val, 
        dim_ptr, 
    )
    return track(out)
}

unique_dim :: proc(
    self: Tensor, 
    dim: i64, 
    sorted: bool = true, 
    return_inverse: bool = false, 
    return_counts: bool = false,
) -> Tensor {
    out: Tensor
    t.atg_unique_dim(
        &out, 
        self, 
        dim, 
        i32(sorted), 
        i32(return_inverse), 
        i32(return_counts),
    )
    return track(out)
}

unique_dim_consecutive :: proc(
    self: Tensor, 
    dim: i64, 
    return_inverse: bool = false, 
    return_counts: bool = false,
) -> Tensor {
    out: Tensor
    t.atg_unique_dim_consecutive(
        &out, 
        self, 
        dim, 
        i32(return_inverse), 
        i32(return_counts),
    )
    return track(out)
}

// UNSQUEEZE

unsqueeze:: proc(self: Tensor, dim: i64) -> Tensor {
    out: Tensor
    t.atg_unsqueeze(&out, self, dim)
    return track(out)
}

unsqueeze_ :: proc(self: Tensor, dim: i64) -> Tensor {
    out: Tensor
    t.atg_unsqueeze_(&out, self, dim)
    return self
}

unsqueeze_copy :: proc(self: Tensor, dim: i64) -> Tensor {
    out: Tensor
    t.atg_unsqueeze_copy(&out, self, dim)
    return track(out)
}

// --- UPSAMPLE BICUBIC 2D ---

upsample_bicubic2d :: proc(
    self: Tensor, 
    output_size: []i64, 
    align_corners: bool, 
    scales_h: Maybe(f64) = nil, 
    scales_w: Maybe(f64) = nil,
) -> Tensor {
    out: Tensor
    
    // Handle optionals
    sh_v, sh_null := _opt(scales_h)
    sw_v, sw_null := _opt(scales_w)
    
    t.atg_upsample_bicubic2d(
        &out, 
        self, 
        raw_data(output_size), 
        i32(len(output_size)), 
        i32(1) if align_corners else i32(0),
        sh_v, sh_null,
        sw_v, sw_null,
    )
    return track(out)
}

upsample_bicubic2d_backward :: proc(
    grad_output: Tensor, 
    output_size: []i64, 
    input_size: []i64, 
    align_corners: bool, 
    scales_h: Maybe(f64) = nil, 
    scales_w: Maybe(f64) = nil,
) -> Tensor {
    out: Tensor
    
    sh_v, sh_null := _opt(scales_h)
    sw_v, sw_null := _opt(scales_w)

    t.atg_upsample_bicubic2d_backward(
        &out, 
        grad_output, 
        raw_data(output_size), i32(len(output_size)),
        raw_data(input_size),  i32(len(input_size)),
        i32(1) if align_corners else i32(0),
        sh_v, sh_null,
        sw_v, sw_null,
    )
    return track(out)
}

upsample_bicubic2d_backward_grad_input :: proc(
    grad_input: Tensor, // This is an argument, likely accumulating or referenced
    grad_output: Tensor, 
    output_size: []i64, 
    input_size: []i64, 
    align_corners: bool, 
    scales_h: Maybe(f64) = nil, 
    scales_w: Maybe(f64) = nil,
) -> Tensor {
    out: Tensor
    
    sh_v, sh_null := _opt(scales_h)
    sw_v, sw_null := _opt(scales_w)

    t.atg_upsample_bicubic2d_backward_grad_input(
        &out,
        grad_input,
        grad_output,
        raw_data(output_size), i32(len(output_size)),
        raw_data(input_size),  i32(len(input_size)),
        i32(1) if align_corners else i32(0),
        sh_v, sh_null,
        sw_v, sw_null,
    )
    return track(out)
}

upsample_bicubic2d_vec :: proc(
    input: Tensor, 
    output_size: []i64, 
    align_corners: bool, 
    scale_factors: []f64,
) -> Tensor {
    out: Tensor
    t.atg_upsample_bicubic2d_vec(
        &out,
        input,
        raw_data(output_size), i32(len(output_size)),
        i32(1) if align_corners else i32(0),
        raw_data(scale_factors), i32(len(scale_factors)),
    )
    return track(out)
}

// --- UPSAMPLE BILINEAR 2D ---

upsample_bilinear2d :: proc(
    self: Tensor, 
    output_size: []i64, 
    align_corners: bool, 
    scales_h: Maybe(f64) = nil, 
    scales_w: Maybe(f64) = nil,
) -> Tensor {
    out: Tensor
    
    sh_v, sh_null := _opt(scales_h)
    sw_v, sw_null := _opt(scales_w)
    
    t.atg_upsample_bilinear2d(
        &out, 
        self, 
        raw_data(output_size), i32(len(output_size)), 
        i32(1) if align_corners else i32(0),
        sh_v, sh_null,
        sw_v, sw_null,
    )
    return track(out)
}

upsample_bilinear2d_backward :: proc(
    grad_output: Tensor, 
    output_size: []i64, 
    input_size: []i64, 
    align_corners: bool, 
    scales_h: Maybe(f64) = nil, 
    scales_w: Maybe(f64) = nil,
) -> Tensor {
    out: Tensor
    
    sh_v, sh_null := _opt(scales_h)
    sw_v, sw_null := _opt(scales_w)

    t.atg_upsample_bilinear2d_backward(
        &out, 
        grad_output, 
        raw_data(output_size), i32(len(output_size)),
        raw_data(input_size),  i32(len(input_size)),
        i32(1) if align_corners else i32(0),
        sh_v, sh_null,
        sw_v, sw_null,
    )
    return track(out)
}

upsample_bilinear2d_backward_grad_input :: proc(
    grad_input: Tensor,
    grad_output: Tensor, 
    output_size: []i64, 
    input_size: []i64, 
    align_corners: bool, 
    scales_h: Maybe(f64) = nil, 
    scales_w: Maybe(f64) = nil,
) -> Tensor {
    out: Tensor
    
    sh_v, sh_null := _opt(scales_h)
    sw_v, sw_null := _opt(scales_w)

    t.atg_upsample_bilinear2d_backward_grad_input(
        &out,
        grad_input,
        grad_output,
        raw_data(output_size), i32(len(output_size)),
        raw_data(input_size),  i32(len(input_size)),
        i32(1) if align_corners else i32(0),
        sh_v, sh_null,
        sw_v, sw_null,
    )
    return track(out)
}

upsample_bilinear2d_vec :: proc(
    input: Tensor, 
    output_size: []i64, 
    align_corners: bool, 
    scale_factors: []f64,
) -> Tensor {
    out: Tensor
    t.atg_upsample_bilinear2d_vec(
        &out,
        input,
        raw_data(output_size), i32(len(output_size)),
        i32(1) if align_corners else i32(0),
        raw_data(scale_factors), i32(len(scale_factors)),
    )
    return track(out)
}

// --- UPSAMPLE LINEAR 1D ---

upsample_linear1d :: proc(
    self: Tensor, 
    output_size: []i64, 
    align_corners: bool, 
    scales: Maybe(f64) = nil, 
) -> Tensor {
    out: Tensor
    
    s_v, s_null := _opt(scales)
    
    t.atg_upsample_linear1d(
        &out, 
        self, 
        raw_data(output_size), i32(len(output_size)), 
        i32(1) if align_corners else i32(0),
        s_v, s_null,
    )
    return track(out)
}

upsample_linear1d_backward :: proc(
    grad_output: Tensor, 
    output_size: []i64, 
    input_size: []i64, 
    align_corners: bool, 
    scales: Maybe(f64) = nil,
) -> Tensor {
    out: Tensor
    
    s_v, s_null := _opt(scales)

    t.atg_upsample_linear1d_backward(
        &out, 
        grad_output, 
        raw_data(output_size), i32(len(output_size)),
        raw_data(input_size),  i32(len(input_size)),
        i32(1) if align_corners else i32(0),
        s_v, s_null,
    )
    return track(out)
}

upsample_linear1d_backward_grad_input :: proc(
    grad_input: Tensor,
    grad_output: Tensor, 
    output_size: []i64, 
    input_size: []i64, 
    align_corners: bool, 
    scales: Maybe(f64) = nil,
) -> Tensor {
    out: Tensor
    
    s_v, s_null := _opt(scales)

    t.atg_upsample_linear1d_backward_grad_input(
        &out,
        grad_input,
        grad_output,
        raw_data(output_size), i32(len(output_size)),
        raw_data(input_size),  i32(len(input_size)),
        i32(1) if align_corners else i32(0),
        s_v, s_null,
    )
    return track(out)
}

upsample_linear1d_vec :: proc(
    input: Tensor, 
    output_size: []i64, 
    align_corners: bool, 
    scale_factors: []f64,
) -> Tensor {
    out: Tensor
    t.atg_upsample_linear1d_vec(
        &out,
        input,
        raw_data(output_size), i32(len(output_size)),
        i32(1) if align_corners else i32(0),
        raw_data(scale_factors), i32(len(scale_factors)),
    )
    return track(out)
}

// UPSAMPLE NEAREST 1D

upsample_nearest1d :: proc(
    self: Tensor, 
    output_size: []i64, 
    scales: Maybe(f64) = nil
) -> Tensor {
    out: Tensor
    s_v, s_null := _opt(scales)

    t.atg_upsample_nearest1d(
        &out, 
        self, 
        raw_data(output_size), i32(len(output_size)), 
        s_v, s_null,
    )
    return track(out)
}

upsample_nearest1d_backward :: proc(
    grad_output: Tensor, 
    output_size: []i64, 
    input_size: []i64, 
    scales: Maybe(f64) = nil
) -> Tensor {
    out: Tensor
    s_v, s_null := _opt(scales)

    t.atg_upsample_nearest1d_backward(
        &out, 
        grad_output, 
        raw_data(output_size), i32(len(output_size)), 
        raw_data(input_size),  i32(len(input_size)), 
        s_v, s_null,
    )
    return track(out)
}

upsample_nearest1d_backward_grad_input :: proc(
    grad_input: Tensor, 
    grad_output: Tensor, 
    output_size: []i64, 
    input_size: []i64, 
    scales: Maybe(f64) = nil
) -> Tensor {
    out: Tensor
    s_v, s_null := _opt(scales)

    t.atg_upsample_nearest1d_backward_grad_input(
        &out, 
        grad_input, 
        grad_output, 
        raw_data(output_size), i32(len(output_size)), 
        raw_data(input_size),  i32(len(input_size)), 
        s_v, s_null,
    )
    return track(out)
}

// UPSAMPLE NEAREST 2D

upsample_nearest2d :: proc(
    self: Tensor, 
    output_size: []i64, 
    scales_h: Maybe(f64) = nil,
    scales_w: Maybe(f64) = nil,
) -> Tensor {
    out: Tensor
    sh_v, sh_null := _opt(scales_h)
    sw_v, sw_null := _opt(scales_w)

    t.atg_upsample_nearest2d(
        &out, 
        self, 
        raw_data(output_size), i32(len(output_size)), 
        sh_v, sh_null,
        sw_v, sw_null,
    )
    return track(out)
}

upsample_nearest2d_backward :: proc(
    grad_output: Tensor, 
    output_size: []i64, 
    input_size: []i64, 
    scales_h: Maybe(f64) = nil,
    scales_w: Maybe(f64) = nil,
) -> Tensor {
    out: Tensor
    sh_v, sh_null := _opt(scales_h)
    sw_v, sw_null := _opt(scales_w)

    t.atg_upsample_nearest2d_backward(
        &out, 
        grad_output, 
        raw_data(output_size), i32(len(output_size)), 
        raw_data(input_size),  i32(len(input_size)), 
        sh_v, sh_null,
        sw_v, sw_null,
    )
    return track(out)
}

upsample_nearest2d_backward_grad_input :: proc(
    grad_input: Tensor, 
    grad_output: Tensor, 
    output_size: []i64, 
    input_size: []i64, 
    scales_h: Maybe(f64) = nil,
    scales_w: Maybe(f64) = nil,
) -> Tensor {
    out: Tensor
    sh_v, sh_null := _opt(scales_h)
    sw_v, sw_null := _opt(scales_w)

    t.atg_upsample_nearest2d_backward_grad_input(
        &out, 
        grad_input, 
        grad_output, 
        raw_data(output_size), i32(len(output_size)), 
        raw_data(input_size),  i32(len(input_size)), 
        sh_v, sh_null,
        sw_v, sw_null,
    )
    return track(out)
}

// UPSAMPLE NEAREST 3D

upsample_nearest3d :: proc(
    self: Tensor, 
    output_size: []i64, 
    scales_d: Maybe(f64) = nil,
    scales_h: Maybe(f64) = nil,
    scales_w: Maybe(f64) = nil,
) -> Tensor {
    out: Tensor
    sd_v, sd_null := _opt(scales_d)
    sh_v, sh_null := _opt(scales_h)
    sw_v, sw_null := _opt(scales_w)

    t.atg_upsample_nearest3d(
        &out, 
        self, 
        raw_data(output_size), i32(len(output_size)), 
        sd_v, sd_null,
        sh_v, sh_null,
        sw_v, sw_null,
    )
    return track(out)
}

upsample_nearest3d_backward :: proc(
    grad_output: Tensor, 
    output_size: []i64, 
    input_size: []i64, 
    scales_d: Maybe(f64) = nil,
    scales_h: Maybe(f64) = nil,
    scales_w: Maybe(f64) = nil,
) -> Tensor {
    out: Tensor
    sd_v, sd_null := _opt(scales_d)
    sh_v, sh_null := _opt(scales_h)
    sw_v, sw_null := _opt(scales_w)

    t.atg_upsample_nearest3d_backward(
        &out, 
        grad_output, 
        raw_data(output_size), i32(len(output_size)), 
        raw_data(input_size),  i32(len(input_size)), 
        sd_v, sd_null,
        sh_v, sh_null,
        sw_v, sw_null,
    )
    return track(out)
}

upsample_nearest3d_backward_grad_input :: proc(
    grad_input: Tensor, 
    grad_output: Tensor, 
    output_size: []i64, 
    input_size: []i64, 
    scales_d: Maybe(f64) = nil,
    scales_h: Maybe(f64) = nil,
    scales_w: Maybe(f64) = nil,
) -> Tensor {
    out: Tensor
    sd_v, sd_null := _opt(scales_d)
    sh_v, sh_null := _opt(scales_h)
    sw_v, sw_null := _opt(scales_w)

    t.atg_upsample_nearest3d_backward_grad_input(
        &out, 
        grad_input, 
        grad_output, 
        raw_data(output_size), i32(len(output_size)), 
        raw_data(input_size),  i32(len(input_size)), 
        sd_v, sd_null,
        sh_v, sh_null,
        sw_v, sw_null,
    )
    return track(out)
}

// UPSAMPLE TRILINEAR 3D

upsample_trilinear3d :: proc(
    self: Tensor, 
    output_size: []i64, 
    align_corners: bool,
    scales_d: Maybe(f64) = nil,
    scales_h: Maybe(f64) = nil,
    scales_w: Maybe(f64) = nil,
) -> Tensor {
    out: Tensor
    sd_v, sd_null := _opt(scales_d)
    sh_v, sh_null := _opt(scales_h)
    sw_v, sw_null := _opt(scales_w)
    
    t.atg_upsample_trilinear3d(
        &out, 
        self, 
        raw_data(output_size), i32(len(output_size)), 
        i32(1) if align_corners else i32(0),
        sd_v, sd_null,
        sh_v, sh_null,
        sw_v, sw_null,
    )
    return track(out)
}

upsample_trilinear3d_backward :: proc(
    grad_output: Tensor, 
    output_size: []i64, 
    input_size: []i64, 
    align_corners: bool,
    scales_d: Maybe(f64) = nil,
    scales_h: Maybe(f64) = nil,
    scales_w: Maybe(f64) = nil,
) -> Tensor {
    out: Tensor
    sd_v, sd_null := _opt(scales_d)
    sh_v, sh_null := _opt(scales_h)
    sw_v, sw_null := _opt(scales_w)

    t.atg_upsample_trilinear3d_backward(
        &out, 
        grad_output, 
        raw_data(output_size), i32(len(output_size)), 
        raw_data(input_size),  i32(len(input_size)), 
        i32(1) if align_corners else i32(0),
        sd_v, sd_null,
        sh_v, sh_null,
        sw_v, sw_null,
    )
    return track(out)
}

upsample_trilinear3d_backward_grad_input :: proc(
    grad_input: Tensor, 
    grad_output: Tensor, 
    output_size: []i64, 
    input_size: []i64, 
    align_corners: bool,
    scales_d: Maybe(f64) = nil,
    scales_h: Maybe(f64) = nil,
    scales_w: Maybe(f64) = nil,
) -> Tensor {
    out: Tensor
    sd_v, sd_null := _opt(scales_d)
    sh_v, sh_null := _opt(scales_h)
    sw_v, sw_null := _opt(scales_w)

    t.atg_upsample_trilinear3d_backward_grad_input(
        &out, 
        grad_input, 
        grad_output, 
        raw_data(output_size), i32(len(output_size)), 
        raw_data(input_size),  i32(len(input_size)), 
        i32(1) if align_corners else i32(0),
        sd_v, sd_null,
        sh_v, sh_null,
        sw_v, sw_null,
    )
    return track(out)
}

// VEC VARIANTS (Array scales)
// These do not use _opt_scale as they accept arrays/slices

upsample_nearest1d_vec :: proc(
    input: Tensor, 
    output_size: []i64, 
    scale_factors: []f64
) -> Tensor {
    out: Tensor
    t.atg_upsample_nearest1d_vec(
        &out, 
        input, 
        raw_data(output_size), i32(len(output_size)), 
        raw_data(scale_factors), i32(len(scale_factors)),
    )
    return track(out)
}

upsample_nearest2d_vec :: proc(
    input: Tensor, 
    output_size: []i64, 
    scale_factors: []f64
) -> Tensor {
    out: Tensor
    t.atg_upsample_nearest2d_vec(
        &out, 
        input, 
        raw_data(output_size), i32(len(output_size)), 
        raw_data(scale_factors), i32(len(scale_factors)),
    )
    return track(out)
}

upsample_nearest3d_vec :: proc(
    input: Tensor, 
    output_size: []i64, 
    scale_factors: []f64
) -> Tensor {
    out: Tensor
    t.atg_upsample_nearest3d_vec(
        &out, 
        input, 
        raw_data(output_size), i32(len(output_size)), 
        raw_data(scale_factors), i32(len(scale_factors)),
    )
    return track(out)
}

upsample_trilinear3d_vec :: proc(
    input: Tensor, 
    output_size: []i64, 
    align_corners: bool,
    scale_factors: []f64
) -> Tensor {
    out: Tensor
    t.atg_upsample_trilinear3d_vec(
        &out, 
        input, 
        raw_data(output_size), i32(len(output_size)), 
        i32(1) if align_corners else i32(0),
        raw_data(scale_factors), i32(len(scale_factors)),
    )
    return track(out)
}

value_selecting_reduction_backward :: proc(
    grad: Tensor, 
    dim: i64, 
    indices: Tensor, 
    sizes: []i64, 
    keepdim: bool = false
) -> Tensor {
    out: Tensor
    keep_int := i32(1) if keepdim else i32(0)
    t.atg_value_selecting_reduction_backward(
        &out, 
        grad, 
        dim, 
        indices, 
        raw_data(sizes), 
        i32(len(sizes)), 
        keep_int,
    )
    return track(out)
}

values :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_values(&out, self)
    return track(out)
}

values_copy :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_values_copy(&out, self)
    return track(out)
}

// Returns the Vandermonde matrix
vander :: proc(x: Tensor, n: int = 0, increasing: bool = false) -> Tensor {
    out: Tensor
    inc_int := i32(1) if increasing else i32(0)
    
    // Logic: if N is 0 (or unspecified), we likely pass nullptr to n_null/n_ptr
    // to let Torch decide the default (usually defaults to size of x).
    n_ptr: rawptr = nil 
    // If user provided a specific N > 0, we might need to handle the pointer logic specific to your C-gen.
    // Assuming 'n_v' is passed as the i64 value, and 'n_null' is the sentinel.
    
    t.atg_vander(&out, x, i64(n), n_ptr, inc_int)
    return track(out)
}

vdot :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_vdot(&out, self, other)
    return track(out)
}

var :: proc(self: Tensor, unbiased: bool = true) -> Tensor {
    out: Tensor
    unb_int := i32(1) if unbiased else i32(0)
    t.atg_var(&out, self, unb_int)
    return track(out)
}

var_dim :: proc(self: Tensor, dim: []i64, unbiased: bool = true, keepdim: bool = false) -> Tensor {
    out: Tensor
    unb_int := i32(1) if unbiased else i32(0)
    keep_int := i32(1) if keepdim else i32(0)
    t.atg_var_dim(&out, self, raw_data(dim), i32(len(dim)), unb_int, keep_int)
    return track(out)
}

var_correction :: proc(self: Tensor, correction: Scalar, keepdim: bool = false) -> Tensor {
    out: Tensor
    keep_int := i32(1) if keepdim else i32(0)
    // NOTE: atg_var_correction signature provided takes dim_data. 
    // If this is the "all" version, we pass empty dims or null? 
    // Usually there is a specific 'atg_var_correction' without dims, but the list only had one.
    // We will assume empty slice means "all".
    empty_dims: []i64
    t.atg_var_correction(&out, self, raw_data(empty_dims), 0, correction, keep_int)
    return track(out)
}

var_correction_dim :: proc(self: Tensor, dim: []i64, correction: Scalar, keepdim: bool = false) -> Tensor {
    out: Tensor
    keep_int := i32(1) if keepdim else i32(0)
    t.atg_var_correction(&out, self, raw_data(dim), i32(len(dim)), correction, keep_int)
    return track(out)
}

// Mean/Var Combined
var_mean :: proc(self: Tensor, unbiased: bool = true) -> Tensor {
    out: Tensor
    unb_int := i32(1) if unbiased else i32(0)
    t.atg_var_mean(&out, self, unb_int)
    return track(out)
}

var_mean_dim :: proc(self: Tensor, dim: []i64, unbiased: bool = true, keepdim: bool = false) -> Tensor {
    out: Tensor
    unb_int := i32(1) if unbiased else i32(0)
    keep_int := i32(1) if keepdim else i32(0)
    t.atg_var_mean_dim(&out, self, raw_data(dim), i32(len(dim)), unb_int, keep_int)
    return track(out)
}

var_mean_correction :: proc(self: Tensor, dim: []i64, correction: Scalar, keepdim: bool = false) -> Tensor {
    out: Tensor
    keep_int := i32(1) if keepdim else i32(0)
    t.atg_var_mean_correction(&out, self, raw_data(dim), i32(len(dim)), correction, keep_int)
    return track(out)
}

view :: proc{view_size, view_dtype}
view_copy :: proc{view_copy_size, view_copy_dtype}

@private
view_size :: proc(self: Tensor, size: []i64) -> Tensor {
    out: Tensor
    t.atg_view(&out, self, raw_data(size), i32(len(size)))
    return track(out)
}

@private
view_dtype :: proc(self: Tensor, dtype: ScalarType) -> Tensor {
    out: Tensor
    t.atg_view_dtype(&out, self, i32(dtype))
    return track(out)
}

view_as :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_view_as(&out, self, other)
    return track(out)
}

view_as_complex :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_view_as_complex(&out, self)
    return track(out)
}

view_as_complex_copy :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_view_as_complex_copy(&out, self)
    return track(out)
}

view_as_real :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_view_as_real(&out, self)
    return track(out)
}

view_as_real_copy :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_view_as_real_copy(&out, self)
    return track(out)
}

@private
view_copy_size :: proc(self: Tensor, size: []i64) -> Tensor {
    out: Tensor
    t.atg_view_copy(&out, self, raw_data(size), i32(len(size)))
    return track(out)
}

@private
view_copy_dtype :: proc(self: Tensor, dtype: ScalarType) -> Tensor {
    out: Tensor
    t.atg_view_copy_dtype(&out, self, i32(dtype))
    return track(out)
}

vstack :: proc(tensors: []Tensor) -> Tensor {
    out: Tensor
    t.atg_vstack(&out, raw_data(tensors), i32(len(tensors)))
    return track(out)
}

where_self :: proc{where_tensor, where_scalar_scalar, where_scalar_tensor, where_tensor_scalar}

@private
where_tensor :: proc(condition: Tensor, self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_where_self(&out, condition, self, other)
    return track(out)
}

@private
where_scalar_scalar :: proc(condition: Tensor, self: Scalar, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_where_scalar(&out, condition, self, other)
    return track(out)
}

@private
where_scalar_tensor :: proc(condition: Tensor, self: Scalar, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_where_scalarself(&out, condition, self, other)
    return track(out)
}

@private
where_tensor_scalar :: proc(condition: Tensor, self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_where_scalarother(&out, condition, self, other)
    return track(out)
}

xlogy :: proc{xlogy_tensor, xlogy_scalar_other, xlogy_scalar_self}
xlogy_ :: proc{xlogy_tensor_, xlogy_scalar_other_}

@private
xlogy_tensor :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_xlogy(&out, self, other)
    return track(out)
}

@private
xlogy_tensor_ :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_xlogy_(&out, self, other)
    return self
}

@private
xlogy_scalar_other :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_xlogy_scalar_other(&out, self, other)
    return track(out)
}

@private
xlogy_scalar_other_ :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_xlogy_scalar_other_(&out, self, other)
    return self
}

@private
xlogy_scalar_self :: proc(self: Scalar, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_xlogy_scalar_self(&out, self, other)
    return track(out)
}

// TODO: XLogY explicit output tensor
xlogy_out :: proc{xlogy_out_tensor, xlogy_out_scalar_other, xlogy_out_scalar_self}

@private
xlogy_out_tensor :: proc(self, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_xlogy_outtensor(&out, self, other)
    return track(out)
}

@private
xlogy_out_scalar_other :: proc(self: Tensor, other: Scalar) -> Tensor {
    out: Tensor
    t.atg_xlogy_outscalar_other(&out, self, other)
    return track(out)
}

@private
xlogy_out_scalar_self :: proc(self: Scalar, other: Tensor) -> Tensor {
    out: Tensor
    t.atg_xlogy_outscalar_self(&out, self, other)
    return track(out)
}

zero :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_zero(&out, self)
    return track(out)
}

zero_ :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_zero_(&out, self)
    return self
}

zeros :: proc(size: []i64, dtype: ScalarType = .Float, device: DeviceType = DEFAULT_DEVICE) -> Tensor {
    out: Tensor
    t.atg_zeros(&out, raw_data(size), i32(len(size)), i32(dtype), i32(device))
    return track(out)
}

zero_grad :: proc(self: Tensor) {
    // TODO: manually zero the gradient
    g := grad(self)
    if defined(g) != 0 {
        zero(g)
    }
}

zeros_like :: proc(self: Tensor) -> Tensor {
    out: Tensor
    t.atg_zeros_like(&out, self)
    return track(out)
}