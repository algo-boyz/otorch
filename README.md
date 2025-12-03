# oTorch

Odin bindings for the C++ api of PyTorch. The goal is to provide a thin wrapper around the C++ PyTorch api (a.k.a. libtorch). Staying as close as possible to the original.

**Downloads required:**

1.  **LibTorch (C++):** Download the **cxx11 ABI** version (Linux) or standard version (Windows) from [pytorch.org](https://pytorch.org/get-started/locally/). Unzip it to `libtorch` in the root of this project.
2.  **tch-rs C api wrapper:** Download ffi files from [tch-rs](https://github.com/LaurentMazare/tch-rs) and put them in `ffi/`.


### Compile C api wrapper 

macos:

```bash
clang++ -std=c++17 -dynamiclib \
    -I libtorch/include \
    -I libtorch/include/torch/csrc/api/include \
    -L libtorch/lib \
    -ltorch -ltorch_cpu -lc10 \
    -Wno-deprecated-declarations \
    -o atg/torch_wrapper.dylib \
    ffi/torch_api.cpp ffi/torch_api_gen.cpp ffi/stubs.cpp \
    -Wl,-rpath,$(pwd)/libtorch/lib
```

linux:

```bash
clang++ -std=c++17 -shared -fPIC \
    -I libtorch/include \
    -I libtorch/include/torch/csrc/api/include \
    -L libtorch/lib \
    -ltorch -ltorch_cpu -lc10 \
    -Wno-deprecated-declarations \
    -o atg/torch_wrapper.so \
    ffi/torch_api.cpp ffi/torch_api_gen.cpp ffi/stubs.cpp \
    -Wl,-rpath,'$ORIGIN/libtorch/lib'
```

### 3\. How to Use
Check out the demo folder:

```odin

odin run demo

Lib Torch via Odin

Creating tensor of shape [2, 3] with data: [1, 2, 3, 4, 5, 6]
Tensor created successfully!

[LibTorch Output]:
 1  2  3
 4  5  6
[ CPUFloatType{2,3} ]

[Filling tensor with 99's]
 99  99  99
 99  99  99
[ CPUFloatType{2,3} ]

Done
```

# TODO: port some more examples

https://github.com/pytorch/examples