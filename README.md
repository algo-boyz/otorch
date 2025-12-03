
<h1 align="center">█▬▬𒄆 (◡̀_◡́)d𓌏nϟ 𒅒▬▬█</h1>
<p align="center">
    <img src="otorch.png" alt="oTorch" width="960">
</p>

Odin bindings for the C++ api of PyTorch. The goal is to provide a thin wrapper around the C++ PyTorch api (a.k.a. libtorch). Staying as close as possible to the original.

**Downloads required:**

  **LibTorch (C++):** Download the **cxx11 ABI** version (Linux) or standard version (Windows) from [pytorch.org](https://pytorch.org/get-started/locally/). Unzip it to `libtorch` in the root of this project.


### Compile C api wrapper

Use `odin run etc` to download libtorch C api headers from [tch-rs](https://github.com/LaurentMazare/tch-rs) to `ffi/` and
automagically compile the shared lib to `atg/`

Alternatively, you can also run the command to compile libtorch abi manually:

macos:

```bash
clang++ -std=c++17 -dynamiclib \
    -I libtorch/include \
    -I libtorch/include/torch/csrc/api/include \
    -L libtorch/lib \
    -ltorch -ltorch_cpu -lc10 \
    -Wno-deprecated-declarations \
    -o atg/torch_wrapper.dylib \
    ffi/torch_api.h ffi/torch_api_gen.h ffi/stubs.cpp \
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
    ffi/torch_api.h ffi/torch_api_gen.h ffi/stubs.cpp \
    -Wl,-rpath,'$ORIGIN/libtorch/lib'
```

windows:
```
TODO should not be all to hard just don't have windows personally. 
Contribution welcome!
```

### 3\. How to Use
Check out the demo folder:

```odin

$ odin run demo/basic

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