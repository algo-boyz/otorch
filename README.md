
**Downloads required:**

1.  **LibTorch (C++):** Download the **cxx11 ABI** version (Linux) or standard version (Windows) from [pytorch.org](https://pytorch.org/get-started/locally/). Unzip it to `libtorch` in the root of this project.
2.  **tch-rs C api wra[[er]]:** Download capi files from [tch-rs](https://www.google.com/search?q=https://github.com/LaurentMazare/tch-rs/tree/main/torch-sys/libtch) and put them in `capi/`.


### 3. Compile C api wrapper

```bash
clang++ -std=c++17 -dynamiclib \
    -I libtorch/include \
    -I libtorch/include/torch/csrc/api/include \
    -L libtorch/lib \
    -ltorch -ltorch_cpu -lc10 \
    -Wno-deprecated-declarations \
    -o libtorch_wrapper.dylib \
    capi/torch_api.cpp capi/torch_api_gen.cpp capi/stubs.cpp \
    -Wl,-rpath,$(pwd)/libtorch/lib
```

### 3\. How to Use
Check demo folder 