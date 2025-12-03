package main

import "core:fmt"
import "core:os"
import "core:strings"
import "core:c/libc"

BASE_URL   :: "https://raw.githubusercontent.com/LaurentMazare/tch-rs/main/torch-sys/libtch/"
TARGET_DIR :: "ffi"
OUTPUT_DIR :: "atg"

FILES_TO_DOWNLOAD :: []string{
    "torch_api.h",
    "torch_api_generated.h",
    "torch_api.cpp",
    "torch_api_generated.cpp",
    "stb_image.h",
    "stb_image_write.h",
    "stb_image_resize.h",
}

main :: proc() {
    ensure_directory(TARGET_DIR)
    ensure_directory(OUTPUT_DIR)

    fmt.println("--- 1. Checking Dependencies ---")
    
    // Check if libtorch exists
    if !os.is_dir("libtorch") {
        fmt.println("[WARNING] 'libtorch/' directory not found!")
        fmt.println("To fix this:")
        fmt.println("  1. Download the LibTorch C++ ZIP (Pre-built) from pytorch.org")
        fmt.println("  2. Unzip it here so you have a folder named 'libtorch'")
        fmt.println("  NOTE: 'git clone' often lacks the pre-built headers structure required by the compiler flags below")
        // Stop here because compilation will definitely fail without it
        return
    } else {
        fmt.println("Found 'libtorch/' directory.")
    }

    fmt.println("\nDownloading Headers & Source Files:")
    for filename in FILES_TO_DOWNLOAD {
        download_file(filename)
    }

    fmt.println("\nBuilding Shared Library...")
    build_library()
}

ensure_directory :: proc(path: string) {
    if !os.is_dir(path) {
        fmt.printf("Creating directory: %s\n", path)
        os.make_directory(path)
    }
}

download_file :: proc(filename: string) {
    url := fmt.tprintf("%s%s", BASE_URL, filename)
    output_path := fmt.tprintf("%s/%s", TARGET_DIR, filename)

    fmt.printf("Downloading %s ... \n", filename)
    
    // curl for dep free http
    command := fmt.tprintf("curl -s -S -L --fail -o \"%s\" \"%s\"", output_path, url)
    c_command := strings.clone_to_cstring(command, context.temp_allocator)
    
    result := libc.system(c_command)
    if result != 0 {
        fmt.printf("Failed to download %s (Exit Code: %d)\n", filename, result)
    }
}

build_library :: proc() {
    cmd := ""
    // Determine OS
    when ODIN_OS == .Darwin {
        fmt.println("Detected OS: macOS")
        cmd = fmt.tprintf(
            "clang++ -std=c++17 -dynamiclib " +
            "-I libtorch/include " +
            "-I libtorch/include/torch/csrc/api/include " +
            "-L libtorch/lib " +
            "-ltorch -ltorch_cpu -lc10 " +
            "-Wno-deprecated-declarations " +
            "-o %s/torch_wrapper.dylib " +
            "%s/torch_api.cpp %s/torch_api_generated.cpp %s/stubs.cpp " +
            "-Wl,-rpath,$(pwd)/libtorch/lib",
            OUTPUT_DIR, TARGET_DIR, TARGET_DIR, TARGET_DIR,
        )
    } else when ODIN_OS == .Linux {
        fmt.println("Detected OS: Linux")
        cmd = fmt.tprintf(
            "clang++ -std=c++17 -shared -fPIC " +
            "-I libtorch/include " +
            "-I libtorch/include/torch/csrc/api/include " +
            "-L libtorch/lib " +
            "-ltorch -ltorch_cpu -lc10 " +
            "-Wno-deprecated-declarations " +
            "-o %s/torch_wrapper.so " +
            "%s/torch_api.cpp %s/torch_api_generated.cpp %s/stubs.cpp " +
            "-Wl,-rpath,'$ORIGIN/libtorch/lib'",
            OUTPUT_DIR, TARGET_DIR, TARGET_DIR, TARGET_DIR,
        )
    } else {
        fmt.println("OS currently unsupported for automatic build, please build manually!")
        return
    }
    fmt.println("Running Clang...")
    fmt.println(cmd)

    c_cmd := strings.clone_to_cstring(cmd, context.temp_allocator)
    result := libc.system(c_cmd)

    if result == 0 {
        fmt.println("Build success")
    } else {
        fmt.printf("Build Failed with code: %d\n", result)
    }
}