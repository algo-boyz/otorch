cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(odin_torch_wrapper LANGUAGES CXX C)

# Setup LibTorch
# We tell CMake to find the Torch package located in our /libtorch folder
find_package(Torch REQUIRED)

# Config
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CRITICAL FOR WINDOWS: 
# Forces CMake to export functions so Odin can see them in the DLL
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON) 

# Build shared wrapper lib (.dll / .so)
add_library(odin_torch SHARED
    capi/torch_api.cpp
    capi/torch_api_gen.cpp
)

# Include paths for "torch_api.h" and LibTorch headers
target_include_directories(odin_torch PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TORCH_INCLUDE_DIRS}
)

# Link wrapper against LibTorch backend
target_link_libraries(odin_torch PUBLIC "${TORCH_LIBRARIES}")
message(STATUS "Torch Libraries: ${TORCH_LIBRARIES}")