# Chapter 3: Setting Up AdaptiveCpp

This chapter covers the installation of AdaptiveCpp via pixi (the recommended approach), CMake project integration, and verifying your setup to ensure everything is working correctly.

## Prerequisites

Before installing AdaptiveCpp, ensure you have the following:

- Linux x86-64 operating system
- pixi package manager (https://prefix.dev/)
- A C++17-capable compiler (GCC 8+, Clang 7+)
- Hardware with an OpenCL/CUDA/ROCm/CPU capable runtime

## Installation via Pixi (Recommended)

The code-accelerate channel provides the acpp-toolchain conda package, which includes AdaptiveCpp and all its dependencies. Create a minimal pixi.toml file:

```toml
[project]
name = "my-acpp-project"
version = "0.1.0"

channels = ["code-accelerate", "conda-forge"]

[dependencies]
acpp-toolchain = "*"
```

Run `pixi install` to fetch and install the toolchain:

```bash
pixi install
```

> [!NOTE]
> `pixi install` downloads AdaptiveCpp plus a matching LLVM and all required runtime libraries automatically - no manual LLVM setup needed.

## Verifying the Installation

After installation, verify that AdaptiveCpp is properly configured:

```bash
pixi run acpp --version
```

Or, if not using pixi tasks:

```bash
pixi shell
acpp --version
```

The output should show:
- AdaptiveCpp version string
- LLVM plugin version
- Installation root directory

To query available devices, you can use the hello_devices example we'll build in this chapter:

```bash
# After building the examples (see later sections)
./build/chapters/03-acpp-setup/examples/hello_devices
```

> [!TIP]
> If `acpp --version` succeeds but no GPU is detected at runtime, the program will fall back to the CPU via the OpenMP backend.

## Building from Source (Advanced)

For CI environments or custom LLVM deployments, you can build AdaptiveCpp from source. Requirements:

- LLVM >= 15
- Boost libraries
- CMake >= 3.20

Standard CMake workflow:

```bash
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/opt/acpp -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm
cmake --build build
cmake --install build
```

> [!NOTE]
> The pixi path handles all this automatically; source builds are for CI or custom LLVM deployments.

## CMake Project Integration

Integrating AdaptiveCpp into your CMake project requires just two lines:

```cmake
cmake_minimum_required(VERSION 3.20)
project(my-sycl-app CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(AdaptiveCpp REQUIRED)

add_executable(my_app main.cpp)
add_sycl_to_target(TARGET my_app)
```

Configure your project with:

```bash
cmake -S . -B build -DACPP_TARGETS=generic
```

> [!IMPORTANT]
> Always pass `-DACPP_TARGETS=generic` when using the SSCP generic flow. This tells AdaptiveCpp to embed backend-agnostic LLVM IR at compile time and JIT-compile to the actual hardware at runtime.

## The SSCP Generic Target: One Binary, Any Hardware

The SSCP (Single-Source Compilation Pipeline) generic target uses a two-stage compilation model:

**Stage 1 (Compile Time):** SYCL kernels are extracted and lowered to backend-independent LLVM IR, which is embedded in the host binary.

**Stage 2 (Runtime):** AdaptiveCpp's runtime JIT compiles the embedded IR to the actual device's native format:
- PTX for NVIDIA GPUs
- amdgcn for AMD GPUs
- SPIR-V for Intel GPUs
- Native code for CPU

Benefits:
- Single binary runs on any supported device
- No recompilation needed to switch hardware
- Enables runtime kernel fusion

> [!NOTE]
> The 'generic' target produces slightly larger binaries and has a small JIT overhead on first kernel launch (subsequent launches use a cache). For production, the cache is warm and overhead is negligible.

## Backend Runtimes

| Backend | Runtime Required | Notes |
|---------|------------------|-------|
| NVIDIA CUDA | CUDA Toolkit >= 10 | Verify with `nvidia-smi` |
| AMD ROCm | ROCm >= 4 | Verify with `rocminfo` |
| Intel OpenCL | Intel Compute Runtime | Verify with `clinfo` |
| CPU (OpenMP) | None (built-in) | Always available |

> [!TIP]
> With the SSCP generic target, your binary will automatically use whichever backends are available at runtime. You do not need to recompile for different hardware.

## Chapter Examples

| Example | Description | File |
|---------|-------------|------|
| hello_devices | Enumerate all platforms and devices | examples/hello_devices.cpp |
| hello_kernel | Vector addition using make_async_buffer | examples/hello_kernel.cpp |

## Building the Examples

To build the chapter examples:

```bash
pixi run configure
pixi run build
```

Run the examples:

```bash
# List available devices
./build/chapters/03-acpp-setup/examples/hello_devices

# Run vector addition with 1M elements
./build/chapters/03-acpp-setup/examples/hello_kernel 1048576
```

## Summary

- pixi provides a one-command AdaptiveCpp install
- The SSCP generic target compiles once and runs on any hardware
- CMake integration is two lines: `find_package` + `add_sycl_to_target`
- Verifying with `acpp --version` and `hello_devices` confirms setup

## Next Steps

Proceed to [Chapter 4: Memory Model](../04-memory-model/README.md) to learn about AdaptiveCpp's memory management system.