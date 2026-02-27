# Introduction to AdaptiveCpp

This guide is a comprehensive introduction to AdaptiveCpp, a community-driven SYCL implementation that enables C++ developers to write performance-portable code for heterogeneous computing systems. We'll explore how AdaptiveCpp's unique Single-Source Compilation Pipeline (SSCP) allows you to compile once and run efficiently across NVIDIA GPUs, AMD GPUs, Intel GPUs, CPUs, and other accelerators without code changes.

## What is AdaptiveCpp?

AdaptiveCpp is a community-driven, open-source SYCL implementation developed primarily at Heidelberg University under the leadership of Aksel Alpay. It has undergone several name changes reflecting its evolution: hipSYCL (2017, originally AMD-focused) → Open SYCL (February 2023, vendor-neutral rebrand) → AdaptiveCpp (September 2023, final name due to legal pressure). 

AdaptiveCpp targets a wide range of computing backends including NVIDIA CUDA, AMD ROCm/HIP, Intel Level Zero/SPIR-V, OpenCL, and CPU backends. It is used in production on some of the world's most powerful supercomputers, including Frontier and LUMI, demonstrating its capability to handle large-scale scientific computing workloads.

## The SSCP Generic Target

The Single-Source Compilation Pipeline (SSCP) is AdaptiveCpp's defining technical innovation. Unlike other SYCL implementations that require multiple compilation passes or separate device compilation steps, AdaptiveCpp parses your source code only once for both host and device code.

The SSCP compilation process works in two stages:

1. **Stage 1**: AdaptiveCpp compiles your C++ source to LLVM IR and embeds device kernel bitcode directly into the host binary.
2. **Stage 2**: At runtime, AdaptiveCpp performs just-in-time (JIT) compilation of the embedded bitcode to the appropriate backend target (NVIDIA PTX, AMD amdgcn, Intel SPIR-V, or CPU native code).

> [!IMPORTANT]
> The result is a single compiled binary that can run on ANY supported hardware without recompilation. These universal binaries are compiled with the flag `--acpp-targets=generic`.

This approach eliminates the need for maintaining multiple builds of your application for different hardware targets, simplifying deployment and distribution.

## AdaptiveCpp vs The Alternatives

| Implementation | Development Model | Binary Portability | Compilation Approach | Primary Focus |
|----------------|-------------------|-------------------|---------------------|---------------|
| AdaptiveCpp | Community-driven | Universal binary | Single-pass SSCP | Multi-vendor portability |
| CUDA | NVIDIA proprietary | NVIDIA only | Separate device compilation | Maximum NVIDIA performance |
| Intel DPC++ | Intel (oneAPI) | Intel-centric | Multi-pass compilation | Intel ecosystem |
| OpenCL | C-based standard | Platform-specific | Runtime compilation | Legacy cross-vendor support |

Each approach has its strengths. CUDA offers the highest performance on NVIDIA hardware but locks you into that ecosystem. Intel DPC++ provides excellent Intel hardware support with industry tools. OpenCL offers broad compatibility but with a more verbose C-based API. AdaptiveCpp strikes a balance by providing competitive performance across multiple vendors with the convenience of universal binaries.

## This Guide: An acpp-First Approach

This guide makes deliberate choices to provide the clearest path to productive AdaptiveCpp programming:

1. We ALWAYS use `--acpp-targets=generic` - no backend-specific targets are covered in this guide.
2. We use AdaptiveCpp buffer factory methods exclusively: `make_sync_buffer`, `make_async_buffer`, `make_sync_view`, `make_async_view`, `make_sync_writeback_view`, and `make_async_writeback_view`. We do NOT use raw `sycl::buffer` constructors directly.
3. We document scoped parallelism (`ACPP_EXT_SCOPED_PARALLELISM_V2`) as the intended primary parallelism abstraction - and use `nd_range` in examples until SSCP support for scoped parallelism lands (see [AdaptiveCpp #1417](https://github.com/AdaptiveCpp/AdaptiveCpp/issues/1417)).
4. We treat buffers as the primary memory model, with USM (Unified Shared Memory) covered as an advanced topic.

> [!NOTE]
> These choices were made to prioritize simplicity and correctness. The buffer factory methods avoid common pitfalls with implicit data movement, scoped parallelism will provide performance portability across CPU and GPU (fiber-free on CPU) once SSCP support lands, and focusing on the generic target ensures your code works everywhere without modification.

## Prerequisites

This guide assumes:
- C++ proficiency at the C++17 level or higher
- Familiarity with compilation and build tools
- A Linux environment (Windows is not covered in this guide)
- No prior GPU programming knowledge is required

AdaptiveCpp installation and environment setup is covered in Chapter 3, so you don't need to have it installed yet to begin reading.

## Guide Structure

This guide consists of 10 chapters that build from foundational concepts to practical implementation:

| Chapter | Title | Focus |
|---------|-------|-------|
| [00](../00-introduction/README.md) | Introduction | Project overview and philosophy |
| [01](../01-heterogeneous-computing/README.md) | Heterogeneous Computing | Background on modern computing architectures |
| [02](../02-sycl-overview/README.md) | SYCL Overview | The SYCL standard and key concepts |
| [03](../03-acpp-setup/README.md) | AdaptiveCpp Setup | Installation, configuration, and first compilation |
| [04](../04-memory-model/README.md) | Memory Model | Buffers, USM, and data management |
| [05](../05-kernels-and-parallelism/README.md) | Kernels and Parallelism | Writing and launching computational kernels |
| [06](../06-acpp-extensions/README.md) | AdaptiveCpp Extensions | Advanced features and specialized capabilities |
| [07](../07-performance/README.md) | Performance | Optimization techniques and best practices |
| [08](../08-footguns/README.md) | Footguns | Common pitfalls and how to avoid them |
| [09](../09-real-world-patterns/README.md) | Real-World Patterns | Practical patterns for production applications |
| [10](../10-atomics/README.md) | Atomics | sycl::atomic_ref, memory orders, memory scopes |

## A Note on Scope

This guide focuses specifically on the SSCP generic target and deliberately omits backend-specific tuning. The approaches presented represent one valid way to program with AdaptiveCpp (the authors' opinionated take), but not the only possible approach. As you gain experience, you may develop your own patterns and preferences.

## Getting Started

If you're new to heterogeneous computing, proceed to [Chapter 1: Heterogeneous Computing](../01-heterogeneous-computing/README.md) for background on modern computing architectures. If you're already familiar with these concepts and want to start coding, jump to [Chapter 3: AdaptiveCpp Setup](../03-acpp-setup/README.md) for AdaptiveCpp setup and your first compilation.