# AdaptiveCpp Programmer's Guide

The first open-source, actively maintained programmer's guide for
[AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp) (acpp) - the performance-portable
SYCL implementation that runs on NVIDIA, AMD, Intel, and CPU targets from a single binary.

This guide targets developers new to heterogeneous computing. It gets you writing real,
performant GPU code as quickly as possible - without requiring a background in CUDA, HIP, or
any other GPU programming model.

> [!WARNING]
> 🚧 This guide is a work in progress. Chapters and examples are being added incrementally.
> Not all sections are complete - expect gaps, placeholders, and rough edges.

> [!NOTE]
> All examples use the SSCP generic target (`--acpp-targets=generic`), AdaptiveCpp's
> "compile once, run anywhere" mode. No other compilation modes are covered.

---

## Why This Guide Exists

SYCL is technically superior to CUDA for platform-agnostic GPU programming. AdaptiveCpp is
the leading open-source SYCL implementation. Yet getting started has been unnecessarily hard:

- Building acpp from source is a monumental task (LLVM plugins, compiler toolchains)
- Intel's oneAPI documentation is thorough but Intel-GPU-centric
- No single resource teaches acpp idiomatically, with its extensions, for newcomers

The `acpp-toolchain` conda package (from the `code-accelerate` channel) and this guide close
that gap. The setup story is: install pixi, run `pixi install`, start writing GPU code.

---

## Getting Started

### Prerequisites

- Linux (x86-64)
- [pixi](https://pixi.sh) package manager

### Install

```sh
git clone https://github.com/nagagroup/CodeAccelerate-SyclProgrammingGuide
cd acpp-tutorial-project
pixi install
```

### Build and Run Examples

```sh
pixi run configure
pixi run build
pixi run test
```

---

## Guide Structure

```
guide/
  00-introduction/              What this guide is, why acpp, how to use it
  01-heterogeneous-computing/   CPU vs GPU architecture, memory hierarchies, parallelism
  02-sycl-overview/             SYCL 2020 concepts (acpp-annotated)
  03-acpp-setup/                Installation, device discovery, first kernel
  04-memory-model/              Buffer factory methods, USM, the library API pattern
  05-kernels-and-parallelism/   parallel_for, nd_range, scoped parallelism
  06-acpp-extensions/           Extension reference: specialized, JIT compile-if, dynamic
                                functions, buffer-USM interop, multi-device, accessor variants
  07-performance/               Adaptivity levels, JIT tuning, backend-specific notes
  08-footguns/                  Comprehensive pitfall reference with mitigations
  09-real-world-patterns/       Matrix multiply, Jacobi solver, iterative patterns
  10-atomics/                   sycl::atomic_ref, memory orders, memory scopes, ACPP_EXT_FP_ATOMICS
```

Read it linearly if you are new to GPU computing. Use individual chapters as reference if
you already know the basics.

---

## Key Design Decisions

**Buffer factory methods exclusively.** Standard SYCL buffer constructors have subtle
footguns (silent writeback, blocking destructor surprises). This guide uses AdaptiveCpp's
`ACPP_EXT_EXPLICIT_BUFFER_POLICIES` factory methods (`make_sync_buffer`, `make_async_buffer`,
`make_sync_view`, `make_async_usm_view`, ...) throughout. Standard constructors are shown
once as a cautionary example and never used again.

**Buffers taught first.** The broader acpp ecosystem recommends USM-first. This guide
deliberately inverts that for newcomers: buffer factory methods deliver SYCL's most
compelling feature - automatic DAG-based execution graph construction and automatic memory
migration - with no additional cost. USM is fully covered as the right tool for explicit
control and interop.

**Scoped parallelism as the intended primary abstraction.** `nd_range` kernels with barriers are a
CPU performance disaster and a common footgun. `ACPP_EXT_SCOPED_PARALLELISM_V2` is the
performance-portable alternative that eliminates fiber overhead on CPU while mapping efficiently to GPU warps/wavefronts.

> [!WARNING]
> `ACPP_EXT_SCOPED_PARALLELISM_V2` is **not yet supported** in `--acpp-targets=generic` (SSCP). This guide documents it as the intended preferred model. For now, all examples that require local memory and barriers use `nd_range`. Tracking issue: [AdaptiveCpp #1417](https://github.com/AdaptiveCpp/AdaptiveCpp/issues/1417).

**No assumed GPU knowledge.** Every concept (SIMT, memory hierarchy, host-device transfer,
asynchronous execution) is explained before it appears in code. This guide is designed to be
the resource a university GPU computing course could assign instead of a CUDA tutorial.

---

## Project Layout

```
acpp-tutorial-project/
  pixi.toml               Pixi workspace (acpp-toolchain, cmake, nushell, catch2)
  CMakeLists.txt          Root CMake - builds all examples
  guide/                  Guide chapters (see above)
  scripts/
    configure.nu          CMake configuration
    build.nu              Build script
    test.nu               Test runner
```

### Pixi Features

| Feature | Contents |
|---------|----------|
| `base` | acpp-toolchain, cmake, nushell, ninja, catch2 |
| `opencl` | pocl-cpu (CPU OpenCL backend) |
| `level-zero` | Intel GPU support |
| `cuda` | NVIDIA GPU support |
| `amd` | AMD GPU support (HIP runtime) |

The default environment activates all features. Use `pixi run configure` to auto-detect
available hardware backends.

---

## Scope

**Covered:**
- SSCP generic target (`--acpp-targets=generic`) on Linux x86-64
- AdaptiveCpp extensions: ACPP_EXT_EXPLICIT_BUFFER_POLICIES, ACPP_EXT_SCOPED_PARALLELISM_V2,
  ACPP_EXT_SPECIALIZED, ACPP_EXT_JIT_COMPILE_IF, ACPP_EXT_DYNAMIC_FUNCTIONS,
  ACPP_EXT_BUFFER_USM_INTEROP, ACPP_EXT_MULTI_DEVICE_QUEUE, ACPP_EXT_ACCESSOR_VARIANTS,
  ACPP_EXT_FP_ATOMICS
- NVIDIA, AMD, Intel GPU, and CPU targets via SSCP
- SYCL 2020 atomics: `sycl::atomic_ref`, memory orders, memory scopes, `sycl::atomic_fence`

**Not covered:**
- Intel oneAPI / DPC++ specifics
- Non-SSCP compilation modes (omp, cuda-smcp, hip-smcp, nvcxx)
- Windows or macOS
- Direct CUDA or HIP comparisons (ecosystem context only)

---

## Contributing

See [PROGRAMMING_GUIDE_IMPLEMENTATION_PLAN.md](PROGRAMMING_GUIDE_IMPLEMENTATION_PLAN.md) for
the agent-facing implementation plan and chapter specifications.

All pull requests must ensure:
- Examples compile with `pixi run build` and pass `pixi run test`
- All code uses `--acpp-targets=generic` (SSCP) exclusively
- Buffer factory methods are used; standard SYCL buffer constructors are not introduced
- No emojis, smart quotes, or decorative symbols in markdown files

---

## License

[MIT](LICENSE)
