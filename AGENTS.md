# AGENTS.md - Project Context for AI Agents

This file provides full context for any AI agent working on this project.
Read it completely before doing any work.

---

## Project Overview

**Name:** AdaptiveCpp Programmer's Guide  
**Purpose:** The first open-source, maintained programmer's guide for AdaptiveCpp (SYCL).
Targets developers new to heterogeneous/GPU computing. Designed to compete with
CUDA's onboarding experience in quality and depth.

**Git remote:** https://github.com/nagagroup/CodeAccelerate-SyclProgrammingGuide  
**Working directory:** `/home/jack/acpp-tutorial-project`  
**License:** MIT, copyright 2026 Jack Myers

---

## Core Philosophy (Non-Negotiable)

These are architectural decisions baked into every example and explanation.
NEVER deviate from them.

### 1. SSCP Generic Target Only

All examples compile with `--acpp-targets=generic`. No other compilation modes
are documented or used. Never show `-DACPP_TARGETS=omp` or platform-specific flags.

### 2. Buffer Factory Methods Only

AdaptiveCpp provides explicit buffer policy factories. Use ONLY these - never
use the standard `sycl::buffer` constructors:

| Factory | Storage | Writeback |
|---------|---------|-----------|
| `make_sync_buffer<T>(range)` | internal | no |
| `make_async_buffer<T>(range)` | internal | no |
| `make_sync_view<T>(ptr, range)` | external | no |
| `make_async_view<T>(ptr, range)` | external | no |
| `make_sync_writeback_view<T>(ptr, range)` | external | yes (on destruction) |
| `make_async_writeback_view<T>(ptr, range, q)` | external | yes (explicit queue) |
| `make_sync_usm_view<T>(ptr, range)` | USM | no |
| `make_async_usm_view<T>(ptr, range)` | USM | no |

Extension macro: `ACPP_EXT_EXPLICIT_BUFFER_POLICIES`

**Why:** Standard `sycl::buffer` has silent writeback on destruction and blocking
destructor semantics that surprise beginners. Factory methods are explicit.

The guide shows the standard constructor ONCE in chapter 08 (footguns) as an
anti-pattern. Never use it elsewhere.

### 3. Scoped Parallelism as Primary Abstraction

`ACPP_EXT_SCOPED_PARALLELISM_V2` is the preferred parallel dispatch model.
`nd_range` with barriers is shown for completeness only, with a clear note that
scoped parallelism is preferred on CPU (avoids fiber overhead).

Scoped parallelism hierarchy:
```
distribute_groups(handler, group_range, [](auto group) {
    distribute_items(group, item_range, [](auto item) {
        // kernel body
    });
});
```

### 4. Buffers First, USM Second

The guide deliberately teaches `sycl::buffer` (via factory methods) BEFORE USM,
despite the broader ecosystem recommendation. USM is fully covered as the
explicit-control model for advanced users.

### 5. Linux Only

No Windows or macOS content. The `acpp-toolchain` conda package targets `linux-64`.

### 6. No Emojis

Do not use emojis in markdown files. The one exception is the WIP warning
callout in the top-level README.md which has a `🚧` - leave it alone.

---

## Repository Structure

```
acpp-tutorial-project/
├── AGENTS.md                  <- you are here
├── README.md                  <- top-level WIP notice
├── PROGRAMMING_GUIDE_IMPLEMENTATION_PLAN.md  <- agent-facing outline
├── CMakeLists.txt             <- root cmake, defines add_acpp_example macro
├── pixi.toml                  <- environment + tasks
├── scripts/
│   ├── configure.nu           <- cmake configure
│   ├── build.nu               <- cmake build
│   └── test.nu                <- ctest
├── chapters/
│   ├── 00-introduction/README.md
│   ├── 01-heterogeneous-computing/README.md
│   ├── 02-sycl-overview/README.md
│   ├── 03-acpp-setup/
│   │   ├── README.md
│   │   └── examples/
│   │       ├── CMakeLists.txt
│   │       ├── hello_devices.cpp
│   │       └── hello_kernel.cpp
│   ├── 04-memory-model/
│   │   ├── README.md
│   │   └── examples/
│   │       ├── CMakeLists.txt
│   │       ├── buffer_policies.cpp
│   │       └── usm_vector_add.cpp
│   ├── 05-kernels-and-parallelism/
│   │   ├── README.md
│   │   └── examples/
│   │       ├── CMakeLists.txt
│   │       ├── nd_range_demo.cpp
│   │       └── scoped_reduction.cpp
│   ├── 06-acpp-extensions/
│   │   ├── README.md
│   │   └── examples/
│   │       ├── CMakeLists.txt
│   │       ├── jit_specialized.cpp
│   │       └── accessor_variants_demo.cpp
│   ├── 07-performance/
│   │   ├── README.md
│   │   └── examples/
│   │       ├── CMakeLists.txt
│   │       └── bandwidth_benchmark.cpp
│   ├── 08-footguns/
│   │   └── README.md
│   └── 09-real-world-patterns/
│       ├── README.md
│       └── examples/
│           ├── CMakeLists.txt
│           ├── matmul.cpp
│           └── jacobi_solver.cpp
│   └── 10-atomics/
│       ├── README.md
│       └── examples/
│           ├── CMakeLists.txt
│           ├── atomic_counter.cpp
│           ├── reduction_fetch_add.cpp
│           ├── compare_exchange.cpp
│           ├── atomic_fence_ordering.cpp
│           └── fp_atomics.cpp
└── build/                     <- cmake build output (not committed)
```

---

## Build System

### CMake Macro

The root `CMakeLists.txt` defines:

```cmake
macro(add_acpp_example name source)
    add_executable(${name} ${source})
    add_sycl_to_target(TARGET ${name})
    install(TARGETS ${name} DESTINATION bin)
endmacro()
```

**Critical:** The AdaptiveCpp CMake function is `add_sycl_to_target(TARGET name)`.
NOT `acpp_add_sycl_to_target`. NOT `target_link_libraries(... AdaptiveCpp::acpp)`.
This was a known bug that was fixed - do not reintroduce it.

### pixi Tasks

```
pixi run configure    # cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
                      #   -DACPP_TARGETS=generic
                      #   -DCMAKE_C_COMPILER=clang
                      #   -DCMAKE_CXX_COMPILER=clang++
pixi run build        # cmake --build build --parallel
pixi run test         # cd build && ctest --output-on-failure
```

**Critical:** `-DCMAKE_CXX_COMPILER=clang++` is required. Without it, CMake picks
up the system gcc instead of the pixi-managed clang 20.1.8.

### Running Executables

```
pixi run ./build/chapters/03-acpp-setup/examples/hello_devices
pixi run ./build/chapters/03-acpp-setup/examples/hello_kernel
```

Use `pixi run <path>` - this ensures the pixi environment (OpenCL runtimes, etc.)
is active when the binary runs.

---

## Environment

### pixi.toml Key Facts

- **Channels:** `code-accelerate`, `conda-forge`
- **Package:** `acpp-toolchain >= 20.1.8_acpp25.10.0` (confirmed published, working)
- **Features:** `base` (default), `opencl` (pocl-cpu), `level-zero`, `cuda`, `amd`
- **SSCP JIT cache:** `~/.acpp/apps/` - clear after acpp-toolchain upgrades

### Confirmed Working (as of Feb 2026)

- `acpp-toolchain 20.1.8_acpp25.10.0` on `code-accelerate` channel
- `pixi run configure && pixi run build` produces binaries correctly
- `pixi run ./build/chapters/03-acpp-setup/examples/hello_devices` - runs
- `pixi run ./build/chapters/03-acpp-setup/examples/hello_kernel` - runs

---

## Chapter Status

| Chapter | README | Examples | Status |
|---------|--------|----------|--------|
| 00-introduction | Done | N/A | Complete |
| 01-heterogeneous-computing | Done | N/A | Complete |
| 02-sycl-overview | Done | N/A | Complete |
| 03-acpp-setup | Done | hello_devices, hello_kernel | Complete |
| 04-memory-model | Done | buffer_policies, usm_vector_add | Complete |
| 05-kernels-and-parallelism | Done | nd_range_demo, scoped_reduction* | Complete |
| 06-acpp-extensions | Done | jit_specialized, accessor_variants_demo | Complete |
| 07-performance | Done | bandwidth_benchmark | Complete |
| 08-footguns | Done | none (prose only) | Complete |
| 09-real-world-patterns | Done | matmul, jacobi_solver | Complete |
| 10-atomics | Done | atomic_counter, reduction_fetch_add, compare_exchange, atomic_fence_ordering, fp_atomics | Complete |

---

## Implementation Workflow

Each chapter follows this sequence:

1. **Research** (librarian agent) - Fetch current AdaptiveCpp docs before writing
2. **Write README** (junior_dev) - Chapter markdown in `chapters/NN-name/README.md`
3. **Write Examples** (junior_dev) - C++ files in `chapters/NN-name/examples/`
4. **Update CMakeLists.txt** (junior_dev) - Add `add_acpp_example()` calls
5. **Verify** (test_runner) - `pixi run configure && pixi run build`, then run each binary

### test_runner Protocol

test_runner MUST:
1. Run `pixi run configure` (if not already done)
2. Run `pixi run build`
3. **Run each compiled executable** with `pixi run ./build/path/to/binary`
4. Verify output is correct (not just that the binary exists)

Do NOT mark verification complete if executables were not actually run.

---

## Key AdaptiveCpp Technical Facts

### SSCP Two-Stage Compilation

1. **Compile-time:** C++ source parsed once, LLVM IR embedded in host binary
2. **Runtime:** IR lowered via JIT to PTX (NVIDIA), amdgcn (AMD), SPIR-V (Intel/CPU)

Universal binary benefit: one binary runs on any supported backend.

### Supported Backends (SSCP generic)

| Backend | IR target | Runtime |
|---------|-----------|---------|
| NVIDIA GPU | PTX | CUDA |
| AMD GPU | amdgcn | HIP |
| Intel GPU | SPIR-V | Level Zero |
| CPU | LLVM JIT | native |
| OpenCL devices | SPIR-V | OpenCL |

### Important Extensions

| Extension macro | Purpose |
|----------------|---------|
| `ACPP_EXT_EXPLICIT_BUFFER_POLICIES` | Buffer factory methods |
| `ACPP_EXT_SCOPED_PARALLELISM_V2` | `distribute_groups`/`distribute_items` |
| `ACPP_EXT_SPECIALIZED` | Zero-overhead JIT constants |
| `ACPP_EXT_JIT_COMPILE_IF` | Branch on target properties at JIT time |
| `ACPP_EXT_DYNAMIC_FUNCTIONS` | Runtime kernel fusion |
| `ACPP_EXT_BUFFER_USM_INTEROP` | Build buffers on USM pointers |
| `ACPP_EXT_MULTI_DEVICE_QUEUE` | Distribute work across multiple devices |
| `ACPP_EXT_COARSE_GRAINED_EVENTS` | Reduced kernel launch latency |
| `ACPP_EXT_ACCESSOR_VARIANTS` | Reduced register pressure |

---

## Known Footguns (Documented in Chapter 08)

These are anti-patterns the guide explicitly teaches readers to avoid:

1. **No default optimization flags** - acpp does not default to `-O3` or `-ffast-math`
2. **Standard `sycl::buffer` constructors** - silent writeback, blocking destructor
3. **Shared USM on AMD without XNACK** - severe performance degradation
4. **Intel discrete GPU + USM** - migrates entire allocations, not pages; set `ACPP_STDPAR_MEM_POOL_SIZE=0`
5. **JIT cache staleness** - clear `~/.acpp/apps/` after acpp-toolchain upgrades
6. **Dynamic functions need `SYCL_EXTERNAL`** - or compiler inlines before dynamic dispatch
7. **Scoped parallelism nesting** - all work items must collectively reach each call
8. **`ACPP_ADAPTIVITY_LEVEL=1` default** - needs 2+ runs for peak performance
9. **libc++ + CUDA backend** - not supported; use libstdc++
10. **nd_range barriers on CPU** - fiber-based, high overhead; use scoped parallelism
11. **Local accessor silent kernel failure** - fixed in >= 25.10.0
12. **Level Zero buffer writeback bug #1799** - always call q.wait() before buffer destruction
13. **Scoped parallelism conditional collectives** - never wrap collective calls in conditionals
14. **Scoped parallelism nesting inside distribute_items** - no collectives; use _and_wait variants
15. **Buffer-USM interop data state** - view() for current data, empty_view() for uninitialized
16. **`seq_cst` on device scope** - forces global GPU sync; use `acq_rel` instead
17. **64-bit atomics without `atomic64` aspect check** - undefined behavior
18. **Wrong scope for cross-group sync** - use `memory_scope::device` not `work_group`
19. **`ACPP_EXT_FP_ATOMICS` silent emulation** - guard with `#ifdef`; provide CAS fallback
20. **Double-fencing around `group_barrier`** - redundant; barrier already provides acq_rel fence
21. **`sub_group` scope on CPU/Intel iGPU** - unsupported; query at runtime
22. **`compare_exchange_weak` without retry loop** - spurious failures; must loop

---

## Reference URLs

- AdaptiveCpp source: https://github.com/AdaptiveCpp/AdaptiveCpp
- AdaptiveCpp docs: https://adaptivecpp.github.io/AdaptiveCpp/
- Extensions reference: https://adaptivecpp.github.io/AdaptiveCpp/extensions/
- SSCP/generic target: https://adaptivecpp.github.io/AdaptiveCpp/generic-sscp/
- Explicit buffer policies: https://adaptivecpp.github.io/AdaptiveCpp/explicit-buffer-policies/
- Buffer-USM interop: https://adaptivecpp.github.io/AdaptiveCpp/buffer-usm-interop/
- Scoped parallelism: https://adaptivecpp.github.io/AdaptiveCpp/scoped-parallelism/

---

## Agent Roles

| Agent | Responsibilities |
|-------|-----------------|
| `tech_lead` | Coordination, planning, writing markdown files only |
| `librarian` | Fetching and synthesizing AdaptiveCpp docs before each chapter |
| `junior_dev` | All C++ code, CMakeLists.txt, and non-markdown file edits |
| `test_runner` | Build verification AND running compiled executables |
| `explore` | Deep codebase analysis when grep/glob/read aren't enough |

**tech_lead cannot edit `.cpp`, `.cmake`, `.toml`, `.json`, or any non-markdown files.**
All such changes must be delegated to `junior_dev`.
