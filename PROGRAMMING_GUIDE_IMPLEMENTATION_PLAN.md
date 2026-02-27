# AdaptiveCpp Programmer's Guide - Implementation Plan

> **Document Purpose:** Agent-facing outline for implementing the first open-source, maintained
> programmer's guide for AdaptiveCpp (acpp). This document describes the overall goal, structure,
> philosophy, and implementation steps. It is intentionally outline-oriented rather than
> content-complete - agents implementing each section MUST use the librarian to research the
> specific topic being written before generating content.

---

## Goal Statement

Create the first open-source, actively maintained programmer's guide for AdaptiveCpp - the
SYCL implementation that should become the gold standard for open-source SYCL projects. The
guide targets developers new to heterogeneous/GPU computing and gets them productive with
AdaptiveCpp quickly. It competes directly with CUDA's famously excellent onboarding experience.

**Scope:**

- Compilation mode: SSCP generic target ONLY (`--acpp-targets=generic`). No other modes documented.
- Implementation: AdaptiveCpp-first. The acpp way is the right way. SYCL spec is background context.
- Format: Markdown guide files + working example code, structured as a pixi project.
- Audience: Developers new to heterogeneous computing (students, engineers exploring GPU programming).
- Ecosystem goal: Close the gap that keeps SYCL fragmented and CUDA dominant. Make acpp as easy
  to start with as CUDA was in its early days.

**Out of scope:**

- Intel oneAPI / DPC++ specifics
- CUDA or HIP direct comparisons (mention ecosystem context only)
- Non-SSCP compilation modes (omp, cuda-smcp, hip-smcp, nvcxx)
- Windows or macOS (linux-64 only, per pixi.toml)

---

## Philosophy and Voice

Agents writing guide content must internalize these principles:

**acpp is king.** When the SYCL spec and acpp diverge, the guide teaches the acpp way. The spec
approach may be mentioned briefly (with rationale for why it is inferior or footgun-prone), then
dismissed in favor of the acpp approach. Example: buffer constructors vs. buffer factory methods.

**Show the footguns.** AdaptiveCpp has real pitfalls. The guide does not hide them. Each footgun
section explains: what the trap is, why it exists, why the acpp extension/approach avoids it,
and what the payoff is (usually a meaningful optimization or correctness guarantee).

**Teach the concepts, not just the syntax.** The guide is for people new to heterogeneous
computing. Concepts like memory hierarchies, work-item/work-group models, host-device data
movement, and asynchronous execution must be explained clearly before code is shown.

**Practical and runnable.** Every code example must compile and run with the project's pixi
environment. Examples are not pseudocode. They are real programs.

**Student-friendly.** Write as if a motivated undergraduate is reading this. Dense jargon without
explanation is a failure mode. The goal is to be the resource that university GPU computing
courses could assign instead of CUDA tutorials.

---

## Project Structure

```
acpp-tutorial-project/
  pixi.toml                          # Pixi workspace (acpp-toolchain, cmake, nushell, catch2)
  PROGRAMMING_GUIDE_IMPLEMENTATION_PLAN.md  # This file
  guide/
    00-introduction/
      README.md                      # Guide overview and how to use it
    01-heterogeneous-computing/
      README.md                      # Concepts: CPUs vs GPUs, memory hierarchies, parallelism
    02-sycl-overview/
      README.md                      # SYCL spec walkthrough (acpp-annotated)
      examples/                      # Runnable code for each concept
    03-acpp-setup/
      README.md                      # Installing via pixi, verifying SSCP, first program
      examples/
    04-memory-model/
      README.md                      # Buffers (factory methods, primary), USM (secondary), accessors
      examples/
    05-kernels-and-parallelism/
      README.md                      # parallel_for, nd_range, scoped parallelism (acpp primary)
      examples/
    06-acpp-extensions/
      README.md                      # Extension index and when to use each
      specialized/
      jit-compile-if/
      scoped-parallelism/
      dynamic-functions/
      buffer-usm-interop/
      multi-device/
      accessor-variants/
    07-performance/
      README.md                      # Adaptivity levels, JIT tuning, backend-specific notes
      examples/
    08-footguns/
      README.md                      # Comprehensive pitfall reference
    09-real-world-patterns/
      README.md                      # Putting it all together: real patterns for real programs
      examples/
  scripts/
    configure.nu                     # CMake configuration script
    build.nu                         # Build script
    test.nu                          # Test runner script
  CMakeLists.txt                     # Root CMake (builds all examples)
```

---

## Pixi Project Configuration

The project uses `pixi.toml` with the `acpp-toolchain` conda package (from the
`code-accelerate` channel). The toolchain is not yet published but the configuration is correct.
Agents must not modify `pixi.toml` without explicit instruction.

Key dependencies:
- `acpp-toolchain >= 20.1.8_acpp25.10.0` - AdaptiveCpp compiler + runtime
- `cmake >= 3.28` - Build system
- `nushell >= 0.110.0` - Script runner
- `ninja >= 1.12` - Build backend
- `catch2 >= 3.8` - Test framework for example tests

Features: `base`, `opencl` (pocl-cpu), `level-zero`, `cuda`, `amd` (hip-runtime-amd)

---

## Implementation Steps

Each step below is a discrete agent task. Steps within a phase may be parallelized.
**Every step that writes guide content must begin by delegating to librarian** to research
the specific topic before writing.

### Phase 0: Project Scaffolding

**Step 0.1 - Create directory structure**
- Create all `guide/` subdirectories and placeholder `README.md` files
- Create `CMakeLists.txt` root with subdirectory includes
- Create `scripts/configure.nu`, `scripts/build.nu`, `scripts/test.nu` stubs
- Acceptance: `pixi run configure` and `pixi run build` succeed (even with empty examples)

**Step 0.2 - CMake infrastructure**
- Root `CMakeLists.txt` that finds acpp, sets SSCP target, includes example subdirs
- Each example directory gets its own `CMakeLists.txt`
- Catch2 integration for test examples
- Acceptance: CMake configures without errors; acpp compiler is found and SSCP mode confirmed

### Phase 1: Guide Introduction and Concepts

**Step 1.1 - `guide/00-introduction/README.md`**
- What this guide is and is not
- Why AdaptiveCpp, why SSCP, why not CUDA/DPC++
- How to use the guide (linear for beginners, reference for experienced)
- Prerequisites (C++17, basic parallel programming concepts helpful but not required)
- How to set up the pixi environment
- Librarian research: AdaptiveCpp project overview, SSCP rationale, ecosystem context

**Step 1.2 - `guide/01-heterogeneous-computing/README.md`**
- What is heterogeneous computing (CPU vs GPU architecture, memory hierarchy)
- SIMD, SIMT, thread/warp/block/work-item/work-group mental models
- Why GPUs are fast for certain workloads (memory bandwidth, massive parallelism)
- Host-device data movement concepts (why it matters, latency vs bandwidth)
- Asynchronous execution model overview
- Librarian research: GPU architecture fundamentals, SIMT model, memory hierarchy concepts

**Step 1.3 - `guide/02-sycl-overview/README.md`**
- Walk through the full SYCL 2020 spec at a conceptual level
- For each major concept: explain the spec approach, note where acpp differs, flag for later
- Topics: platform/device/context/queue model, command groups, kernels, memory model,
  synchronization, error handling, interoperability
- Tone: "Here is what SYCL defines. In the acpp chapters, we will show you the better way."
- Librarian research: SYCL 2020 specification overview, Khronos SYCL docs

### Phase 2: Getting Started with acpp

**Step 2.1 - `guide/03-acpp-setup/README.md`**
- Installing via pixi (the one-command setup story - this is our answer to `apt install cuda`)
- Verifying the installation: `acpp --version`, device discovery
- First program: hello world that queries available devices
- Compiling with SSCP: `acpp --acpp-targets=generic`
- Understanding the JIT cache (`~/.acpp/apps/`) and when to clear it
- Librarian research: AdaptiveCpp installation docs, device discovery API, SSCP compilation flags

**Step 2.2 - `guide/03-acpp-setup/examples/`**
- `hello_devices.cpp` - enumerate and print all available devices
- `hello_kernel.cpp` - simplest possible kernel (vector add)
- `CMakeLists.txt` for these examples

### Phase 3: Memory Model (Critical Chapter)

**Step 3.1 - `guide/04-memory-model/README.md`**

This is one of the most important chapters. Structure:

*Section A: Buffers - The Primary Model (acpp Way)*

Buffers are taught first because they deliver SYCL's most compelling features for newcomers:
automatic memory migration between host and device, and automatic DAG-based execution graph
construction (no manual event management). This is what makes SYCL exciting for someone who
has never written a single line of GPGPU code.

- Brief: what the SYCL spec buffer constructor does and why it is a footgun
  (silent writeback behavior, blocking destructor surprises, ambiguous semantics)
- The acpp buffer factory methods (full reference):
  - `make_sync_buffer` / `make_async_buffer` (internal storage, no writeback)
  - `make_sync_view` / `make_async_view` (external storage, no writeback)
  - `make_sync_writeback_view` / `make_async_writeback_view` (external storage, writeback)
  - USM interop variants: `make_sync_usm_view` / `make_async_usm_view`
- Rule: **The guide uses factory methods exclusively from this point forward.**
  Standard buffer constructors are shown once, explained as footguns, then never used again.
- Accessors: how they work with factory-created buffers
- `ACPP_EXT_ACCESSOR_VARIANTS` for reduced register pressure
- Librarian research: `ACPP_EXT_EXPLICIT_BUFFER_POLICIES`, `ACPP_EXT_BUFFER_USM_INTEROP`,
  accessor variant extension docs

*Section B: USM (Unified Shared Memory) - Explicit Control*
- `sycl::malloc_device`, `sycl::malloc_shared`, `sycl::malloc_host`
- When to use each allocation type
- Explicit data movement with `queue::memcpy`
- Shared USM caveats (AMD XNACK, Intel discrete page migration)
- Librarian research: SYCL USM spec, acpp USM docs

*Section C: The Bridge - USM Views and the Library API Pattern*

This section explains why the buffer-primary approach does NOT mean you must choose one or
the other. The `make_sync_usm_view` / `make_async_usm_view` factory methods build a buffer
around an existing USM pointer (with or without taking ownership), eliminating the lazy
allocation overhead that would otherwise make buffers costly.

This enables a powerful library design pattern, supporting both buffer and USM public APIs with ease:
- Implement all internal logic using the buffer API (automatic DAG, automatic migration)
- Expose a USM-facing public API by wrapping caller-provided USM pointers in usm_view buffers
- Result: no double implementation, no performance penalty, both APIs work correctly

Agents writing this section must explain this pattern with a concrete example: a library
function that accepts either a buffer or a USM pointer and dispatches to the same kernel.

*Section D: Choosing Between Buffers and USM*
- Decision guide: buffers for new code and library internals (automatic DAG, migration),
  USM for interop with existing allocations, C APIs, or when explicit control is required
- Note the guide's deliberate deviation from the acpp/ecosystem USM-first recommendation,
  with rationale: factory methods make buffer costs negligible, and the automatic DAG is
  SYCL's killer feature for newcomers

**Step 3.2 - `guide/04-memory-model/examples/`**
- `buffer_factory.cpp` - demonstrating each factory method variant
- `usm_device.cpp` - device USM with explicit memcpy
- `usm_shared.cpp` - shared USM, implicit migration
- `usm_view_bridge.cpp` - the library API pattern: buffer internals + USM-facing public API
  using `make_sync_usm_view` / `make_async_usm_view` to wrap caller USM pointers
- `buffer_vs_usm.cpp` - side-by-side comparison of same algorithm (framed as: when to reach for USM)
- `CMakeLists.txt`

### Phase 4: Kernels and Parallelism

**Step 4.1 - `guide/05-kernels-and-parallelism/README.md`**

*Section A: Basic Kernels*
- `queue::submit` and command groups
- `parallel_for` with `range` (basic)
- `parallel_for` with `nd_range` (work-groups, local memory)
- Local memory (`sycl::local_accessor`)
- Synchronization: barriers, fences

*Section B: Scoped Parallelism (acpp Primary Abstraction)*
- Why `nd_range` with barriers is a CPU performance disaster
- `ACPP_EXT_SCOPED_PARALLELISM_V2` as the performance-portable solution
- `distribute_groups`, `distribute_items`, `single_item`
- `memory_environment` for portable local memory
- Nesting rules and footguns (strict hierarchy, collective reach requirements)
- When to use scoped parallelism vs nd_range
- Librarian research: `ACPP_EXT_SCOPED_PARALLELISM_V2` full docs, performance-portability rationale

**Step 4.2 - `guide/05-kernels-and-parallelism/examples/`**
- `basic_parallel_for.cpp`
- `nd_range_local_mem.cpp`
- `scoped_parallelism.cpp` - same algorithm as nd_range but with scoped parallelism
- `CMakeLists.txt`

### Phase 5: AdaptiveCpp Extensions Deep Dive

Each extension gets its own subdirectory under `guide/06-acpp-extensions/`. Each is a
separate agent task. All must use librarian to research the specific extension before writing.

**Step 5.0 - `guide/06-acpp-extensions/README.md`**
- Extension index with one-line descriptions and links to subdirectories
- How to check if an extension is available at runtime
- Extension naming convention (`ACPP_EXT_*`)

**Step 5.1 - `specialized/README.md` + examples**
- `ACPP_EXT_SPECIALIZED`: zero-overhead kernel specialization via JIT constants
- When to use: loop bounds, tile sizes, algorithm variants known at launch time
- Footgun: only works with SSCP/generic target (already our only mode - no issue)
- Librarian research: `ACPP_EXT_SPECIALIZED` docs

**Step 5.2 - `jit-compile-if/README.md` + examples**
- `ACPP_EXT_JIT_COMPILE_IF`: branch on target properties at JIT time (zero runtime cost)
- Queryable properties: vendor ID, architecture, forward progress guarantees, backend type
- Footgun: `knows()` must be checked before querying; not all properties available everywhere
- Librarian research: `ACPP_EXT_JIT_COMPILE_IF` docs

**Step 5.3 - `scoped-parallelism/README.md` + examples** (deep dive, referenced from Phase 4)
- Full reference for `ACPP_EXT_SCOPED_PARALLELISM_V2`
- All nesting rules, all footguns, all memory environment variants
- Librarian research: scoped parallelism full spec

**Step 5.4 - `dynamic-functions/README.md` + examples**
- `ACPP_EXT_DYNAMIC_FUNCTIONS`: runtime kernel fusion, dynamic dispatch at JIT time
- Requires `SYCL_EXTERNAL` on definition functions (footgun if forgotten)
- Librarian research: `ACPP_EXT_DYNAMIC_FUNCTIONS` docs

**Step 5.5 - `buffer-usm-interop/README.md` + examples**
- `ACPP_EXT_BUFFER_USM_INTEROP`: build buffers on USM pointers, query allocations
- `get_pointer`, `has_allocation`, `own_allocation`, `disown_allocation`
- Librarian research: `ACPP_EXT_BUFFER_USM_INTEROP` docs

**Step 5.6 - `multi-device/README.md` + examples**
- `ACPP_EXT_MULTI_DEVICE_QUEUE`: distribute work across multiple devices automatically
- `ACPP_EXT_CG_PROPERTY_RETARGET`: retarget command groups to different devices at runtime
- Footgun: retarget requires queue constructed with `AdaptiveCpp_retargetable` property
- Librarian research: multi-device queue and retarget extension docs

**Step 5.7 - `accessor-variants/README.md` + examples**
- `ACPP_EXT_ACCESSOR_VARIANTS` + `ACPP_EXT_ACCESSOR_VARIANT_DEDUCTION`
- Reduced register pressure via type-encoded accessor purpose
- CTAD deduction guides
- Librarian research: accessor variant extension docs

### Phase 6: Performance

**Step 6.1 - `guide/07-performance/README.md`**
- Adaptivity levels (`ACPP_ADAPTIVITY_LEVEL=1` vs `2`) and what they mean
- JIT cache management (location, when to clear, staleness after upgrades)
- Compilation flags: acpp does NOT default to `-O3` or `-ffast-math` (footgun vs nvcc)
- Backend-specific performance notes:
  - NVIDIA: mature, excellent
  - AMD: ROCm driver quality, XNACK for shared USM
  - Intel discrete: page-granularity migration, `ACPP_STDPAR_MEM_POOL_SIZE=0`
  - CPU: scoped parallelism required for good performance, NUMA awareness
- `ACPP_EXT_COARSE_GRAINED_EVENTS` for reduced kernel launch latency
- `ACPP_EXT_SYNCHRONOUS_MEM_ADVISE` for efficient memory hints
- Allocation tracking overhead (`ACPP_ALLOCATION_TRACKING`)
- Librarian research: AdaptiveCpp performance guide, adaptivity docs, backend-specific notes

**Step 6.2 - `guide/07-performance/examples/`**
- `benchmark_buffer_vs_usm.cpp` - measure overhead difference (framed as: buffers via factory methods are not slower)
- `adaptivity_demo.cpp` - show JIT specialization improving performance across runs
- `CMakeLists.txt`

### Phase 7: Footguns Reference

**Step 7.1 - `guide/08-footguns/README.md`**

Comprehensive, searchable reference of all known acpp pitfalls. Organized by category.
Each entry: what the trap is, minimal reproducer, why it happens, mitigation.

Categories:
- Memory management footguns (buffer destructor blocking, USM shared on AMD/Intel)
- Compilation footguns (missing `-O3`, libc++ with CUDA, JIT cache staleness)
- Performance footguns (allocation tracking, adaptivity levels, CPU nd_range barriers)
- Extension footguns (dynamic functions needing `SYCL_EXTERNAL`, retarget needing property,
  scoped parallelism nesting violations, JIT reflection `knows()` check)
- Correctness footguns (buffer reinterpretation granularity, USM modification during buffer lifetime)

Librarian research: AdaptiveCpp performance guide, GitHub issues, extension docs for footgun details

### Phase 8: Real-World Patterns

**Step 8.1 - `guide/09-real-world-patterns/README.md`**
- Putting it all together: patterns for real programs
- Pattern: matrix multiplication (USM + scoped parallelism + JIT specialization)
- Pattern: image processing pipeline (multi-device queue)
- Pattern: iterative solver (async buffers + automatic DAG, no manual event management)
- Pattern: stdpar integration (C++ standard algorithms on GPU)
- Each pattern: problem statement, acpp-idiomatic solution, footguns to watch for
- Librarian research: each pattern individually before writing

---

## Key Architectural Decisions

**Decision: SSCP only**
All examples compile with `--acpp-targets=generic`. No other targets are mentioned in code.
This is the "compile once, run anywhere" story that makes acpp compelling.

**Decision: Buffers primary, USM secondary (deliberate deviation from acpp/ecosystem recommendation)**
The acpp project and broader SYCL ecosystem recommend USM-first. This guide deliberately
inverts that for newcomers: buffers (via factory methods) are taught first because they
deliver SYCL's most compelling features - automatic memory migration and automatic DAG-based
execution graph construction. Not having to reason about events is a huge win for someone
new to GPU programming.

The acpp buffer factory methods (`ACPP_EXT_EXPLICIT_BUFFER_POLICIES`) make this cost-free:
`make_sync_usm_view` / `make_async_usm_view` build a buffer around an existing USM pointer
with no lazy allocation overhead. This also enables the library API pattern: implement
internals in the buffer API, expose a USM-facing public API via usm_view wrappers - no
double implementation required.

The guide acknowledges the ecosystem recommendation and explains the rationale for the
deviation. USM is fully covered as the right tool for explicit control, interop with C APIs,
and cases where the automatic DAG is not needed.

**Decision: Buffer factory methods exclusively**
Standard SYCL buffer constructors are shown exactly once (in the memory model chapter) as a
cautionary example. Every subsequent buffer in the guide uses acpp factory methods.

**Decision: Scoped parallelism as the primary parallelism abstraction**
`nd_range` with barriers is shown for completeness and SYCL spec coverage. Scoped parallelism
(`ACPP_EXT_SCOPED_PARALLELISM_V2`) is the recommended approach for all new code.

**Decision: Pixi for environment management**
The guide's setup story is: install pixi, run `pixi install`, start writing GPU code.
This is the answer to `apt install cuda-toolkit`. The `acpp-toolchain` conda package
(from the `code-accelerate` channel) handles the entire compiler toolchain.

---

## Acceptance Criteria for the Complete Guide

- A developer with no GPU programming experience can follow the guide from start to finish
  and write a working, performant SYCL program using AdaptiveCpp
- All code examples compile with `pixi run build` and pass `pixi run test`
- Every acpp extension covered has at least one runnable example
- Every documented footgun has a minimal reproducer and clear mitigation
- The guide is usable as a university course reference (clear explanations, no assumed GPU knowledge)
- The guide is findable and linkable (good markdown structure, clear headings, no broken links)
- Buffer factory methods are used exclusively after their introduction - no standard constructors
- All examples use `--acpp-targets=generic` (SSCP) exclusively

---

## Research Resources for Agents

When implementing any section, agents MUST use librarian with these starting URLs:

- AdaptiveCpp GitHub: `https://github.com/AdaptiveCpp/AdaptiveCpp`
- AdaptiveCpp Docs: `https://adaptivecpp.github.io/AdaptiveCpp/`
- Extensions Reference: `https://adaptivecpp.github.io/AdaptiveCpp/extensions/`
- SSCP Docs: `https://adaptivecpp.github.io/AdaptiveCpp/generic-sscp/`
- Compilation Model: `https://adaptivecpp.github.io/AdaptiveCpp/compilation/`
- Performance Guide: `https://adaptivecpp.github.io/AdaptiveCpp/performance/`
- Scoped Parallelism: `https://adaptivecpp.github.io/AdaptiveCpp/scoped-parallelism/`
- Buffer Policies: `https://adaptivecpp.github.io/AdaptiveCpp/explicit-buffer-policies/`
- Buffer-USM Interop: `https://adaptivecpp.github.io/AdaptiveCpp/buffer-usm-interop/`
- SYCL 2020 Spec: `https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html`

---

## Notes on Ecosystem Context (for Introduction Chapter)

The guide should briefly acknowledge why this project exists:

- SYCL is technically superior to CUDA for platform-agnostic GPU programming
- The ecosystem is fragmented: Intel oneAPI has great docs but is Intel-GPU-centric
- Building acpp from source is a monumental task (LLVM plugins, compiler toolchains)
- The `acpp-toolchain` conda package + this guide closes that gap
- The goal: make acpp as easy to start with as CUDA was in 2010
- Target the student-to-industry pipeline: if students learn acpp, they bring it to work

This context belongs in the introduction, not repeated throughout the guide.
