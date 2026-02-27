# Chapter 07: Performance

AdaptiveCpp SSCP compiles to native PTX/amdgcn/SPIR-V, so vendor profilers work directly
with the resulting binaries. The primary levers for performance are JIT adaptivity, the kernel
cache, memory access patterns, and compiler optimization flags.

## The Most Important Flag: -O3

> [!CAUTION]
> AdaptiveCpp does NOT apply -O3 by default, unlike nvcc (-O2) or hipcc (-O3). Always compile
> with -O3 for production.

The `add_acpp_example` macro defined in this project's root `CMakeLists.txt` does not set
optimization flags - it only wires up the SYCL toolchain integration. To enable full
optimization, pass the flag at configure time:

```sh
cmake ... -DCMAKE_CXX_FLAGS=-O3
```

Additional flags worth considering for CPU code paths: `-ffp-contract=fast` (enables fused
multiply-add) and `-march=native` (generates CPU-specific vector instructions).

## JIT Compilation and the Kernel Cache

SSCP uses a two-stage compilation model: at build time, your C++ source is compiled once to
LLVM IR, which is embedded in the host binary. At first run, the SSCP runtime JIT-compiles
that IR to the target backend (PTX for NVIDIA, amdgcn for AMD, SPIR-V for Intel/CPU). The
result is cached at `~/.acpp/apps/` (overridable via `ACPP_APPDB_DIR`).

This means the first execution of a new binary carries a JIT overhead of roughly 100-300 ms.
All subsequent runs hit the cache and incur negligible JIT cost. When you have finished
development and are benchmarking, run the application 2-3 times until the "new binaries being
JIT-compiled" warning stops appearing.

After upgrading `acpp-toolchain`, always clear the cache to avoid stale PTX from a prior
compiler version:

```sh
rm -rf ~/.acpp/apps/*
```

## ACPP_ADAPTIVITY_LEVEL

AdaptiveCpp can track how kernel arguments change across calls and re-specialize the JIT when
invariant values (sizes, strides, scale factors) are detected. This is controlled by the
`ACPP_ADAPTIVITY_LEVEL` environment variable:

| Level | Behavior | Notes |
|-------|----------|-------|
| 0 | Disabled, no JIT recompile | Fastest startup, no adaptive optimization |
| 1 | Default, basic IADS, peak by run 2 | Good balance of startup and performance |
| 2 | Aggressive IADS, peak by run 3-4 | Best for kernels with invariant arguments |
| 3+ | Not yet implemented | Reserved for future use |

IADS stands for Invariant Argument Detection and Specialization. When an argument - such as an
array length or a floating-point scale - remains constant across calls, AdaptiveCpp hard-codes
it as a JIT constant in subsequent compilations, giving the optimizer more to work with.

> [!NOTE]
> Level 2 can yield a 10-30% speedup for kernels with invariant arguments like sizes or
> strides. Run the application 3-4 times to reach convergence.

Set the level via the environment:

```sh
export ACPP_ADAPTIVITY_LEVEL=2
```

## Memory Access Patterns

### Coalesced Access

On NVIDIA hardware, a warp consists of 32 threads that execute together. On AMD hardware, the
equivalent wavefront is 64 threads. Both architectures deliver maximum memory bandwidth when
consecutive threads in a warp or wavefront access consecutive memory addresses - a pattern
called coalesced access. Non-coalesced access can impose up to a 32x bandwidth penalty on
NVIDIA hardware.

The following shows a correctly coalesced pattern, where thread `i` accesses element `i`:

```cpp
c[i] = a[i] + b[i];  // Consecutive threads access consecutive addresses
```

The following is a non-coalesced pattern that serializes memory transactions:

```cpp
c[i] = a[i*32] + b[i*32];  // Each thread accesses memory 32 elements apart
```

### Local Memory Bank Conflicts

Local (shared) memory is divided into 32 banks on NVIDIA and 64 banks on AMD. When multiple
threads in a warp access different addresses that map to the same bank, those accesses are
serialized into bank conflicts. Avoid stride-2+ access patterns in local memory to keep all
banks busy in parallel.

### USM vs Buffers for Performance

Device USM carries the lowest overhead: you manage transfers explicitly with `q.memcpy()` and
the runtime has no DAG scheduling to perform. The buffer+accessor model, by contrast, builds
an execution graph that inserts dependencies and manages transfers automatically - a powerful
feature for correctness, but one that adds runtime overhead that is visible at fine granularity.

This guide teaches buffers first because their automatic dependency tracking is genuinely
compelling for learning. For maximum throughput in production code, however, device USM with
explicit `memcpy` transfers is the right tool.

## Work Group Sizing

When using scoped parallelism, the backend selects the physical work group size automatically,
giving portable performance without manual tuning. When using `nd_range` kernels, you must
specify the local size explicitly.

For `nd_range` kernels, prefer group sizes that are multiples of the hardware warp or
wavefront size: 64, 128, 256, or 512 are typical sweet spots on modern GPUs. Larger groups
allow more in-flight warps and better latency hiding, but the optimal size depends on register
pressure and local memory usage.

> [!NOTE]
> When using nd_range, prefer group sizes that are multiples of the hardware warp/wavefront:
> 64, 128, 256, 512.

## Kernel Launch Latency

Out-of-order queues route submissions through the DAG scheduler, adding 100 us or more of
overhead per kernel launch. In-order queues bypass that scheduler entirely, cutting latency to
10-50 us. Combining an in-order queue with the `AdaptiveCpp_coarse_grained_events` property
reduces event overhead further, at the cost of coarser synchronization granularity. This
combination is the right choice for tight dispatch loops:

```cpp
sycl::queue q{sycl::default_selector_v, 
    sycl::property_list{sycl::property::queue::in_order{},
                        sycl::property::queue::AdaptiveCpp_coarse_grained_events{}}};
```

## Profiling

Because AdaptiveCpp SSCP generates native PTX, amdgcn, and SPIR-V, vendor profiling tools
see AdaptiveCpp binaries as ordinary CUDA, HIP, or OpenCL applications. No special wrappers
or flags are required.

| Tool | Hardware | Command |
|------|----------|---------|
| NVIDIA Nsight Compute | NVIDIA GPU | `ncu -o results ./binary` |
| NVIDIA Nsight Systems | NVIDIA GPU | `nsys profile ./binary` |
| AMD rocprof | AMD GPU | `rocprof --stats ./binary` |
| Linux perf | CPU | `perf record -g ./binary` |

## The Bandwidth Benchmark Example

The `bandwidth_benchmark` example measures device memory bandwidth by performing a vector add
over three arrays of `N` floats (total data movement: `3 * N * 4` bytes) and reporting
throughput in GB/s. Because this workload performs very little arithmetic relative to the
data it touches, the result is a direct measure of memory bandwidth, not compute throughput.

Run it with different adaptivity levels to see JIT specialization at work:

```sh
pixi run configure && pixi run build
pixi run ./build/chapters/07-performance/examples/bandwidth_benchmark
ACPP_ADAPTIVITY_LEVEL=2 pixi run ./build/chapters/07-performance/examples/bandwidth_benchmark
```

On the second command, run the binary two or three times until the JIT converges.

## Summary

- Always compile with `-O3` for production code
- The JIT cache at `~/.acpp/apps/` eliminates first-run overhead on subsequent executions
- `ACPP_ADAPTIVITY_LEVEL=2` provides the best performance for kernels with invariant arguments
- Memory coalescing is critical: consecutive threads should access consecutive memory addresses
- Device USM provides lower overhead than buffers for performance-critical, throughput-sensitive code
- Work group sizes should be multiples of the warp or wavefront size (32/64)
- In-order queues with coarse-grained events minimize kernel launch latency
- Vendor profilers work directly with AdaptiveCpp SSCP binaries

---

[<- Chapter 06: AdaptiveCpp Extensions](../06-acpp-extensions/README.md) | [Up to Guide](../../README.md) | [Chapter 08: Footguns ->](../08-footguns/README.md)
