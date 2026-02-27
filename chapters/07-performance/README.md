# Chapter 07: Performance

AdaptiveCpp SSCP compiles to native PTX/amdgcn/SPIR-V - vendor profilers work directly. Key levers: JIT adaptivity, kernel cache, memory access patterns, compiler flags.

## The Most Important Flag: -O3

> [!CAUTION]
> AdaptiveCpp does NOT apply -O3 by default, unlike nvcc (-O2) or hipcc (-O3). Always compile with -O3 for production.

Note that this guide's CMakeLists use add_acpp_example which doesn't set -O3 by default. Add -O3 via: cmake ... -DCMAKE_CXX_FLAGS=-O3

Additional recommended flags: -ffp-contract=fast, -march=native (for CPU paths)

## JIT Compilation and the Kernel Cache

SSCP two-stage: IR embedded at compile time, lowered to PTX/amdgcn/SPIR-V at first run

Cache at ~/.acpp/apps/ (override: ACPP_APPDB_DIR)

First run: ~100-300ms overhead; subsequent runs: cache hit, negligible JIT cost

When done developing: run app 2-3 times until 'new binaries being JIT-compiled' warning disappears

Clear cache after upgrades: rm -rf ~/.acpp/apps/*

## ACPP_ADAPTIVITY_LEVEL

| Level | Behavior | Notes |
|-------|----------|-------|
| 0 | disabled, no JIT recompile | Fastest startup, no optimization |
| 1 | default, basic, peak by run 2 | Good balance of startup and performance |
| 2 | aggressive IADS, peak by run 3-4 | Best performance for invariant arguments |
| 3+ | not yet implemented | Reserved for future use |

IADS = Invariant Argument Detection and Specialization: tracks args that stay constant across calls and hard-codes them as JIT constants

Set via env: export ACPP_ADAPTIVITY_LEVEL=2

> [!NOTE]
> Level 2 can yield 10-30% speedup for kernels with invariant arguments like sizes or strides. Run the application 3-4 times to reach convergence.

## Memory Access Patterns

### Coalesced Access

NVIDIA warp=32 threads, AMD wavefront=64; consecutive threads must access consecutive memory addresses for max bandwidth. Non-coalesced = 32x bandwidth penalty on NVIDIA.

Good pattern (linear idx):
```cpp
c[i] = a[i] + b[i];  // Consecutive threads access consecutive addresses
```

Bad pattern (stride-32 idx):
```cpp
c[i] = a[i*32] + b[i*32];  // Each thread accesses memory 32 elements apart
```

### Local Memory Bank Conflicts

32 banks on NVIDIA, 64 on AMD; avoid stride-2+ patterns in local memory access

### USM vs Buffers for Performance

USM > Buffers for performance; device USM is lowest overhead; buffers have runtime DAG scheduling overhead. Note this guide teaches buffers first for clarity but for maximum throughput use device USM with explicit memcpy.

## Work Group Sizing

Scoped parallelism lets the backend choose physical group size (portable)

For nd_range: multiples of warp size (32 NVIDIA, 64 AMD); typical sweet spot 128-256 on modern GPUs; larger groups enable more latency hiding

> [!NOTE]
> When using nd_range, prefer group sizes that are multiples of the hardware warp/wavefront: 64, 128, 256, 512.

## Kernel Launch Latency

In-order queue bypasses DAG scheduler: lower latency (10-50 us vs 100+ us)

Combine with coarse-grained events for minimum overhead: sycl::property::queue::in_order{} + sycl::property::queue::AdaptiveCpp_coarse_grained_events{}

Code pattern for latency-sensitive loop:
```cpp
sycl::queue q{sycl::default_selector_v, 
    sycl::property_list{sycl::property::queue::in_order{},
                        sycl::property::queue::AdaptiveCpp_coarse_grained_events{}}};
```

## Profiling

AdaptiveCpp binaries appear as native CUDA/HIP/OpenCL to vendor tools

| Tool | Hardware | Command |
|------|----------|---------|
| NVIDIA Nsight Compute | NVIDIA GPU | ncu -o results ./binary |
| NVIDIA Nsight Systems | NVIDIA GPU | nsys profile ./binary |
| AMD rocprof | AMD GPU | rocprof --stats ./binary |
| Linux perf | CPU | perf record -g ./binary |

## The Bandwidth Benchmark Example

Explains what bandwidth_benchmark measures: vector add (3 arrays * N * 4 bytes), reports GB/s

How to interpret: memory bandwidth limited, not compute limited

Show how to run with different ACPP_ADAPTIVITY_LEVEL settings

## Building and Running

```
pixi run configure && pixi run build
pixi run ./build/chapters/07-performance/examples/bandwidth_benchmark
ACPP_ADAPTIVITY_LEVEL=2 pixi run ./build/chapters/07-performance/examples/bandwidth_benchmark
```

## Summary

- Always compile with -O3 for production code
- JIT cache at ~/.acpp/apps/ reduces startup overhead after first run
- ACPP_ADAPTIVITY_LEVEL=2 provides best performance for kernels with invariant arguments
- Memory coalescing is critical: consecutive threads should access consecutive memory addresses
- USM provides lower overhead than buffers for performance-critical code
- Work group sizes should be multiples of warp/wavefront size (32/64)
- In-order queues with coarse-grained events minimize kernel launch latency
- Vendor profilers work directly with AdaptiveCpp SSCP binaries

---

[<- Chapter 06: AdaptiveCpp Extensions](../06-acpp-extensions/README.md) | [Up to Guide](../../README.md) | [Chapter 08: Footguns ->](../08-footguns/README.md)