# Chapter 08: Common Pitfalls and Footguns

AdaptiveCpp SYCL is a powerful and portable programming model, but it has several sharp edges that are not obvious from the documentation. This chapter documents the most common mistakes and how to avoid them.

---

## 1. Forgetting -O3

**The Pitfall:** AdaptiveCpp does not apply any optimization flags by default. Your kernel runs unoptimized.

```bash
# BAD: compiles with -O0 (default, like g++)
acpp --acpp-targets=generic -o my_app my_app.cpp

# GOOD: always add -O3
acpp --acpp-targets=generic -O3 -o my_app my_app.cpp
```

**Why it matters:** Forgetting `-O3` can make kernels 5-20x slower than expected. This is especially misleading when comparing against CUDA (`nvcc` defaults to `-O2`) or HIP (`hipcc` defaults to `-O3`). If you benchmark AdaptiveCpp without `-O3` against CUDA with default flags, the comparison is unfair.

**Fix:** Always compile with `-O3`. When using CMake:

```cmake
cmake -S . -B build -DCMAKE_CXX_FLAGS=-O3 -DACPP_TARGETS=generic
```

> [!CAUTION]
> This is the single most common source of "AdaptiveCpp is slow" reports. Verify your compilation flags before drawing conclusions about performance.

---

## 2. Raw `sycl::buffer` Constructors - Silent Writeback and Blocking Destructor

**The Pitfall:** The standard `sycl::buffer` constructor writes back data to the host pointer on destruction, and the destructor blocks until all queued work completes.

```cpp
// BAD: silent writeback, blocking destructor
float* host_ptr = new float[1024];
{
    sycl::buffer<float, 1> buf(host_ptr, sycl::range<1>{1024});
    // submit kernels...
} // destructor blocks here AND silently modifies host_ptr - surprising!
```

**Why it matters:** The blocking destructor can introduce unexpected stalls in your application. The silent writeback means reading `host_ptr` after the scope exit gives modified data - but only sometimes, depending on whether the `buffer` was given a host pointer. This inconsistency confuses beginners who expect `host_ptr` to be unchanged.

**Fix:** Use AdaptiveCpp's explicit buffer policy factory methods. The writeback behavior is encoded in the factory function name:

```cpp
// Explicit: internal storage, no writeback
auto buf = sycl::make_sync_buffer<float>(sycl::range<1>{1024});

// Explicit: views external ptr, NO writeback on destruction
auto view = sycl::make_sync_view(host_ptr, sycl::range<1>{1024});

// Explicit: views external ptr, writeback on destruction (sync)
auto wbv = sycl::make_sync_writeback_view(host_ptr, sycl::range<1>{1024});
```

> [!WARNING]
> The standard `sycl::buffer` constructor is shown once in this chapter as an anti-pattern. It is never used in the rest of this guide for this reason.

---

## 3. Verifying Host Data Before Buffer Destruction

**The Pitfall:** Reading host data after `q.wait()` but before the buffer goes out of scope returns the original (pre-kernel) values.

```cpp
std::vector<float> result(N);
auto buf = sycl::make_async_writeback_view(result.data(), sycl::range<1>{N}, q);
q.submit(/* kernel that writes to buf */);
q.wait();

// BUG: result[] still has original values here!
// Writeback hasn't happened because buf is still alive.
if (result[0] != expected) { /* always false */ }
```

**Why it matters:** The kernel has completed, so `q.wait()` returns immediately. But `make_async_writeback_view` only copies data back to `result.data()` when the buffer object itself is destroyed.

**Fix:** Wrap the buffer, submit, and wait in a nested scope. Verify after the scope ends:

```cpp
std::vector<float> result(N);
{
    auto buf = sycl::make_async_writeback_view(result.data(), sycl::range<1>{N}, q);
    q.submit(/* kernel */);
    q.wait();
} // buf destroyed here -> writeback to result.data() occurs

// NOW it is safe to read result[]
if (result[0] != expected) { /* correct check */ }
```

> [!NOTE]
> `make_sync_writeback_view` has the same behavior - writeback occurs when the buffer object is destroyed. The `sync`/`async` distinction refers to whether destruction itself blocks (`sync` = blocking) or the writeback is queued asynchronously (`async` = non-blocking destruction).

---

## 4. Shared USM on AMD Without XNACK

**The Pitfall:** Using `sycl::malloc_shared` on AMD GPUs that lack XNACK support causes severe performance degradation or runtime errors.

```cpp
// RISKY on AMD: depends on XNACK support
float* data = sycl::malloc_shared<float>(N, q);
```

**Why it matters:** Shared USM relies on hardware page migration (the GPU can fault on unmapped pages and request migration from the CPU). On AMD GPUs without XNACK (the hardware mechanism that enables this), the driver must either disable shared USM entirely or fall back to a very slow emulation path.

**Check before using shared USM on AMD:**

```bash
rocminfo | grep xnack
# "xnack+" means XNACK is enabled (safe to use shared USM)
# "xnack-" means XNACK is disabled (avoid shared USM)
```

**Fix:** Use `sycl::malloc_device` with explicit `q.memcpy()` calls instead:

```cpp
float* d_data = sycl::malloc_device<float>(N, q);
q.memcpy(d_data, h_data, N * sizeof(float)).wait();
// use d_data in kernels
q.memcpy(h_data, d_data, N * sizeof(float)).wait();
sycl::free(d_data, q);
```

> [!WARNING]
> This affects consumer AMD GPUs (RDNA1/RDNA2 typically ship without XNACK). Data center AMD GPUs (MI series) typically have XNACK enabled.

---

## 5. Intel Discrete GPU + USM Memory Pool

**The Pitfall:** On Intel discrete GPUs, shared USM allocations migrate entire allocations at once (not page-by-page). Combined with AdaptiveCpp's default memory pool (40% of device global memory), this can cause performance cliffs when the pool fills.

**Fix:** Disable the memory pool for Intel discrete GPU workloads:

```bash
export ACPP_STDPAR_MEM_POOL_SIZE=0
```

Or use device USM with explicit transfers to avoid shared migration entirely.

---

## 6. JIT Cache Staleness After Upgrades

**The Pitfall:** After upgrading `acpp-toolchain`, LLVM, or GPU drivers, cached kernel binaries in `~/.acpp/apps/` may be stale or incompatible.

**Symptoms:**
- Runtime errors immediately after an upgrade
- Performance regression after an upgrade
- Incorrect results that were correct before the upgrade

**Fix:** Clear the cache after any acpp-toolchain or driver upgrade:

```bash
rm -rf ~/.acpp/apps/*
```

Then re-run your application once to repopulate the cache with fresh binaries.

> [!CAUTION]
> The cache location can be overridden with `ACPP_APPDB_DIR`. If you use a non-default location, clear that directory instead.

---

## 7. Dynamic Functions Require `SYCL_EXTERNAL`

**The Pitfall:** Functions used as dynamic function definitions (via `ACPP_EXT_DYNAMIC_FUNCTIONS`) must be marked `SYCL_EXTERNAL`. Without it, the compiler may inline or discard them before runtime dispatch can occur.

```cpp
// BAD: compiler may inline or eliminate this
void my_operation(float* data, sycl::item<1> idx) {
    data[idx.get(0)] *= 2.0f;
}

// GOOD: SYCL_EXTERNAL prevents unwanted inlining/elimination
SYCL_EXTERNAL void my_operation(float* data, sycl::item<1> idx) {
    data[idx.get(0)] *= 2.0f;
}
```

**Additionally:** Default implementations (the placeholder that gets replaced) should use `__attribute__((noinline))` to prevent the compiler from inlining them before the dynamic dispatch can substitute them:

```cpp
__attribute__((noinline))
void placeholder_op(float* data, sycl::item<1> idx) {
    sycl::AdaptiveCpp_jit::arguments_are_used(data, idx); // prevent DCE
}
```

---

## 8. Scoped Parallelism Requires All Work Items to Reach Each Synchronization Point

**The Pitfall:** In scoped parallelism, every work item in a group must collectively reach every `distribute_items_and_wait()` or `group_barrier()` call. Diverging control flow that causes some items to skip a barrier results in undefined behavior.

```cpp
// BAD: some items may skip the barrier
distribute_items(group, [&](auto item) {
    if (item.get_local_id(group, 0) < 16) {
        // Only 16 items do this...
        scratch[item.get_local_id(group, 0)] = compute(item);
    }
});
// THIS IS WRONG if the if-branch contains a barrier
group_barrier(group); // Some items skipped the branch above - undefined behavior!
```

**Fix:** Ensure all items always reach every barrier. Use the condition inside the branch body, not around the barrier:

```cpp
// GOOD: all items reach the barrier
distribute_items(group, [&](auto item) {
    auto lid = item.get_local_id(group, 0);
    scratch[lid] = (lid < 16) ? compute(item) : 0.0f; // all items participate
});
group_barrier(group); // all items reach this - correct
```

> [!WARNING]
> This is a fundamental constraint of GPU barrier semantics and applies equally to `nd_range` barriers (`sycl::group_barrier(ndi.get_group())`). It is not specific to AdaptiveCpp.

---

## 9. ACPP_ADAPTIVITY_LEVEL Needs Multiple Runs

**The Pitfall:** Running a benchmark once and concluding "AdaptiveCpp is slow" when `ACPP_ADAPTIVITY_LEVEL=1` (default) is in effect.

```bash
# Misleading: first run compiles kernels, includes JIT time
time ./my_benchmark   # "slow" 

# Fair: second run uses cached kernels
time ./my_benchmark   # "fast"
```

**Why it matters:** At level 1, the first invocation compiles kernels to the JIT cache. Subsequent invocations load from the cache. If you benchmark only the first run, you are measuring compilation overhead, not steady-state performance.

For level 2 (aggressive invariant argument detection), the runtime needs 3-4 runs to converge on the best specialization.

**Fix:** Always benchmark steady-state (after the "new binaries being JIT-compiled" warning disappears):

```bash
export ACPP_ADAPTIVITY_LEVEL=2

./my_benchmark   # Run 1: compiles, slower
./my_benchmark   # Run 2: begins specializing
./my_benchmark   # Run 3: approaching peak
./my_benchmark   # Run 4: peak performance (no more JIT warning)
```

---

## 10. `libc++` + CUDA Backend Not Supported

**The Pitfall:** Attempting to compile AdaptiveCpp code for NVIDIA CUDA backends with `libc++` (LLVM's standard library) instead of `libstdc++` (GCC's standard library).

**Symptoms:** Link errors involving `std::__1::` symbols (libc++ ABI) mixed with `libstdc++` symbols from the CUDA runtime.

**Fix:** Use `libstdc++` when targeting CUDA:

```bash
acpp --acpp-targets=generic -stdlib=libstdc++ -o app app.cpp
```

This is rarely an issue in practice on Linux since `libstdc++` is the default, but it can surface when building in environments (e.g., some conda setups) that default to `libc++`.

---

## 11. `nd_range` Barriers on CPU: High Overhead via Fibers

**The Pitfall:** Using `nd_range` with `group_barrier` on the CPU backend. AdaptiveCpp implements CPU barriers via cooperative fibers, which have significant context-switch overhead.

```cpp
// Works on GPU, but slow on CPU
q.parallel_for(sycl::nd_range<1>{N, group_size}, [=](sycl::nd_item<1> ndi) {
    // ...
    sycl::group_barrier(ndi.get_group()); // -> fiber context switch on CPU
    // ...
});
```

**Why it matters:** The fiber context-switch overhead can be 100-1000x higher than a GPU warp barrier. For CPU-heavy workloads this makes `nd_range` + barrier essentially unusable.

**Fix:** Use scoped parallelism instead for CPU-friendly barriers. Note that scoped parallelism is not yet supported in `--acpp-targets=generic` (SSCP) mode - this is a known limitation tracked in [AdaptiveCpp issue #1417](https://github.com/AdaptiveCpp/AdaptiveCpp/issues/1417). Until SSCP support lands, work around with `nd_range` for GPU and keep CPU paths barrier-free where possible.

> [!NOTE]
> When scoped parallelism SSCP support arrives, it will be the recommended path for all CPU and GPU work, automatically avoiding fiber overhead on CPU backends.

---

## 12. Local Accessor Kernel Execution Failure on CUDA

**The Pitfall:** Using `sycl::local_accessor` for shared memory can cause kernels to silently fail to execute on NVIDIA CUDA backends. No error or exception is raised - the kernel submission appears to succeed but the kernel never launches.

**Why it matters:** Silent non-execution is one of the hardest bugs to diagnose. Your output arrays contain their original (uninitialized or zero) values with no indication of what went wrong.

**Diagnosis:** Use `cuda-gdb` or NVIDIA Nsight to verify the kernel actually launches. If using `cuda-gdb`, set a breakpoint at the kernel entry point and check whether it is ever hit.

**Status:** This was a confirmed bug in AdaptiveCpp (GitHub issue [#1434](https://github.com/AdaptiveCpp/AdaptiveCpp/issues/1434)), fixed in recent versions. Ensure you are on `acpp-toolchain >= 25.10.0`.

**Fix:** Upgrade to a fixed version. As a diagnostic, temporarily replace `sycl::local_accessor` with global memory to confirm whether the issue is local accessor specific.

> [!WARNING]
> Always verify that kernels using local/shared memory actually execute and produce results. A quick sanity check (e.g., writing a sentinel value to one output element and checking it) catches silent non-execution early.

---

## 13. Level Zero Backend: Buffer Destruction Does Not Guarantee Writeback

**The Pitfall:** On Intel Level Zero backends, buffer destruction does NOT automatically synchronize pending operations. Modified data may not be written back to the host pointer after the buffer goes out of scope.

```cpp
std::vector<float> result(N);
{
    auto buf = sycl::make_async_writeback_view(result.data(), sycl::range<1>{N}, q);
    q.submit(/* kernel */);
    // NOTE: q.wait() is missing here
} // On Level Zero: data may NOT be written back to result.data()!
```

**Fix:** Always call `q.wait()` before the buffer is destroyed, regardless of backend. This is good practice in general and is mandatory on Level Zero:

```cpp
{
    auto buf = sycl::make_async_writeback_view(result.data(), sycl::range<1>{N}, q);
    q.submit(/* kernel */);
    q.wait(); // required before scope exit on Level Zero
} // Now writeback is guaranteed
```

> [!CAUTION]
> This is a known open bug on the Level Zero backend (GitHub issue [#1799](https://github.com/AdaptiveCpp/AdaptiveCpp/issues/1799)). Defensive coding (always calling `q.wait()` before buffer destruction) protects against this on all backends.

---

## 14. Scoped Parallelism: Collective Calls Cannot Be Conditional

**The Pitfall:** Scoped parallelism functions (`distribute_items()`, `distribute_groups()`, `single_item()`, `group_barrier()`) are **collective** - every physical work item in the group must reach every call site. Wrapping them in a conditional causes deadlock or undefined behavior.

```cpp
// BAD: only the group leader reaches distribute_items - deadlock
sycl::distribute_groups(q_handler, num_groups, [&](auto group) {
    if (group.leader()) {                          // WRONG
        sycl::distribute_items(group, [&](auto item) {
            // Only leader calls this - other items are stuck
        });
    }
});
```

**Why it matters:** The `group.leader()` pattern is common in standard SYCL `nd_range` code for single-item operations. It is not safe to use before scoped parallelism collective calls.

**Fix:** Use `single_item()` for leader-only work, which is itself a collective that all items participate in:

```cpp
// GOOD: all items reach single_item(), but only leader executes the body
sycl::distribute_groups(q_handler, num_groups, [&](auto group) {
    sycl::distribute_items(group, [&](auto item) {
        // all items do work
    });
    sycl::single_item(group, [&]() {
        // only group leader executes this - but all items reach the call
    });
});
```

---

## 15. Scoped Parallelism: No Collective Calls Inside `distribute_items()`

**The Pitfall:** You cannot call `distribute_items()`, `distribute_groups()`, `single_item()`, or `group_barrier()` from within a `distribute_items()` lambda. These are top-level collective operations, not nestable.

```cpp
// BAD: calling group_barrier() inside distribute_items() is undefined
sycl::distribute_items(group, [&](sycl::s_item<1> idx) {
    scratch[idx.get_local_id(group, 0)] = input[idx];
    sycl::group_barrier(group);    // UNDEFINED BEHAVIOR
    output[idx] = scratch[...];    // may see garbage
});
```

**Fix:** Use `distribute_items_and_wait()` for the first phase, then continue at the outer group level:

```cpp
// GOOD: barrier is implicit in distribute_items_and_wait
sycl::distribute_items_and_wait(group, [&](sycl::s_item<1> idx) {
    scratch[idx.get_local_id(group, 0)] = input[idx];
    // implicit barrier here - all items finish before continuing
});
// Now at group scope: safe to read scratch written by all items
sycl::distribute_items(group, [&](sycl::s_item<1> idx) {
    output[idx] = scratch[...];
});
```

> [!WARNING]
> The nesting rules for scoped parallelism are strict: collective calls must appear at the group scope level, never inside `distribute_items()` lambdas.

---

## 16. Buffer-USM Interop: Wrong Data State Hint Causes Silent Migration

**The Pitfall:** When constructing buffers on top of USM pointers using `ACPP_EXT_BUFFER_USM_INTEROP`, the choice between `buffer_allocation::empty_view()` and `buffer_allocation::view()` affects whether the runtime thinks the data is current or stale.

```cpp
float* d_ptr = sycl::malloc_device<float>(N, q);
// ... fill d_ptr with kernel results ...

// BAD: empty_view tells the runtime the data is EMPTY/INVALID
// Runtime will NOT use d_ptr's existing contents as a data source
auto alloc = hipsycl::sycl::buffer_allocation::empty_view(d_ptr, q.get_device());

// GOOD: view tells the runtime the data IS CURRENT
auto alloc = hipsycl::sycl::buffer_allocation::view(d_ptr, q.get_device());
```

**Why it matters:** Using `empty_view()` when data is already current causes the runtime to treat `d_ptr` as uninitialized, potentially overwriting valid results or triggering unnecessary data migrations.

**Fix:** Use `buffer_allocation::view()` when the USM pointer already holds valid data you want the buffer system to read. Use `buffer_allocation::empty_view()` only when the allocation is logically uninitialized (e.g., freshly allocated output buffers).

---

## Atomics Footguns

### 17. `seq_cst` on Device Scope

**The Pitfall:** Using `sycl::memory_order::seq_cst` with `sycl::memory_scope::device` forces a global synchronization point on most GPU backends. The operation is correct but serializes all GPU memory traffic.

```cpp
// BAD: seq_cst at device scope - correct but very expensive on GPU
sycl::atomic_ref<int, sycl::memory_order::seq_cst, sycl::memory_scope::device,
                 sycl::access::address_space::global_space> ref{counter};
ref.fetch_add(1);

// GOOD: acq_rel at device scope - sufficient for almost all use cases
sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                 sycl::access::address_space::global_space> ref{counter};
ref.fetch_add(1);
```

**Why it matters:** `seq_cst` implies a global memory order visible to all threads across all work groups simultaneously. On PTX (NVIDIA) and amdgcn (AMD), this translates to a fence that stalls the entire GPU pipeline. `acq_rel` with the appropriate scope achieves the same correctness guarantee at a fraction of the cost for the vast majority of patterns.

**Fix:** Use `memory_order::acq_rel` with an explicit scope. Reserve `seq_cst` only for the rare case where you need total global order across all devices.

---

### 18. 64-bit Atomics Without `atomic64` Aspect Check

**The Pitfall:** Using `sycl::atomic_ref` with `long`, `unsigned long long`, or `double` on a device that does not support 64-bit atomics produces undefined behavior. No exception is raised.

```cpp
// BAD: assumes atomic64 is available - undefined behavior if not
sycl::atomic_ref<long, sycl::memory_order::relaxed, sycl::memory_scope::device,
                 sycl::access::address_space::global_space> ref{val};

// GOOD: check first
if (!q.get_device().has(sycl::aspect::atomic64)) {
    // Fall back to int or use a software reduction
}
```

**Why it matters:** Most modern discrete GPUs support `atomic64`, but integrated GPUs (Intel, ARM Mali) and some older hardware do not. Skipping the check is a portability bug that silently produces wrong results.

**Fix:** Always check `device.has(sycl::aspect::atomic64)` before using 64-bit types in `atomic_ref`.

---

### 19. Wrong Scope for Cross-Group Synchronization

**The Pitfall:** Using `memory_scope::work_group` when you need to synchronize across multiple work groups. An atomic at `work_group` scope only guarantees ordering within a single work group - other work groups may observe a different order.

```cpp
// BAD: work_group scope for a counter incremented by all work groups
sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                 sycl::access::address_space::global_space> ref{global_counter};
ref.fetch_add(1); // no ordering guarantee across groups

// GOOD: device scope for global coordination
sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                 sycl::access::address_space::global_space> ref{global_counter};
ref.fetch_add(1);
```

**Why it matters:** This is a silent correctness bug. The code compiles and runs without error; it just gives wrong answers intermittently.

**Fix:** Use `memory_scope::device` for any atomic that multiple work groups access.

---

### 20. `ACPP_EXT_FP_ATOMICS` Silent Emulation

**The Pitfall:** Defining `ACPP_EXT_FP_ATOMICS` and using `float` or `double` `fetch_add` on a backend that does not have native hardware floating-point atomic support. AdaptiveCpp silently emulates the operation in software, which can be many times slower than expected.

```cpp
// This compiles and runs on all backends - but may be emulated
#define ACPP_EXT_FP_ATOMICS
#include <sycl/sycl.hpp>

sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                 sycl::access::address_space::global_space> ref{result};
ref.fetch_add(val); // hardware on NVIDIA/AMD; emulated on CPU, Intel iGPU
```

**Why it matters:** Silent emulation can turn an expected O(1) hardware instruction into a slow CAS retry loop, degrading throughput by orders of magnitude without any diagnostic.

**Fix:** Guard `ACPP_EXT_FP_ATOMICS` usage with a `#ifdef` and always provide a CAS-loop fallback. Profile on each target you care about.

---

### 21. Double-Fencing Around `group_barrier`

**The Pitfall:** Manually calling `sycl::atomic_fence` immediately before or after `sycl::group_barrier`. The barrier already acts as an `acq_rel` fence at `work_group` scope - the manual fence is redundant and clutters the code.

```cpp
// BAD: redundant fence
sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::work_group);
sycl::group_barrier(g); // already provides acq_rel fence

// GOOD: barrier is sufficient
sycl::group_barrier(g);
```

**Why it matters:** While functionally harmless, the redundant fence confuses readers into thinking the barrier alone is insufficient, and may add measurable latency on some backends.

**Fix:** Use `sycl::group_barrier` alone. Reserve `sycl::atomic_fence` for patterns where you need a fence **without** a barrier (producer/consumer with flags, as shown in Chapter 10).

---

### 22. `memory_scope::sub_group` on CPU and Some Intel GPUs

**The Pitfall:** Using `memory_scope::sub_group` on a CPU backend or on Intel integrated GPUs where the scope is not supported. The behavior is implementation-defined when an unsupported scope is used.

```cpp
// BAD: sub_group scope assumed to work everywhere
sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::sub_group,
                 sycl::access::address_space::global_space> ref{val};

// GOOD: check at runtime
auto caps = ctx.get_info<sycl::info::context::atomic_memory_scope_capabilities>();
bool has_sub_group = std::find(caps.begin(), caps.end(),
    sycl::memory_scope::sub_group) != caps.end();
```

**Why it matters:** CPUs have no hardware sub-group concept. AdaptiveCpp's LLVM JIT backend does not support `sub_group` scope. Intel integrated GPUs vary by generation.

**Fix:** Always query `sycl::info::context::atomic_memory_scope_capabilities` before using `sub_group` or `system` scope in portable code. See [Chapter 10: Atomics and Memory Ordering](../10-atomics/README.md) for the full scope support matrix.

---

### 23. `compare_exchange_weak` Without a Retry Loop

**The Pitfall:** Using `compare_exchange_weak` as a one-shot operation. Unlike `compare_exchange_strong`, the weak variant is permitted to fail spuriously even when the expected value matches. Without a retry loop, the CAS silently does nothing on a spurious failure.

```cpp
// BAD: no retry loop - silently fails on spurious failure
float expected = 0.0f;
sycl::atomic_ref<float, ...> ref{val};
ref.compare_exchange_weak(expected, new_val); // may do nothing!

// GOOD: retry loop for weak CAS
float expected = ref.load(sycl::memory_order::relaxed);
float desired;
do {
    desired = compute_new_value(expected);
} while (!ref.compare_exchange_weak(expected, desired));
// compare_exchange_weak updates 'expected' on failure - retry is correct
```

**Why it matters:** `compare_exchange_weak` is valid only inside a loop that retries on failure. A one-shot use is a latent correctness bug that appears to work on most hardware (spurious failures are rare in practice on x86/PTX) but can fail on ARM or when under high contention.

**Fix:** Use `compare_exchange_strong` for one-shot operations. Use `compare_exchange_weak` only inside a do-while retry loop, as shown in the [Chapter 10 compare_exchange example](../10-atomics/examples/compare_exchange.cpp).

---

## Summary

| # | Footgun | Fix |
|---|---------|-----|
| 1 | No -O3 by default | Always compile with `acpp -O3` or set `-DCMAKE_CXX_FLAGS=-O3` |
| 2 | Raw `sycl::buffer` writeback | Use AdaptiveCpp buffer factory methods |
| 3 | Verify before buffer destruction | Wrap buffer+kernel+wait in scope `{}`, verify after |
| 4 | Shared USM on AMD without XNACK | Use device USM + explicit memcpy; check `rocminfo \| grep xnack` |
| 5 | Intel discrete GPU USM pool | Set `ACPP_STDPAR_MEM_POOL_SIZE=0` |
| 6 | Stale JIT cache | `rm -rf ~/.acpp/apps/*` after upgrades |
| 7 | Dynamic functions need `SYCL_EXTERNAL` | Mark functions `SYCL_EXTERNAL` + use `__attribute__((noinline))` for placeholders |
| 8 | Scoped parallelism barrier divergence | All items must reach every barrier; no conditional barriers |
| 9 | Benchmark on first run only | Warm up cache; use `ACPP_ADAPTIVITY_LEVEL=2`, run 4 times |
| 10 | libc++ + CUDA backend | Use `-stdlib=libstdc++` |
| 11 | nd_range barriers on CPU | Avoid barriers on CPU; use scoped parallelism (when SSCP support lands) |
| 12 | Local accessor silent kernel failure | Upgrade to acpp-toolchain >= 25.10.0; verify kernel executes |
| 13 | Level Zero buffer writeback bug | Always call `q.wait()` before buffer destruction |
| 14 | Scoped parallelism: conditional collectives | Never wrap collective calls in conditionals; use `single_item()` |
| 15 | Scoped parallelism: nesting inside distribute_items | No collective calls inside `distribute_items()`; use `_and_wait` variants |
| 16 | Buffer-USM interop data state | Use `view()` for current data, `empty_view()` for uninitialized |
| 17 | `seq_cst` on device scope | Use `acq_rel` with explicit scope instead |
| 18 | 64-bit atomics without `atomic64` aspect | Check `device.has(sycl::aspect::atomic64)` before using `long`/`double` |
| 19 | Wrong scope for cross-group sync | Use `memory_scope::device` for atomics accessed by multiple work groups |
| 20 | `ACPP_EXT_FP_ATOMICS` silent emulation | Guard with `#ifdef`; provide CAS-loop fallback; profile on each target |
| 21 | Double-fencing around `group_barrier` | Use `group_barrier` alone; `atomic_fence` only where no barrier exists |
| 22 | `sub_group` scope on CPU/Intel iGPU | Query `atomic_memory_scope_capabilities` before using `sub_group` scope |
| 23 | `compare_exchange_weak` without retry | Use `strong` for one-shot; use `weak` only inside a do-while retry loop |

---

[<- Chapter 07: Performance](../07-performance/README.md) | [Up to Guide](../../README.md) | [Chapter 09: Real-World Patterns ->](../09-real-world-patterns/README.md)
