# Chapter 10: Atomics and Memory Ordering

## Why Atomics?

When multiple work-items access the same memory location without proper synchronization, data races occur. A classic example is two work-items incrementing a counter without atomics:

```cpp
// Both work-items read count = 5
item1: count = count + 1;  // Writes 6
item2: count = count + 1;  // Writes 6 (lost increment!)
```

The result is 6 instead of the expected 7. Atomics provide indivisible read-modify-write operations that prevent such race conditions.

> [!NOTE]
> Atomics coordinate data access between work-items, while barriers coordinate control flow. They solve different problems: atomics prevent data races, barriers ensure all work-items reach a point before proceeding.

> [!IMPORTANT]
> `sycl::atomic<T>` is deprecated in SYCL 2020. Always use `sycl::atomic_ref` which wraps existing variables rather than allocating separate atomic storage.

## sycl::atomic_ref

`atomic_ref` creates an atomic view of an existing variable:

```cpp
template <typename T,
          memory_order DefaultOrder,
          memory_scope DefaultScope,
          access::address_space AddressSpace>
class atomic_ref;
```

Construction example:
```cpp
int counter = 0;
atomic_ref<int, memory_order::relaxed, memory_scope::device, 
           access::address_space::global_space> ref{counter};
ref.fetch_add(1);  // Atomic increment
```

Supported types:
- Integral types: `int`, `unsigned`, `long`, etc.
- Floating-point types: `float`, `double` (with `ACPP_EXT_FP_ATOMICS`)
- 64-bit types require device `atomic64` aspect support

## Memory Orders

Memory orders define how atomic operations synchronize with other operations:

| Order | Description | When to Use |
|-------|-------------|-------------|
| `relaxed` | No ordering guarantees, only atomicity | Simple counters where order doesn't matter |
| `acquire` | Prevents subsequent memory ops from moving before | Reading flags/locks |
| `release` | Prevents preceding memory ops from moving after | Writing flags/locks |
| `acq_rel` | Combines acquire and release semantics | Read-modify-write operations |
| `seq_cst` | Sequentially consistent - total ordering | Default choice, but may be expensive |

> [!CAUTION]
> `seq_cst` can force global GPU synchronization and significantly impact performance. Prefer `acq_rel` for most read-modify-write operations.

## Memory Scopes

Memory scopes define which work-items see the effects of an atomic operation:

| Scope | Synchronizes | Typical Use |
|-------|--------------|-------------|
| `work_item` | Only the executing work-item | Rarely useful |
| `sub_group` | Work-items in the same sub-group | Sub-group collective operations |
| `work_group` | Work-items in the same work-group | Work-group synchronization |
| `device` | All work-items on the device | Most common choice |
| `system` | Across multiple devices | Multi-device coordination |

> [!WARNING]
- `sub_group` scope is unsupported on CPU backends and behavior varies on Intel integrated GPUs
- `system` scope is not supported on NVIDIA or AMD GPU backends
- In production code, check `sycl::info::context::atomic_memory_scope_capabilities` before using specific scopes

## Operations Reference

Core atomic operations:

- `load(order)` - Atomically read value
- `store(value, order)` - Atomically write value
- `exchange(value, order)` - Atomically replace value and return old
- `compare_exchange_weak/strong(expected, desired, order)` - CAS operation
  - `weak` may spuriously fail (use in retry loops)
  - `strong` only fails if value actually changed (use for simple cases)
- `fetch_add(value, order)` - Add and return old value (integral only)
- `fetch_sub(value, order)` - Subtract and return old value (integral only)
- `fetch_min/max(value, order)` - Update min/max and return old (integral only)
- `fetch_and/or/xor(value, order)` - Bitwise operations and return old (integral only)
- `is_lock_free()` - Check if operation is lock-free

## sycl::atomic_fence

Standalone fence without an atomic variable:

```cpp
atomic_fence(memory_order::order, memory_scope::scope);
```

Used in producer/consumer patterns:
```cpp
// Producer
data[id] = value;
atomic_fence(memory_order::release, memory_scope::device);
flag[id] = 1;

// Consumer
while (flag[id] == 0) { /* spin */ }
atomic_fence(memory_order::acquire, memory_scope::device);
value = data[id];
```

> [!NOTE]
- `group_barrier` already acts as an `acq_rel` fence - don't double-fence
- Fences are rarely needed with `atomic_ref` operations that include ordering

## ACPP_EXT_FP_ATOMICS

AdaptiveCpp extension for floating-point atomics:

```cpp
#define ACPP_EXT_FP_ATOMICS
#include <sycl/sycl.hpp>
```

Enables:
- `fetch_add` for `float` and `double`
- Direct FP atomic operations on supported backends

> [!WARNING]
- On backends without native FP atomics, operations are emulated
- Use `#ifdef ACPP_EXT_FP_ATOMICS` guards for portable code
- CAS fallback pattern provides compatibility when extension unavailable

## SSCP/Generic Backend Notes

SSCP (Single Source C++ Compilation) with the generic target:

1. **Compilation**: C++ source parsed once, LLVM IR embedded in host binary
2. **Runtime**: IR lowered via JIT to PTX (NVIDIA), amdgcn (AMD), SPIR-V (Intel/CPU)

Memory orders and scopes are preserved through the JIT process. The same binary runs on any supported backend with appropriate atomic operations.

> [!IMPORTANT]
- 64-bit atomics require checking device `atomic64` aspect at runtime
- Use `device.has(aspect::atomic64)` before `long` or `unsigned long long` atomics
- Some backends may emulate unsupported atomic operations

## Examples

1. **atomic_counter.cpp** - Demonstrates race condition without atomics and correct counting with `atomic_ref`
2. **reduction_fetch_add.cpp** - Two-phase sum reduction using atomic `fetch_add` with integer types
3. **compare_exchange.cpp** - Atomic max for float using compare-exchange loop
4. **atomic_fence_ordering.cpp** - Producer/consumer pattern with release/acquire fences
5. **fp_atomics.cpp** - Floating-point atomics with `ACPP_EXT_FP_ATOMICS` and CAS fallback

## Summary

- Atomics prevent data races in concurrent memory access
- Use `atomic_ref` to wrap existing variables (avoid deprecated `atomic<T>`)
- Choose appropriate memory order: `relaxed` for simple counters, `acq_rel` for RMW operations
- Device scope is most common; check capabilities before using sub_group or system
- FP atomics require `ACPP_EXT_FP_ATOMICS` extension or CAS fallback
- SSCP preserves atomic semantics through JIT compilation to target backends
- Always check device aspects (like `atomic64`) before using 64-bit atomics

---

[<- Chapter 09: Real-World Patterns](../09-real-world-patterns/README.md) | [Up to Guide](../../README.md)