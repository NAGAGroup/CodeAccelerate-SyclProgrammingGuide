# Chapter 04: Memory Model

This chapter covers the two memory models available in SYCL and how they work in AdaptiveCpp: the buffer+accessor model and Unified Shared Memory (USM).

## Introduction

SYCL provides two distinct approaches to managing data between host and device:

1. **Buffer+Accessor Model**: High-level, automatic data management with opaque ownership
2. **USM (Unified Shared Memory)**: Low-level, explicit control with pointer-based access

Both models have their place in heterogeneous computing. The buffer model is generally safer for beginners, while USM provides the control needed for performance-critical applications and interoperability with existing codebases.

## The Buffer+Accessor Model

The buffer+accessor model is SYCL's original approach to memory management. It separates data ownership from data access:

- `sycl::buffer`: An opaque ownership object that manages data across host and device
- `sycl::accessor`: The mechanism to read/write buffer data inside command groups

### Buffers: Opaque Data Ownership

A `sycl::buffer` represents data that can be accessed by the device. The buffer owns the data and manages its lifetime, handling transfers between host and device automatically. Buffers are opaque - you cannot directly access their underlying storage from host code.

### Accessors: Controlled Data Access

Accessors provide controlled access to buffer data within command groups. They are the bridge between your kernel code and the buffer's data.

> [!IMPORTANT]
> Accessors MUST be created inside the command group lambda, not outside. The command group handler (`cgh`) is required to establish the data dependencies correctly.

```cpp
q.submit([&](sycl::handler& cgh) {
    // Correct: Create accessor inside command group
    auto acc = buf.get_access<sycl::access_mode::read_write>(cgh);
    
    cgh.parallel_for(/*...*/, [&](auto item) {
        // Use accessor in kernel
        acc[item] = /* ... */;
    });
});
```

## Buffer Factory Methods

AdaptiveCpp provides explicit buffer factory methods that make the behavior of buffers clear and predictable. These are the ONLY buffer constructors you should use in this guide.

> [!NOTE]
> Include the explicit buffer policies header and define the extension macro:
> ```cpp
> #include <hipSYCL/sycl/extensions/explicit_buffer_policies.hpp>
> #define ACPP_EXT_EXPLICIT_BUFFER_POLICIES
> ```

### Factory Method Overview

| Factory | Storage | Writeback | Destructor Blocks |
|---------|---------|-----------|-------------------|
| `make_sync_buffer<T>(range)` | internal | no | yes |
| `make_async_buffer<T>(range)` | internal | no | no |
| `make_sync_view<T>(ptr, range)` | external | no | yes |
| `make_async_view<T>(ptr, range)` | external | no | no |
| `make_sync_writeback_view<T>(ptr, range)` | external | yes | yes |
| `make_async_writeback_view<T>(ptr, range, q)` | external | yes | no |

### Understanding the Properties

The factory methods vary along three orthogonal properties:

1. **External Storage**: Views use existing host memory, buffers create internal storage
2. **Writes Back**: Writeback views copy data back to host memory on destruction
3. **Destructor Blocks**: Sync variants block the host until operations complete

### Example: Using Buffer Factory Methods

```cpp
#include <sycl/sycl.hpp>
#include <hipSYCL/sycl/extensions/explicit_buffer_policies.hpp>
#define ACPP_EXT_EXPLICIT_BUFFER_POLICIES

int main() {
    sycl::queue q;
    
    // Input: read-only view of existing host data
    std::vector<float> input(1024, 3.0f);
    auto input_buf = sycl::make_sync_view<const float>(
        input.data(), sycl::range<1>{1024});
    
    // Output: writeback view that updates host data
    std::vector<float> output(1024, 0.0f);
    {
        auto output_buf = sycl::make_sync_writeback_view<float>(
            output.data(), sycl::range<1>{1024});
        
        q.submit([&](sycl::handler& cgh) {
            auto in_acc = input_buf.get_access<sycl::access_mode::read>(cgh);
            auto out_acc = output_buf.get_access<sycl::access_mode::write>(cgh);
            
            cgh.parallel_for(sycl::range<1>{1024}, [&](auto item) {
                out_acc[item] = in_acc[item] * 2.0f;
            });
        });
    } // output_buf destructor writes back to output vector
    
    // output now contains the results
    return 0;
}
```

## Accessor Modes

Accessors support different access modes that control how data can be accessed:

| Mode | Description | Use Case |
|------|-------------|----------|
| `access_mode::read` | Read-only access | Input data that won't be modified |
| `access_mode::write` | Write-only access | Output buffers (no initial values needed) |
| `access_mode::read_write` | Read and write access | In-place operations or data that needs both read and write access |

> [!TIP]
> Use the most restrictive access mode that meets your needs. This enables better optimization and clearer code intent.

## USM (Unified Shared Memory)

USM provides a pointer-based approach to memory management that feels more like traditional C++ programming. USM allocations can be accessed from both host and device code using regular pointers.

### USM Allocation Types

| Type | Description | Access Pattern |
|------|-------------|----------------|
| `malloc_device<T>(count, q)` | Device-only memory | Host cannot access directly |
| `malloc_host<T>(count, q)` | Host-accessible memory | Slower device access |
| `malloc_shared<T>(count, q)` | Unified memory | Both host and device can access |

### USM Example

```cpp
#include <sycl/sycl.hpp>

int main() {
    sycl::queue q;
    const size_t N = 1024;
    
    // Allocate device memory
    float* a = sycl::malloc_device<float>(N, q);
    float* b = sycl::malloc_device<float>(N, q);
    float* c = sycl::malloc_device<float>(N, q);
    
    // Host staging buffers
    std::vector<float> h_a(N, 1.0f);
    std::vector<float> h_b(N, 2.0f);
    
    // Copy data to device
    q.memcpy(a, h_a.data(), N * sizeof(float));
    q.memcpy(b, h_b.data(), N * sizeof(float));
    q.wait(); // Ensure copies complete
    
    // Submit kernel
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> i) {
            c[i] = a[i] + b[i];
        });
    });
    
    // Copy result back
    std::vector<float> h_c(N);
    q.memcpy(h_c.data(), c, N * sizeof(float));
    q.wait();
    
    // Clean up
    sycl::free(a, q);
    sycl::free(b, q);
    sycl::free(c, q);
    
    return 0;
}
```

> [!WARNING]
> Shared USM on AMD without XNACK has severe performance issues. See Chapter 08 for details on this and other performance pitfalls.

## Buffers vs USM: When to Use Each

| Aspect | Buffers | USM |
|--------|---------|-----|
| **Data Management** | Automatic | Manual |
| **Safety** | High (no dangling pointers) | Lower (manual management) |
| **Learning Curve** | Easier for beginners | Steeper (requires explicit sync) |
| **Performance Control** | Limited | Full control |
| **Interop with C++** | Limited | Excellent (pointer-based) |
| **Existing Code Integration** | Difficult | Easy |

**Choose Buffers when:**
- You're new to SYCL/heterogeneous computing
- You want automatic data management
- Safety is more important than performance optimization
- You don't need to interoperate with existing pointer-based code

**Choose USM when:**
- You need explicit control over data transfers
- You're integrating with existing C++ codebases
- Performance optimization requires fine-grained control
- You need pointer-based access patterns

## Buffer-USM Interop

AdaptiveCpp provides extensions for interoperability between buffers and USM:

```cpp
#define ACPP_EXT_BUFFER_USM_INTEROP
#include <hipSYCL/sycl/extensions/buffer_usm_interop.hpp>

// Get USM pointer for a specific device
auto* ptr = buffer.get_pointer(device);

// Check if buffer has allocation for device
bool has_alloc = buffer.has_allocation(device);
```

This is an advanced topic covered in more detail in Chapter 06 (AdaptiveCpp Extensions).

## Summary and Next Steps

This chapter covered the two memory models in SYCL:

1. **Buffer+Accessor Model**: High-level, automatic data management with factory methods for explicit behavior
2. **USM**: Pointer-based memory management with explicit control

Key takeaways:
- Use buffer factory methods (`make_sync_buffer`, `make_async_view`, etc.) - never raw `sycl::buffer` constructors
- Accessors must be created inside command group lambdas
- USM provides more control but requires careful synchronization
- Choose the model that matches your experience and performance needs

Next, Chapter 05 covers kernels and parallelism, where we'll explore how to actually execute code on devices using the memory models we've learned about here.

> [!NOTE]
> This chapter covers memory **storage** - how data is allocated, transferred, and managed between host and device. Memory **ordering** - how atomic operations synchronize concurrent work-items, and how `sycl::atomic_fence` coordinates fine-grained access - is a separate concern covered in [Chapter 10: Atomics and Memory Ordering](../10-atomics/README.md). If you need cross-work-group synchronization or concurrent data access without barriers, read Chapter 10 after Chapter 05.

---

[<- Chapter 03: AdaptiveCpp Setup](../03-acpp-setup/README.md) | [Up to Guide](../../README.md) | [Chapter 05: Kernels and Parallelism ->](../05-kernels-and-parallelism/README.md)