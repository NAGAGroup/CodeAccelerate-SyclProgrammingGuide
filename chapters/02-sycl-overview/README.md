# SYCL Overview

SYCL is a C++ abstraction layer for heterogeneous computing, standardized by the Khronos Group. It provides a single-source programming model where the same C++ code can execute on host processors (CPU) and a wide range of accelerators (GPUs, FPGAs, etc.). AdaptiveCpp is one implementation of the SYCL 2020 specification, extending it with additional features and optimizations.

## The Core SYCL Objects

| Object | Description |
|--------|-------------|
| sycl::queue | Primary interface for submitting work to devices |
| sycl::device | Hardware abstraction for compute devices |
| sycl::platform | Group of devices from the same vendor/implementation |
| sycl::context | Execution context managing devices and memory |
| sycl::buffer | Data container managing host-device synchronization |
| sycl::accessor | Interface for accessing buffer data in kernels |
| sycl::handler | Command group builder for kernel submission |
| sycl::event | Handle for tracking asynchronous operations |

### sycl::queue

The queue is the primary interface for submitting work to SYCL devices. Key characteristics:

1. A queue is bound to a specific device
2. The default constructor selects the best available device
3. Device selectors include: gpu_selector_v, cpu_selector_v, default_selector_v
4. queue.submit() is how you submit work to the device
5. queue.wait() blocks until all submitted work completes

```cpp
#include <sycl/sycl.hpp>

int main() {
    // Create a queue bound to a GPU device
    sycl::queue q(sycl::gpu_selector_v);
    
    // Print the device name
    std::cout << "Device: " 
              << q.get_device().get_info<sycl::info::device::name>() 
              << std::endl;
    
    return 0;
}
```

### sycl::device and sycl::platform

Devices represent the actual hardware accelerators available in the system. A platform groups together devices from the same vendor or SYCL implementation. You can enumerate available devices:

```cpp
// Get all GPU devices from all platforms
auto gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);

for (const auto& device : gpu_devices) {
    std::cout << "GPU: " << device.get_info<sycl::info::device::name>() << std::endl;
}
```

## Buffers and Accessors: The SYCL Data Model

### Buffers

A buffer owns data and manages synchronization between host and device memory. The SYCL runtime decides WHEN to copy data to/from the device - you just declare your intent via accessors. In AdaptiveCpp, we use factory methods for explicit policy control:

```cpp
#include <hipsycl/sycl.hpp>

// Method 1: Explicit namespace
using namespace hipsycl::sycl;
float* host_data = new float[1024];
auto buf = make_sync_buffer<float>(host_data, sycl::range<1>(1024));

// Method 2: Fully qualified
auto buf2 = hipsycl::sycl::make_async_buffer<float>(host_data, sycl::range<1>(1024));
```

> [!NOTE]
> This guide uses make_sync_buffer/make_async_buffer (AdaptiveCpp extensions) rather than raw sycl::buffer constructors for explicit policy control. The factory methods make the synchronization policy clear in the code.

### Accessors

Accessors are "requests" for data access. Creating an accessor inside a command group tells the runtime "this kernel needs this data in this mode". Access modes include:

- read: Data is loaded from host, kernel can read but not write
- write: Data is not loaded, kernel must initialize, host copy is discarded
- read_write: Full bidirectional access, data is loaded and copied back

```cpp
auto acc = buf.get_access<sycl::access_mode::read_write>(cgh);
```

The runtime uses accessor information to BUILD A TASK GRAPH automatically. If kernel A writes buffer X and kernel B reads buffer X, the runtime inserts a dependency between them.

## The Command Group Pattern

The command group is THE fundamental SYCL programming pattern. All work submission follows this structure:

```cpp
q.submit([&](sycl::handler& cgh) {
    // 1. Declare accessors (data dependencies)
    auto acc = buf.get_access<sycl::access_mode::read_write>(cgh);
    
    // 2. Optionally declare event dependencies
    // cgh.depends_on(previous_event);
    
    // 3. Submit work
    cgh.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
        acc[idx] *= 2.0f;
    });
});
```

> [!IMPORTANT]
> Accessors MUST be created inside the command group lambda, not outside. The handler (cgh) is how the runtime tracks dependencies between operations.

## Kernel Submission Models

SYCL provides three primary kernel submission models:

### single_task

Executes once on the device, typically used for setup or initialization:

```cpp
cgh.single_task([=]() {
    // Initialize some global state
    result[0] = 42;
});
```

### parallel_for with sycl::range<1>(N)

Simplest parallelism, one work-item per element:

```cpp
cgh.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
    output[idx] = input[idx] * 2.0f;
});
```

### parallel_for with sycl::nd_range<1>(global, local)

Explicit work-group control, required for local memory and barriers:

```cpp
cgh.parallel_for(sycl::nd_range<1>(global_size, local_size), 
    [=](sycl::nd_item<1> item) {
        auto group = item.get_group();
        auto local_id = item.get_local_id();
        
        // Use local memory
        local_mem[local_id] = input[item.get_global_id()];
        
        item.barrier(sycl::access::fence_space::local_space);
        
        // Continue computation
    });
```

## In-Order vs Out-of-Order Queues

By default, SYCL queues are out-of-order, allowing maximum parallelism for independent work:

```cpp
// Out-of-order queue (default)
sycl::queue q;  // Independent kernels may execute in any order
```

For simpler reasoning or when explicit ordering is needed:

```cpp
// In-order queue
sycl::queue q(sycl::property::queue::in_order());
// Commands execute in submission order
```

For beginners, in_order() is simpler to reason about. For performance, out-of-order with explicit dependencies is better.

## sycl::event and Synchronization

SYCL operations are asynchronous. submit() returns a sycl::event that represents the operation:

```cpp
sycl::event e = q.submit([&](sycl::handler& cgh) {
    // Kernel work
});

// Wait for specific operation
e.wait();

// Wait for all operations
q.wait();
```

Explicit dependencies can be created:

```cpp
cgh.depends_on(previous_event);
```

When buffer scope ends, host can read data - synchronization happens automatically.

## Putting It Together: A First SYCL Program (Conceptual)

```cpp
#include <sycl/sycl.hpp>
#include <hipsycl/sycl.hpp>
#include <vector>

int main() {
    const size_t N = 1024;
    
    // 1. Allocate host data
    std::vector<float> host_data(N);
    for (size_t i = 0; i < N; ++i) {
        host_data[i] = static_cast<float>(i);
    }
    
    // 2. Create buffer with AdaptiveCpp factory method
    using namespace hipsycl::sycl;
    auto buf = make_sync_buffer<float>(host_data.data(), sycl::range<1>(N));
    
    // 3. Create queue and submit work
    sycl::queue q(sycl::gpu_selector_v);
    
    q.submit([&](sycl::handler& cgh) {
        // 4. Create accessor declaring data dependency
        auto acc = buf.get_access<sycl::access_mode::read_write>(cgh);
        
        // 5. Submit kernel
        cgh.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
            acc[idx] *= 2.0f;  // Double each element
        });
    });
    
    // 6. Synchronize (either explicit or via buffer destruction)
    q.wait();  // Explicit wait
    
    // 7. Results are now available in host_data
    // (synchronization happened when buffer went out of scope or we called wait())
    
    return 0;
}
```

## Summary

- SYCL provides a single-source C++ model for heterogeneous computing
- The queue is the primary interface for submitting work to devices
- Buffers manage data ownership and synchronization
- Accessors declare data access patterns and create implicit dependencies
- The command group pattern is fundamental to SYCL programming
- Three kernel submission models: single_task, parallel_for with range, parallel_for with nd_range
- Queues can be in-order or out-of-order
- Events provide fine-grained synchronization control

## Next Steps

- [Chapter 3: AdaptiveCpp Setup](../03-acpp-setup/README.md) - Installation and configuration
- [Chapter 4: Memory Model](../04-memory-model/README.md) - Deep dive into buffers, USM, and data management

---

[<- Chapter 01: Heterogeneous Computing](../01-heterogeneous-computing/README.md) | [Up to Guide](../../README.md) | [Chapter 03: AdaptiveCpp Setup ->](../03-acpp-setup/README.md)