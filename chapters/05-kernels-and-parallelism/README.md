# Kernels and Parallelism

SYCL exposes three kernel submission models; this chapter covers nd_range (the current recommended approach for generic target), scoped parallelism v2 (preferred once SSCP support lands), and single_task for serial work.

## Kernel Submission Models

### single_task

The `single_task` kernel submits a single instance of work to the device. Use this for initialization, serial fallback code, or when you need exactly one work item.

```cpp
queue.submit([&](sycl::handler& h) {
    h.single_task<KernelName>([=](){
        // Serial code that runs once on the device
        result = initialize_data();
    });
});
```

> [!NOTE]
> Use `single_task` sparingly - it doesn't leverage the parallel nature of accelerator devices.

### parallel_for with range<D>

The simplest parallel launch model creates independent work items across a global range. Each work item gets a unique global ID but has no concept of work groups or local memory.

```cpp
queue.submit([&](sycl::handler& h) {
    h.parallel_for<class SimpleKernel>(sycl::range<1>{N}, [=](sycl::id<1> i) {
        buf[i] = i * 2;  // Each work item processes one element
    });
});
```

> [!IMPORTANT]
- No local memory access
- No barriers between work items
- Each work item runs independently

### nd_range parallel_for

The `nd_range` model explicitly organizes work items into groups, enabling local memory access and barriers. Required for algorithms that need cooperation between work items.

```cpp
sycl::nd_range<1> nd_range{sycl::range<1>{N}, sycl::range<1>{group_size}};
queue.submit([&](sycl::handler& h) {
    h.parallel_for<class GroupedKernel>(nd_range, [=](sycl::nd_item<1> it) {
        // Work item code with access to group information
        sycl::group<1> g = it.get_group();
        size_t lid = it.get_local_id(0);
        
        // Synchronize within the work group
        sycl::group_barrier(g);
    });
});
```

> [!WARNING]
> On CPU backends, nd_range barriers are implemented using fibers. Each work item requires its own stack and context-switch overhead. For CPU workloads, prefer scoped parallelism.

## Scoped Parallelism v2 (Roadmap: Not Yet Supported in SSCP Generic)

### Why Scoped Parallelism

> [!WARNING]
> **Not yet supported in `--acpp-targets=generic` (SSCP).** `ACPP_EXT_SCOPED_PARALLELISM_V2` is documented here as the intended future primary model, but the SSCP backend does not yet implement the `scoped_parallel_for` kernel launcher. Until this is resolved upstream, **use `nd_range parallel_for`** for workloads requiring local memory and barriers. See [AdaptiveCpp issue #1417](https://github.com/AdaptiveCpp/AdaptiveCpp/issues/1417).

> [!NOTE]
> This is a known gap on the AdaptiveCpp roadmap. Once SSCP support for scoped parallelism lands, it will become the preferred model over nd_range for all backends - eliminating fiber overhead on CPU while maintaining GPU efficiency.

Scoped parallelism separates physical from logical parallelism:
- You specify a logical iteration space
- The backend decides the physical mapping to hardware

On CPU (non-SSCP): items become loop iterations (vectorizable)
On GPU (non-SSCP): maps to warps/wavefronts

This avoids fiber overhead on CPU backends while maintaining GPU efficiency.

### The Dispatch Hierarchy

The scoped parallelism model uses a hierarchical dispatch pattern:

```cpp
q.submit([&](sycl::handler& cgh) {
    cgh.parallel<class KernelName>(sycl::range<1>{num_groups}, sycl::range<1>{group_size}, [=](auto group) {
        sycl::distribute_items(group, [&](sycl::s_item<1> it) {
            // Kernel body - each work item executes this
            size_t gid = it.get_global_id(0);
            // Process data at global position gid
        });
    });
});
```

Key components:
- `cgh.parallel<class KernelName>()` - Entry point for scoped parallelism, called inside `q.submit()`
- Kernel name template argument is required (use a unique tag struct per kernel)
- First range: number of groups to launch
- Second range: logical size of each group
- `distribute_items()` - Distribute work across physical work items

### Local Memory with memory_environment

Scoped parallelism provides explicit control over local memory allocation:

```cpp
q.submit([&](sycl::handler& cgh) {
    cgh.parallel<class KernelName>(sycl::range<1>{num_groups}, sycl::range<1>{group_size}, [=](auto group) {
        sycl::memory_environment(group, sycl::require_local_mem<float[GROUP_SIZE]>(),
            [&](auto& scratch) {
                // scratch is a local memory array shared by the group
                sycl::distribute_items(group, [&](sycl::s_item<1> it) {
                    size_t lid = it.get_local_id(group, 0);
                    scratch[lid] = input_data[it.get_global_id(0)];
                });
            });
    });
});
```

- `require_local_mem<T[size]>()` - Allocates group-shared memory
- Variables inside `distribute_items` are per-item private
- Memory is shared across all work items in the group

### Synchronization

Scoped parallelism provides two synchronization approaches:

```cpp
// Approach 1: Implicit synchronization with distribute_items_and_wait
sycl::distribute_items_and_wait(group, [&](sycl::s_item<1> it) {
    // All work items must complete before continuing
    size_t lid = it.get_local_id(group, 0);
    scratch[lid] = input_data[it.get_global_id(0)];
});

// Approach 2: Explicit barrier outside distribute_items
sycl::distribute_items(group, [&](sycl::s_item<1> it) {
    // Work that doesn't require synchronization
});
sycl::group_barrier(group);  // ALL physical work items must reach here
sycl::distribute_items(group, [&](sycl::s_item<1> it) {
    // Work after barrier
});
```

> [!IMPORTANT]
> Rule 3: `group_barrier` must be called outside `distribute_items` - ALL physical work items must collectively reach it.

## Comparison Table

| Model | Local Mem | Barriers | CPU Overhead | Best For |
|-------|-----------|----------|--------------|----------|
| single_task | No | No | Low | Initialization, serial fallback |
| range parallel_for | No | No | Low | Simple data-parallel operations |
| nd_range | Yes | Yes | High on CPU (fibers) | Current recommendation for generic target |
| scoped parallelism | Yes | Yes | Low | All parallel workloads (when SSCP support lands) |

## Examples

- **scoped_reduction** - Parallel sum reduction using local memory + scoped parallelism
  - Note: scoped_reduction compiles but will fail at runtime with the generic target until upstream SSCP support lands. Run nd_range_demo for a working comparison.
- **nd_range_demo** - Same reduction using nd_range for comparison

### Building and Running

```bash
# Configure and build
pixi run configure
pixi run build

# Run the examples
pixi run ./build/chapters/05-kernels-and-parallelism/examples/scoped_reduction
pixi run ./build/chapters/05-kernels-and-parallelism/examples/nd_range_demo
```

## Summary

- SYCL provides three kernel submission models: single_task, parallel_for with range, and nd_range
- Scoped parallelism v2 is the intended primary model - it eliminates fiber overhead and provides physical/logical separation
- **Current status:** ACPP_EXT_SCOPED_PARALLELISM_V2 is not yet supported in `--acpp-targets=generic`; use nd_range until upstream support lands
- CPU workloads should prefer scoped parallelism (when available) to avoid fiber overhead
- The dispatch hierarchy provides explicit control over work group organization