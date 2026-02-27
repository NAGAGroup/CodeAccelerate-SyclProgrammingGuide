# Chapter 09: Real-World Patterns

This chapter synthesizes everything from the guide into two complete, realistic examples. Choosing between nd_range tiling (local memory, work groups) and async-buffer DAG (no manual events, automatic dependency tracking).

## Pattern 1: Tiled Matrix Multiplication

Matrix multiplication is the canonical example of compute-bound workloads that benefit from data reuse. The key insight is that global memory bandwidth is the limiting factor - each element is read multiple times during the computation.

The tiling strategy loads tiles of the input matrices into local (shared) memory per work group. This allows each work item to reuse the tile data multiple times, reducing global memory accesses.

> [!NOTE]
> nd_range is used here because scoped_parallel_for is not yet supported in --acpp-targets=generic SSCP mode.

The pattern uses:
- Compile-time TILE_SIZE constant for work group dimensions
- Two barriers per tile iteration (load-then-compute barrier, then post-compute barrier)
- nd_range<2> dispatch with 2D work groups
- local_accessor<float, 2> for shared memory tiles

```cpp
// Tiled matrix multiplication skeleton
constexpr int TILE_SIZE = 16;
sycl::nd_range<2> range{
    sycl::range<2>{N, N}, 
    sycl::range<2>{TILE_SIZE, TILE_SIZE}
};

cgh.parallel_for<class MatMulKernel>(range, [=](sycl::nd_item<2> item) {
    int row = item.get_global_id(0);
    int col = item.get_global_id(1);
    int local_row = item.get_local_id(0);
    int local_col = item.get_local_id(1);
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < N / TILE_SIZE; ++t) {
        // Load tiles into local memory
        a_tile[local_row][local_col] = a_acc[row][t * TILE_SIZE + local_col];
        b_tile[local_row][local_col] = b_acc[t * TILE_SIZE + local_row][col];
        
        // Barrier: ensure tiles are loaded before computation
        sycl::group_barrier(item.get_group());
        
        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += a_tile[local_row][k] * b_tile[k][local_col];
        }
        
        // Barrier: ensure computation completes before next tile load
        sycl::group_barrier(item.get_group());
    }
    
    c_acc[row][col] = sum;
});
```

## Pattern 2: Jacobi Iterative Solver with Automatic DAG

The Jacobi method is an iterative algorithm for solving systems of linear equations. It demonstrates how AdaptiveCpp's automatic dependency tracking eliminates the need for explicit event management.

AdaptiveCpp provides page-based dependency tracking: accessor conflicts between kernels in the same queue automatically become DAG edges. No manual `handler::depends_on()` calls are needed.

The pattern uses:
- make_async_buffer for internal work buffers (non-blocking destructor)
- make_async_writeback_view for result buffers (writeback on destruction to host pointer)
- Two-kernel-per-iteration pattern with automatic dependencies
- Reduction for convergence checking

```cpp
// Jacobi iteration skeleton
for (int iter = 0; iter < max_iter; ++iter) {
    // Kernel 1: Update solution
    q.submit([&](sycl::handler& cgh) {
        auto x_cur_acc = sycl::accessor{x_cur_buf, cgh, sycl::read_only};
        auto x_new_acc = sycl::accessor{x_new_buf, cgh, sycl::write_only};
        
        cgh.parallel_for<class UpdateKernel>(sycl::range<1>{N}, [=](sycl::id<1> i) {
            // Update x_new[i] based on x_cur neighbors
        });
    });
    
    // Kernel 2: Copy new solution back (automatic dependency)
    q.submit([&](sycl::handler& cgh) {
        auto x_new_acc = sycl::accessor{x_new_buf, cgh, sycl::read_only};
        auto x_cur_acc = sycl::accessor{x_cur_buf, cgh, sycl::write_only};
        
        cgh.parallel_for<class CopyKernel>(sycl::range<1>{N}, [=](sycl::id<1> i) {
            x_cur_acc[i] = x_new_acc[i];
        });
    });
    
    // No explicit event dependency needed - AdaptiveCpp DAG tracks
    // accessor conflicts automatically between kernels
}
```

> [!NOTE]
> Convergence check requires `q.wait()` because we need to read the norm on the host. The reduction kernel computes the norm on the device, but we must synchronize before accessing the result.

## Putting It Together

| Example | Key technique | What it demonstrates | Buffer strategy |
|---------|---------------|---------------------|-----------------|
| matmul | nd_range tiling with local memory | Data reuse, work group synchronization, manual barriers | make_sync_view for inputs, make_sync_writeback_view for output |
| jacobi_solver | Automatic DAG with async buffers | Implicit dependencies, iterative algorithms, reduction operations | make_async_buffer for work buffers, explicit copy for result |

> [!NOTE]
> The Jacobi solver uses `sycl::reduction` for the convergence norm - a built-in reduction that operates within a single kernel invocation. For the complementary pattern of accumulating per-work-group partial results into a single global value across multiple work groups using `sycl::atomic_ref::fetch_add`, see [Chapter 10: Atomics and Memory Ordering](../10-atomics/README.md). That chapter's `reduction_fetch_add` example demonstrates the two-phase approach: local partial sum per work-item, then one atomic add per work-group to a global result.

## Building and Running

```bash
# Configure and build
pixi run configure && pixi run build

# Run matrix multiplication
pixi run ./build/chapters/09-real-world-patterns/examples/matmul

# Run Jacobi solver
pixi run ./build/chapters/09-real-world-patterns/examples/jacobi_solver
```

## Summary

This chapter completes the AdaptiveCpp Programmer's Guide with two realistic examples that demonstrate the key patterns you'll encounter in real-world SYCL development. The tiled matrix multiplication shows how to extract performance from compute-bound workloads through careful data reuse and work group synchronization. The Jacobi solver demonstrates how AdaptiveCpp's automatic dependency tracking simplifies complex multi-kernel workflows.

With these patterns and the concepts from previous chapters, you now have the foundation to build efficient, correct heterogeneous applications using AdaptiveCpp and SYCL.