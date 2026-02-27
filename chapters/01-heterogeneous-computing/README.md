# Heterogeneous Computing

This chapter introduces the fundamental concepts of heterogeneous computing that underpin SYCL and AdaptiveCpp programming. We'll explore why modern systems combine different processor types, how CPUs and GPUs differ architecturally, and why these differences matter for programming parallel applications. Understanding these concepts is essential before diving into SYCL code, as they explain the "why" behind the programming models we'll use.

## What is Heterogeneous Computing?

Heterogeneous computing refers to systems that contain multiple processor types with different strengths and specializations. In the context of SYCL and AdaptiveCpp, this typically means combining a CPU (Central Processing Unit) with a GPU (Graphics Processing Unit) in a single system.

The CPU excels at control flow, complex decision-making, and latency-sensitive tasks. The GPU, originally designed for graphics rendering, has evolved into a massively parallel processor ideal for data-parallel computation. In SYCL terminology, the CPU-side code runs on the "host" while the GPU-side code runs on the "device."

> [!NOTE]
> While heterogeneous systems can include FPGAs, DSPs, and other accelerators, this guide focuses primarily on CPU+GPU combinations, which are the most common target for SYCL applications.

## CPU Architecture: Latency-Optimized

CPU design philosophy centers on minimizing latency—the time to complete a single task or respond to an event. Every aspect of CPU architecture serves this goal:

- **Few powerful cores**: Modern CPUs typically have 4-64 cores, each capable of handling complex instruction sequences and sophisticated control flow.
- **Large caches**: CPUs feature multi-level cache hierarchies with L3 caches often measuring tens of megabytes to keep frequently accessed data close to the cores.
- **Sophisticated control logic**: Branch prediction, speculative execution, out-of-order execution, and register renaming all work to keep the pipeline full and avoid stalls.
- **High clock speeds**: CPUs typically run at 3-5 GHz, executing each instruction as quickly as possible.

```
CPU Architecture (Simplified)
┌─────────────────────────────────────────────────────┐
│                 L3 Cache (32MB)                      │
├─────────────────────────────────────────────────────┤
│  Core 1  │  Core 2  │  Core 3  │  Core 4  │  ...    │
│ ┌───────┐ │ ┌───────┐ │ ┌───────┐ │ ┌───────┐ │        │
│ │ L1/L2 │ │ │ L1/L2 │ │ │ L1/L2 │ │ │ L1/L2 │ │        │
│ │ Cache  │ │ │ Cache  │ │ │ Cache  │ │ │ Cache  │ │        │
│ └───────┘ │ └───────┘ │ └───────┘ │ └───────┘ │        │
│ ┌───────┐ │ ┌───────┐ │ ┌───────┐ │ ┌───────┐ │        │
│ │Branch │ │ │Branch │ │ │Branch │ │ │Branch │ │        │
│ │Pred.  │ │ │Pred.  │ │ │Pred.  │ │ │Pred.  │ │        │
│ └───────┘ │ └───────┘ │ └───────┘ │ └───────┘ │        │
│ ┌───────┐ │ ┌───────┐ │ ┌───────┐ │ ┌───────┐ │        │
│ │Out-of │ │ │Out-of │ │ │Out-of │ │ │Out-of │ │        │
│ │Order  │ │ │Order  │ │ │Order  │ │ │Order  │ │        │
│ └───────┘ │ └───────┘ │ └───────┘ │ └───────┘ │        │
└─────────────────────────────────────────────────────┘
```

All this complexity exists to make individual tasks complete as quickly as possible, whether that's responding to user input, handling an interrupt, or executing a complex algorithm with lots of branching.

## GPU Architecture: Throughput-Optimized

GPU design philosophy takes the opposite approach: maximizing throughput—the total amount of work completed per second, even if individual tasks take longer. This is achieved through massive parallelism:

- **Thousands of simpler cores**: Modern GPUs contain thousands to tens of thousands of smaller, more efficient cores.
- **Smaller per-core caches**: Individual cores have minimal caches, relying more on shared memory and high bandwidth access to global memory.
- **High memory bandwidth**: GPUs feature memory systems with an order of magnitude higher bandwidth than CPUs (500-1000+ GB/s vs 50-100 GB/s).
- **Moderate clock speeds**: GPUs typically run at 1-2 GHz, slower than CPUs but compensated by massive parallelism.

```
GPU Architecture (Simplified)
┌─────────────────────────────────────────────────────┐
│            High Bandwidth Memory (HBM)              │
├─────────────────────────────────────────────────────┤
│  SM 1     │  SM 2     │  SM 3     │  ...  │  SM N   │
│ ┌───────┐ │ ┌───────┐ │ ┌───────┐ │        │ ┌───────┐│
│ │Core   │ │ │Core   │ │ │Core   │ │        │ │Core   ││
│ │Array  │ │ │Array  │ │ │Array  │ │        │ │Array  ││
│ │(32-64)│ │ │(32-64)│ │ │(32-64)│ │        │ │(32-64)││
│ └───────┘ │ └───────┘ │ └───────┘ │        │ └───────┘│
│ ┌───────┐ │ ┌───────┐ │ ┌───────┐ │        │ ┌───────┐│
│ │Shared │ │ │Shared │ │ │Shared │ │        │ │Shared ││
│ │Memory │ │ │Memory │ │ │Memory │ │        │ │Memory ││
│ └───────┘ │ └───────┘ │ └───────┘ │        │ └───────┘│
│ ┌───────┐ │ ┌───────┐ │ ┌───────┐ │        │ ┌───────┐│
│ │Scheduler│ │ │Scheduler│ │ │Scheduler│ │        │ │Scheduler││
│ └───────┘ │ └───────┘ │ └───────┘ │        │ └───────┘│
└─────────────────────────────────────────────────────┘
```

The GPU sacrifices single-thread performance and latency optimization to achieve vastly higher aggregate throughput for parallel workloads.

## The SIMT Execution Model

GPUs use SIMT (Single Instruction Multiple Threads) execution, which differs from the SIMD (Single Instruction Multiple Data) model used by CPU vector instructions:

1. **Warp execution**: GPU threads execute in groups called warps (typically 32 threads on NVIDIA GPUs or wavefronts of 64 threads on AMD GPUs).
2. **Lockstep instruction execution**: All threads in a warp execute the same instruction simultaneously on different data elements.
3. **Scalar programming model**: You write scalar code as if each thread runs independently, and the GPU automatically vectorizes execution across the warp.
4. **Warp divergence**: When threads in a warp take different branches (if/else), the GPU serializes execution paths and masks off inactive threads.

```
SIMT Execution Example
┌─────────────────────────────────────────────────────┐
│ Warp (32 threads executing same instruction)        │
│ Thread: 0  1  2  3  4  5  6  7  8  9 10 11 12 13 ... │
│ Instr:  ADD  ADD  ADD  ADD  ADD  ADD  ADD  ADD  ADD  │
│ Data:   A0  A1  A2  A3  A4  A5  A6  A7  A8  A9 ...   │
│         B0  B1  B2  B3  B4  B5  B6  B7  B8  B9 ...   │
│         R0  R1  R2  R3  R4  R5  R6  R7  R8  R9 ...   │
└─────────────────────────────────────────────────────┘
```

> [!WARNING]
> Warp divergence can significantly impact performance. If threads in a warp take different paths through conditional code, the GPU must execute each path separately, effectively reducing parallelism. Minimizing divergence is crucial for GPU performance.

## Latency Hiding via Parallelism

GPUs use a fundamentally different strategy than CPUs to handle memory access latency:

- **Warp scheduling**: When a warp stalls waiting for memory (which is slow even on GPUs), the Streaming Multiprocessor (SM) instantly switches to execute a different warp that is ready to run.
- **Massive thread count**: GPUs keep dozens or hundreds of warps "in flight" simultaneously, ensuring there's always work to do.
- **No caching for latency**: Unlike CPUs that cache to reduce latency, GPUs use parallelism to hide latency entirely.

This approach explains why GPUs need thousands of threads—not just for computation, but specifically to hide memory access latency. With enough parallel work, the GPU can keep all its execution units busy despite the high latency of memory operations.

> [!TIP]
> The rule of thumb for GPU programming is to have enough active warps to hide memory latency. For most applications, this means launching thousands of threads rather than dozens or hundreds.

## Data-Parallel Workloads: The GPU Sweet Spot

GPUs excel at specific types of problems, particularly those with these characteristics:

1. **Same operation on many data elements**: Applying identical computations to large datasets (matrix operations, image processing, physics simulation).
2. **High arithmetic intensity**: Performing many calculations per memory access (dense linear algebra, signal processing).
3. **Regular memory access patterns**: Accessing memory in predictable, contiguous patterns that enable coalesced memory access.
4. **Minimal control flow**: Avoiding complex branching and conditional logic within the parallel computation.

```
GPU-Friendly vs CPU-Friendly Workloads

GPU-Friendly:
┌─────────────────────────────────────────────────────┐
│ Matrix Multiplication: C[i,j] += A[i,k] * B[k,j]    │
│ - Same operation on all elements                     │
│ - Regular memory access patterns                    │
│ - High arithmetic intensity                         │
└─────────────────────────────────────────────────────┘

CPU-Friendly:
┌─────────────────────────────────────────────────────┐
│ Graph Traversal: DFS/BFS on irregular data structures│
│ - Complex control flow and branching                 │
│ - Pointer-chasing memory access                      │
│ - Low arithmetic intensity                           │
└─────────────────────────────────────────────────────┘
```

In contrast, CPUs handle workloads with complex control flow, irregular data structures, small datasets, and pointer-chasing operations more efficiently.

## The Host-Device Programming Model

The heterogeneous programming model divides responsibilities between host and device:

1. **Host (CPU)**: Orchestrates computation, manages I/O, runs control logic, and prepares data for processing.
2. **Device (GPU)**: Executes computationally intensive kernels in parallel across thousands of threads.
3. **Kernel**: A function that runs on the device, executing the same code across many threads in parallel.
4. **Memory separation**: Host and device have separate memory spaces by default, requiring explicit data transfer.

```
Host-Device Memory Model
┌─────────────────┐    PCIe Bus    ┌─────────────────┐
│   Host RAM      │ ←──────────→   │  Device VRAM    │
│   (CPU Memory)  │                │   (GPU Memory)  │
└─────────────────┘                └─────────────────┘
       ↑                                    ↓
       │                                    │
┌─────────────────┐                ┌─────────────────┐
│   Host Code     │                │   Device Code   │
│   (Control)     │                │   (Kernels)     │
└─────────────────┘                └─────────────────┘
```

The classic workflow involves:
1. Allocate device memory
2. Copy data from host to device
3. Launch kernel to process data
4. Copy results from device back to host
5. Free device memory

> [!IMPORTANT]
> This explicit memory management model is the foundation of GPU programming frameworks like CUDA. SYCL provides higher-level abstractions that automate much of this process, as we'll see in the next chapter.

## Why SYCL Helps

Manual host-device programming (as in CUDA) requires verbose, error-prone code for memory management and data movement:

```cpp
// CUDA-style manual memory management (conceptual example)
float *host_data = ...;
float *device_data;
cudaMalloc(&device_data, size);
cudaMemcpy(device_data, host_data, size, cudaMemcpyHostToDevice);
my_kernel<<<blocks, threads>>>(device_data);
cudaMemcpy(host_data, device_data, size, cudaMemcpyDeviceToHost);
cudaFree(device_data);
```

SYCL's buffer and accessor model automates data movement and expresses dependencies declaratively. AdaptiveCpp builds on this with convenient buffer factory methods that further simplify the code while maintaining performance.

> [!TIP]
> SYCL allows you to focus on algorithm logic rather than low-level memory management, while still giving you control when needed. This balance of productivity and performance is why SYCL has become popular for heterogeneous computing.

## Summary

Key takeaways from this chapter:

- **Heterogeneous systems** combine CPUs (latency-optimized) and GPUs (throughput-optimized) for different types of work.
- **CPU architecture** prioritizes single-thread performance with sophisticated control logic and large caches.
- **GPU architecture** maximizes throughput through thousands of simpler cores and high memory bandwidth.
- **SIMT execution** runs threads in warps that execute the same instruction on different data.
- **Latency hiding** via massive parallelism is the GPU's strategy for handling memory delays.
- **Data-parallel workloads** with high arithmetic intensity and regular memory patterns are ideal for GPUs.
- **Host-device model** separates control logic (host) from parallel computation (device).
- **SYCL** simplifies heterogeneous programming by automating memory management while preserving performance.

Understanding these architectural differences is essential for writing efficient SYCL code that properly leverages both CPU and GPU capabilities.