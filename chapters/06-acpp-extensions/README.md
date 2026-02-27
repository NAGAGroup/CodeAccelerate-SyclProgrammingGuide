# Chapter 06: AdaptiveCpp Extensions

AdaptiveCpp provides several SSCP-specific extensions beyond standard SYCL. All work with --acpp-targets=generic.

## Overview

| Extension | Macro | Purpose | Status |
|-----------|-------|---------|--------|
| ACPP_EXT_SPECIALIZED | None | JIT-time constant specialization | Stable |
| ACPP_EXT_JIT_COMPILE_IF | None | Target-aware code generation | Stable |
| ACPP_EXT_DYNAMIC_FUNCTIONS | None | Runtime kernel fusion | Stable/SSCP-only |
| ACPP_EXT_MULTI_DEVICE_QUEUE | None | Work distribution across devices | Experimental |
| ACPP_EXT_COARSE_GRAINED_EVENTS | None | Reduced launch latency | Stable |
| ACPP_EXT_ACCESSOR_VARIANTS | None | Register pressure reduction | Stable |

## ACPP_EXT_SPECIALIZED: JIT-Time Constants

`sycl::specialized<T>` hints to the SSCP JIT compiler that a value should be treated as a compile-time constant. No enable macro needed.

```cpp
sycl::specialized<float> scale{2.5f};
// Use in kernel as a capture
q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(range<1>{N}, [=](sycl::id<1> idx) {
        result[idx] = input[idx] * scale;  // scale becomes JIT constant
    });
});
```

> [!NOTE]
> Unlike SYCL 2020 specialization constants which can hurt AOT performance, sycl::specialized<T> is zero-overhead - if the JIT cannot specialize, it falls back to a regular runtime variable.

Requires ACPP_ADAPTIVITY_LEVEL >= 1 (default).

## ACPP_EXT_JIT_COMPILE_IF: Target-Aware Kernels

`compile_if`/`compile_if_else` and `reflect<>()` queries in `sycl::AdaptiveCpp_jit` namespace enable target-aware code generation.

```cpp
#ifdef __acpp_if_target_sscp()
using namespace sycl::AdaptiveCpp_jit;

q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(range<1>{N}, [=](sycl::id<1> idx) {
        float result = 0.0f;
        
        compile_if(reflect<vendor_id>() == vendor_id::nvidia) {
            // NVIDIA-specific optimization
            result = input[idx] * 2.0f;
        }
        .compile_if_else(reflect<vendor_id>() == vendor_id::amd) {
            // AMD-specific optimization
            result = input[idx] * 1.5f;
        }
        .compile_else {
            // Generic fallback
            result = input[idx] * 1.0f;
        };
        
        output[idx] = result;
    });
});
#endif
```

> [!IMPORTANT]
> Always wrap JIT_COMPILE_IF code with __acpp_if_target_sscp() for portability. Without this guard, the code will fail to compile on non-SSCP backends.

## ACPP_EXT_DYNAMIC_FUNCTIONS: Runtime Kernel Fusion

`dynamic_function_config.define()` and `define_as_call_sequence()` enable runtime kernel fusion.

```cpp
#ifdef __acpp_if_target_sscp()
using namespace sycl::AdaptiveCpp_jit;

// Function definitions MUST be marked SYCL_EXTERNAL
SYCL_EXTERNAL float multiply_add(float a, float b, float c) {
    return a * b + c;
}

// Configure dynamic functions
dynamic_function_config config;
config.define("multiply_add", multiply_add<float>);

// Use in kernel
q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(range<1>{N}, [=](sycl::id<1> idx) {
        // Runtime fusion of multiply_add with other operations
        result[idx] = dynamic_function<float>("multiply_add")(x[idx], y[idx], z[idx]);
    });
});
#endif
```

> [!WARNING]
> Functions used as dynamic function definitions MUST be marked SYCL_EXTERNAL, otherwise the compiler may inline or discard them before dynamic dispatch can occur.

Exclusively for --acpp-targets=generic.

## ACPP_EXT_MULTI_DEVICE_QUEUE: Multi-Device Dispatch

`sycl::multi_gpu_selector_v` enables work distribution across multiple devices.

```cpp
sycl::queue q{sycl::multi_gpu_selector_v};
auto devices = q.get_devices();

std::cout << "Found " << devices.size() << " devices\n";
for (const auto& device : devices) {
    std::cout << "  " << device.get_info<sycl::info::device::name>() << "\n";
}

// Work is automatically distributed across available devices
q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(range<1>{N}, [=](sycl::id<1> idx) {
        // Kernel executes on distributed devices
        result[idx] = input[idx] * 2.0f;
    });
});
```

> [!WARNING]
> ACPP_EXT_MULTI_DEVICE_QUEUE is highly experimental. The scheduling algorithm is primitive and not suitable for production workloads. It does not support handler::copy(), buffer host accessors, or USM transfers.

## ACPP_EXT_COARSE_GRAINED_EVENTS: Launch Latency

`sycl::property::queue::AdaptiveCpp_coarse_grained_events{}` reduces kernel launch latency.

```cpp
sycl::queue q{
    sycl::default_selector_v,
    sycl::property_list{
        sycl::property::queue::in_order{},
        sycl::property::queue::AdaptiveCpp_coarse_grained_events{}
    }
};

// Lighter-weight events, may sync more operations than strictly needed
auto event = q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(range<1>{N}, [=](sycl::id<1> idx) {
        result[idx] = input[idx] * 2.0f;
    });
});
```

Tradeoff: lighter events, may sync more ops than strictly needed. Best for in-order queues.

## ACPP_EXT_ACCESSOR_VARIANTS: Register Pressure

`raw_accessor<T>`, `unranged_accessor<T>`, `ranged_accessor<T>` reduce register pressure.

```cpp
#define ACPP_EXT_ACCESSOR_VARIANT_DEDUCTION

// Raw accessor - pointer only, lowest register pressure
raw_accessor<float> raw_acc{buf, cgh};

// Unranged accessor - pointer + shape, medium register pressure
unranged_accessor<float> unranged_acc{buf, cgh};

// Ranged accessor - pointer + shape + offset + range, highest register pressure
ranged_accessor<float> ranged_acc{buf, cgh, sycl::range<1>{N}};
```

| Variant | Storage | Register Pressure | Use Case |
|---------|---------|-------------------|----------|
| raw_accessor | Pointer only | Lowest | 1D kernels, no sub-buffer access |
| unranged_accessor | Pointer + shape | Medium | Multi-dimensional access |
| ranged_accessor | Pointer + shape + offset + range | Highest | Sub-buffer access, offset queries |

> [!NOTE]
> Raw accessors do not support offset or range queries. They are most effective for 1D kernels with no sub-buffer access.

## Examples in This Chapter

This chapter includes two examples:
- `jit_specialized` (demonstrates sycl::specialized<T> for JIT constant optimization)
- `accessor_variants_demo` (demonstrates raw vs unranged accessor register pressure reduction)

## Building and Running

```bash
pixi run configure && pixi run build
pixi run ./build/chapters/06-acpp-extensions/examples/jit_specialized
pixi run ./build/chapters/06-acpp-extensions/examples/accessor_variants_demo
```

## Summary

- ACPP_EXT_SPECIALIZED provides zero-overhead JIT-time constants via sycl::specialized<T>
- ACPP_EXT_JIT_COMPILE_IF enables target-aware code generation with compile_if/reflect<>()
- ACPP_EXT_DYNAMIC_FUNCTIONS allows runtime kernel fusion with SYCL_EXTERNAL functions
- ACPP_EXT_MULTI_DEVICE_QUEUE distributes work across multiple devices (experimental)
- ACPP_EXT_COARSE_GRAINED_EVENTS reduces kernel launch latency with lighter events
- ACPP_EXT_ACCESSOR_VARIANTS reduces register pressure with raw/unranged/ranged accessors