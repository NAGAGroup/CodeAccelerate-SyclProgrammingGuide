# Adding Atomics to the AdaptiveCpp Programmer's Guide

> **Document Purpose:** Supplementary planning document for the AdaptiveCpp Programmer's Guide.
> The main guide (`PROGRAMMING_GUIDE_IMPLEMENTATION_PLAN.md`) does not cover atomics. This document
> instructs the implementing agent on how to incorporate atomics into the existing guide structure.
> **Do not execute this document until the main guide (all 10 chapters) is complete.**

---

## Summary of the Gap

The main guide's 10-chapter structure covers the memory model, kernels and parallelism, extensions,
performance, footguns, and real-world patterns - but never mentions atomics. This is a significant
omission: atomics are the primary mechanism for correct inter-work-item communication and are
needed for any non-trivial parallel algorithm (reductions, histograms, lock-free queues, barriers
built on top of SYCL, etc.).

SYCL 2020 ships a complete, modern atomics model via `sycl::atomic_ref`. AdaptiveCpp implements
it faithfully under SSCP/generic and adds one extension (`ACPP_EXT_FP_ATOMICS`). There is enough
material for a dedicated chapter, and several existing chapters need minor amendments.

---

## Agent Instructions

### Step 0: Mandatory Librarian Research

> [!IMPORTANT]
> Before writing a single line of guide content or code, the implementing agent MUST delegate
> to the librarian to fetch current AdaptiveCpp and SYCL 2020 atomics documentation.
> The librarian session should cover all items listed under "Research Targets" below.
> Do NOT proceed to writing until the librarian has returned its findings.

**Minimum librarian research targets:**

- `https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html` - Section on
  `atomic_ref`, `memory_order`, `memory_scope`, and `atomic_fence`
- `https://adaptivecpp.github.io/AdaptiveCpp/` - Top-level docs for any atomics-specific notes
- `https://adaptivecpp.github.io/AdaptiveCpp/extensions/` - Verify `ACPP_EXT_FP_ATOMICS` details
  and any other atomics-related extensions that may have been added since this document was written
- `https://github.com/AdaptiveCpp/AdaptiveCpp` - Check recent commits and docs for atomics changes

**Key questions the librarian must answer before you write:**

1. Is `ACPP_EXT_FP_ATOMICS` still the correct macro name? What is its exact header placement?
2. Has AdaptiveCpp added any new atomics extensions beyond FP_ATOMICS?
3. Does SSCP/generic have any known limitations on atomics that are documented?
4. What is the current status of `memory_scope::sub_group` on each major backend (NVIDIA, AMD,
   Intel GPU, CPU)?
5. Are there any known correctness bugs or caveats with atomics under the current acpp-toolchain
   version (>= 20.1.8_acpp25.10.0)?

---

### Step 1: Assess the Existing Guide

Before creating new content, read each of the following files to understand what is already there
and what needs amending. Use the read tool, not bash.

- `chapters/04-memory-model/README.md` - Check whether memory coherence and ordering are mentioned
- `chapters/05-kernels-and-parallelism/README.md` - Check whether barriers/fences are explained
  and whether any mention of atomics appears
- `chapters/08-footguns/README.md` - Check which footgun categories exist; atomics footguns must
  be added as a new category if not present
- `chapters/09-real-world-patterns/README.md` - Check which patterns are present; determine if
  any reference atomics (reductions are the most likely candidate)

Document your findings. If any of these files do not yet exist (the chapter is still pending),
note that the atomics content for that chapter must be authored at the same time as that chapter.

---

### Step 2: Create the New Atomics Chapter

Create `chapters/10-atomics/` with the following structure:

```
chapters/10-atomics/
  README.md
  examples/
    CMakeLists.txt
    atomic_counter.cpp
    reduction_fetch_add.cpp
    compare_exchange.cpp
    atomic_fence_ordering.cpp
    fp_atomics.cpp              (guarded by ACPP_EXT_FP_ATOMICS check)
```

#### README.md Content Outline

The README must follow the guide's voice: teach the concept before showing code, be explicit about
footguns, and use acpp-idiomatic patterns throughout. All code uses SSCP (`--acpp-targets=generic`).
All buffers use factory methods (no standard `sycl::buffer` constructors).

**Section A: Why Atomics**

- Explain data races in parallel kernels with a minimal illustrative example (two work-items
  incrementing the same counter without synchronization, producing incorrect results)
- Define the problem atomics solve: indivisible read-modify-write operations
- Clarify the relationship to barriers: barriers coordinate control flow, atomics coordinate data
- Note: atomic_ref is the SYCL 2020 approach; the older `sycl::atomic<T>` is deprecated and must
  not be used in examples or recommendations

**Section B: sycl::atomic_ref - The Core API**

Explain the four template parameters:

```
atomic_ref<T, DefaultOrder, DefaultScope, AddressSpace>
```

- `T`: supported integral and floating-point types; note 64-bit requires `sycl::aspect::atomic64`
- `DefaultOrder`: the memory order used when the caller does not specify one explicitly
- `DefaultScope`: the synchronization scope; choosing this correctly is performance-critical
- `AddressSpace`: usually `generic_space`; rarely needs changing

Show construction from an existing variable (the atomic_ref wraps an existing object, it does not
own it):

```cpp
int counter = 0;
sycl::atomic_ref<int,
                 sycl::memory_order::relaxed,
                 sycl::memory_scope::work_group> ref{counter};
```

**Section C: Memory Orders**

Teach each order in order of increasing strength:

| Order | Description | When to use |
|-------|-------------|-------------|
| `relaxed` | No ordering guarantees; only atomicity | Simple counters with no data dependencies |
| `acquire` | Load; prevents later ops moving before this | Reading a flag that guards other data |
| `release` | Store; prevents earlier ops moving after this | Writing data before publishing a flag |
| `acq_rel` | Both; for read-modify-write ops | fetch_add, CAS when both sides need ordering |
| `seq_cst` | Total global order across all devices | Last resort; very expensive on GPU |

> [!CAUTION]
> `seq_cst` forces a global synchronization barrier on most GPU backends. It is correct but
> expensive. Use `acq_rel` with the appropriate scope in almost all cases. If you find yourself
> reaching for `seq_cst` on device scope, reconsider whether a group barrier would be cleaner.

**Section D: Memory Scopes**

Explain the hierarchy and the cost/correctness tradeoff:

| Scope | Synchronizes | Typical use |
|-------|-------------|-------------|
| `work_item` | Within one work-item (ordering only) | Rare; mostly for fence semantics |
| `sub_group` | All work-items in a sub-group (warp/wave) | Intra-warp coordination |
| `work_group` | All work-items in a work-group | Local reductions, shared memory flags |
| `device` | All work-items on the device | Cross-group synchronization |
| `system` | Device + host | Rarely correct; avoid unless required |

> [!WARNING]
> Not all backends support all scopes equally. `memory_scope::sub_group` is available on
> NVIDIA and AMD, but behavior varies on Intel integrated GPUs and CPU backends. Always verify
> support via `sycl::info::context::atomic_memory_scope_capabilities` in production code.
> `memory_scope::system` is not supported on NVIDIA or AMD GPU backends.

**Section E: Operations Reference**

Cover each operation family with a brief example:

- `load` / `store` - basic read and write
- `exchange` - atomic swap
- `compare_exchange_weak` / `compare_exchange_strong` - CAS; explain the weak vs strong tradeoff
  (weak may spuriously fail but is faster on some architectures; use strong in simple cases,
  weak inside a retry loop)
- `fetch_add`, `fetch_sub` - integral and float (with FP caveat)
- `fetch_min`, `fetch_max` - integral only
- `fetch_and`, `fetch_or`, `fetch_xor` - integral only
- `is_lock_free` - query; show how result is scope-dependent

**Section F: sycl::atomic_fence**

```cpp
sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::work_group);
```

- Standalone fence without a corresponding atomic variable
- Useful for ordering non-atomic loads/stores around an atomic signal
- Show the pattern: producer writes data, calls release fence, stores flag; consumer checks flag,
  calls acquire fence, reads data
- Note that `group_barrier` already acts as an `acq_rel` fence; do not double-fence

**Section G: ACPP_EXT_FP_ATOMICS**

- The macro must be defined before `#include <sycl/sycl.hpp>`
- Enables floating-point `fetch_add` on `float` and `double`
- Not all backends support this natively; some fall back to software emulation
- Show how to guard usage with `#ifdef ACPP_EXT_FP_ATOMICS` for portability
- Show the alternative for portable float reduction: CAS-loop pattern

**Section H: Backend Notes for SSCP/Generic**

Briefly document what the SSCP JIT does with atomics at runtime:

- Atomic operations are embedded in LLVM IR at compile time
- JIT lowers them to PTX atomics (NVIDIA), AMD GCN atomics, SPIR-V atomics (Intel/OpenCL), or
  native LLVM atomics (CPU)
- Memory orders and scopes are preserved through the JIT pipeline
- 64-bit atomic support requires `sycl::aspect::atomic64`; check at runtime before using `long`
  or `unsigned long long` types

---

### Step 3: Write the Example Programs

All examples must:

- Compile with `--acpp-targets=generic` (enforced by the project CMake)
- Use buffer factory methods (no `sycl::buffer` standard constructors)
- Include a clear comment block explaining what the example demonstrates
- Print output that verifies correctness (not just that it ran)

**`atomic_counter.cpp`**

Demonstrate the simplest correct atomic pattern: parallel increment of a shared counter. Show the
broken version (no atomic) and the correct version side-by-side, with output demonstrating that
the non-atomic version gives wrong results and the atomic version is correct. Use
`memory_order::relaxed` and `memory_scope::device`.

**`reduction_fetch_add.cpp`**

Implement a sum reduction using `fetch_add` with `memory_order::acq_rel` and
`memory_scope::work_group`. Compare approach: each work-item accumulates a local partial sum,
then one atomic add per work-group to a global result. This is the canonical pattern and must be
shown clearly with comments explaining the two-phase approach.

**`compare_exchange.cpp`**

Demonstrate `compare_exchange_strong` in a retry loop to implement a custom atomic max (since
`fetch_max` is integers-only). Use `float` as the type. This example is also a useful introduction
to the CAS pattern for readers who need to implement operations not covered by the built-ins.

**`atomic_fence_ordering.cpp`**

Demonstrate the producer/consumer pattern using `atomic_fence`:
- Producer work-items write to a shared buffer, then call a `release` fence, then set a flag
- Consumer work-items poll the flag, call an `acquire` fence, then read the shared buffer
- Show that without the fences the reads may observe stale data

**`fp_atomics.cpp`**

Demonstrate `ACPP_EXT_FP_ATOMICS` for float `fetch_add`, guarded by the macro. If the macro is
not defined, fall back to a CAS loop. This teaches both the extension and the portable fallback.

---

### Step 4: Update CMakeLists.txt

Create `chapters/10-atomics/examples/CMakeLists.txt` using the project macro:

```cmake
add_acpp_example(atomic_counter atomic_counter.cpp)
add_acpp_example(reduction_fetch_add reduction_fetch_add.cpp)
add_acpp_example(compare_exchange compare_exchange.cpp)
add_acpp_example(atomic_fence_ordering atomic_fence_ordering.cpp)
add_acpp_example(fp_atomics fp_atomics.cpp)
```

Also update the root `CMakeLists.txt` to include this new subdirectory:

```cmake
add_subdirectory(chapters/10-atomics/examples)
```

Use the junior_dev agent for all CMakeLists.txt edits. The tech_lead cannot edit non-markdown files.

---

### Step 5: Amend Existing Chapters

The following chapters need targeted amendments after the atomics chapter is written. These are
small additions, not rewrites. Delegate to junior_dev only for non-markdown files; tech_lead can
edit the markdown directly.

**`chapters/04-memory-model/README.md` - Add memory coherence note**

Near the end of the chapter (or as a new subsection after the Buffers/USM sections), add a short
note that the memory model in SYCL 2020 defines acquire/release semantics for atomic operations,
and that this chapter covers memory *storage* (buffers and USM) while Chapter 10 covers memory
*ordering* (atomics, fences, scopes). This sets up the reader for the atomics chapter without
duplicating content.

**`chapters/05-kernels-and-parallelism/README.md` - Cross-reference barriers and atomics**

In the section on barriers and fences (group_barrier), add a cross-reference to Chapter 10. Note
that `group_barrier` acts as an `acq_rel` fence at work-group scope, making it sufficient for
most within-work-group synchronization, and that Chapter 10 covers explicit atomics and fences for
patterns where barriers are not enough (cross-work-group coordination, fine-grained per-item
synchronization).

**`chapters/08-footguns/README.md` - Add atomics footgun category**

Add a new category "Atomics Footguns" with entries for each of the following:

1. **`seq_cst` on device scope** - explanation and mitigation (use `acq_rel` with explicit scope)
2. **64-bit atomics without aspect check** - undefined behavior without `atomic64` aspect check
3. **Wrong scope for cross-group sync** - using `work_group` scope when `device` is needed
4. **FP atomics non-portability** - `ACPP_EXT_FP_ATOMICS` may emulate silently on some backends
5. **Double-fencing around group_barrier** - redundant and confusing; barrier already fences
6. **Sub-group scope on CPU/integrated GPU** - may not be supported; runtime query required
7. **CAS spurious failure with `compare_exchange_weak`** - must use retry loop; show the pattern

**`chapters/09-real-world-patterns/README.md` - Add atomic reduction pattern**

If a parallel reduction pattern is not already present, add it here as a real-world pattern.
If a reduction pattern is already present but uses barriers only (local reduction within a
work-group), extend it to show the second phase: accumulating work-group results into a global
result using `fetch_add` at `device` scope.

---

### Step 6: Verification

After all content and code is written, delegate to test_runner to:

1. Run `pixi run configure`
2. Run `pixi run build` - all five new examples must compile cleanly
3. Run each binary with `pixi run ./build/chapters/10-atomics/examples/<name>`
4. Verify output: each example must print a result and the result must be correct (not just that
   the binary ran without crashing)

test_runner must NOT mark verification complete without having run all five binaries and observed
their output.

---

## Reference: Key Technical Facts (From Librarian Research)

The following facts were gathered from SYCL 2020 spec and AdaptiveCpp docs research conducted
at the time this supplementary document was written. The implementing agent MUST verify these
against current docs via the librarian before writing (see Step 0).

### sycl::atomic_ref Template Signature

```cpp
template <typename T,
          sycl::memory_order DefaultOrder,
          sycl::memory_scope DefaultScope,
          sycl::access::address_space AddressSpace = sycl::access::address_space::generic_space>
class atomic_ref;
```

### Memory Order Enum

- `sycl::memory_order::relaxed`
- `sycl::memory_order::acquire`
- `sycl::memory_order::release`
- `sycl::memory_order::acq_rel`
- `sycl::memory_order::seq_cst`

### Memory Scope Enum

- `sycl::memory_scope::work_item`
- `sycl::memory_scope::sub_group`
- `sycl::memory_scope::work_group`
- `sycl::memory_scope::device`
- `sycl::memory_scope::system`

### Available Atomic Operations

- All types: `load`, `store`, `exchange`, `compare_exchange_weak`, `compare_exchange_strong`,
  `is_lock_free`
- Integral only: `fetch_add`, `fetch_sub`, `fetch_min`, `fetch_max`, `fetch_and`, `fetch_or`,
  `fetch_xor` (plus operator overloads `+=`, `-=`)
- Float/double (with `ACPP_EXT_FP_ATOMICS`): `fetch_add`

### ACPP_EXT_FP_ATOMICS

```cpp
// Must appear BEFORE the sycl.hpp include
#define ACPP_EXT_FP_ATOMICS
#include <sycl/sycl.hpp>
```

No additional header required. Enables floating-point `fetch_add` on backends that support it.

### 64-bit Atomic Aspect Check

```cpp
if (!device.has(sycl::aspect::atomic64)) {
    // Do not use long/unsigned long long/double in atomic_ref
}
```

### Backend Scope Support Summary (as of research date)

| Backend | sub_group | device | system |
|---------|-----------|--------|--------|
| NVIDIA (PTX) | yes | yes | no |
| AMD (amdgpu) | yes (wave) | yes | no |
| Intel GPU (SPIR-V) | varies | varies | no |
| CPU (LLVM JIT) | no | yes | yes |

### sycl::atomic_fence Signature

```cpp
void sycl::atomic_fence(sycl::memory_order order, sycl::memory_scope scope);
```

`memory_order::relaxed` is a no-op. Minimum supported scopes: `work_item`, `sub_group`, `work_group`.

---

## Chapter Number and Guide Structure

The new atomics chapter is numbered `10`. The existing chapter list ends at `09-real-world-patterns`.
The agent implementing this must:

1. Create `chapters/10-atomics/`
2. Update `PROGRAMMING_GUIDE_IMPLEMENTATION_PLAN.md` (or the equivalent current plan doc) to
   add chapter 10 to the chapter status table
3. Update `README.md` at the project root if it has a chapter list or table of contents

The supplementary nature of this document means it does NOT change the numbering or structure of
chapters 00 through 09. It adds one chapter at the end.

---

## Acceptance Criteria for This Addition

- All five example programs compile and produce correct output
- Chapter 10 README follows the guide's voice: explains concepts before showing code, is
  accessible to a reader with no GPU programming background
- `memory_order` and `memory_scope` are explained with enough depth that a reader understands
  why choosing the wrong scope is a performance bug, not just a style issue
- The deprecated `sycl::atomic<T>` is mentioned once (as what not to use) and never used in examples
- All examples use SSCP (enforced by project CMake) and buffer factory methods where buffers appear
- All cross-references from chapters 04, 05, 08, and 09 are present and accurate
- `ACPP_EXT_FP_ATOMICS` usage is guarded by `#ifdef` so examples compile without the extension
  enabled
- test_runner has run and verified output from all five binaries
