# Revision Plan: AdaptiveCpp Programmer's Guide

**Date:** 2026-02-27  
**Status:** Planning  
**Scope:** Major revision pass addressing depth, canonical patterns, and structural rework

---

## Goal Statement

Address three categories of feedback from a full read-through of the guide:

1. **Missing canonical pattern** -- the USM-public/buffer-internal bridge that justifies the guide's buffer-first recommendation was never shown
2. **Insufficient concept depth** -- the guide currently reads as a feature catalogue ("here's A, here's B") with no in-depth treatment of individual concepts
3. **Structural gap in chapter 06** -- the extensions chapter is a monolithic overview rather than the per-extension deep-dives it needs to be

Additionally, incorporate maintainer feedback from AdaptiveCpp discussion #1981 regarding scoped parallelism stance and dual-version strategy.

---

## Research Findings

### Buffer-USM Interop API

The `sycl::buffer_allocation` namespace is **always enabled** -- no `#define` required. Key API:

- `view(T* ptr, device dev, no_ownership)` -- wrap a pointer that holds current valid data; runtime skips transfers
- `empty_view(T* ptr, device dev, no_ownership)` -- wrap an uninitialized pointer; runtime transfers on first access
- `no_ownership` / `take_ownership` -- controls whether the buffer frees the pointer on destruction
- Buffer introspection: `get_pointer(dev)`, `has_allocation(dev)`, `for_each_allocation(handler)`

When a buffer is constructed on an existing device allocation for the same queue's device, **no data transfer occurs**. This is the key property that makes the bridge pattern zero-cost.

### Maintainer Feedback (illuhad, discussion #1981)

- **Scoped parallelism:** Not being implemented for SSCP. SSCP's compiler transformations already provide efficient CPU support. The original motivation is gone. `nd_range` is the primary model going forward.
- **Buffers for beginners:** Maintainer discourages exposing the buffer-accessor API directly to beginners due to complexity and footguns. Recommends in-order queues + USM as the lower-friction onboarding path.
- **Dual-version strategy:** Standalone guide (buffer-first, author's convictions) and an upstreamed version (USM-first, nd_range primary, SYCL Academy references) are both acceptable. Author committed to pursuing both.
- **SYCL Academy:** Complementary, not competing. Academy covers spec-compliant SYCL; acpp guide should cover AdaptiveCpp-specific extensions, compilation model, and ecosystem tooling.
- **stdpar:** Suggested as a beginner-friendly addition worth considering.

### SYCL Academy Pedagogical Model

- Concept introduction paired with hands-on exercise (source.cpp skeleton + solution.cpp)
- Progressive scaffolding: foundational -> intermediate -> advanced
- Explicit learning objectives per exercise
- Self-contained repository with no external dependencies
- Separation of theory (Lesson_Materials/) and practice (Code_Exercises/)

The guide should draw inspiration from this structure: every major concept section should have a clear "what is this, why does it exist, when do you need it" before showing code.

### Namespace / Include Correctness (Non-Negotiable)

- All code: `sycl::` namespace only, **never** `hipsycl::sycl::`
- Include: `<sycl/sycl.hpp>` not `<hipSYCL/sycl/sycl.hpp>`
- Buffer-USM interop: always enabled, no `#define` needed
- Explicit buffer policies: requires `#define ACPP_EXT_EXPLICIT_BUFFER_POLICIES` before include
- Other extensions vary -- always check whether a `#define` is required

---

## Codebase Audit Findings

### Chapter 04 (Memory Model) -- Buffer-USM Interop Gap

The bridge pattern is **entirely absent**. The chapter mentions `ACPP_EXT_BUFFER_USM_INTEROP` briefly (lines 205-221) with only introspection API (`get_pointer`, `has_allocation`), but contains no explanation of the canonical pattern and no example demonstrating it.

`buffer_policies.cpp` covers factory method variants well but has no USM-interop example. `usm_vector_add.cpp` covers basic USM allocation and is correct.

### Chapter 06 (Extensions) -- Structural Rework Required

**Current state:** Single monolithic `README.md` (~200 lines) covering all extensions in summary form with 2 examples total (`jit_specialized.cpp`, `accessor_variants_demo.cpp`).

**Planned state (per PROGRAMMING_GUIDE_IMPLEMENTATION_PLAN.md):** Individual subdirectories per extension, each with its own README and example(s).

This is the single largest structural gap in the guide.

### Chapter 02 (SYCL Overview) -- Execution DAG Needs Depth

Events are covered at a surface level (lines 180-202). The execution DAG concept is mentioned in one sentence (line 93) but is not explained as a foundational concept. A reader who doesn't understand the DAG will not understand why buffers are valuable.

### Chapter 05 (Kernels & Parallelism) -- Mostly Good, Minor Gaps

Local memory, private memory, work groups, and barriers are covered well. Missing: an explicit "what is a work group" definition stated plainly for absolute beginners before diving into code.

### Chapter 07 (Performance) -- Kernel Fusion Missing

Coarse-grained events are covered well. Kernel fusion via `ACPP_EXT_DYNAMIC_FUNCTIONS` is **not covered at all**.

---

## The Canonical Pattern: USM-Public / Buffer-Internal Bridge

This pattern is the cornerstone justification for the guide's buffer-first approach. It must be prominently documented. The key insight: **the public API stays pointer-based (USM), so callers interact with the familiar CUDA-like interface; the internal implementation uses buffer accessors for automatic DAG construction with no manual event chaining.**

```cpp
// Public API: caller passes USM pointers, gets an event back
sycl::event complex_algorithm(
    sycl::queue& q,
    float* input1,   // device USM, read-only
    float* input2,   // device USM, read-only
    float* output,   // device USM, written during algorithm
    size_t n,
    std::span<const sycl::event> deps)
{
    using namespace sycl::buffer_allocation;

    // Async views: non-blocking destructors, no writeback, no data transfer
    // (buffers constructed on pre-existing device allocations for this device)
    auto input1_buf = sycl::make_async_usm_view<float, 1>(
        view(input1, q.get_device(), no_ownership), sycl::range<1>(n));
    auto input2_buf = sycl::make_async_usm_view<float, 1>(
        view(input2, q.get_device(), no_ownership), sycl::range<1>(n));
    auto output_buf = sycl::make_async_usm_view<float, 1>(
        view(output, q.get_device(), no_ownership), sycl::range<1>(n));

    // Internal implementation uses buffer API: accessor conflicts become
    // DAG edges automatically, no manual event chaining needed.
    return complex_algorithm_impl(q, input1_buf, input2_buf, output_buf, deps);
    // Async view destructors fire here: non-blocking, no writeback.
}

// Synchronous variant: caller just waits for output to be ready
void complex_algorithm_sync(
    sycl::queue& q,
    float* input1, float* input2, float* output,
    size_t n, std::span<const sycl::event> deps)
{
    using namespace sycl::buffer_allocation;

    auto input1_buf = sycl::make_async_usm_view<float, 1>(
        view(input1, q.get_device(), no_ownership), sycl::range<1>(n));
    auto input2_buf = sycl::make_async_usm_view<float, 1>(
        view(input2, q.get_device(), no_ownership), sycl::range<1>(n));

    // Sync view for output: destructor BLOCKS until all writes complete
    auto output_buf = sycl::make_sync_usm_view<float, 1>(
        view(output, q.get_device(), no_ownership), sycl::range<1>(n));

    complex_algorithm_impl(q, input1_buf, input2_buf, output_buf, deps);
    // sync output_buf destructor fires here: blocks until output is ready
}

// Internal: pure buffer API, automatic dependency graph
sycl::event complex_algorithm_impl(
    sycl::queue& q,
    sycl::buffer<float, 1>& input1,
    sycl::buffer<float, 1>& input2,
    sycl::buffer<float, 1>& output,
    std::span<const sycl::event> deps)
{
    step1(q, input1, output, deps);   // DAG edge: output write
    step2(q, input2, output, {});     // DAG edge: output read-write conflict with step1
    return step3(q, input1, input2, output, {}); // DAG edges: chained automatically
}
```

**Properties of this pattern:**
- Public contract: pointer-based, familiar to CUDA developers
- Zero allocations, zero data movement on pre-existing device memory
- Automatic DAG: accessor read-write conflicts create dependency edges without manual event chains
- Non-owning semantics: caller retains full pointer lifetime control
- Async variant: returns event, caller chains further work
- Sync variant: blocks on destruction, simplest for one-shot calls

---

## Recommended Approach

### 1. Fix Scoped Parallelism Stance

Update AGENTS.md and chapter 05 to reflect maintainer's guidance: `nd_range` is the primary model. Scoped parallelism is historical context (ACPP_EXT_SCOPED_PARALLELISM_V2 was never implemented for SSCP and there are no plans to do so). The "scoped parallelism as primary abstraction" core philosophy entry must be revised.

### 2. Add Execution DAG as a First-Class Concept (Chapter 02)

Before explaining buffers, explain the DAG. Readers need to understand:
- What a command group is and when it executes
- What an accessor conflict means (read-write, write-write)
- How the runtime builds edges from accessor conflicts
- Why this eliminates manual event chaining for complex multi-kernel algorithms
- What `sycl::event` is and when you need it (vs. when the DAG handles it)

### 3. Add the Bridge Pattern to Chapter 04

After introducing USM, add a new section: "Combining Buffers and USM: The Bridge Pattern". This section explains:
- Why you might want buffer internals with a USM-facing public API
- The `sycl::buffer_allocation` namespace (always-on, no `#define`)
- `view()` vs. `empty_view()` and when to use each
- `no_ownership` as the default and why it matters
- Zero-copy construction on pre-existing device allocations
- The async vs. sync variant tradeoff
- A complete worked example (`usm_bridge_pattern.cpp`)

### 4. Restructure Chapter 06: Per-Extension Subsections

Break the monolithic `README.md` into a chapter index plus individual subsection files. Each extension gets its own `README.md` with:
- What problem this extension solves (motivation)
- When to use it (vs. alternatives)
- Complete API surface with `sycl::` namespace
- Whether a `#define` is required or the extension is always-on
- Annotated example
- Known footguns or caveats

**Directory structure:**

```
chapters/06-acpp-extensions/
    README.md                        <- chapter index / overview
    01-explicit-buffer-policies/
        README.md
        (examples already exist in parent examples/)
    02-buffer-usm-interop/
        README.md
        examples/
            usm_bridge_pattern.cpp   <- new: the canonical bridge pattern
            CMakeLists.txt
    03-jit-specialization/
        README.md
        (jit_specialized.cpp already exists)
    04-jit-compile-if/
        README.md
        (no example yet -- add one or reference jit_specialized)
    05-dynamic-functions/
        README.md
        examples/
            kernel_fusion_demo.cpp   <- new
            CMakeLists.txt
    06-accessor-variants/
        README.md
        (accessor_variants_demo.cpp already exists)
    07-coarse-grained-events/
        README.md
        (no dedicated example -- add one or reference bandwidth_benchmark)
    08-multi-device-queue/
        README.md
        examples/
            multi_device_demo.cpp    <- new
            CMakeLists.txt
    09-scoped-parallelism/
        README.md  <- historical context only; not recommended for SSCP
```

### 5. Flesh Out Concept Depth Across All Chapters

Every chapter needs a consistent "concept -> why -> how -> example" structure. Specific gaps to address:

**Chapter 02:**
- Execution DAG: what it is, how accessor conflicts create edges, why it matters
- Events: what they are, when you get them back, how to chain manually (and when you don't need to)
- In-order vs. out-of-order queues: explicit tradeoff discussion

**Chapter 04:**
- Explain the mental model for "why buffers": automatic scheduling vs. manual pointer management
- The bridge pattern section (see step 3 above)

**Chapter 05:**
- Add explicit definitions for: work item, work group, local memory, private memory, sub-group
- Define these as a mini-glossary before showing nd_range code
- Explain why local memory matters for performance (bandwidth vs. latency hierarchy)
- `nd_range` as primary; scoped parallelism as historical note

**Chapter 06:**
- Full restructure (see step 4 above)

**Chapter 07:**
- Add kernel fusion section covering `ACPP_EXT_DYNAMIC_FUNCTIONS`
- Explain what fusion means (eliminating intermediate memory round-trips)
- When fusion helps and when it doesn't

### 6. Draw from SYCL Academy's Structure

For each concept section, adopt the SYCL Academy model:
- Open with a plain-English "what is this?" paragraph (written for someone who has never heard the term)
- Follow with "why does this exist?" (motivation, what problem it solves)
- Then "how do you use it?" (API walkthrough)
- Then annotated example code
- Close with "common mistakes" or "when NOT to use this"

---

## Implementation Steps

### Phase 1: Foundation Fixes (High Priority)

1. **Update AGENTS.md** -- revise "Scoped Parallelism as Primary Abstraction" core philosophy to reflect that `nd_range` is primary; scoped parallelism is historical context for SSCP users.

2. **Update chapter 05 README** -- flip nd_range to primary, reframe scoped parallelism section as "historical context / CPU-heavy workloads on non-SSCP targets only"; add explicit definitions for work item, work group, local memory, private memory.

3. **Deepen chapter 02 README** -- add a dedicated "Execution DAG" section explaining command groups, accessor conflicts as dependency edges, and the event model. Add an "Events and Explicit Dependencies" section with depth.

### Phase 2: Chapter 04 Bridge Pattern (High Priority)

4. **Add bridge pattern section to chapter 04 README** -- new section "Combining Buffers and USM: The Bridge Pattern" with full explanation of `sycl::buffer_allocation`, `view()` / `empty_view()`, `no_ownership`, zero-copy construction, and async vs. sync variant tradeoff.

5. **Add `usm_bridge_pattern.cpp` example** -- a complete, self-contained example demonstrating the canonical pattern: a multi-step algorithm with USM public API and buffer internal implementation. Include both async (returns event) and sync (blocking destructor) variants.

6. **Update chapter 04 CMakeLists.txt** -- add `add_acpp_example` entry for the new example.

### Phase 3: Chapter 06 Restructure (High Priority)

7. **Create per-extension subdirectory structure** -- create all subdirectories listed in step 4 of Recommended Approach above.

8. **Write per-extension README files** -- each extension gets motivation + API + example + caveats. Cover at minimum:
   - `01-explicit-buffer-policies/README.md`
   - `02-buffer-usm-interop/README.md`
   - `03-jit-specialization/README.md`
   - `04-jit-compile-if/README.md`
   - `05-dynamic-functions/README.md`
   - `06-accessor-variants/README.md`
   - `07-coarse-grained-events/README.md`
   - `08-multi-device-queue/README.md`
   - `09-scoped-parallelism/README.md`

9. **Rewrite chapter 06 index README** -- make it a navigable overview that links to each subsection with a one-sentence description and "use when" guidance.

10. **Add `kernel_fusion_demo.cpp` example** -- demonstrates `ACPP_EXT_DYNAMIC_FUNCTIONS` for runtime kernel fusion with before/after memory bandwidth comparison.

11. **Add CMakeLists.txt entries** -- for any new examples under chapter 06.

### Phase 4: Remaining Concept Depth (Medium Priority)

12. **Deepen chapter 07 README** -- add kernel fusion section covering `ACPP_EXT_DYNAMIC_FUNCTIONS`; explain what fusion means and when it's beneficial.

13. **Audit remaining chapters for surface-level mentions** -- scan each chapter for terms that are used but not defined (kernel fusion, local memory tiling, sub-groups, memory fence, etc.) and add plain-English definitions at first use.

### Phase 5: AGENTS.md + Build Verification (Medium Priority)

14. **Update chapter status table in AGENTS.md** -- mark chapters affected by the restructure as needing reverification.

15. **Build and run all new examples** -- `pixi run configure && pixi run build`, then `pixi run ./build/...` for each new binary to verify correct output.

---

## Key Files to Modify

| File | What Changes |
|------|-------------|
| `AGENTS.md` | Revise "Scoped Parallelism as Primary Abstraction" entry; update chapter status table |
| `chapters/02-sycl-overview/README.md` | Add execution DAG section; deepen events explanation |
| `chapters/04-memory-model/README.md` | Add bridge pattern section with full `sycl::buffer_allocation` API explanation |
| `chapters/04-memory-model/examples/CMakeLists.txt` | Add `usm_bridge_pattern` entry |
| `chapters/04-memory-model/examples/usm_bridge_pattern.cpp` | New: canonical USM-public/buffer-internal example |
| `chapters/05-kernels-and-parallelism/README.md` | Flip nd_range to primary; add concept glossary; reframe scoped parallelism |
| `chapters/06-acpp-extensions/README.md` | Rewrite as navigable chapter index |
| `chapters/06-acpp-extensions/01-explicit-buffer-policies/README.md` | New: deep-dive |
| `chapters/06-acpp-extensions/02-buffer-usm-interop/README.md` | New: deep-dive |
| `chapters/06-acpp-extensions/02-buffer-usm-interop/examples/usm_bridge_pattern.cpp` | New: or symlink/reference ch04 example |
| `chapters/06-acpp-extensions/03-jit-specialization/README.md` | New: deep-dive |
| `chapters/06-acpp-extensions/04-jit-compile-if/README.md` | New: deep-dive |
| `chapters/06-acpp-extensions/05-dynamic-functions/README.md` | New: deep-dive |
| `chapters/06-acpp-extensions/05-dynamic-functions/examples/kernel_fusion_demo.cpp` | New |
| `chapters/06-acpp-extensions/06-accessor-variants/README.md` | New: deep-dive |
| `chapters/06-acpp-extensions/07-coarse-grained-events/README.md` | New: deep-dive |
| `chapters/06-acpp-extensions/08-multi-device-queue/README.md` | New: deep-dive |
| `chapters/06-acpp-extensions/09-scoped-parallelism/README.md` | New: historical context |
| `chapters/06-acpp-extensions/examples/CMakeLists.txt` | Add new example targets |
| `chapters/07-performance/README.md` | Add kernel fusion section |

---

## Acceptance Criteria

- [ ] `AGENTS.md` no longer lists "Scoped Parallelism as Primary Abstraction" as a core philosophy; `nd_range` is primary
- [ ] Chapter 02 contains a dedicated "Execution DAG" section that explains what accessor conflicts are and how they become dependency edges
- [ ] Chapter 02 contains an "Events" section that explains what `sycl::event` is, when you get one, and when the DAG makes manual chaining unnecessary
- [ ] Chapter 04 contains a "Bridge Pattern" section explaining the entire `sycl::buffer_allocation` namespace at tutorial depth
- [ ] `usm_bridge_pattern.cpp` compiles and runs correctly, demonstrating both async (event-returning) and sync (blocking-destructor) variants
- [ ] Chapter 06 is restructured into per-extension subdirectory files; the monolithic README is replaced with a chapter index
- [ ] Each per-extension README follows the "what / why / how / example / gotchas" structure
- [ ] `kernel_fusion_demo.cpp` compiles and runs correctly
- [ ] Chapter 05 explicitly defines work item, work group, local memory, private memory before using these terms in code examples
- [ ] Chapter 07 covers kernel fusion via `ACPP_EXT_DYNAMIC_FUNCTIONS`
- [ ] All new code uses `sycl::` namespace exclusively (no `hipsycl::sycl::`)
- [ ] All new code uses `<sycl/sycl.hpp>` include path
- [ ] `pixi run configure && pixi run build` succeeds with no errors
- [ ] All new executables produce correct output when run via `pixi run ./build/...`

---

## Out of Scope for This Revision

- stdpar (C++ standard parallelism offloading) -- noted as a future addition based on maintainer suggestion; not part of this pass
- Creating an upstreamed USM-first variant of the guide -- a separate future effort; this revision focuses on the standalone buffer-first guide
- SYCL Academy integration or cross-referencing -- future effort once upstreamed version strategy is finalized
- Chapter 10 (atomics) -- considered complete; no changes planned
- Chapter 09 (real-world patterns) -- `jacobi_solver.cpp` already demonstrates a mixed buffer/USM pattern; may add a note pointing to bridge pattern once ch04 section is written, but no major rework
