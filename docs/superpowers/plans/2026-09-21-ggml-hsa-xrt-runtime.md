# ggml-hsa XRT runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `ggml-hsa` (in `../ggml`, branch base `hsa-backend`) run its JIT-compiled NPU kernels through XRT instead of ROCR/HSA, so it works with the in-tree `amdxdna` kernel driver on this host.

**Architecture:** Add a compile-time option `GGML_HSA_RUNTIME` (`HSA` default, `XRT`). The HSA-specific code (agent discovery, memory pools, queue/signal/kernarg dispatch, kernel file loading) moves behind a small internal interface `ggml-hsa/runtime.hpp` with two implementations (`runtime-hsa.cpp`, `runtime-xrt.cpp`). The ggml op selection, tensor-extra/kernel cache and IRON JIT stay shared. In XRT mode the JIT gains a post-step that packages an xclbin with `aiecc` + `xclbinutil`.

**Tech Stack:** C++17, CMake, XRT 2.25 (`xrt::device/hw_context/kernel/bo`), mlir-aie 1.4.0 (`aiecc`, Peano), `xclbinutil` (built from XRT source), Python 3.12 embedded via pybind11.

**Spec:** moth issue `nom4q` (decision record, "Decision: port ggml-hsa dispatch to XRT" and "Step 1 result"). Working notes: this plan.

## Global Constraints

- Work in `/home/doo/projects/ggml` on a feature branch `xrt-runtime` created from `hsa-backend`; never commit to `hsa-backend` directly. No git worktrees.
- Default build (`GGML_HSA_RUNTIME=HSA`) must behave exactly as before; Task 2 is a pure refactor.
- XRT mode targets only aie2p (XDNA2). No new kernels, no quantised types, no decode (n=1) support — out of scope.
- The gemm kernel takes its input dtype from A only, so B (`src1`) must be bf16; kernel argument order is opcode `3`, instruction-stream BO, instruction count, then tensor BOs (src0, src1, dst); XRT kernel name is `MLIR_AIE`.
- Verified environment facts (from spikes): system XRT libs are `/usr/lib/x86_64-linux-gnu/libxrt_coreutil.so.2` / `libxrt_core.so.2` (no dev symlinks); XRT headers are in `/home/doo/projects/1bit-MONSTER/.local/xrt/usr/include`; `xclbinutil` is built at `<scratch>/xcu-build/xclbinutil`; JIT build needs the IRON venv on system Python 3.12 (`PYTHONPATH=<venv>/lib/python3.12/site-packages`, `PEANO_INSTALL_DIR=<venv>/lib/python3.12/site-packages/llvm-aie`) and `LD_LIBRARY_PATH` including `/usr/lib/x86_64-linux-gnu`; `aiecc` needs `--no-xchesscc --no-xbridge` and outputs are requested with `--get-xclbin --get-npu-insts`.
- Correctness bar: NPU result equals CPU reference (max abs error ≤ 1e-3 for bf16 inputs, K ≤ 512) on tile-aligned shapes.
- **Known top risk:** an XRT `hw_context` is created per xclbin and the NPU supports only a few concurrent contexts. Task 5 handles this with a small LRU; if that proves insufficient, fall back to `aiecc --xclbin-input` (multi-kernel xclbin) — do not silently proceed.

## File Structure

- Create `src/ggml-hsa/runtime.hpp` — internal interface both runtimes implement (device enumeration, memory, context, kernel load/dispatch, wait).
- Create `src/ggml-hsa/runtime-hsa.cpp` — existing HSA logic moved out of `ggml-hsa.cpp`, `aie-kernel.cpp`, `kernel-discovery.cpp`.
- Create `src/ggml-hsa/runtime-xrt.cpp` — XRT implementation.
- Modify `src/ggml-hsa/common.hpp` — `device_info` / `ggml_backend_hsa_context` lose raw HSA members behind the interface.
- Modify `src/ggml-hsa/ggml-hsa.cpp`, `aie-kernel.{cpp,hpp}`, `kernel-discovery.cpp` — call the interface.
- Modify `src/ggml-hsa/CMakeLists.txt` — `GGML_HSA_RUNTIME` option, XRT link/include.
- Modify `src/ggml-hsa/kernels/build_iron.py` — xclbin post-step in XRT mode.
- Modify `src/ggml-hsa/README.md` — document the XRT runtime.
- Create `src/ggml-hsa/tests/xrt-mul-mat.cpp` — standalone NPU-vs-CPU MUL_MAT check (from the spike's `mm.cpp`).

---

### Task 1: Feature branch, baseline, and build option

**Files:**
- Modify: `src/ggml-hsa/CMakeLists.txt`

**Interfaces:**
- Produces: CMake cache var `GGML_HSA_RUNTIME` (`HSA`|`XRT`), compile definition `GGML_HSA_RUNTIME_XRT` when XRT.

- [ ] **Step 1: Create the branch and record the baseline**

```bash
cd /home/doo/projects/ggml && git checkout -b xrt-runtime hsa-backend
S=/tmp/claude-1000/-home-doo-projects-jeb/964fbb73-2e7c-498c-bcd3-9d6eb752f3a0/scratchpad
P=/home/doo/projects/1bit-MONSTER/.venv/lib/python3.13/site-packages/_rocm_sdk_core
cmake -S . -B $S/b-hsa -DGGML_HSA=ON -DGGML_HSA_JIT_COMPILE=OFF -Dhsa-runtime64_DIR=/opt/rocm/lib/cmake/hsa-runtime64 -DCMAKE_CXX_FLAGS="-I$P/include" -DCMAKE_BUILD_TYPE=Release
cmake --build $S/b-hsa -j8 --target test-backend-ops
```
Expected: builds; `LD_LIBRARY_PATH=/opt/rocm/lib:/usr/lib/x86_64-linux-gnu $S/b-hsa/bin/test-backend-ops support -b HSA0 -o MUL_MAT | head -5` prints device `HSA0` and `not supported` lines (same as spike 2).

- [ ] **Step 2: Add the option to `CMakeLists.txt`**

Replace the line `find_package(hsa-runtime64 1.0 REQUIRED)` and the `target_link_libraries(ggml-hsa PRIVATE ggml-base hsa-runtime64::hsa-runtime64)` line with:

```cmake
set(GGML_HSA_RUNTIME "HSA" CACHE STRING "ggml-hsa: NPU runtime (HSA or XRT)")
set_property(CACHE GGML_HSA_RUNTIME PROPERTY STRINGS HSA XRT)

if (GGML_HSA_RUNTIME STREQUAL "HSA")
    find_package(hsa-runtime64 1.0 REQUIRED)
elseif (GGML_HSA_RUNTIME STREQUAL "XRT")
    set(XRT_INCLUDE_DIR "" CACHE PATH "directory containing xrt/xrt_device.h")
    set(XRT_LIBRARIES "" CACHE STRING "XRT libraries to link (xrt_coreutil, xrt_core)")
    if (NOT XRT_INCLUDE_DIR OR NOT XRT_LIBRARIES)
        message(FATAL_ERROR "GGML_HSA_RUNTIME=XRT requires XRT_INCLUDE_DIR and XRT_LIBRARIES")
    endif()
else()
    message(FATAL_ERROR "GGML_HSA_RUNTIME must be HSA or XRT")
endif()
```

and (after `ggml_add_backend_library`):

```cmake
target_link_libraries(ggml-hsa PRIVATE ggml-base)
if (GGML_HSA_RUNTIME STREQUAL "HSA")
    target_link_libraries(ggml-hsa PRIVATE hsa-runtime64::hsa-runtime64)
else()
    target_compile_definitions(ggml-hsa PRIVATE GGML_HSA_RUNTIME_XRT)
    target_include_directories(ggml-hsa PRIVATE ${XRT_INCLUDE_DIR})
    target_link_libraries(ggml-hsa PRIVATE ${XRT_LIBRARIES})
endif()
```

- [ ] **Step 3: Rebuild HSA variant, confirm unchanged**

Run the Step 1 build again. Expected: identical output to baseline.

- [ ] **Step 4: Commit**

```bash
git add src/ggml-hsa/CMakeLists.txt
git commit -m "ggml-hsa: add GGML_HSA_RUNTIME build option (HSA default)"
```

---

### Task 2: Extract the runtime interface (pure refactor, HSA only)

**Files:**
- Create: `src/ggml-hsa/runtime.hpp`, `src/ggml-hsa/runtime-hsa.cpp`
- Modify: `src/ggml-hsa/common.hpp`, `src/ggml-hsa/ggml-hsa.cpp`, `src/ggml-hsa/aie-kernel.cpp`, `src/ggml-hsa/aie-kernel.hpp`, `src/ggml-hsa/kernel-discovery.cpp`, `src/ggml-hsa/CMakeLists.txt`

**Interfaces:**
- Produces (`runtime.hpp`), all in namespace-free `ggml_hsa_rt_*` style to match the codebase:

```cpp
#pragma once
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include "ggml.h"

struct ggml_hsa_device_info;
struct ggml_backend_hsa_context;

/// Fills @p info with the NPU devices reachable through this runtime.
void ggml_hsa_rt_enumerate(ggml_hsa_device_info & info);

/// Allocates @p size bytes usable as tensor storage on device @p device (aligned to the device
/// alignment). Returns nullptr on failure.
void * ggml_hsa_rt_alloc(std::int32_t device, std::size_t size);
void ggml_hsa_rt_free(std::int32_t device, void * ptr);

/// Per-backend dispatch state (queue+signal+kernargs for HSA; hw_context cache for XRT).
struct ggml_hsa_rt_context;
std::unique_ptr<ggml_hsa_rt_context, void (*)(ggml_hsa_rt_context *)>
ggml_hsa_rt_context_create(std::int32_t device);

/// Kernel artifact loaded from files produced by the JIT / precompiled kernel directory.
class ggml_hsa_rt_kernel;
std::shared_ptr<ggml_hsa_rt_kernel> ggml_hsa_rt_load_kernel(std::int32_t device,
                                                            const std::filesystem::path & dir,
                                                            const std::string & exported_name);
/// True if the JIT output for @p exported_name exists in @p dir for this runtime.
bool ggml_hsa_rt_kernel_files_exist(const std::filesystem::path & dir,
                                    const std::string & exported_name);

ggml_status ggml_hsa_rt_dispatch(ggml_hsa_rt_context & ctx,
                                 const ggml_hsa_rt_kernel & kernel,
                                 ggml_tensor * const src[],
                                 std::size_t nsrc,
                                 ggml_tensor & dst);
void ggml_hsa_rt_flush(ggml_hsa_rt_context & ctx);
void ggml_hsa_rt_wait(ggml_hsa_rt_context & ctx);
```

- [ ] **Step 1: Write the characterization check first**

Before touching code, capture current behavior of the HSA build so the refactor can be compared:

```bash
S=/tmp/claude-1000/-home-doo-projects-jeb/964fbb73-2e7c-498c-bcd3-9d6eb752f3a0/scratchpad
LD_LIBRARY_PATH=/opt/rocm/lib:/usr/lib/x86_64-linux-gnu $S/b-hsa/bin/test-backend-ops support -b HSA0 > $S/support-before.txt 2>&1; wc -l $S/support-before.txt
```

- [ ] **Step 2: Create `runtime.hpp`** with exactly the interface above.

- [ ] **Step 3: Create `runtime-hsa.cpp`** by moving, without editing bodies, these existing pieces (use the current line numbers on `hsa-backend`; they may shift by a few lines):
  - `ggml-hsa.cpp` `ggml_hsa_find_hsa_agents`, `ggml_hsa_init` device loop, memory-pool helpers (lines ~226–480) → `ggml_hsa_rt_enumerate`.
  - `ggml-hsa.cpp` `hsa_amd_memory_pool_allocate` / `ggml_hsa_delete` uses in buffer allocation (~1022–1046) and `allocate_internal_storage` (~701) → `ggml_hsa_rt_alloc/free`.
  - `ggml_backend_hsa_context` constructor/destructor, `ggml_hsa_flush_dispatches`, `ggml_hsa_wait_dispatches` (~727–823) → `ggml_hsa_rt_context` + `ggml_hsa_rt_flush/wait`.
  - `aie-kernel.cpp` `ggml_hsa_aie_kernel::dispatch` → `ggml_hsa_rt_dispatch`; `ggml_hsa_aie_buffer`/`ggml_hsa_aie_kernel` → `ggml_hsa_rt_kernel`.
  - `kernel-discovery.cpp` `ggml_hsa_find_aie_kernel_files`, `ggml_hsa_load_file`, `ggml_hsa_create_aie_kernel` → `ggml_hsa_rt_kernel_files_exist` / `ggml_hsa_rt_load_kernel`.

  `common.hpp`: keep `hsa_agent_t`, memory-pool members only inside `#ifndef GGML_HSA_RUNTIME_XRT` blocks in `ggml_hsa_device_info::device_info`; replace direct `ctx.queue` / `ctx.dispatch_signal` / `ctx.kernargs` accesses in `ggml-hsa.cpp` with `ggml_hsa_rt_*` calls. `ggml_backend_hsa_context` holds `std::unique_ptr<ggml_hsa_rt_context, ...> rt`.

- [ ] **Step 4: Add `runtime-hsa.cpp` to `GGML_SOURCES_HSA`** (only when `GGML_HSA_RUNTIME STREQUAL "HSA"`) in `CMakeLists.txt`.

- [ ] **Step 5: Build and compare**

```bash
cmake --build $S/b-hsa -j8 --target test-backend-ops
LD_LIBRARY_PATH=/opt/rocm/lib:/usr/lib/x86_64-linux-gnu $S/b-hsa/bin/test-backend-ops support -b HSA0 > $S/support-after.txt 2>&1
diff $S/support-before.txt $S/support-after.txt && echo SAME
```
Expected: `SAME`.

- [ ] **Step 6: Commit**

```bash
git add -A src/ggml-hsa && git commit -m "ggml-hsa: move HSA specifics behind a runtime interface (no behavior change)"
```

---

### Task 3: JIT emits an xclbin in XRT mode

**Files:**
- Modify: `src/ggml-hsa/kernels/build_iron.py`, `src/ggml-hsa/kernel-compiler.cpp` (pass mode flag), `src/ggml-hsa/CMakeLists.txt`

**Interfaces:**
- Consumes: env `GGML_HSA_XCLBINUTIL` (path to `xclbinutil`), env `PEANO_INSTALL_DIR`.
- Produces: for each kernel `<name>`: `<name>.xclbin` (kernel `MLIR_AIE`) and `<name>_insts.bin` in the same output directory (in addition to `<name>.pdi`).

- [ ] **Step 1: Failing check** — run the spike build for a known kernel and assert the xclbin exists:

```bash
S=/tmp/claude-1000/-home-doo-projects-jeb/964fbb73-2e7c-498c-bcd3-9d6eb752f3a0/scratchpad
rm -rf $S/k2 && mkdir $S/k2
GGML_HSA_XCLBIN=1 GGML_HSA_XCLBINUTIL=$S/xbin/xclbinutil PYTHONPATH=/home/doo/projects/ggml/src/ggml-hsa \
  $S/iron-venv/bin/python /home/doo/projects/ggml/src/ggml-hsa/kernels/build.py --op_name MUL_MAT --arch aie2p \
  --input_tensors "(512,512,1,1)/bf16" "(512,512,1,1)/bf16" --output_tensor "(512,512,1,1)/f32" \
  --exported_name mm512 --output_directory $S/k2
ls $S/k2/mm512.xclbin
```
Expected now: FAIL (`No such file`).

- [ ] **Step 2: Implement** in `build_iron.py`, after `compile_mlir_module(...)`:

```python
    if os.environ.get("GGML_HSA_XCLBIN") == "1":
        xclbinutil = os.environ.get("GGML_HSA_XCLBINUTIL", "xclbinutil")
        peano = os.environ.get("PEANO_INSTALL_DIR")
        cmd = [
            "aiecc", "--no-progress", "--no-xchesscc", "--no-xbridge",
            "--alloc-scheme=basic-sequential", "--get-xclbin", "--get-npu-insts",
            f"--xclbin-name={output_directory / (exported_name + '.xclbin')}",
            "--xclbin-kernel-name=MLIR_AIE",
            f"--npu-insts-name={insts_path}",
            f"--xclbinutil-path={xclbinutil}",
            f"--tmpdir={work_dir / 'xclbin-prj'}",
            str(mlir_path),
        ]
        env = dict(os.environ)
        if peano:
            env["PEANO_INSTALL_DIR"] = peano
        subprocess.run(cmd, check=True, cwd=work_dir, env=env)
```
(add `import os, subprocess` at the top of the file if missing; `cwd=work_dir` is required because the core `.o` files are linked by relative path — this was the spike's failure mode.)

- [ ] **Step 3: Run Step 1 again.** Expected: `mm512.xclbin` exists (~135 KB).

- [ ] **Step 4:** In `kernel-compiler.cpp`, when compiled with `GGML_HSA_RUNTIME_XRT`, call `setenv("GGML_HSA_XCLBIN", "1", 1)` before importing the Python build module.

- [ ] **Step 5: Commit** — `git commit -am "ggml-hsa: JIT packages an xclbin in XRT mode"`.

---

### Task 4: XRT runtime — enumeration, memory, kernel load

**Files:**
- Create: `src/ggml-hsa/runtime-xrt.cpp`
- Modify: `src/ggml-hsa/CMakeLists.txt` (add source when `GGML_HSA_RUNTIME STREQUAL "XRT"`)

**Interfaces:**
- Implements `ggml_hsa_rt_enumerate`, `ggml_hsa_rt_alloc/free`, `ggml_hsa_rt_kernel_files_exist`, `ggml_hsa_rt_load_kernel` from Task 2.
- Produces: `class ggml_hsa_rt_kernel` (XRT) holding `xrt::xclbin xclbin`, `std::vector<std::uint32_t> insts`, `std::string exported_name`, and lazily created `xrt::hw_context`/`xrt::kernel` owned by the context (Task 5).

- [ ] **Step 1: Failing test** — enumeration test in `tests/xrt-enum.cpp`: link the XRT build and expect one device named `aie2p`:

```cpp
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
int main() {
    ggml_backend_load_all();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto d = ggml_backend_dev_get(i);
        if (!strcmp(ggml_backend_dev_name(d), "HSA0")) { puts(ggml_backend_dev_description(d)); return 0; }
    }
    return 1;
}
```
Expected before implementation: exit 1.

- [ ] **Step 2: Implement.** Device enumeration: one `xrt::device dev(0)`; name from `dev.get_info<xrt::info::device::name>()`, mapped to arch `aie2p` when the name contains `npu` (this host reports `RyzenAI-npu6`); alignment 4096; memory size = 62554 MiB (query from `/proc/meminfo` `MemTotal`, since the NPU uses system memory). Memory: `ggml_hsa_rt_alloc` = `std::aligned_alloc(4096, round_up(size, 4096))`, `ggml_hsa_rt_free` = `std::free`. Kernel files: `<dir>/<name>.xclbin` and `<dir>/<name>_insts.bin`. Load: `xrt::xclbin{path}`, read the insts as `std::vector<uint32_t>`.

```cpp
static std::vector<std::uint32_t> ggml_hsa_rt_read_words(const std::filesystem::path & p) {
    std::ifstream f(p, std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), {});
    std::vector<std::uint32_t> words(raw.size() / 4);
    std::memcpy(words.data(), raw.data(), words.size() * 4);
    return words;
}
```

- [ ] **Step 3: Build** with:

```bash
cmake -S . -B $S/b-xrt -DGGML_HSA=ON -DGGML_HSA_RUNTIME=XRT -DGGML_HSA_JIT_COMPILE=ON \
  -DXRT_INCLUDE_DIR="/home/doo/projects/1bit-MONSTER/.local/xrt/usr/include;/nix/store/0y71r0ad2bvq63066a12y2s5d53imgzq-util-linux-minimal-static-x86_64-unknown-linux-musl-2.42-dev/include" \
  -DXRT_LIBRARIES="/usr/lib/x86_64-linux-gnu/libxrt_coreutil.so.2;/usr/lib/x86_64-linux-gnu/libxrt_core.so.2" -DCMAKE_BUILD_TYPE=Release
cmake --build $S/b-xrt -j8
```

- [ ] **Step 4: Run Step 1's test.** Expected: prints the device description, exit 0.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "ggml-hsa: XRT runtime enumeration, memory and kernel loading"`.

---

### Task 5: XRT runtime — context and dispatch

**Files:**
- Modify: `src/ggml-hsa/runtime-xrt.cpp`

**Interfaces:**
- Implements `ggml_hsa_rt_context_create`, `ggml_hsa_rt_dispatch`, `ggml_hsa_rt_flush`, `ggml_hsa_rt_wait`.
- `ggml_hsa_rt_context` (XRT) holds: `xrt::device dev`, an LRU list (capacity 4, constant `k_max_live_contexts`) of `{kernel-name, xrt::hw_context, xrt::kernel}`, a `std::vector<xrt::run>` of in-flight runs, and a `std::unordered_map<void*, xrt::bo>` cache of user-pointer BOs.

- [ ] **Step 1: Failing test** — `tests/xrt-mul-mat.cpp`: copy the spike program `scratchpad/t/mm.cpp` (ggml graph `ggml_mul_mat(bf16 A[K,M], f32 B[K,N])` on HSA0 vs CPU, prints max abs err) and add `if (err > 1e-3) return 1;`. Run with K=M=N=512. Expected before implementation: `not supported`/abort.

- [ ] **Step 2: Implement dispatch.**

```cpp
ggml_status ggml_hsa_rt_dispatch(ggml_hsa_rt_context & ctx, const ggml_hsa_rt_kernel & k,
                                 ggml_tensor * const src[], std::size_t nsrc, ggml_tensor & dst) {
    auto & entry = ctx.acquire(k);              // creates/evicts hw_context + xrt::kernel (LRU)
    auto bo_insts = ctx.bo_for(k.insts.data(), k.insts.size() * 4, entry.kernel.group_id(1),
                               XCL_BO_FLAGS_CACHEABLE);
    // arg order: opcode 3, insts BO, insts count, then src..., dst
    // BO creation from user pointers: xrt::bo(device, userptr, size, flags, group)
    auto run = xrt::run(entry.kernel);
    run.set_arg(0, 3);
    run.set_arg(1, bo_insts);
    run.set_arg(2, static_cast<unsigned>(k.insts.size()));
    int arg = 3;
    for (std::size_t i = 0; i < nsrc; ++i) {
        run.set_arg(arg, ctx.bo_for(src[i]->data, ggml_nbytes(src[i]), entry.kernel.group_id(arg),
                                    XRT_BO_FLAGS_HOST_ONLY));
        ++arg;
    }
    run.set_arg(arg, ctx.bo_for(dst.data, ggml_nbytes(&dst), entry.kernel.group_id(arg),
                                XRT_BO_FLAGS_HOST_ONLY));
    run.start();
    ctx.inflight.push_back(std::move(run));
    return GGML_STATUS_SUCCESS;
}
void ggml_hsa_rt_wait(ggml_hsa_rt_context & ctx) {
    for (auto & r : ctx.inflight) { r.wait(); }
    ctx.inflight.clear();
}
void ggml_hsa_rt_flush(ggml_hsa_rt_context &) {}  // runs are started immediately
```

  `acquire()` must call `ggml_hsa_rt_wait(*this)` before evicting an entry, because hw_context destruction must not race in-flight runs. BOs from user pointers require the pointer to be page-aligned and the size a multiple of 4096 in the spike's XRT version; if a tensor is not, `bo_for` must fall back to a bounce BO (allocate an `xrt::bo`, copy in before start, copy out after wait) — implement the bounce path in this task, not later, because ggml tensors inside a shared buffer are only 64-byte aligned.

- [ ] **Step 3: Sync BOs.** Before `start()`, `sync(XCL_BO_SYNC_BO_TO_DEVICE)` for inputs; after `wait()`, `sync(XCL_BO_SYNC_BO_FROM_DEVICE)` for outputs (record outputs in `ctx.pending_outputs`).

- [ ] **Step 4: Run the test** (env: IRON venv, `GGML_HSA_XCLBINUTIL`, `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu`, `PYTHONPATH`, `PEANO_INSTALL_DIR` as in Global Constraints):

```bash
$S/b-xrt/bin/xrt-mul-mat 512 512 512
```
Expected: `max abs err` ≤ 1e-3, exit 0. If B arrives as f32 and the result is wrong, the bf16 conversion is missing — see Task 6.

- [ ] **Step 5: Commit** — `git commit -am "ggml-hsa: XRT context and dispatch with hw_context LRU"`.

---

### Task 6: bf16 src1 handling and validation

**Files:**
- Modify: `src/ggml-hsa/ggml-hsa.cpp` (only if the existing tensor-extra path does not already convert), `src/ggml-hsa/README.md`

- [ ] **Step 1: Check whether conversion already happens.** In `ggml_backend_hsa_tensor_extra`'s constructor (`ggml-hsa.cpp` ~575–680) look for the `substitute_fp16_bf16` / temporary-conversion path for `src1`. Run the Task 5 test with K=M=N=512. If error ≤ 1e-3, conversion exists: skip to Step 3.

- [ ] **Step 2 (only if needed): add the conversion.** Mark `src1` of `GGML_OP_MUL_MAT` as needing an f32→bf16 host conversion node (same mechanism the tensor extra uses for other dtype substitutions), sized `ggml_nelements(src1) * 2` bytes of temporary storage via `ggml_hsa_rt_alloc`.

- [ ] **Step 3: Full validation.**

```bash
$S/b-xrt/bin/test-backend-ops -b HSA0 -o MUL_MAT -p 'type_a=bf16,type_b=f32' 2>&1 | tail -20
$S/b-xrt/bin/xrt-mul-mat 512 512 512
$S/b-xrt/bin/xrt-mul-mat 1024 1024 1024
```
Expected: tile-aligned cases pass; unaligned/gemv cases report `not supported` (out of scope), never a wrong result or a crash. Record avg ms per shape vs the CPU column printed by the test.

- [ ] **Step 4: Docs.** Add an "XRT runtime" section to `src/ggml-hsa/README.md`: when to use it (in-tree `amdxdna`), the build flags from Task 4 Step 3, the `xclbinutil`/`aiecc` requirements, and the known limits (tile-aligned bf16 gemm only).

- [ ] **Step 5: Commit** — `git commit -am "ggml-hsa: validate XRT runtime, document build and limits"`.

---

### Task 7: Report back in moth

- [ ] **Step 1:** `moth show nom4q`, append a "Step 2–6 result" section (measured error, ms vs CPU, LRU behavior, what did not work) to the existing description via `moth update`, keeping all prior text. Do not run `moth done` or `moth start`.

## Self-Review

- **Spec coverage:** seam (Tasks 1–2), JIT post-step (3), XRT device/memory/kernel/dispatch (4–5), bf16 contract + validation vs CPU (6), decision-record update (7). Out-of-scope items (decode kernels, quantised types) are stated in Global Constraints. The three risks in the decision record map to: xclbin packaging → Task 3; BO/pointer handling → Task 5 bounce path; argument order → Global Constraints + Task 5 code.
- **Placeholders:** Task 2 Step 3 references existing line ranges rather than restating ~500 lines of moved code; the instruction is a pure move, verified by the before/after diff in Steps 1 and 5. `runtime-xrt.cpp` helper `ctx.acquire`/`ctx.bo_for` are specified by contract in Task 5's Interfaces block and must be written in that task.
- **Type consistency:** interface names (`ggml_hsa_rt_*`) are identical across Tasks 2, 4 and 5; kernel-name constant `MLIR_AIE`, arg order and opcode `3` match the working spike program.
