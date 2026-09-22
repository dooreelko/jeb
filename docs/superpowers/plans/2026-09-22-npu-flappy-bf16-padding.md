# NPU-backed Flappy Inference (bf16 model + N-padded prefill matmul) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get Flappy Bird's move-decision inference actually running on the AMD XDNA NPU (via `ggml-hsa`'s XRT backend) and measure whether it's a real speedup over CPU.

**Architecture:** Convert Flappy's model to bf16 (sidesteps quantization support, out of scope). Add N-padding to `ggml-hsa`'s shared `MUL_MAT` dispatch path so short prefill calls (Flappy's prompts are well under the NPU kernel's 128-token tiling requirement) still reach the NPU. Refresh a local `llama.cpp` checkout's existing (stale) `ggml-hsa` overlay to the `xrt-runtime` branch and rebuild it; point Flappy's `llama-cpp-python` at that prebuilt `libllama.so` via the `LLAMA_CPP_LIB_PATH` env var it already supports — no pip rebuild, no changes to `src/jev.py`/`src/common.py`. Validate with a CPU-vs-NPU Flappy run.

**Tech Stack:** C++ (ggml-hsa backend, `/home/doo/projects/ggml` repo, `xrt-runtime` branch), CMake, Python/llama-cpp-python (ctypes bindings), bash.

**Spec:** `docs/superpowers/specs/2026-09-22-npu-flappy-bf16-padding-design.md`

## Global Constraints

- All `ggml-hsa` C++ changes happen in `/home/doo/projects/ggml`, on the existing `xrt-runtime` branch (do not create a new branch there — it's still unmerged from the prior `nom4q` work).
- N-padding logic lives in `ggml-hsa`'s shared tensor-extra code path (`src/ggml-hsa/ggml-hsa.cpp` / `src/ggml-hsa/host-ops.cpp`), **not** gated behind `GGML_HSA_RUNTIME_XRT` — it is a kernel-tiling requirement, not XRT-specific.
- The NPU GEMM kernel tiling constants on this host (aie2p): `m=16, n=16, k=16, n_aie_rows=4, n_aie_cols=8`. The activation-batch dimension N (= `src1->ne[1]` = `dst->ne[1]` for `MUL_MAT`) must be a multiple of `n * n_aie_cols = 128`.
- `/home/doo/projects/llama.cpp` is a plain upstream checkout (not a fork) used only as a local build site for a prebuilt `libllama.so`; it is already fast-forwarded to `origin/master` (`ec5a12b85`, 2026-09-21) and already carries a fresh minimal `ggml-hsa` overlay (copied from `../ggml`'s `xrt-runtime` branch, plus the 2-line CMakeLists patch) as of this plan being written — Task 2 starts from that state, it does not need to redo the ff/overlay.
- XRT paths on this host: `XRT_INCLUDE_DIR=/home/doo/projects/1bit-MONSTER/.local/xrt/usr/include`, `XRT_LIBRARIES=/usr/lib/x86_64-linux-gnu/libxrt_coreutil.so.2;/usr/lib/x86_64-linux-gnu/libxrt_core.so.2`.
- `jeb` repo work happens on a new feature branch (never worktrees, never directly on `main` — see `CLAUDE.md`).
- No decode/autoregressive generation support, no general quantized-weight (Q4_K dequant-on-load) support, no model other than Flappy's — all explicitly out of scope.

---

### Task 1: Produce the bf16 Flappy model

**Files:**
- Create: `models/Qwen3.5-0.8B-bf16.gguf` (binary artifact, not committed to git — confirm `models/*.gguf` is already gitignored before running; if not, add it)

**Interfaces:**
- Consumes: `models/Qwen3.5-0.8B-Q4_K_M.gguf` (existing)
- Produces: `models/Qwen3.5-0.8B-bf16.gguf`, consumed by Task 5's validation run and (after validation) as Flappy's new default model

- [ ] **Step 1: Confirm `models/` is gitignored**

```bash
cd /home/doo/projects/jeb
git check-ignore -v models/Qwen3.5-0.8B-Q4_K_M.gguf || echo "NOT IGNORED"
```

If it prints "NOT IGNORED", add `models/*.gguf` to `.gitignore` and commit that on its own before continuing.

- [ ] **Step 2: Convert to bf16 with `llama-quantize`**

`/home/doo/projects/llama.cpp/build/bin/llama-quantize` requires `LD_LIBRARY_PATH` set to its own bin dir to find `libllama.so`:

```bash
cd /home/doo/projects/jeb
LD_LIBRARY_PATH=/home/doo/projects/llama.cpp/build/bin \
  /home/doo/projects/llama.cpp/build/bin/llama-quantize \
  models/Qwen3.5-0.8B-Q4_K_M.gguf models/Qwen3.5-0.8B-bf16.gguf BF16
```

If this fails (e.g. the existing `build/` predates the fresh `ggml-hsa` overlay from Task 2's prerequisite state and needs a rebuild first), rebuild `llama-quantize` alone before retrying:

```bash
cmake --build /home/doo/projects/llama.cpp/build --target llama-quantize -j"$(nproc)"
```

- [ ] **Step 3: Sanity-check the output loads**

```bash
cd /home/doo/projects/jeb
LD_LIBRARY_PATH=/home/doo/projects/llama.cpp/build/bin \
  /home/doo/projects/llama.cpp/build/bin/llama-cli \
  -m models/Qwen3.5-0.8B-bf16.gguf -p "hello" -n 8 --no-warmup 2>&1 | tail -20
```

Expected: the model loads and produces 8 tokens of output without error (content doesn't matter, only that loading and one prefill+decode succeed — this is a CPU-only sanity check, unrelated to the NPU path).

- [ ] **Step 4: Commit**

Nothing to commit in `jeb` from this task except the `.gitignore` fix if Step 1 needed one:

```bash
cd /home/doo/projects/jeb
git add .gitignore 2>/dev/null || true
git diff --cached --quiet || git commit -m "flappy: gitignore GGUF models"
```

---

### Task 2: Rebuild the NPU-enabled llama.cpp and wire `scripts/flappy-npu.sh`

**Files:**
- Modify (in `/home/doo/projects/llama.cpp`, not `jeb`): rebuild `build/` with the NPU backend enabled
- Create (in `jeb`): `scripts/flappy-npu.sh`

**Interfaces:**
- Consumes: the `ggml-hsa` overlay already present in `/home/doo/projects/llama.cpp` (see Global Constraints — already current as of this plan)
- Produces: `LLAMA_CPP_LIB_PATH`-compatible build output at `/home/doo/projects/llama.cpp/build/bin/`, and `scripts/flappy-npu.sh` for Task 5's validation run

- [ ] **Step 1: Configure the build with the NPU backend**

```bash
cd /home/doo/projects/llama.cpp
cmake -S . -B build \
  -DGGML_HSA=ON -DGGML_HSA_RUNTIME=XRT \
  -DXRT_INCLUDE_DIR=/home/doo/projects/1bit-MONSTER/.local/xrt/usr/include \
  -DXRT_LIBRARIES="/usr/lib/x86_64-linux-gnu/libxrt_coreutil.so.2;/usr/lib/x86_64-linux-gnu/libxrt_core.so.2" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TOOLS=ON
```

Expected: configure completes with `-- Including HSA backend` printed and no `GGML_HSA_RUNTIME=XRT requires XRT_INCLUDE_DIR and XRT_LIBRARIES` fatal error (this exact configuration was already spiked successfully against the vendored `llama-cpp-python` copy of `ggml`; this is the same `ggml-hsa` source against a full `llama.cpp` checkout).

If `ggml-hsa`'s own kernel build step needs the IRON venv / Peano toolchain (JIT-compiling the GEMM kernel at first dispatch, not at CMake configure time — confirm against `../ggml`'s `src/ggml-hsa/README.md` "Required tooling" section, which nom4q already set up on this host), no extra CMake flags are needed for that; it resolves the toolchain paths itself at runtime.

- [ ] **Step 2: Build**

```bash
cmake --build build -j"$(nproc)" --target llama-cli llama-quantize
```

Expected: builds `libllama.so`, `libggml*.so` (including `libggml-hsa.so`), `llama-cli`, `llama-quantize` into `build/bin/`. This can take a while (pybind11 embed + AIE kernel compiler machinery); if it fails on a missing tool, cross-check against `../ggml/src/ggml-hsa/README.md`'s "Required tooling" table — every tool it lists was already installed for `nom4q`, so a failure here means an env var (e.g. `PEANO_INSTALL_DIR`) isn't set in this shell.

- [ ] **Step 3: Smoke-test the NPU backend loads and enumerates**

```bash
cd /home/doo/projects/jeb
LD_LIBRARY_PATH=/home/doo/projects/llama.cpp/build/bin \
  /home/doo/projects/llama.cpp/build/bin/llama-cli \
  -m models/Qwen3.5-0.8B-bf16.gguf -p "hello" -n 8 --no-warmup 2>&1 | grep -iE "hsa|npu|xrt|device" | head -20
```

Expected: log lines showing an HSA/NPU device was enumerated (exact wording depends on `ggml-hsa`'s device-registration logging, established in `nom4q`'s `runtime-xrt.cpp`). This does not yet confirm any op actually dispatched to it — that's Task 5.

- [ ] **Step 4: Write `scripts/flappy-npu.sh`**

```bash
#!/usr/bin/env bash
# Runs Flappy Bird (scripts/flappy.sh) against the NPU-enabled llama.cpp build at
# /home/doo/projects/llama.cpp instead of this project's own .venv llama-cpp-python build.
# See docs/superpowers/specs/2026-09-22-npu-flappy-bf16-padding-design.md.
# usage: scripts/flappy-npu.sh [flappy options]     (works from any directory)
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/_common.sh"
ensure_nix_shell "$@"
cd "$ROOT"

NPU_LLAMA_BIN=/home/doo/projects/llama.cpp/build/bin
if [[ ! -f "$NPU_LLAMA_BIN/libllama.so" ]]; then
  echo "error: $NPU_LLAMA_BIN/libllama.so not found -- build it first (see the implementation plan's Task 2)" >&2
  exit 1
fi
export LLAMA_CPP_LIB_PATH="$NPU_LLAMA_BIN"
export LD_LIBRARY_PATH="$NPU_LLAMA_BIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec scripts/flappy.sh "$@"
```

```bash
chmod +x scripts/flappy-npu.sh
```

- [ ] **Step 5: Verify `LLAMA_CPP_LIB_PATH` is actually honored**

```bash
cd /home/doo/projects/jeb
scripts/flappy-npu.sh --episodes 1 --max-steps 30 --no-watch --model models/Qwen3.5-0.8B-bf16.gguf 2>&1 | tail -20
```

Expected: runs to completion (a short 1-episode, 30-step game), printing an `episode 0: score N steps M` line — proving the custom `libllama.so` loaded and ran end to end. This does not yet prove NPU dispatch happened (Task 5 measures that); it proves the build/wiring works.

- [ ] **Step 6: Commit**

```bash
cd /home/doo/projects/jeb
git add scripts/flappy-npu.sh
git commit -m "flappy: add NPU-backed launch script (LLAMA_CPP_LIB_PATH wiring)"
```

---

### Task 3: N-padding for `MUL_MAT` in `ggml-hsa`

**Files:**
- Modify: `/home/doo/projects/ggml/src/ggml-hsa/ggml-hsa.cpp` (tensor-extra constructor, `ggml_backend_hsa_graph_compute`)
- Modify: `/home/doo/projects/ggml/src/ggml-hsa/host-ops.cpp` and `host-ops.hpp` (new copy helpers)
- Test: `/home/doo/projects/ggml/tests/ggml-hsa/xrt-mul-mat.cpp` (extend with a padded-N case)

**Interfaces:**
- Consumes: `ggml_backend_hsa_tensor_extra` struct, `src_nodes[]`, `node.tensor`, `requires_sync`, `ggml_hsa_set_contiguous_strides` (all existing, from `ggml-hsa.cpp`); `ggml_hsa_copy_tensor(const ggml_tensor*, ggml_tensor*)` (existing, from `host-ops.hpp` — requires equal shape, cannot be reused as-is for this task)
- Produces: `ggml_hsa_copy_tensor_pad_n(const ggml_tensor * src, ggml_tensor * dst)` and `ggml_hsa_copy_tensor_trim_n(const ggml_tensor * src, ggml_tensor * dst, std::int64_t real_n)`, new functions in `host-ops.hpp`/`host-ops.cpp`

This task requires reading the current file before editing — the line numbers below are from the
`xrt-runtime` branch state at the time this plan was written; re-locate the exact blocks by their
surrounding comments if they've shifted.

- [ ] **Step 1: Add the padding decision to the tensor-extra constructor**

In `ggml-hsa.cpp`, the existing bf16-conversion block for `MUL_MAT` (guarded by
`#ifdef GGML_HSA_RUNTIME_XRT`, around line 365-388) already leaves `src_nodes[1]` (the
activations) as bf16 with `convert_dtype = true` and `requires_sync = true` when needed.
Immediately after that `#endif // GGML_HSA_RUNTIME_XRT` block (so it applies regardless of
runtime, per the Global Constraints), add:

```cpp
    // The NPU GEMM kernel (kernels/iron_kernels/gemm.py) requires the activation-batch
    // dimension N (src1->ne[1], which is also the output's ne[1]) to be a multiple of
    // n * n_aie_cols. On this host's aie2p kernel that is 16 * 8 = 128. A short prefill call
    // (Flappy's prompts are well under 128 tokens) would otherwise never reach the kernel at
    // all -- pad both the activation input and the output to the next multiple of 128 here,
    // and trim the output back to the real N after dispatch (see graph_compute below).
    constexpr std::int64_t k_gemm_n_tile = 128;
    if ((node.tensor.op == GGML_OP_MUL_MAT) && (nsrcs == 2)) {
        auto & src_node = src_nodes[1];
        const std::int64_t real_n = src_node.tensor.ne[1];
        const std::int64_t padded_n = GGML_PAD(real_n, k_gemm_n_tile);
        if (padded_n != real_n) {
            src_node.tensor.ne[1] = padded_n;
            ggml_hsa_set_contiguous_strides(src_node.tensor);
            update_src_buffer_size[1] = true;
            src_node.pad_n = true;
            src_node.real_n = real_n;

            node.tensor.ne[1] = padded_n;
            ggml_hsa_set_contiguous_strides(node.tensor);
            node.pad_n = true;
            node.real_n = real_n;
        }
    }
```

This needs two new fields on the per-node struct used for both `node` and each entry of
`src_nodes[]` (find its definition near the top of `ggml-hsa.cpp` or in `common.hpp` --
it's the same struct that already has `.tensor`, `.convert_dtype`, `.buffer_size`): add
`bool pad_n = false;` and `std::int64_t real_n = 0;` alongside the existing `convert_dtype`
field.

Because `node.tensor.ne[1]` (the output) now differs from the real output tensor's `ne[1]`,
the output also needs temporary storage -- the existing `update_src_buffer_size` array only
covers `src_nodes`, not `node` itself. Find where `node`'s own buffer requirement is decided
(search for where `node.convert_dtype` triggers `requires_sync`, or -- if output temp
storage isn't already a case this code handles -- add it explicitly): after the block above,
if `node.pad_n`, set `node.tensor.data = nullptr` and account for
`GGML_PAD(ggml_nbytes(&node.tensor), dev_info.alignment)` in the same total that
`allocate_internal_storage` sums over `src_nodes[]` (extend that function to also include the
output's buffer size when `node.pad_n` is set, mirroring how it already walks `src_nodes[]`).
Set `requires_sync = true` in this case too, matching the existing pattern for `src1`
conversion.

- [ ] **Step 2: Add the padded-copy helpers**

In `host-ops.hpp`, alongside the existing `ggml_hsa_copy_tensor` declaration:

```cpp
/**
 * @brief Copies @p src into @p dst, which has the same leading dimension and a larger ne[1]
 * (the padded activation-batch dimension for a MUL_MAT N-padding case). Rows beyond @p src's
 * ne[1] are zero-filled. Converts types as ggml_hsa_copy_tensor does.
 */
ggml_status ggml_hsa_copy_tensor_pad_n(const ggml_tensor * src, ggml_tensor * dst);

/**
 * @brief Copies the first @p real_n rows (dim 1) of @p src into @p dst, which has ne[1] ==
 * real_n. The inverse of ggml_hsa_copy_tensor_pad_n's row selection, used to trim a padded
 * MUL_MAT output back down after dispatch.
 */
ggml_status ggml_hsa_copy_tensor_trim_n(const ggml_tensor * src, ggml_tensor * dst,
                                        std::int64_t real_n);
```

In `host-ops.cpp`, implement both using the same type-trait dispatch pattern as
`ggml_hsa_copy_tensor_to_cont_tensor_f` above it (same file, ~line 72) -- same nested-loop
structure over `ne[3]/ne[2]/ne[1]/ne[0]`, but:
- `ggml_hsa_copy_tensor_pad_n`: loop `i01` from `0` to `dst->ne[1]`; for `i01 < src->ne[1]`
  copy-convert as the existing helper does; for `i01 >= src->ne[1]`, write a zero of `dst`'s
  element type (use `dst_traits::from_fp32(0.0f)` for non-fundamental types like bf16, or
  `dst_type{}` for fundamental ones -- follow the same `if constexpr` branching already used
  for conversion in the neighboring functions).
- `ggml_hsa_copy_tensor_trim_n`: identical loop structure to `ggml_hsa_copy_tensor`, but
  bounded by `real_n` instead of `src->ne[1]` (assert `dst->ne[1] == real_n` and
  `src->ne[1] >= real_n` instead of `ggml_are_same_shape`).

Register both through the same `ggml_hsa_assign` dispatch helper the existing functions use
(pass the same template-instantiation machinery -- follow `ggml_hsa_copy_tensor`'s definition
at `host-ops.cpp:196` as the template for how `ggml_hsa_assign` is invoked with a functor).

- [ ] **Step 3: Wire the helpers into the dispatch and readback paths**

In `ggml-hsa.cpp`'s `ggml_backend_hsa_graph_compute`:

- Where `requires_sync` triggers a source copy (around line 1016,
  `ggml_hsa_copy_tensor(node->src[src_idx], internal_node.src[src_idx])`): if
  `tensor_extra.src_nodes[src_idx].pad_n`, call `ggml_hsa_copy_tensor_pad_n` instead of
  `ggml_hsa_copy_tensor` for that source.
- Where `convert_dtype` triggers an output copy-back (around line 1037-1046,
  `ggml_hsa_copy_tensor(&internal_node, node)`): if `tensor_extra.node.pad_n`, call
  `ggml_hsa_copy_tensor_trim_n(&internal_node, node, tensor_extra.node.real_n)` instead.
  Note the existing `if (tensor_extra.node.convert_dtype)` guard around this call needs to
  also trigger when `tensor_extra.node.pad_n` is set but `convert_dtype` isn't (a padded-but
  -not-dtype-converted output still needs the trim-copy-back) -- change that condition to
  `if (tensor_extra.node.convert_dtype || tensor_extra.node.pad_n)`.

- [ ] **Step 4: Write the padded-N test**

In `tests/ggml-hsa/xrt-mul-mat.cpp`, add a test case alongside the existing ones (follow
the file's existing structure/helpers for building a `MUL_MAT` graph and running it) using
an M, K, N shape where N is small and not a multiple of 128 (e.g. M=64, K=16, N=17 -- the
minimum tiling-legal M and K, an N Flappy-shaped). Assert the result matches a CPU-computed
reference for exactly the real N rows (do not check the padded rows -- they're discarded and
their content is unspecified).

- [ ] **Step 5: Build and run the ggml-hsa test suite**

```bash
cd /home/doo/projects/ggml
cmake --build build --target ggml-hsa-xrt-mul-mat 2>&1 | tail -30   # adjust target name to match tests/ggml-hsa/CMakeLists.txt
ctest --test-dir build -R xrt- --output-on-failure
```

Expected: all `xrt-` tests pass, including the new padded-N case.

- [ ] **Step 6: Commit**

```bash
cd /home/doo/projects/ggml
git add src/ggml-hsa/ggml-hsa.cpp src/ggml-hsa/host-ops.cpp src/ggml-hsa/host-ops.hpp \
        src/ggml-hsa/common.hpp tests/ggml-hsa/xrt-mul-mat.cpp
git commit -m "ggml-hsa: pad MUL_MAT's N dimension to the kernel's 128-row tile requirement"
```

---

### Task 4: Rebuild `/home/doo/projects/llama.cpp` with the padding change and re-verify

**Files:**
- Rebuild (in `/home/doo/projects/llama.cpp`): refresh the `ggml-hsa` overlay from Task 3's commit, rebuild.

**Interfaces:**
- Consumes: Task 3's commit in `/home/doo/projects/ggml` (`xrt-runtime` branch)
- Produces: an updated `build/bin/libllama.so` with the padding fix, for Task 5

- [ ] **Step 1: Refresh the overlay**

```bash
cp -r /home/doo/projects/ggml/src/ggml-hsa/. /home/doo/projects/llama.cpp/ggml/src/ggml-hsa/
```

- [ ] **Step 2: Rebuild**

```bash
cmake --build /home/doo/projects/llama.cpp/build -j"$(nproc)" --target llama-cli llama-quantize
```

- [ ] **Step 3: Re-run the Task 2 Step 5 smoke test**

```bash
cd /home/doo/projects/jeb
scripts/flappy-npu.sh --episodes 1 --max-steps 30 --no-watch --model models/Qwen3.5-0.8B-bf16.gguf 2>&1 | tail -20
```

Expected: still runs to completion with the rebuilt library.

No commit for this task (the source change was already committed in Task 3; this is just a rebuild of the separate, non-`jeb`, non-`../ggml` checkout).

---

### Task 5: Validate — CPU vs NPU, bf16 model, real Flappy episodes

**Files:**
- Modify: `flappy.md` (append results)

**Interfaces:**
- Consumes: `models/Qwen3.5-0.8B-bf16.gguf` (Task 1), `scripts/flappy-npu.sh` (Task 2), the padding fix (Tasks 3-4)

- [ ] **Step 1: Baseline run — bf16 model, CPU/HIP build (this project's own `.venv`)**

```bash
cd /home/doo/projects/jeb
scripts/flappy.sh --seed 1000 --episodes 3 --max-steps 900 --no-watch \
  --model models/Qwen3.5-0.8B-bf16.gguf
```

Record: the printed `episode N: score S steps T` lines, the `mean score` line, and wall-clock
(`time scripts/flappy.sh ...` if not already captured by the shell).

- [ ] **Step 2: NPU run — same model, same seed**

```bash
cd /home/doo/projects/jeb
time scripts/flappy-npu.sh --seed 1000 --episodes 3 --max-steps 900 --no-watch \
  --model models/Qwen3.5-0.8B-bf16.gguf
```

- [ ] **Step 3: Confirm NPU dispatch actually happened, not silent CPU fallback**

Re-run a short NPU episode with `ggml-hsa`'s own device/dispatch logging visible (check
`../ggml/src/ggml-hsa/README.md` or `ggml-hsa.cpp`'s `GGML_HSA_LOG_INFO` calls for the env
var that controls log verbosity, e.g. `GGML_HSA_LOG_LEVEL` -- follow whatever the nom4q work
established), and confirm dispatch log lines appear for the `MUL_MAT` ops during Flappy's
`jev.probabilities` calls, not just at model load.

- [ ] **Step 4: Compare and record**

Compare Step 1 vs Step 2: scores (should match or be within bf16-rounding noise -- same
model weights, same seed) and wall-clock per-step latency (NPU expected faster if the
padding overhead doesn't dominate at Flappy's small M/K/N -- this is the open question the
whole spec exists to answer).

Append a new dated section to `flappy.md`, following its existing results-section format,
recording: the two runs' scores/means, the wall-clock comparison, and the conclusion (is the
NPU path a net win at Flappy's scale, yes or no, with the numbers). If NPU is not a win,
that's a complete, useful finding, not a bug to chase.

- [ ] **Step 5: If validated as a real speedup, switch Flappy's default model**

Only if Step 4 shows a clear NPU win: change `src/flappy.py`'s `--model` default from
`models/Qwen3.5-0.8B-Q4_K_M.gguf` to `models/Qwen3.5-0.8B-bf16.gguf`, and update `README.md`
wherever it documents the default model.

- [ ] **Step 6: Commit**

```bash
cd /home/doo/projects/jeb
git add flappy.md src/flappy.py README.md 2>/dev/null
git commit -m "flappy: NPU vs CPU validation results (bf16 model, N-padded prefill)"
```

---

## Final note for the executor

Task 3 is the highest-risk task (new tensor-shape-changing dispatch logic in a backend that
already had a Critical use-after-free found and fixed during `nom4q`'s review). Give it the
same rigor: careful review of the buffer-lifetime interaction between `pad_n`'s new temp
storage and the existing `convert_dtype` temp storage (a `MUL_MAT` call can need both at once
-- bf16 conversion AND N-padding on the same `src1` tensor), and a real hardware test run
(Step 5), not just a compile check.
