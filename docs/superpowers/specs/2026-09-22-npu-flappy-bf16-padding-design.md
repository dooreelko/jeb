# NPU-backed Flappy inference: bf16 model + N-padded prefill matmul

**Moth issue:** q8kte — N Padded Prefill Matmul On Npu For Flappy
**Depends on:** nom4q (ggml-hsa XRT runtime, complete, unmerged branch `../ggml#xrt-runtime`)

## Problem

The NPU (`ggml-hsa`, XRT runtime) can run bf16 GEMM correctly and fast (established in
nom4q), but nothing in this project actually reaches it yet. Two blockers stand between
"NPU backend exists" and "Flappy uses the NPU":

1. **Quantization.** Flappy's models are Q4_K_M. `ggml-hsa`'s GEMM kernel types both
   matmul operands from the weight tensor's dtype, and only has bf16 kernels — it does not
   dequantize. A Q4_K_M model never reaches the NPU path at all.
2. **Tiling.** The kernel requires the activation-batch dimension N (token count for a
   prefill call) to be a multiple of 128 on this host's NPU (aie2p: `n=16, n_aie_cols=8`).
   Flappy's prompts are single-digit-to-low-double-digit tokens — every real call falls
   back to CPU silently, even once weights are bf16.

Flappy is the right first target because its inference shape is simple: one prefill pass
per game step (`jev.probabilities`), no autoregressive decode loop — and this kernel design
doesn't support decode (N must be ≥128) regardless, so a target that never decodes sidesteps
that limitation entirely.

## Scope

In scope: a bf16 Flappy model, N-padding in `ggml-hsa`'s shared dispatch path, a way to
actually build/run `llama-cpp-python` against the `ggml-hsa` XRT backend, and a wall-clock
comparison of NPU vs CPU on real Flappy episodes.

Out of scope: general quantized-weight (Q4_K dequant-on-load) support, decode/autoregressive
generation on the NPU, any model other than Flappy's.

## Design

### 1. Model: bf16 conversion

Convert the existing `models/Qwen3.5-0.8B-Q4_K_M.gguf` lineage to bf16 using llama.cpp's own
tooling, output `models/Qwen3.5-0.8B-bf16.gguf`. If a bf16 (or f16/f32) intermediate of this
model isn't already available, regenerate from source with `convert_hf_to_gguf.py
--outtype bf16`; otherwise `llama-quantize <in>.gguf <out>.gguf bf16` on a non-quantized
ancestor. No new model is downloaded. Flappy's `--model` default in `src/flappy.py` changes
to the new path only after validation (Section 4) confirms the NPU path is actually
exercised and scores stay consistent.

### 2. Build integration: llama-cpp-python overlay

Flappy runs on `llama-cpp-python` (via `src/jev.py` → `src/common.py`'s `Model`), which is
built by `scripts/build-llama.sh` today via `pip install --no-binary llama-cpp-python`
against **`llama-cpp-python`'s own vendored `llama.cpp`/`ggml` copy** — not `../ggml`
directly. Confirmed by spike: `ggml-hsa`'s footprint outside its own source directory is two
lines (`option(GGML_HSA ...)` in `ggml/CMakeLists.txt`, `ggml_add_backend(HSA)` in
`ggml/src/CMakeLists.txt`), so overlaying it onto the vendored copy is sufficient — no fork
to maintain, no change to `src/jev.py`/`src/common.py`.

New script `scripts/build-llama-npu.sh`, same shape as `build-llama.sh`:

1. Clone `llama-cpp-python` at the pinned tag matching the project's current version
   (`0.3.35`) plus its `vendor/llama.cpp` submodule into a scratch dir (git-ignored, under
   e.g. `.build/llama-cpp-python-npu/`, recreated each run — same disposability as the
   existing pip `--no-cache` rebuild).
2. Copy `../ggml`'s `src/ggml-hsa/` (from the `xrt-runtime` branch) into the scratch dir's
   `vendor/llama.cpp/ggml/src/ggml-hsa/`, overwriting any prior copy.
3. Patch the two CMakeLists lines (option + `ggml_add_backend(HSA)`) into
   `vendor/llama.cpp/ggml/CMakeLists.txt` and `vendor/llama.cpp/ggml/src/CMakeLists.txt` if
   not already present (idempotent — `grep -q` guard before appending).
4. `pip install --python .venv/bin/python --reinstall --no-cache --no-binary
   llama-cpp-python <scratch-dir>` with:
   ```
   CMAKE_ARGS="-DGGML_HSA=ON -DGGML_HSA_RUNTIME=XRT \
               -DXRT_INCLUDE_DIR=<xrt include path> \
               -DXRT_LIBRARIES=<xrt libs> \
               -DCMAKE_BUILD_TYPE=Release"
   ```
   XRT paths are host-specific (this host's are under
   `1bit-MONSTER/.local/xrt` and `/usr/lib/x86_64-linux-gnu`); the script resolves them the
   same defensive way `ggml-hsa`'s own kernel build script resolves `aiecc`/`xclbinutil`
   (explicit path checks, hard-fail with a clear message if not found), rather than
   hardcoding this machine's paths.

This leaves the existing `build-llama.sh` (HIP/ROCm) untouched; the two backends are
mutually exclusive builds selected by which script the user runs, matching how the project
already treats CPU vs HIP builds as separate `scripts/build-llama*.sh` entry points.

### 3. N-padding (shared tensor-extra logic in `ggml-hsa.cpp`)

Same location as the existing f32→bf16 `src1` conversion added in nom4q (the tensor-extra
constructor for `MUL_MAT` ops), and **not** gated behind `GGML_HSA_RUNTIME_XRT` — padding is
a kernel-tiling requirement, not an XRT-specific concern, so it benefits HSA too if that
runtime ever becomes usable on this host.

When a `MUL_MAT`'s src1 (activations, already bf16 by this point via the existing
conversion) has N (its row count / token count) not a multiple of 128:

- Allocate a padded bf16 buffer sized to the next multiple of 128 rows, same K columns.
- Copy the real N rows in, zero-fill the pad rows.
- Dispatch against the padded buffer; the output tensor is sized to padded-N accordingly.
- On readback, copy only the first real-N rows of the result into the destination tensor —
  the pad rows' output is computed but discarded, never written back.

This reuses the existing temp-storage/`requires_sync` machinery the bf16 conversion already
established (same pattern, extended to also handle a size change, not just a dtype change).

### 4. Validation

Run `scripts/flappy.sh` with fixed `--seed`/`--episodes` twice: once against the current
CPU/HIP build, once against the `build-llama-npu.sh` build with the bf16 model. Compare:

- **Correctness:** scores should match or be within noise (same model weights modulo bf16
  vs. original quantization's numerical difference — the bf16 conversion itself, not the
  NPU path, accounts for any score drift; the NPU path is expected to be bit-close to a
  bf16-CPU baseline, so a third run — bf16 model on CPU — isolates that if scores don't
  match).
- **Performance:** per-step wall-clock (the `jev.probabilities` call), NPU vs CPU, on the
  bf16 model.

If the NPU path shows no real speedup (padding overhead may dominate at these small N /
short K, M sizes — Flappy's model dims are on the small side), that's a valid, useful
finding: it means the padding approach doesn't pay off at Flappy's scale, and any further
NPU investment should look at either genuinely larger-batch workloads or the bigger
quantized-weight-support piece instead. This validation step is the deliverable, not just a
sanity check.

## Out of scope (explicit)

- Q4_K (or any quantized format) dequant-on-load support in `ggml-hsa` — a separate, larger
  piece of work, only worth doing if this spec's validation shows the NPU path is a real win.
- Decode/autoregressive generation on the NPU — this kernel design doesn't support it
  (N must be ≥128), and Flappy doesn't need it.
- Any model other than Flappy's.
