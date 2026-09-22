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

### 2. Build integration: prebuilt libllama.so via LLAMA_CPP_LIB_PATH

Flappy runs on `llama-cpp-python` (via `src/jev.py` → `src/common.py`'s `Model`), which
loads its native libraries through ctypes. Its loader (`llama_cpp/_ctypes_extensions.py`,
used from `llama_cpp/llama_cpp.py`) honors an `LLAMA_CPP_LIB_PATH` environment variable that
points it at a directory containing a prebuilt `libllama.so` (and its sibling
`libggml*.so`s) instead of the package's own bundled copy — no pip rebuild, no vendoring
`llama-cpp-python` at all. This supersedes the pip-sdist-overlay approach explored in an
earlier spike (confirmed to configure, but unnecessarily heavy next to this).

`/home/doo/projects/llama.cpp` is a plain upstream `llama.cpp` checkout (`origin/master`, no
fork) that already carries an uncommitted overlay of `ggml-hsa` (`ggml/src/ggml-hsa/` plus
the same two `CMakeLists.txt` lines from Section above) from earlier HSA/ROCR-era work, built
once with `GGML_HSA=ON` but not XRT. Refresh that overlay from `../ggml`'s `xrt-runtime`
branch and rebuild:

1. Replace `ggml/src/ggml-hsa/` in the `llama.cpp` checkout with `../ggml`'s `src/ggml-hsa/`
   (from `xrt-runtime`); reconcile the two `CMakeLists.txt` lines if upstream `llama.cpp` has
   moved past what the existing overlay patched (diff against the current overlay's own
   patch to `ggml/CMakeLists.txt` / `ggml/src/CMakeLists.txt` first).
2. Reconfigure and rebuild `build/` with `-DGGML_HSA=ON -DGGML_HSA_RUNTIME=XRT
   -DXRT_INCLUDE_DIR=<xrt include path> -DXRT_LIBRARIES=<xrt libs> -DCMAKE_BUILD_TYPE=Release`
   (XRT paths resolved defensively, not hardcoded, same as `ggml-hsa`'s kernel build script
   resolves `aiecc`/`xclbinutil`).
3. New script `scripts/flappy-npu.sh`, a thin wrapper: `export LLAMA_CPP_LIB_PATH=<build
   dir>/bin`, `export LD_LIBRARY_PATH=<build dir>/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}`
   (needed because sibling `.so` resolution wasn't observed to work automatically on this
   host — confirmed by hand when running the checkout's `llama-quantize`), then
   `exec scripts/flappy.sh "$@"`. `scripts/flappy.sh` itself is untouched.

This leaves the existing `build-llama.sh` (HIP/ROCm, pip-installed) untouched; switching
between backends is just which script launches Flappy.

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
