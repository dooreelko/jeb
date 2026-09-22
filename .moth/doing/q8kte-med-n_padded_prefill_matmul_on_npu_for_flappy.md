Follow-up to nom4q (ggml-hsa XRT runtime, complete). Goal: get the NPU actually doing useful work
in this project, starting with Flappy's move-decision inference (one prefill pass per game step,
no decode loop).

Two blockers found investigating this:
1. Flappy's models are Q4_K_M quantized; the NPU GEMM kernel needs weights in bf16.
2. The NPU GEMM kernel requires the activation-batch dimension (N, i.e. token count) to be a
   multiple of 128 on this NPU (aie2p: n=16, n_aie_cols=8). Flappy's prompts are well under 128
   tokens, so every real call currently falls back to CPU silently.
   M (weight output dim) and K are already compatible for typical model dims.

Decision: start with the smaller piece. Switch Flappy to a bf16 (non-quantized) model to sidestep
the dequant question, and add N-padding (pad short prompts to a multiple of 128, trim the output
back) to ggml-hsa's MUL_MAT dispatch so short prefill calls actually reach the NPU. Validate
whether this is a real speedup before investing in general quantized-weight support (dequant-on-
load), which is a separate, bigger piece of work if this proves worthwhile.

Out of scope for this issue: decode/autoregressive generation (Flappy doesn't need it), general
quantized-weight support (Q4_K dequant-on-load), any model other than Flappy's.

Design decisions from brainstorming:
- bf16 model: produce via llama.cpp's own GGUF-to-bf16 conversion tooling on a model already in
  models/, not a separately downloaded pre-made bf16 GGUF.
- N-padding: lives in ggml-hsa's shared tensor-extra logic (ggml-hsa.cpp), same place as the
  existing XRT-only f32->bf16 src1 conversion — not XRT-runtime-only, so it benefits any future
  runtime.
- llama-cpp-python integration (spiked and confirmed): this project's Flappy runs on
  llama-cpp-python, whose --no-binary build compiles its own vendored llama.cpp/ggml copy — not
  ../ggml directly. ggml-hsa's footprint outside its own directory is tiny (one CMakeLists option
  line, one ggml_add_backend(HSA) line), so a small scripted overlay (clone llama-cpp-python at
  the pinned version + its vendored llama.cpp submodule, copy in src/ggml-hsa/ from ../ggml
  xrt-runtime, patch those two lines, pip install from that local checkout with
  GGML_HSA=ON/GGML_HSA_RUNTIME=XRT) is sufficient. Confirmed by spike: CMake configure reaches
  "Including HSA backend" cleanly with XRT paths accepted. No fork to maintain, no changes needed
  to src/jev.py or src/common.py — same integration shape as the existing HIP build path
  (scripts/build-llama.sh), just a different source overlay and CMAKE_ARGS.


Full technical design: docs/superpowers/specs/2026-09-22-npu-flappy-bf16-padding-design.md
