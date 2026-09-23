# 07 NPU offload

2026-09-21 to 2026-09-22 · moth nom4q (1bit Monster), q8kte (N-padded prefill matmul on NPU for Flappy)

## Hypothesis

The Ryzen NPU (XDNA2, aie2p) can take over the prefill matmuls of jeb's one-pass readout through a
custom llama.cpp backend (ggml-hsa), making Flappy's per-step inference faster or freeing the GPU.

## Setup

- Engineering spikes, not a benchmark. ggml-hsa (ggml's HSA/XDNA backend) built into llama.cpp;
  ROCR could not reach the NPU with the in-tree kernel driver, so its dispatch was ported to XRT
  (branch `xrt-runtime` in the local `../ggml`, not pushed).
- Kernels JIT-compiled via MLIR-AIE/IRON; bf16 only, tile-aligned shapes; short prompts N-padded to 128.
- Flappy with a bf16 Qwen3.5 0.8B through llama-cpp-python.

## Data

- Isolated MUL_MAT on the NPU: 512³ exact, ~1-2 ms vs ~22-47 ms CPU; 1024³ error 1e-5, ~7.7 ms vs ~270 ms CPU.
- In a real forward pass: output garbled at the first N=1 matmul (layer 3); the same kernel, shape and
  exact bytes compute correctly in isolation. Not root-caused; worked around by refusing N=1 on the NPU.
- Flappy 5-step episode: 45.9 s "NPU" (actually CPU + occasional NPU matmul, no GPU compiled in) vs
  1.98 s GPU. With GPU and NPU both compiled in: 2.4 s, but the NPU is unused (0 bytes placed on it).

## Conclusion

- The NPU works for isolated large matmuls (real speedup over CPU), after fixing many real integration
  bugs (registry, embedded Python, toolchain, host buffer abort, negative kernel cache).
- Dead end next to a GPU: llama.cpp places weights on a GPU-type device over an ACCEL-type one, so the NPU
  is only used when no GPU exists. Forcing the NPU as a primary device hits a scheduler abort (sparse op
  coverage). Parked until llama.cpp has proper NPU support; jeb stays on vanilla llama.cpp + HIP.
- Follow-up idea only (moth ed5rs): precompiled FastFlowLM kernels instead of the JIT path.
