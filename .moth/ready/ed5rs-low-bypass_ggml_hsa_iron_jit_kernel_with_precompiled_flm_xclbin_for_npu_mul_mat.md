Follow-up to q8kte. ggml-hsa's NPU MUL_MAT path (IRON/MLIR-AIE JIT-compiled kernels via an
embedded Python interpreter) has a known, narrowed-but-unresolved correctness bug: a real
model's single-token (N=1) MUL_MAT computes wrong results as part of a real multi-layer forward
pass, even though the exact same kernel/shape/weight/activation bytes compute correctly when
dispatched in isolation (see q8kte's investigation). q8kte works around this by refusing N=1
dispatch (falls back to CPU), which is safe for prefill-only use (Flappy) but leaves decode/
autoregressive generation unable to use the NPU at all, and leaves the underlying bug
unexplained -- a real risk for other N values that were never explicitly validated (only N=2 and
N=512 are hardware-confirmed correct; nothing between).

Idea: bypass ggml-hsa's own IRON-JIT kernel-compilation path entirely for MUL_MAT, and instead
wrap precompiled, vendor-validated kernels the way /home/doo/projects/1bit-MONSTER does --
FastFlowLM (FLM)'s prebuilt dequant.xclbin/mm.xclbin (default root /home/doo/.local/flm-v0946,
overridable via FLM_ROOT; see 1bit-MONSTER's engine/npu/src/npu_engine_bf16_mm_bridge.cpp for the
reference integration). Note FLM's own kernel only computes 128 correct M-rows per invocation and
requires explicit batching/splitting by the caller (bf16mm_gemm_2batch) -- not a pad-and-discard
design like ggml-hsa's current N-padding, so this isn't a drop-in swap, it's a new dispatch path.

Goal: a MUL_MAT path on ggml-hsa (probably runtime-xrt.cpp, alongside/instead of the existing
kernel-compiler.cpp JIT path) that dispatches to FLM's precompiled kernels instead of compiling
its own via IRON, removing the current unexplained correctness risk and also removing the whole
embedded-Python-interpreter dependency (kernel-compiler.cpp, PEANO/aiecc/xclbinutil toolchain)
from the runtime path entirely -- kernels would be prebuilt artifacts, not JIT-compiled.

Out of scope for now: fixing IRON-JIT's actual bug (that's what would make this issue
unnecessary if solved instead); any non-MUL_MAT op.

Not started -- this is speculative future work, not committed to. Decide whether to pursue this
or keep chasing the IRON-JIT bug directly when it's next picked up.


----- AI agent updates -------

q8kte closed with NPU integration parked as a dead end for now (see its closure notes), pending
llama.cpp having full/proper NPU support. Two findings from that closure that bear directly on
this idea if it's ever picked up:

1. **GPU-vs-NPU offload is not automatic and not easily fixed by a kernel swap alone.**
   llama.cpp's automatic backend selection always prefers a real GPU-type device over an
   ACCEL-type device (which is what any ggml-hsa NPU backend registers as, HSA0 today,
   regardless of what's compiled underneath it) for weight placement -- confirmed empirically:
   compiling the same llama.cpp checkout with both GGML_HIP=ON and GGML_HSA=ON results in
   "HSA0 compute buffer size is 0.0000 MiB" (fully bypassed) the moment a GPU is present,
   because ggml's offload heuristic decides by device *type*, not by which backend or kernels
   sit behind it. This means: if this idea is implemented as "swap ggml-hsa's IRON-JIT kernel
   source for FLM's precompiled kernels, same ACCEL registration, same integration point" --
   the exact scope this issue currently describes -- it inherits the SAME GPU-bypass problem
   q8kte hit, unchanged. Achieving real GPU+NPU coexistence (GPU for most ops, NPU for MUL_MAT)
   would need something further: either targeted `tensor_buft_overrides` forcing specifically
   the MUL_MAT-only weight tensors onto ACCEL while leaving everything else on GPU's default
   path (untried, unclear if it avoids the separate scheduler-abort issue q8kte also found for
   genuine ACCEL placement), or accepting NPU-only (no GPU compiled in) as this idea's target
   configuration too, same as q8kte's dead-end state.

2. **If instead this idea means adopting 1bit-MONSTER's own standalone engine (not just its
   kernel binaries) as Flappy's inference engine**, that's a materially different, larger scope
   than "swap ggml-hsa's kernel source" -- a different API surface entirely, not a llama.cpp/
   ggml integration point at all, so it sidesteps finding 1 above completely (no competing GPU
   backend in the same scheduler, because it isn't llama.cpp's scheduler). But: Flappy's design
   (src/mc.py) reads raw last-position logits at specific label-token IDs via a direct in-process
   C API call (`llama._ctx.get_logits()`); 1bit-MONSTER's own HTTP server does NOT implement real
   per-token logprobs at all (`rest_handler.cpp` always returns `logprobs: null`, tracked
   upstream as unimplemented, #81) -- so this route would need embedding 1bit-MONSTER's engine
   in-process (not its HTTP API), and would need confirming it supports Qwen3.5's specific hybrid
   linear-attention/SSM architecture (its docs claim broad Qwen family support, not confirmed for
   this specific hybrid variant).

Net: this idea's current scope (kernel swap only) is now known to not solve the practical
GPU-vs-NPU tradeoff that ultimately closed q8kte; only the larger, different-scoped "adopt
1bit-MONSTER's own engine" variant might, and that has its own open questions (logits access
route, architecture support). Still not started, still speculative -- this note exists so a
future pickup doesn't have to rediscover the above.
