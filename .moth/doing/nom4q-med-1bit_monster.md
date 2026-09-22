so we should try to build a custom llamacpp with HSA or NPU support. the wip of HSA is in ../ggml
and we can use the 1-bit monster that has reverse engineered amd's FLM and thus has support for AMD NPU
let's see if we can integrate with it using ../1bit-MONSTER

it seems like one of the approaches could be to expand vulkan's use of GPU to NPU.




## Spike findings (2026-09-21)

Decision: pursue path (a) — build ggml's `hsa-backend` (ggml-hsa, XDNA NPU backend on ROCR/HSA) into llama.cpp. Fallback (b): wrap 1bit-MONSTER's NPU FFN kernels as a ggml backend.

Why:
- ggml-hsa is a real, upstream-shaped ggml backend; Krackan/XDNA2 listed as supported. ROCm 7.2.1 `rocminfo` already exposes the NPU as an HSA agent (`aie2`, AIE-ML) next to the gfx1150 GPU — the main ROCR blocker is cleared. (Reported as aie2, not aie2p: check which kernel arch is needed.)
- Its op coverage is thin: MUL_MAT (I8/I16/I32/BF16; F16 converted, F32 emulated), elementwise, SOFT_MAX. No quantised matmul, RMS_NORM, ROPE, flash-attn, SILU/GELU. Expect partial offload with CPU/Vulkan fallback.
- Kernels are JIT-built via MLIR-AIE/IRON (Python venv, heavy); prebuilt kernel dir option exists.
- 1bit-MONSTER's NPU path is XRT + FastFlowLM xclbins, FFN only, validated only on Strix Halo; FLM engines are closed prebuilt libs and /opt/fastflowlm is not installed here. Not ggml-shaped.
- Vulkan -> NPU (c) rejected: Vulkan has no NPU device model.

Open questions / next spike: does ROCR 7.2.1 dispatch to the NPU (else build ROCR at bd52f48); can IRON/mlir-aie be installed via nix or venv; does BF16 MUL_MAT on NPU beat CPU/Vulkan (`test-backend-ops` + llama-bench).

Out of scope: rewriting 1bit-MONSTER, custom Vulkan NPU device, running FLM closed engines.



## Spike 2 findings (2026-09-21)

Built ggml `hsa-backend` out-of-tree (JIT off) and ran `test-backend-ops`:
- Stock ROCm 7.2.1 lacks `hsa/hsa_ext_amd_aie.h`; compile fails. The header exists in the pip ROCm SDK in 1bit-MONSTER's venv (`_rocm_sdk_core`); adding its include dir makes ggml-hsa build.
- Runtime works: backend registers device `HSA0` (`aie2`, 62 GB visible) — the NPU is reachable through ROCR without building ROCR from source.
- Every MUL_MAT case reports "not supported": supports_op only says yes when a compiled kernel exists, and there are no prebuilt kernels (JIT off, no `GGML_HSA_KERNEL_DIR`). Nothing executed on the NPU yet.

Next: install IRON/mlir-aie (venv), rebuild with JIT on, rerun MUL_MAT (bf16) correctness + perf vs CPU/Vulkan. Watch the aie2 vs aie2p arch question.



## Spike 3 findings (2026-09-21)

Installed IRON (mlir_aie 1.4.0 + llvm-aie) in a scratch venv and rebuilt ggml-hsa with JIT on:
- Device now reports `aie2p` (the earlier `aie2` was the JIT-off build) — arch question settled: XDNA2 kernels apply.
- The JIT pipeline works end to end. A 512x512x512 bf16xf32 MUL_MAT compiles via IRON to a PDI + instruction stream for aie2p.
- The stock `test-backend-ops` MUL_MAT cases all fail to compile (tiling assertion: M must be a multiple of tile-m x AIE rows; the suite's small shapes, and gemv n=1, don't fit). The large perf shapes (m=4096, k=14336) are reported unsupported too. So the gemm kernel handles only tile-aligned, batched-matmul shapes; decode-time gemv is not covered.
- Nothing has been executed on the NPU yet; correctness and perf vs CPU/Vulkan are still unmeasured.

Practical notes: the JIT build embeds system Python 3.12, so the IRON venv must match that version and be on PYTHONPATH; runtime needed libelf/libdrm/libnuma/libz not on the default loader path here (nix-shell environment).

Implication: ggml-hsa is a viable base for prefill-style bf16 GEMM offload, but decode (n=1) has no kernel, and quantised weights are unsupported. Path (b) (reusing 1bit-MONSTER's FFN/GEMV NPU kernels) may be needed to cover decode.

Next: execute the compiled 512^3 kernel on the NPU via a tiny ggml test program and time it against CPU/Vulkan.



## Spike 4 findings (2026-09-21)

Wrote a throwaway ggml program (512x512x512 bf16 x f32 MUL_MAT, HSA0 vs CPU) against the JIT build:
- Everything up to dispatch works: HSA0 (`aie2p`, RyzenAI-npu6) registers, supports_op says yes for the tile-aligned shape, the kernel JIT-compiles, and the graph is submitted to the NPU queue.
- Submission segfaults inside ROCR (`XdnaDriver::PrepareBOs` <- `SubmitCmdChain` <- `AieAqlQueue::SubmitPackets`), reproducibly, with both the installed ROCm 7.2.1 libhsa-runtime64 and another 7.2 build. This is the AIE submit path in the runtime itself, not ggml-hsa or the kernel.
- Consistent with ggml-hsa's README, which says to build ROCR from source at commit bd52f48 because NPU support in released ROCR is in flux. So building ROCR from source is now a confirmed prerequisite, not a hedge.

Still unmeasured: NPU correctness and speed vs CPU/Vulkan.

Next: build ROCR at bd52f48 (rocm-systems, rocr-runtime), point the test at it, rerun the 512^3 program.



## Spike 5 findings (2026-09-21)

Built ROCR from source at bd52f48 (needs ClangConfig from a pip ROCm SDK; libnuma/libelf found via explicit CMAKE_LIBRARY_PATH) and reran the 512^3 NPU test:
- The earlier ROCR segfault is gone, but queue creation now fails: `hsa_queue_create` -> INVALID_ARGUMENT.
- strace shows the cause: `DRM_IOCTL_AMDXDNA_CREATE_HWCTX` on /dev/accel/accel0 returns EINVAL. The kernel here is 7.0.0-30 with the in-tree `amdxdna` module; ROCR bd52f48 (and ggml-hsa's README) target the out-of-tree AMD XDNA driver 1.6. Likely UAPI mismatch between the two — not verified further.
- So on this host ggml-hsa/ROCR cannot reach the NPU with the in-tree driver. XRT (libxrt_driver_xdna 2.25, already installed) does talk to this driver, and 1bit-MONSTER's NPU path is XRT-based.

Options now: (1) install the out-of-tree xdna-driver 1.6 (needs root/DKMS, replaces the in-tree module); (2) make ggml-hsa dispatch through XRT instead of ROCR (a real port of its queue/dispatch layer); (3) fall back to path (b): 1bit-MONSTER's XRT-based NPU kernels as a ggml backend.

Nothing has yet run on the NPU via ggml. Correctness/perf still unmeasured.



## Decision (2026-09-21): port ggml-hsa dispatch to XRT

Chosen over installing the out-of-tree xdna driver (system-invasive) and over wrapping 1bit-MONSTER kernels (Strix-Halo-tuned, FFN only). Reason: reuses ggml-hsa's ggml backend, kernel cache and IRON JIT; no system changes.

Plan shape:
1. Spike: load the JIT-built PDI + instruction stream through XRT in a standalone program and run it on the NPU. Gate: if XRT cannot take JIT output directly (needs xclbin wrapping), revisit the design.
2. Put dispatch, memory and device enumeration behind a thin seam with HSA (existing) and XRT (new) implementations, chosen at build time.
3. XRT implementation: device + hw_context, host-visible BO-backed ggml buffers, one run per op, wait on run handle.
4. Validate with test-backend-ops MUL_MAT (tile-aligned bf16) and a 512^3 comparison vs CPU for error and timing.

Work happens in ../ggml on a feature branch. Out of scope: decode (n=1) kernels, quantised types, changes to 1bit-MONSTER.

Risks: XRT PDI loading may need an xclbin; pointer kernargs become BO handles (copies for foreign tensors); argument order must match the IRON kernel's layout.



## Step 1 progress (2026-09-21): packaging is the gap

Goal was to run the JIT-built mm512 kernel through XRT. XRT loads an xclbin (or an ELF via the newer module flow), not the raw PDI + insts that the ggml-hsa JIT emits.
- `aiecc` can package an xclbin, but needs `xclbinutil`, which is not installed on this host (nor in 1bit-MONSTER's local XRT).
- With Peano (`--no-xchesscc --no-xbridge`, PEANO_INSTALL_DIR = llvm-aie in the venv, core objects copied next to the MLIR) aiecc builds the per-core ELFs but not a combined `--full-elf-name` ELF; that step needs aiebu tooling that is absent.
- 1bit-MONSTER's local XRT build ships `libaiebu.a` + headers (`aiebu_assembler.h`), so assembling the ELF ourselves via the aiebu API is a possible route; building `xclbinutil` from XRT source is the other.

So the design gate ("can XRT take JIT output directly?") answers: no, an extra packaging step is required. Design implication: the XRT backend needs a JIT post-step producing an xclbin/ELF (xclbinutil or aiebu) alongside the PDI/insts, adding a build-time dependency.



## Step 1 result (2026-09-21): XRT runs the ggml-hsa JIT kernel on the NPU — gate passed

Built `xclbinutil` standalone from XRT source (boost/rapidjson via nix-shell; only needs `XRT_SOURCE_DIR=<XRT>/src`), then `aiecc --no-xchesscc --no-xbridge --get-xclbin --get-npu-insts` on the MLIR the JIT already emits produced an xclbin + insts. A standalone XRT program (system libxrt 2.25, in-tree amdxdna driver) loaded it and ran the 512x512x512 MUL_MAT on the NPU:
- Result matches the CPU reference exactly (max abs error 0.0) with ggml-native layouts (A = MxK bf16, B = NxK, C[n*M+m]).
- ~0.48 ms per run steady state (~0.55 TFLOP/s, bf16 in / f32 out).
- Two contract findings for the port: the gemm kernel takes its input dtype from A only, so B must be bf16 too (the backend has to convert f32 src1); and the kernel's argument order is instr stream, A, B, C with opcode 3 (XRT kernel name MLIR_AIE).

So the design holds: the XRT backend needs a JIT post-step (aiecc -> xclbin via xclbinutil) and dispatches via xrt::kernel. New build-time dependency: xclbinutil. Next: step 2 (seam between HSA and XRT dispatch) on a feature branch in ../ggml.



## Implementation plan (2026-09-21)

Detailed plan: `docs/superpowers/plans/2026-09-21-ggml-hsa-xrt-runtime.md` (7 tasks: branch + build option, runtime-interface refactor, JIT xclbin post-step, XRT enumeration/memory/kernel load, XRT context/dispatch, bf16 + validation, moth report). Known top risk recorded there: XRT hw_contexts are per-xclbin and few can be live, so dispatch uses a small LRU; fallback is a multi-kernel xclbin.



## Implementation result (2026-09-22): XRT runtime works end to end on the NPU

Executed the 7-task plan via subagent-driven development on `../ggml` branch `xrt-runtime` (base `hsa-backend`), commits `58c2ff5..623c795`. Not merged, not pushed.

What works:
- `GGML_HSA_RUNTIME=HSA|XRT` build option; default HSA build verified byte-identical throughout (repeated before/after `support -b HSA0` diffs).
- HSA-specific code moved behind a `ggml-hsa/runtime.hpp` interface; HSA and XRT implementations in `runtime-hsa.cpp` / `runtime-xrt.cpp`.
- JIT gained an xclbin/insts packaging post-step (aiecc + a standalone-built `xclbinutil`) for XRT mode.
- XRT runtime: device enumeration (aie2p / RyzenAI-npu6), memory, kernel load, dispatch with a 4-entry hw_context LRU (device allows 16 live contexts; cap 4 is safe headroom), bounce-BO fallback for non-page-aligned tensors, sticky failure latch on a bad run.
- f32->bf16 src1 conversion added for MUL_MAT (the JIT gemm kernel types both operands from A), gated to XRT only — HSA/ROCR has the same latent bug, left undocumented-fixed since it can't be validated on hardware here.
- Validated: `xrt-mul-mat` 512^3 (err 0.0, ~1-2 ms vs ~22-47 ms CPU depending on load) and 1024^3 (err 1e-5, ~7.7 ms vs ~270 ms CPU) run correctly on the NPU. `test-backend-ops -b HSA0 -o MUL_MAT` gives 0 OK / all "not supported" / 0 FAIL — every generated shape is untileable for this gemm kernel (n=1 decode, unaligned m/k, batched/permuted); the numeric proof is `xrt-mul-mat` + `xrt-alias`, not that suite.
- `src/ggml-hsa/README.md` documents the XRT runtime: build flags, required tooling (mlir_aie/llvm-aie matching the embedded Python version, xclbinutil built from XRT source, aiecc/PEANO/GGML_HSA_XCLBINUTIL env), and known limits.

Known limits / left as future work: decode (n=1) and quantised types are not supported (out of scope per the design); host f32->bf16 conversion is a scalar loop, roughly doubling small-matmul dispatch time; per-context state assumes buffer free doesn't race compute on the same context (documented assumption, not enforced); HSA/ROCR carries the same gemm.py bf16xf32 typing bug, unfixed.

Process note: caught and fixed during review — a Critical use-after-free in XRT dispatch when an op's input/output tensors alias the same data pointer (fixed with shared_ptr'd BO cache entries + per-dispatch staging); an instruction-stream BO that could outlive the kernel cache's backing memory; missing TO_DEVICE sync of the destination buffer; inconsistent locking of the context registry; and a wait() path that could copy back data from a failed NPU run silently.

Final whole-branch review pending before considering this ready to merge/present.



## Final review (2026-09-22)

Whole-branch review clean after one fix wave. Fixed: a device-run failure detected only at the trailing flush of a graph did not surface as GGML_STATUS_FAILED (silently stale destination data) — added `ggml_hsa_rt_failed()` to the runtime interface, checked after the trailing flush and mid-dispatch. Plus minor polish (test linkage, zero-size alloc parity, dead code comments). Branch: ../ggml `xrt-runtime` (base hsa-backend @ 4c5c8fc), head 9d808fb, 20 commits. Not merged, not pushed.


## Closing (2026-09-22)

Extended further during q8kte (bf16 Flappy + N-padded NPU matmul, done): the XRT runtime got
several more real, hardware-verified fixes and additions on top of the head noted above --
a backend-registry gap found in a downstream llama.cpp checkout using this branch, a crash in
the embedded Python interpreter under any real (non-standalone) consumer, environment/toolchain
documentation, two new hardware stress tests (multi-kernel hw_context eviction, cross-backend
scheduler interleave), a host-buffer-type abort fix, a negative cache for doomed kernel compiles,
and a correctness gate refusing single-token (N=1) MUL_MAT dispatch (a real, narrowed-but-
unexplained numerical bug found during q8kte). Branch head is now b47e268, still on ../ggml
`xrt-runtime` (base hsa-backend @ 4c5c8fc).

Marking this done on its own terms: the goal here was building custom llama.cpp NPU support via
ggml-hsa's XRT runtime, and that was achieved and hardware-validated (correct MUL_MAT results,
real speedup over CPU for large tile-aligned shapes). q8kte's later finding that NPU can't
coexist with a GPU backend in the same llama.cpp process is a separate, downstream integration
concern (see q8kte's closure) -- it doesn't undo what this task set out to do and accomplished.

../ggml `origin` is ypapadop-amd/ggml, not a remote this project controls -- `xrt-runtime` stays
a local-only branch there by design; nothing pushed or merged upstream as part of closing this.
