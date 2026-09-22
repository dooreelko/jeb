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
- llama-cpp-python integration, corrected during plan-writing: llama_cpp.py honors a
  LLAMA_CPP_LIB_PATH env var to load a prebuilt libllama.so directly -- no pip rebuild/overlay
  needed (supersedes an earlier spike of a pip-sdist-overlay approach, which worked but was
  unnecessarily heavy). /home/doo/projects/llama.cpp is a plain upstream checkout that already
  carries an uncommitted ggml-hsa overlay from earlier HSA/ROCR-era work (built once with
  GGML_HSA=ON, not XRT) -- refresh that overlay to ../ggml's xrt-runtime branch, fast-forward
  the checkout to current upstream first (it's an old pull), rebuild with GGML_HSA_RUNTIME=XRT,
  and point Flappy at the build via a new scripts/flappy-npu.sh wrapper setting
  LLAMA_CPP_LIB_PATH/LD_LIBRARY_PATH. No changes to src/jev.py or src/common.py.

Full technical design: docs/superpowers/specs/2026-09-22-npu-flappy-bf16-padding-design.md

----- AI agent updates -------

Scope expansion (still the same goal, broadened): the user wants this NPU integration to be a
generally NPU-capable llama.cpp, reusable by future tasks beyond Flappy, not a one-off. This
doesn't change what was decided above, it changes how durable the fix needs to be: prefer
fixing the shared llama.cpp checkout / the standalone ggml-hsa branch itself over anything
Flappy-specific, and prefer a small reusable Python loading helper over inlining device-selection
logic into Flappy's own code.

Implementation plan executed (docs/superpowers/plans/2026-09-22-npu-flappy-bf16-padding.md), via
subagent-driven development on branch q8kte-npu-flappy:
- Task 1 (bf16 model) and Task 3 (N-padding in ggml-hsa) complete, hardware-verified.
- Task 2 (build wiring) and Task 4 (rebuild) complete, but validation work (Task 5) surfaced that
  the whole premise underneath them was incomplete — see below.

Validation (Task 5) found the NPU was never actually being used by any prior run in this issue,
despite everything "working" functionally. Root causes found, in the order they were found:
1. ggml-hsa's NPU device registers as an ACCEL-type ggml backend device, not GPU. llama.cpp's
   automatic `n_gpu_layers`-based offload path only considers GPU-type devices; ACCEL devices are
   structurally excluded from it (they're meant to integrate as an opportunistic accelerator for
   the CPU backend, similar to BLAS/AMX, not as a primary offload target). Flappy's model loading
   (src/common.py) relies entirely on that automatic path.
2. `llama-cpp-python`'s ctypes bindings never populate the one field
   (`llama_model_params.devices`) that can bypass that automatic filtering with an explicit
   device list -- the Python API marks it unused.
3. Investigating (2) by building a device list ourselves via direct ctypes calls uncovered a
   separate, real bug: the llama.cpp checkout's central backend registry never learned about the
   HSA backend at all (a required wiring step that lives outside ggml-hsa's own source directory,
   so it was never covered when ggml-hsa was copied into this checkout). Fixed directly in the
   checkout, confirmed against the standalone ggml repo already having the equivalent wiring on
   its own copy of that file -- this was a copy-out gap, not a design gap in ggml-hsa itself.
4. With (2) worked around and (3) fixed, the NPU device is now visible and the model loads
   successfully with it explicitly listed, but the model's compute buffers still aren't actually
   placed on it (confirmed via llama.cpp's own buffer-size accounting log line showing 0 bytes
   used on the NPU device). This is the open question: ACCEL-type devices integrate through a
   different mechanism (an "extra buffer type" hook the CPU backend can opportunistically use)
   than the GPU offload path, and something in that mechanism isn't picking up the NPU either.
   Investigation of this ongoing.

Nothing has been committed for the backend-registry fix (item 3) or the ctypes-based
device-targeting workaround yet -- both currently exist only as local, uncommitted changes in
/home/doo/projects/llama.cpp and a scratch script, pending the item-4 investigation landing
somewhere workable before being written up properly (as either a change to
scripts/flappy-npu.sh's build recipe, a small reusable Python loading module, or both).


Chased the item-4 open question ("why doesn't the NPU device actually get used") to a full
resolution. Found and fixed, in order: (1) llama.cpp's central backend registry was missing the
HSA entry -- committed to the standalone ../ggml repo's README as an integration note (the
registry file itself lives outside ggml-hsa's own directory, so it wasn't part of the overlay);
(2) the kernel JIT compiler's embedded-Python-interpreter setup unconditionally assumed no
Python was already running, which crashes for any real Python-based consumer (llama-cpp-python,
i.e. every real use of this backend) -- fixed in ggml-hsa itself
(src/ggml-hsa/kernel-compiler.cpp, committed to ../ggml's xrt-runtime branch: f3a6d9c) to attach
to an already-running interpreter instead of always trying to own one; (3) three more
environment-level gaps (kernels/ directory not colocated with the loaded library at runtime,
missing PEANO_INSTALL_DIR/aiecc on PATH, missing xclbinutil, missing IRON packages in the
Python environment that actually runs the consumer) -- all documented in ../ggml's README
(825e754) since they're operational requirements any future Python-based consumer of this
backend will hit, not just this project.

With all of that fixed, the NPU is now genuinely used: weight tensors get placed on its buffer
type, kernels compile and dispatch on hardware, a real model loads and runs end-to-end through
llama-cpp-python's standard API (verified with a scratch script using ctypes to populate the two
llama_model_params fields -- devices, tensor_buft_overrides -- that llama-cpp-python's Python
bindings never expose, both required to route weight placement to an ACCEL-type device; see the
spec if this needs productionizing).

New, separate finding: output is reproducibly wrong (garbled, even for a single generated
token) once the NPU is actually exercised across the model's full 24-layer forward pass. Task 3
validated individual isolated MUL_MAT calls exactly; nothing has yet validated a full
multi-layer mixed-CPU/NPU forward pass. This is a numerical correctness question, separate from
all the dispatch-plumbing fixes above, and is the next thing being investigated.


Correctness investigation, three hardware tests plus one methodology correction:

1. New standalone test (xrt-multi-kernel-stress.cpp, committed 53e081a): dispatches 5 distinct
   real-model MUL_MAT shapes in one graph_compute() call, forcing hw_context LRU eviction
   (k_max_live_contexts=4). All 5 matched CPU exactly. Rules out hw_context eviction as the
   correctness bug's cause.
2. New standalone test (xrt-sched-interleave.cpp, committed 6c1e8b7): chains HSA->CPU->HSA through
   ggml_backend_sched with both backends genuinely registered. Matched an all-CPU run closely
   (err 0.00009, consistent with bf16 noise). Rules out cross-backend scheduler data hand-off --
   *for that code path specifically* (see correction below, this path turned out not to be the one
   actually exercised in the real Flappy/llama-cpp-python run).
3. Root-caused kernel-compiler.cpp's embedded-interpreter fix (already committed, f3a6d9c) plus
   traced runtime-xrt.cpp's BO staging/sync logic (bo_for/ggml_hsa_rt_dispatch) by hand for the
   weight-tensor-reuse case (load once via plain memcpy, dispatch many times) -- looks correct,
   every dispatch that stages a BO (new or cache-hit) unconditionally syncs it to device.

Correction to the "confirmed working end-to-end" claim two updates ago: the ctypes scratch script
used to prove that, and reused for these hardware tests' baseline, patched
`llama_cpp.llama_model_default_params` on the top-level `llama_cpp` package -- but `llama.py`
internally does `import llama_cpp.llama_cpp as llama_cpp`, a separate module object with its own
attribute namespace. The patch never reached the code that actually builds model/context params.
So `devices`/`tensor_buft_overrides` were never actually set in any prior run this issue. Every
"NPU is used" observation so far (948 MiB HSA0 weight buffer, garbled output, all three hardware
tests' baselines) came entirely through the *opportunistic extra-buffer-type* path (ACCEL as a
CPU-backend fallback buffer type), never through genuine primary-device scheduling. This deflates
finding 2 above: xrt-sched-interleave proved cross-backend hand-off is correct in the abstract, but
production traffic doesn't take that path at all -- it stays on the CPU backend, which reaches into
HSA's buffer type directly for supported ops.

Fixing the patch (also patch `llama_cpp.llama_cpp`, not just `llama_cpp`) and forcing genuine
`devices=[ACCEL]` placement surfaced two more real bugs, in order:
1. `ggml_hsa_host_malloc` (ggml-hsa.cpp) called `NOT_IMPLEMENTED()` (aborts) instead of returning
   nullptr, even though its caller already has a designed CPU-buffer fallback for a null result.
   Any context that treats HSA0 as a genuine device asks every device for its host buffer type at
   construction (`llama_context::output_reserve`) and aborted immediately. Fixed (now returns
   nullptr, letting the existing fallback run) -- not yet committed to ../ggml.
2. With that fixed, model construction now aborts one step later, inside ggml's own scheduler
   (`ggml_backend_sched_backend_id_from_cur`, called from `ggml_backend_sched_split_graph` during
   `llama_context::graph_reserve`/`sched_reserve`) -- consistent with the scheduler being unable to
   find a valid backend assignment once ACCEL is a real scheduling participant, most likely because
   ggml-hsa doesn't implement enough ops for genuine mixed-backend graph splitting (as opposed to
   the CPU-backend-with-extra-buft path, which only ever calls HSA for MUL_MAT and never asks the
   scheduler to place anything there). Not yet root-caused; next thing to investigate.

Net effect: the numerical-correctness question (garbled output) was being chased on the
opportunistic extra-buffer-type path, which is real production traffic (Flappy's actual code path,
via automatic n_gpu_layers offload + ACCEL extra-buft) and remains open. The genuine
primary-device path is a separate, currently more broken avenue (two hard aborts, one fixed one
not) that was never actually working before despite earlier notes to the contrary -- correcting the
record here so a future session doesn't rely on the "devices/tensor_buft_overrides workaround
proven end-to-end" claim above without knowing it was based on a script bug.


Garbled-output root-cause investigation, resolved to a strong, evidence-backed narrowing (not yet
a final fix). Built a per-node instrumentation tool (llama_context_params.cb_eval, ctypes-bound,
dumps every graph node's name/shape/first-few-values) and ran it NPU vs CPU on the actual
Qwen3.5-0.8B model, one real generation step. Findings, in order:

1. Node-by-node diff shows the two runs match closely (bf16-noise-level, ~1e-6) all the way
   through the first 3 attention layers, until node 228: layer 3's attn_output/o_proj MUL_MAT
   (K=2048, M=1024, real N=1 padded to 128). Its input (node 227) matches closely; its output is
   wildly wrong -- not a uniform scale error, ratios vary per-element and one element even flips
   sign. This is the first real divergence, everything before it is fine.
2. Traced host-side dispatch by hand at that exact call: the weight tensor's raw bytes in HSA's
   buffer match the GGUF file's bytes exactly; the activation's f32->bf16 conversion and zero-
   padding (127 of 128 rows are padding, since real N=1) match a manual reference conversion
   exactly. Every host-side input to this dispatch is provably correct.
3. Isolated this exact shape (K=2048, M=1024, N=1) in a standalone two-tensor test
   (xrt-square-n1.cpp, committed dbcb5f6) with random data: passes cleanly (err 0.00001).
4. Same isolated test, but loaded with the EXACT real weight bytes (from the GGUF file) and EXACT
   real activation bytes (captured from the same dispatch via the cb_eval tool): ALSO passes
   cleanly (err 0.00000), computing the correct value that the real run got wrong.

Conclusion: the bug is not in the compiled kernel, not in this shape, and not in this data -- the
identical kernel, shape, and bytes compute correctly when dispatched alone. It only goes wrong as
node ~228 of a real ~1405-node forward pass. This implicates something stateful in the dispatch
history leading up to it that the existing xrt-multi-kernel-stress.cpp test doesn't reproduce:
that test churns 5 distinct shapes once through the 4-slot hw_context LRU (k_max_live_contexts in
runtime-xrt.cpp) and found no error, but a real forward pass cycles through a dozen-plus distinct
MUL_MAT shapes (Kcur/Qcur_full/Vcur/attn_output/ffn_gate/ffn_up/ffn_down, each layer having its
own K/M combination) well before reaching layer 3, i.e. a much more demanding
churn/eviction/BO-cache pattern than anything hardware-verified so far. Next step: reproduce with
a synthetic multi-shape churn sequence that matches the real run's actual shape cycling order and
count, to find the specific interaction.

Also found and fixed a real, unrelated bug along the way: `ggml_hsa_host_malloc` called
`NOT_IMPLEMENTED()` (aborts) instead of returning nullptr, even though its only caller already has
a designed CPU-buffer fallback for a null result. This aborted any context that treats HSA0 as a
genuine primary device (as opposed to the opportunistic CPU-extra-buffer-type path that automatic
`n_gpu_layers` offload actually uses for ACCEL-type devices) at construction time, inside
`llama_context::output_reserve()`. Fixed and committed (ed2a89e).


Root-caused the scheduler abort (genuine primary-device path, item 2 from two updates ago).
`ggml_backend_sched_backend_id_from_cur` (ggml/src/ggml-backend.cpp:927-931) hard-aborts
("pre-allocated tensor ... in a buffer ... that cannot run the operation") whenever a tensor is
pre-allocated into a backend's buffer but that backend doesn't support the requested op on it --
by design, with no fallback, once a buffer has strict backend affinity in the scheduler.

This is architectural, not a bug to patch: ggml-hsa's XRT buffers are plain host memory
(ggml_hsa_rt_alloc is a std::aligned_alloc, not device-exclusive VRAM -- confirmed in
runtime-xrt.cpp). That's exactly why the opportunistic CPU-extra-buffer-type path (what real
production traffic actually uses, via automatic n_gpu_layers offload) works fine for every op:
the CPU backend owns scheduling for everything and can read/compute directly on HSA-buffer memory
for any op it supports, special-casing only MUL_MAT out to the NPU. Genuine primary-device
placement (`model_params.devices` forcing ACCEL into sched->backends[]) breaks this: the
scheduler then enforces strict buffer affinity with no soft fallback, and ggml-hsa only
implements a handful of ops (chiefly MUL_MAT) -- any weight-adjacent op it doesn't support
(RMS_NORM, ROPE, softmax, etc. reading a buffer the scheduler now considers HSA-owned) aborts
outright.

Conclusion: the `devices`/`tensor_buft_overrides` ctypes workaround from two updates ago was never
the right integration lever for this backend as currently scoped, independent of the host_malloc
bug already fixed. The opportunistic extra-buffer-type path already does the right thing and is
what should be relied on/documented as the supported integration route; genuine primary-device
placement would need ggml-hsa to implement comprehensive op coverage first (a much larger scope
than this issue), or the scheduler to gain a soft-fallback for buffer-affinity op mismatches
(a ggml-upstream change, out of scope here).

Net position on this issue: the garbled-output bug on the actual-production (opportunistic) path
remains open and is narrowed to a dispatch-history-dependent NPU-context/BO-cache interaction
(previous update); the genuine-device path is understood but not viable without a much bigger
scope of work, so it's set aside rather than pursued further here.
