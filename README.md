# Let's see how close can we get to Jev

See [IDEA.md](IDEA.md) for the core concept.

Short version: take a plain LLM, don't generate. Do one forward pass and read the next-token
distribution over the answer tokens. This repo compares two ways of turning those logits into a
distribution over options, and measures accuracy, calibration and latency against ordinary
greedy generation. Findings and decisions are tracked in the moth tasks (`moth ls`), not here.

## Getting started

Prerequisite: [nix](https://nixos.org/). `shell.nix` provides everything else: `uv`, `moth`,
`cmake`/`gcc`, and the ROCm stack (HIP compiler, hipBLAS, rocBLAS, ...) for the AMD GPU.

```bash
nix-shell                          # optional: an interactive shell with all of the above
uv sync                            # Python deps; llama-cpp-python comes out CPU-only

scripts/build-llama.sh             # rebuild llama-cpp-python with the ROCm/HIP backend (long)

# fetch a model (GGUF, Q4_K_M is a good default)
uv run hf download unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf --local-dir models

# run the comparison
scripts/run.sh                                  # defaults: Qwen3.5 4B, 200 examples
scripts/run.sh models/Qwen3.5-9B-Q4_K_M.gguf 100
```

The scripts work from any directory and re-enter the nix shell by themselves, so plain
`scripts/run.sh` is enough; no need to be in `nix-shell` first. A relative model path is taken
from your current directory if the file is there, and from the project root otherwise.
If `uv` reinstalls `llama-cpp-python`, run `scripts/build-llama.sh` again to get the GPU back.

**Watching progress.** `run.sh` prints `logging to <path>` on its first line and tees all raw
output to `logs/<timestamp>-<model>-n<N>.log`. Follow it with:

```bash
tail -f "$(ls -t logs/*.log | head -1)"
```

It prints `i/N` every 10 examples, and the full report when done.

## Configuration

| What | Where | Default |
|---|---|---|
| Model file | 1st arg of `scripts/run.sh` | `models/Qwen3.5-4B-Q4_K_M.gguf` |
| Number of examples | 2nd arg | 200 |
| CPU threads, context size | `Model.__init__` in `src/common.py` | 16, 2048 |
| GPU offload | `Model.__init__` | always on, all layers (no switch) |
| GPU architecture | `HSA_OVERRIDE_GFX_VERSION` in `shell.nix`, `AMDGPU_TARGETS` in `scripts/build-llama.sh` | `gfx1150` (see below) |
| Classes | `CLASSES` in `src/common.py` | the 4 `ag_news` topics |
| Dataset / sampling | `main()` in `src/eval.py` | `fancyzhx/ag_news` test split, shuffle seed 0, articles cut to 600 chars |

**The GPU setup is specific to this machine.** The Radeon 860M is `gfx1152`, for which the
rocBLAS/hipBLASLt kernels in nixpkgs do not exist, so `shell.nix` presents it as `gfx1150`
(`HSA_OVERRIDE_GFX_VERSION=11.5.0`) and llama.cpp is built for `gfx1150`. On other AMD hardware,
change both. The GPU is an integrated one that shares system RAM, so it is only slightly faster
than the CPU cores for prefill; the point of using it is to free the CPUs.

Scores for every run are saved to `scores_<model file>.npz` (`y`, `gen`, `mc`, the four `yn_*`
variants, `mass`, and per-question latencies `lat_*`), so metrics can be recomputed without
re-running inference.

## How it works

```
scripts/
  run.sh          run src.eval in the nix shell, tee the output to logs/
  build-llama.sh  rebuild llama-cpp-python with the HIP backend
  _common.sh      shared helper: finds the project root, re-enters the nix shell
src/
  common.py   Model wrapper: load, chat template, last_logits(); CLASSES; shared prompt framing
  mc.py       score_mc: one pass, options labelled A/B/C/D, read the label-token logits
  yn.py       score_yn: one Y/N pass per option (the IDEA.md approach), four score variants
  gen.py      baseline: greedy generation of up to 8 tokens, parsed back to a class
  metrics.py  accuracy, NLL, ECE, temperature fit
  eval.py     runner
```

**Prompts.** They are rendered with the chat template embedded in the GGUF, so any model family
works. Control tokens are tokenized as real special tokens (the default is to spell them out as
plain text, which silently degrades the prompt). Reasoning mode is switched off by prefilling the
empty reasoning block, so the very next token is the answer. Both scorers share the same
system-message style and article/options framing and differ only where the method requires.

**Readout.** `Model.last_logits` runs one forward pass and returns the vocab logits at the last
position. It reads them from the llama.cpp context (`llm._ctx.get_logits()`, a private API)
because `llm.scores` stays all zeros unless `logits_all=True`. Getting that wrong shows up as
exactly uniform probabilities. Reading the logits also forces the asynchronous GPU work to
finish, which the latency measurement relies on.

**The two scorers**

- `mc` (multiple choice): one pass per article. The prompt lists the options as A, B, C, D;
  the score per option is the logit of its label token.
- `yn` (IDEA.md): one pass per option with the full option list in the context, asking whether
  that option is the most likely topic. Four ways of scoring the result are compared, each
  followed by a softmax across options:
  - `yn_gap`: `logit(Y) - logit(N)`
  - `yn_p_yn`: `log p(Y)` after a softmax over just {Y, N}
  - `yn_y_logit`: `logit(Y)` alone, no N involved
  - `yn_p_full`: `log p(Y)` over the whole vocabulary, no N involved

  The probability mass on {Y, N} is recorded too, to check the tokens are actually used.

**Metrics** (`metrics.py`). The scores are split in half by index. A temperature `T` is fitted on
the first half by minimising NLL, and everything is reported on the second half.

- accuracy
- NLL and ECE (10 bins), both raw (`T=1`) and after temperature scaling (`cal`)
- latency per question for `mc`, `yn` and the generation baseline (after a warm-up pass)

The generation baseline is there for reference only. It has no probabilities, so it only gets
accuracy and latency.

## Caveats

- `yn` costs one pass per option and currently re-processes the whole prompt each time, so its
  latency is not representative of what it could be. The planned fix is llama.cpp's
  multi-`seq_id` batching: the shared prompt in every sequence, each option suffix in its own,
  so all options run in one `llama_decode`.
- Only the held-out half (100 examples at the default 200) is scored, so accuracy differences of
  a few points are within noise.
- The models are 4-bit quantized and run on a shared-memory iGPU. Whether the gap to Jev is method
  or model size needs the same run across sizes, which is what the model ladder is for.
