# Let's see how close can we get to Jev

See [IDEA.md](IDEA.md) for the core concept.

Short version: take a plain LLM, don't generate. Do one forward pass and read the next-token
distribution over the answer tokens. The approach is **multiple choice**: options are labelled
`A, B, C, ...`, and the probabilities are a softmax over just those letters' logits. This repo
measures accuracy, calibration and latency against ordinary greedy generation.
Findings and decisions are tracked in the moth tasks (`moth ls`, `moth show <id>`), not here.

An earlier per-option Y/N scorer is still in the code behind a switch, but that direction was
dropped (see the moth tasks).

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
scripts/run.sh                                  # defaults: Qwen3.5 4B, 200 examples, ag_news
scripts/run.sh models/Qwen3.5-9B-Q4_K_M.gguf 500 --dataset dbpedia_14 --no-yn
```

The scripts work from any directory and re-enter the nix shell by themselves, so plain
`scripts/run.sh` is enough; no need to be in `nix-shell` first. A relative model path is taken
from your current directory if the file is there, and from the project root otherwise.
Options after the example count are passed through to `src/eval.py`.
If `uv` reinstalls `llama-cpp-python`, run `scripts/build-llama.sh` again to get the GPU back.

**Watching progress.** `run.sh` prints `logging to <path>` on its first line and tees all raw
output to `logs/<timestamp>-<model>-n<N>[-options].log`. Follow it with:

```bash
tail -f "$(ls -t logs/*.log | head -1)"
```

It prints `i/N` every 10 examples, and the full report when done.

## Configuration

| What | Where | Default |
|---|---|---|
| Model file | 1st arg of `scripts/run.sh` | `models/Qwen3.5-4B-Q4_K_M.gguf` |
| Number of examples | 2nd arg | 200 |
| Dataset | `--dataset` | `ag_news` (4 classes); also `dbpedia_14` (14 classes) |
| Skip the Y/N scorer | `--no-yn` | off (Y/N needs one pass per class, so skip it for many classes) |
| CPU threads, context size | `Model.__init__` in `src/common.py` | 16, 2048 |
| GPU offload | `Model.__init__` | always on, all layers (no switch) |
| GPU architecture | `HSA_OVERRIDE_GFX_VERSION` in `shell.nix`, `AMDGPU_TARGETS` in `scripts/build-llama.sh` | `gfx1150` (see below) |
| Sampling | `load()` in `src/data.py` | test split, shuffle seed 0, texts cut to 600 chars |

**Adding a dataset** is one entry in `DATASETS` in `src/data.py`: the Hugging Face id, the split,
the class names (index = label id, written as the model should read them) and how to turn a row
into text. MC supports up to 26 classes.

**The GPU setup is specific to this machine.** The Radeon 860M is `gfx1152`, for which the
rocBLAS/hipBLASLt kernels in nixpkgs do not exist, so `shell.nix` presents it as `gfx1150`
(`HSA_OVERRIDE_GFX_VERSION=11.5.0`) and llama.cpp is built for `gfx1150`. On other AMD hardware,
change both. The GPU is an integrated one that shares system RAM, so it is only slightly faster
than the CPU cores for prefill; the point of using it is to free the CPUs.

Scores for every run are saved to `scores_<dataset>_<model file>.npz` (`y`, `gen`, `mc`, per-question
latencies `lat_*`, and, unless `--no-yn`, the four `yn_*` variants and `mass`), so metrics can be
recomputed without re-running inference. Note that a run overwrites the file of the same
dataset and model.

## How it works

```
scripts/
  run.sh          run src.eval in the nix shell, tee the output to logs/
  build-llama.sh  rebuild llama-cpp-python with the HIP backend
  _common.sh      shared helper: finds the project root, re-enters the nix shell
src/
  common.py   Model wrapper: load, chat template, last_logits(); shared prompt framing
  data.py     dataset registry and loader
  mc.py       score_mc: one pass, options labelled A, B, C, ..., read the label-token logits
  gen.py      baseline: greedy generation of up to 8 tokens, parsed back to a class
  yn.py       legacy: one Y/N pass per option, four score variants
  metrics.py  accuracy, NLL, ECE, temperature fit
  eval.py     runner
```

**Prompts.** They are rendered with the chat template embedded in the GGUF, so any model family
works. Control tokens are tokenized as real special tokens (the default is to spell them out as
plain text, which silently degrades the prompt). Reasoning mode is switched off by prefilling the
empty reasoning block, so the very next token is the answer.

**Readout.** `Model.last_logits` runs one forward pass and returns the vocab logits at the last
position. It reads them from the llama.cpp context (`llm._ctx.get_logits()`, a private API)
because `llm.scores` stays all zeros unless `logits_all=True`. Getting that wrong shows up as
exactly uniform probabilities. Reading the logits also forces the asynchronous GPU work to
finish, which the latency measurement relies on.

**The scorer** (`mc.py`). One pass per article. The prompt lists the classes as `A. ...`,
`B. ...`, the instruction names exactly the letters in use, and the score per class is the
logit of its letter. The Y/N scorer (`yn.py`) asked one yes/no question per class instead and
compared the answers afterwards; it is kept for reference only.

**Metrics** (`metrics.py`). The scores are split in half by index. A temperature `T` is fitted on
the first half by minimising NLL, and everything is reported on the second half.

- accuracy, next to the chance level
- NLL and ECE (10 bins), both raw (`T=1`) and after temperature scaling (`cal`)
- latency per question for `mc` and the generation baseline (after a warm-up pass)

The generation baseline is there for reference only. It has no probabilities, so it only gets
accuracy and latency.

## Caveats

- Only the held-out half is scored for calibration (250 examples at 500), so accuracy
  differences of a few points are within noise.
- The generation baseline is the best case for generation: a terse answer capped at 8 tokens
  with thinking off. One-pass readout therefore does not beat it on latency here. A chat-style
  baseline with long answers, and shared-context batching of several questions, are not measured.
- Abstention ("none of the above"), several questions per pass, and non-enum outputs are not
  tested.
- The models are 4-bit quantized and run on a shared-memory iGPU.

## License

GPL-3.0, see [LICENSE](LICENSE).
