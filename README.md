# Let's see how close can we get to Jev

See [IDEA.md](IDEA.md) for the core concept.

Short version: take a plain LLM, don't generate. Do one forward pass and read the next-token
distribution over the answer tokens. The approach is **multiple choice**: options are labelled
`A, B, C, ...`, and the probabilities are a softmax over just those letters' logits. An optional
last option, "None of the above", lets the model abstain. This repo measures accuracy,
calibration and latency against ordinary greedy generation, and tests the abstention.
Findings and decisions are tracked in the moth tasks (`moth ls`, `moth show <id>`), not here.

## Getting started

Prerequisite: [nix](https://nixos.org/). `shell.nix` provides everything else: `uv`, `moth`,
`cmake`/`gcc`, and the ROCm stack (HIP compiler, hipBLAS, rocBLAS, ...) for the AMD GPU.

```bash
nix-shell                          # optional: an interactive shell with all of the above
uv sync                            # Python deps; llama-cpp-python comes out CPU-only

scripts/build-llama.sh             # rebuild llama-cpp-python with the ROCm/HIP backend (long)

# fetch a model (GGUF, Q4_K_M is a good default)
uv run hf download unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf --local-dir models

# accuracy / calibration / latency against greedy generation
scripts/run.sh                                  # defaults: Qwen3.5 4B, 200 examples, ag_news
scripts/run.sh models/Qwen3.5-9B-Q4_K_M.gguf 500 --dataset dbpedia_14

# abstention test: hide 3 of the classes and see whether "None of the above" catches them
scripts/run.sh models/Qwen3.5-9B-Q4_K_M.gguf 500 --dataset dbpedia_14 --hide 3
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
| Abstention test | `--hide K` | off; K classes are hidden from the options |
| Which classes get hidden | `--hide-seed` | 0 |
| CPU threads, context size | `Model.__init__` in `src/common.py` | 16, 2048 |
| GPU offload | `Model.__init__` | always on, all layers (no switch) |
| GPU architecture | `HSA_OVERRIDE_GFX_VERSION` in `shell.nix`, `AMDGPU_TARGETS` in `scripts/build-llama.sh` | `gfx1150` (see below) |
| Sampling | `load()` in `src/data.py` | test split, shuffle seed 0, texts cut to 600 chars |

**Adding a dataset** is one entry in `DATASETS` in `src/data.py`: the Hugging Face id, the split,
the class names (index = label id, written as the model should read them) and how to turn a row
into text. MC supports up to 26 options, including the abstain option.

**The GPU setup is specific to this machine.** The Radeon 860M is `gfx1152`, for which the
rocBLAS/hipBLASLt kernels in nixpkgs do not exist, so `shell.nix` presents it as `gfx1150`
(`HSA_OVERRIDE_GFX_VERSION=11.5.0`) and llama.cpp is built for `gfx1150`. On other AMD hardware,
change both. The GPU is an integrated one that shares system RAM, so it is only slightly faster
than the CPU cores for prefill; the point of using it is to free the CPUs.

Scores are saved so metrics can be recomputed without re-running inference. A run overwrites the
file of the same name.

- standard run: `scores_<dataset>_<model file>.npz` with `y`, `gen`, `mc` and latencies `lat_*`
- abstention run: `scores_<dataset>_hide<K>s<seed>_<model file>.npz` with `A` (scores with the
  abstain option last), `B` (scores over the visible classes only), `y` (labels over the visible
  classes, -1 = out of scope), `y_full`, `hidden` (ids of the hidden classes) and `lat`

## How it works

```
scripts/
  run.sh          run src.eval in the nix shell, tee the output to logs/
  flappy.sh       play Flappy Bird (src.flappy), tee the output to logs/
  build-llama.sh  rebuild llama-cpp-python with the HIP backend
  _common.sh      shared helper: finds the project root, re-enters the nix shell
src/
  common.py   Model wrapper: load, chat template, last_logits(); shared prompt framing
  data.py     dataset registry and loader; hide_classes for the abstention test
  mc.py       score_mc: one pass, options labelled A, B, C, ..., read the label-token logits
  gen.py      baseline: greedy generation of up to 8 tokens, parsed back to a class
  metrics.py  accuracy, NLL, ECE, temperature fit
  abstain.py  the abstention test and its metrics
  eval.py     runner (dispatches to abstain.py for --hide)
  jev.py      Jev: probabilities over any list of options (a thin wrapper on score_mc)
  flappy.py   turn-based Flappy Bird, ASCII view, policies, balanced-state metric
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
logit of its letter. With `abstain=True` a last option, "None of the above", is appended and
scored like any other.

**Metrics** (`metrics.py`). The scores are split in half by index. A temperature `T` is fitted on
the first half by minimising NLL, and everything is reported on the second half.

- accuracy, next to the chance level
- NLL and ECE (10 bins), both raw (`T=1`) and after temperature scaling (`cal`)
- latency per question for `mc` and the generation baseline (after a warm-up pass)

The generation baseline is there for reference only. It has no probabilities, so it only gets
accuracy and latency.

## The abstention test

To get out-of-scope examples with known labels, `--hide K` removes K classes from the options.
Their examples are then out of scope, and the right answer for them is to abstain. Two arms are
scored on the same examples (`abstain.py`):

- **A**: the visible classes plus "None of the above"; abstaining means predicting that option.
- **B**: the visible classes only; abstaining means the top probability is below a threshold.

The report gives the in-scope accuracy of both arms, arm A's abstention recall and false-abstain
rate, the AUROC of each arm at separating in-scope from out-of-scope (p(None) for A, one minus
the top probability for B), the recall of each at a fixed 5% false-abstain rate, and the recall
per hidden class. Which classes are hidden matters a lot, since a hidden class next to a visible
lookalike (a Film beside an Album) is close to impossible to reject, so treat one draw as one
sample and vary `--hide-seed`.

## Flappy Bird vs openjev

[openjev](https://huggingface.co/AlexWortega/openjev) plays Flappy Bird with a fine-tuned NLI model
(about 27.5 pipes out of a possible 28). We run the same game, turn-based, through the multiple-choice
readout (`scripts/flappy.sh`, `--watch` draws it in the terminal). Details in [flappy.md](flappy.md).

- With raw numbers in the prompt the 4B scores at most 8.33 pipes, and 0 with openjev's verbatim text.
- With the state bucketed in code and sent as words ("The bird is far below the centre of the gap and is
  rising fast."), as the Jev 1.13 docs recommend, it scores **28 in all six episodes** at plain argmax,
  matching openjev's number, and again on six fresh seeds.
- **Best approach so far:** a semantic state (named buckets, only the fields the decision needs, no
  numbers), options worded like the state ("The bird is below/above the centre of the gap"), plain
  argmax. Tricks such as thresholds come last.
- Read it carefully: the model executes a stated comparison, it does not infer the move. With
  "Flap" / "Do nothing" and only a game description it scores 0. The statement-to-move mapping is
  hand-written, and openjev's model reads raw numbers, so this is not a like-for-like comparison.
  6 episodes, one model.

## Caveats

- Only the held-out half is scored for calibration (250 examples at 500), so accuracy
  differences of a few points are within noise.
- The generation baseline is the best case for generation: a terse answer capped at 8 tokens
  with thinking off. One-pass readout therefore does not beat it on latency here. A chat-style
  baseline with long answers, and shared-context batching of several questions, are not measured.
- Several questions per pass and non-enum outputs are not tested.
- The models are 4-bit quantized and run on a shared-memory iGPU.

## License

GPL-3.0, see [LICENSE](LICENSE).
