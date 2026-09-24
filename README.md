# Let's see how close can we get to Jev

[Jev](https://docs.typesafe.ai) is a product that answers questions with a choice from a fixed list
and a probability for each option, instead of free text, so a program can act on the answer
directly. This repo asks whether an ordinary open language model, run with plain
[llama.cpp](https://github.com/ggml-org/llama.cpp), can do the same. See [IDEA.md](IDEA.md) for the
starting idea.

The trick: a language model works by predicting the next word, with a likelihood for every possible
word. Instead of letting it write an answer, we label the options `A`, `B`, `C`, … and look at how
likely it finds each letter as the next word. One step, no writing, and a probability per option.
An extra last option, "None of the above", lets it decline.

![Playing Flappy bird](./flappy.gif)
![Playing doom](./doom.gif)

## Experiments

- **Sorting text into categories**: news topics and the type of a Wikipedia article.
- **[Flappy Bird](flappy.md)**: the bird flaps or not, one decision per frame.
- **[Doom](doom.md)**: turn left, turn right or shoot, against monsters coming from all sides.

Every experiment is written up in [experiments/](experiments/README.md) (question, setup, data,
conclusion), with an overall reading in [experiments/LESSONS.md](experiments/LESSONS.md). Decisions
are tracked in the moth tasks (`moth ls`, `moth show <id>`).

## Findings

**Goal.** Get jev-like answers (a choice plus a probability per option) out of a vanilla model on
plain llama.cpp, and compare with jev and with [openjev](https://huggingface.co/AlexWortega/openjev),
an open imitation of jev that uses a specially trained model.

**How close we got.** For sorting text into categories, close. For making decisions in games, only
partly.

- *Sorting.* Given a Wikipedia article and 14 possible types (company, artist, village, film, …),
  4B and 9B models pick the right one 96-97% of the time, as often as when they write the answer out.
  The probabilities are honest: when the model is unsure it is usually wrong, so the least confident fifth of
  the answers holds most of the mistakes and can be sent to a human. With a "None of the above"
  option it rejects 95-99% of articles whose type is not on the list, while wrongly rejecting only
  about 1-2% of the rest. A bigger model did not do better.
- *Flappy Bird.* 28 out of 28 pipes, matching openjev, but only because the description of the game
  already contains the answer: "the bird is below the centre of the gap" is a sentence the model just
  has to agree with, and code turns that into "flap". Told only the rules ("flapping pushes the bird
  up"), it fails completely.
- *Doom.* openjev gets about 11 kills per game. Our best is 7.6 on average (one game 12) with the
  largest model we can run (27B). The smallest (0.8B) does worse than pressing buttons at random.

**Most important findings.**

1. **Model size matters for decisions, not for sorting.** A 4B sorts as well as a 27B, but in Doom
   the same question gives 0 kills with the 0.8B and 5.8 with the 27B.
2. **Ask about each option separately.** "Which move: A. turn left, B. turn right, C. attack?" makes
   a small model pick the first option almost always. Asking "is attack the right move?", "is turn
   left…?", "is turn right…?" separately and taking the most confident yes lifted the 0.8B above
   random play and the 27B from 5.8 to 7.6 kills. openjev also judges each move separately.
3. **Small models match words instead of reading the situation.** If the scene says "a monster left
   of the crosshair", the small model turns left because "left" appears in both. Say "a monster at
   11 o'clock" instead, and it turns the wrong way. The big model handles the clock positions and plays
   the most sensible game (it never turns away from a monster and never shoots at an empty screen),
   but it gets left and right backwards when the positions are given as made-up symbols.
4. **What the description says matters more than how the question is asked.** Plain statements
   worked out in code ("far below", "rising fast") work best. Raw numbers the model has to compare
   itself do not: given the bird's height and the gap's edges, it cannot tell whether the bird is
   below the gap. Short radio-style descriptions work as well as full sentences for the big model.
5. **Letting the model think first did not help.** The small model wrote "aim at the Demon and fire"
   and then turned left anyway. The big one did not need it.
6. **Small models react to wording details.** "Is 'turn right' the right move right now?" uses
   "right" in two meanings, and that alone blurred the answer; a trailing full stop shifted Flappy's
   decisions.

**Where the gap probably is.** openjev adds a small trained layer on top of its model to score each
option; we use models as they are. That is the most likely reason for the remaining difference in
Doom. We have not tested it.

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

## License

GPL-3.0, see [LICENSE](LICENSE).
