# 05 Flappy Bird, numeric state

2026-09-21 · moth cjd4t · details in `../flappy.md`

## Hypothesis

With openjev's physics and openjev's own numeric state text, jeb's MC readout plays Flappy Bird about as
well as openjev (27.5-28 of 28 pipes).

## Setup

- Turn-based Flappy Bird (the game waits, so latency cannot matter), openjev's physics, 900 steps = 28 pipes.
- State: numbers (heights, velocity, gap span, signed offset from the gap centre). Options: position
  statements ("The bird is below/above the centre of the gap", option A = flap) or actions (flap / do nothing).
- Qwen3.5 4B (and 9B), argmax, 6 episodes. Reading quality: balanced accuracy and AUROC on 100 states
  balanced below/above the centre (plain agreement is misleading: never flapping agrees ~90%).
- References: never flap 0, random ~0.17, lookahead oracle ~25, perfect rule 28, openjev 27.5-28.
- Reproduce: `scripts/flappy-experiments.sh --state numbers ...` (flags per row in flappy.md).

## Data

| state text | balanced acc at 0.5 | game score mean (max) |
|---|---|---|
| shortened copy of openjev's | 0.96 | 8.33 (18) |
| shortened, 9B | - | 5.83 (9) |
| openjev's verbatim | 0.76 | 0 |
| verbatim minus goal sentence | 0.89 | 2.33 (5) |
| verbatim minus position and goal, no final period | 0.90 | 3.33 (8) |
| same, flap at p ≥ 0.6 | - | 22.00 (28) |
| same, sampled at T = 0.5 / 1 / 2 | - | 0 |
| raw heights only (offset and position dropped) | 0.50 (AUROC 0.46) | - |

Actions wording: ~chance (0.52).

## Conclusion

- Far below openjev with the identical prompt (0 vs 27.5).
- The signed offset is the essential input; raw heights alone are at chance: the model cannot subtract
  heights in one pass.
- Wording mostly moves the decision boundary, not the ranking (goal sentence, trailing period push
  toward "flap"). A fitted threshold recovers a lot, but that patches the boundary.
- Every death is a near-tie spurious flap while above the gap: a few percent per-tick error compounds.
- 9B no better than 4B. Sampling is worse than argmax.
- Led to 06, after the Jev 1.13 docs said named buckets beat numbers.
