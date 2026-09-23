# 06 Flappy Bird, semantic state

2026-09-21 · moth cjd4t · details in `../flappy.md`

## Hypothesis

Following the Jev 1.13 docs ("pass in either the computed number or a named bucket", "send only the
fields the question needs"), a state computed in code as named buckets makes the readout play perfectly.
Second question: can the model infer the move from the game, or only execute a stated comparison?

## Setup

- State: two buckets computed in code, "The bird is {far below … far above} the centre of the gap and
  is {rising fast … falling fast}." No numbers, no goal sentence.
- Option styles: position statements (worded like the bucket); actions with a plain game description;
  actions plus a goal line; actions plus an explicit rule (leaks the policy, never a comparison result).
- Qwen3.5 4B and 0.8B, plain argmax, 6 episodes on tuning seeds 1000-1005 and fresh seeds 2000-2005.
- Reproduce: `scripts/flappy.sh` (the winning setup) and `scripts/flappy-experiments.sh --style ...`.

## Data

| options | balanced acc | game score mean |
|---|---|---|
| position statements | 0.99 | **28** in all six; also on fresh seeds; also with 0.8B |
| actions, plain description | 0.50 | 0 |
| actions + goal line (fresh seeds) | 0.50 | 0 |
| actions + explicit rule | 0.99 | 28 |

## Conclusion

- 28/28, but it is not "the model decides the game": the pipeline states the comparison ("slightly
  below the centre"), and the statement-to-move mapping is hand-written. The model executes a stated
  comparison perfectly and cannot get from a goal or a game description to the move.
- Not like-for-like with openjev, whose model reads raw numbers.
- This saturation is the motivation for the doom experiments (08+): what can be done without telling
  the engine what to decide.
