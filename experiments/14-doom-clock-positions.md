# 14 Doom: clock positions

2026-09-23 · moth blbci

## Hypothesis

Clock positions ("a Demon at 11 o'clock (far)") describe the scene perceptually, need no arithmetic, and
share no word with the actions, yet use a convention every model has seen. If the symbolic scene (13)
failed because the convention had to be learned from the prompt, a known convention should restore
correct turning without word matching.

## Setup

- openjev's wording with "left of / on / right of the crosshair" replaced by the clock bearing
  (screen offset → bearing using vizdoom's 90° horizontal field of view, half-hour steps of 15°,
  12 o'clock straight ahead: e.g. "11 o'clock", "between 11 and 12 o'clock"). Distance words unchanged.
- Everything else as the best English combination in 12 (bare instruction, quoted frame, yes/no tokens,
  "the right move right now"), same 160 labelled states, same metrics.
- Caveat: the 12 o'clock bin (±7.5°) is coarser than openjev's "exactly on" (±0.015 of screen width),
  so the attack information differs slightly from the English row.
- Reproduce: `scripts/doom-experiments.sh --agree 40 --clock [--model …27B…]`.

## Data

| model | scene | AUROC turn left / right / attack | mean | acc | attack when none |
|---|---|---|---|---|---|
| 0.8B | English | 0.74 / 0.80 / 0.73 | 0.75 | 0.36 | 0.38 |
| 0.8B | clock | **0.34 / 0.35** / 0.79 | 0.49 | 0.33 | 0.38 |
| 27B | English | 0.90 / 0.87 / 0.71 | 0.83 | 0.57 | 0.00 |
| 27B | clock | **0.94 / 0.92** / 0.77 | **0.88** | 0.52 | 0.00 |
| 27B | symbolic (13) | 0.31 / 0.29 / 0.75 | 0.45 | 0.25 | 0.30 |

## Conclusion

- 27B: the best readout on any prompt so far, turning in the right direction without shared vocabulary.
  Its inversion in 13 came from a convention it had to learn from the prompt, not from spatial
  reasoning as such.
- 0.8B: inverted with clock positions too. Once "left" is not shared between scene and action it gets the
  direction backwards, whatever the encoding. Its correct turning in English was word matching.
- So a decision-free-ish, perceptual scene is readable, but only by the big model.
