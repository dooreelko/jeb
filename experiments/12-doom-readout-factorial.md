# 12 Doom: readout agreement factorial

2026-09-23 · moth blbci

## Hypothesis

Variant 6 lost to variant 5 because of one of the three things that differ between them: persona
("You are a decision maker…" vs a bare instruction), framing (scene quoted in jeb's "Given a context
of" frame vs plain), or readout (A/B letters vs yes/no tokens). Scoring every combination on fixed,
labelled states, without playing, isolates it.

## Setup

- Labelled states from random play on separate seeds, target from raw geometry: attack when the crosshair
  is inside a monster's box, else turn toward the monster nearest the crosshair, "none" when nothing is
  visible. 40 states per target (160), scene in openjev's wording.
- Per action and combination: AUROC of p(yes) separating states where that action is right from states
  where it is clearly wrong (signal); mean p(yes) (bias); argmax accuracy; attack rate on empty screens.
- Qwen3.5 0.8B, all 2×2×2 combinations. AUROC ≈ ±0.06 at this size.
- Reproduce: `scripts/doom-experiments.sh --agree 40`.

## Data

| persona | framing | readout | AUROC left / right / attack | mean | acc | attack when none | |
|---|---|---|---|---|---|---|---|
| decision | quoted | letters | 0.72 / 0.57 / 0.46 | 0.58 | 0.38 | 1.00 | = variant 5 |
| decision | quoted | tokens | 0.44 / 0.51 / 0.50 | 0.49 | 0.33 | 0.45 | |
| decision | plain | letters | 0.81 / 0.40 / 0.40 | 0.54 | 0.50 | 1.00 | |
| decision | plain | tokens | 0.68 / 0.57 / 0.29 | 0.51 | 0.31 | 1.00 | |
| bare | quoted | letters | 0.73 / 0.45 / 0.40 | 0.53 | 0.48 | 0.50 | |
| bare | quoted | tokens | 0.74 / 0.80 / 0.73 | **0.75** | 0.36 | 0.38 | best |
| bare | plain | letters | 0.81 / 0.29 / 0.46 | 0.52 | 0.38 | 0.53 | |
| bare | plain | tokens | 0.64 / 0.51 / 0.48 | 0.54 | 0.40 | 1.00 | = variant 6 |

## Conclusion

- Neither variant 5 nor 6 really reads the scene: nearly all signal is "turn left"; "turn right" and
  "attack" sit at or below chance in most combinations. Variant 5 won in play mainly via a stronger
  turn-left signal. No single factor explains it; they interact.
- Suspected cause of the left/right asymmetry: the question "is "turn right" the right move right now?"
  uses "right" three times in two meanings. "Turn left" has no collision, so the scene's "left of" gets through.
- Attack is capped by the context, not the readout: "exactly on" only for a near-dead-centre enemy, so
  many crosshair-on-target states read "left of"/"right of".
- One combination reads all three actions (bare, quoted, tokens), but its per-action levels are not
  comparable across actions (argmax accuracy 0.36): usable in play only with calibration (15).
- Method lesson: fixed labelled states + AUROC separate signal from bias in seconds; gameplay is a slow,
  noisy measure of readout quality.
