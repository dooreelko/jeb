# 17 Doom: brevity (radio-call) scene

2026-09-23 · moth blbci

## Hypothesis

We are not talking to a human, so the scene can be condensed into a terse military/aviation radio call.
The information is the same; the question is whether LLMs handle a denser, well-known register as well as
prose, or whether dropping function words (which bind attributes to objects) hurts, especially small models.

## Setup

- Same clock bins as 14 (half-hour steps, 12 o'clock straight ahead), only the style changes:
  - prose (14): "Doom, defending the center. Ammo: 18. Health: 100. You see a Demon at between 10 and 11
    o'clock (very close)." / "No enemies are visible right now."
  - brevity: "Doom, defend the center. Ammo 18, health 100. Contact: Demon, 10-11 o'clock, very close."
    / "No contact." (also fixes the "at between" slip).
- Question, instruction and readout unchanged (bare instruction, quoted frame, yes/no tokens), so only
  the scene style varies. Same 160 labelled states and metrics as 12-14.
- Reproduce: `scripts/doom-experiments.sh --agree 40 --clock --brevity [--model …27B…]`.

## Data

| model | scene | AUROC turn left / right / attack | mean | acc | attack when none |
|---|---|---|---|---|---|
| 0.8B | clock prose | 0.34 / 0.35 / 0.79 | 0.49 | 0.33 | 0.38 |
| 0.8B | brevity | 0.39 / 0.45 / 0.79 | 0.54 | 0.34 | 0.85 |
| 27B | clock prose | 0.94 / 0.92 / 0.77 | 0.88 | 0.52 | 0.00 |
| 27B | brevity | 0.98 / 0.89 / 0.76 | 0.88 | 0.48 | 0.00 |

## Conclusion

- 27B: no difference within noise. Same information, same result, prose or radio call: density is free for
  a big model (and a shorter prompt is cheaper to prefill).
- 0.8B: no rescue, turns still at or below chance; and it now fires at 85% of empty screens (from 38%).
  "No contact." appears to pull it toward attack: again the small model reacts to form, not content.
- Practical rule so far: for the big model, pick the scene style for cost; for the small model, any
  wording change is a new experiment.
