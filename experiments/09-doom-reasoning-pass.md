# 09 Doom: reasoning pass before the readout

2026-09-22 · moth blbci

## Hypothesis

openjev uses two steps, so jeb can too: a short free-text "plan" generated first, then the same
letter readout, lets the model reason its way past the first-option bias.

## Setup

- Variant 4: openjev's wording, then one greedy generation of ~24 tokens ("In one short phrase, what
  should the player do right now and why?"), appended as "Plan: …" before the MC readout.
- 0.8B and 27B, seeds 1000-1004. Reproduce: `scripts/doom-experiments.sh --variant 4`.
- Note: openjev's two steps are not reasoning; they are one classifier pass per action (see 10).

## Data

| model | kills | vs variant 1 |
|---|---|---|
| 0.8B | 0 0 0 0 0, identical steps to variants 1-3 | 0.0 → 0.0 |
| 27B | 5 6 6 5 5 → 5.40 | 5.8 → 5.4 |

0.8B's own plan text was sensible ("aim at the Demon and fire") and the readout still chose "turn left".

## Conclusion

- Helps neither size. The small model's readout ignores its own reasoning; chaining a weak component
  onto itself adds no capability. The big model does not need it and pays extra latency per decision.
