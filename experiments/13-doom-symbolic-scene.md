# 13 Doom: symbolic scene language

2026-09-23 · moth blbci

## Hypothesis

(The user's idea.) Separating the language of the scene from the language of the decision removes word
matching: scene and question in meaningless symbols, glossed once in English, answer T/N. A model that
understands the scene still decides correctly; one that only matched "left of" to "turn left" drops to chance.

## Setup

- Abstract symmetric Unicode symbols (● ■ ◆ ★ ⬢ ♣ …; no arrows, half-filled shapes, weapons, faces or
  yes/no-looking marks), re-drawn per state so no symbol's own bias sticks. First draft used Slavic-looking
  codes (WRG ≈ *vrag*, LVO ≈ *levo*) that a multilingual model would read, so symbols replaced them.
- Prompt: "You are a decision maker in a symbolic language. You answer only T for yes or N for no."; game
  line with health/ammo in words; glossary (enemy, left of / on / right of the crosshair, far / close / very
  close, no enemies, the 3 actions); scene lines like `⬢ ☾ ★`; question `♠?`. One pass per action.
- Same information as openjev's wording, same labelled states and metrics as 12, next to the best English
  combination. T/N hold ~80% of the mass on 0.8B (rest mostly a symbol).
- Reproduce: `scripts/doom-experiments.sh --agree 40 --coded [--model …27B…]`.

## Data

| model | prompt | AUROC turn left / right / attack | acc | attack when none |
|---|---|---|---|---|
| 0.8B | English (bare, quoted, tokens) | 0.74 / 0.80 / 0.73 | 0.36 | 0.38 |
| 0.8B | symbolic | 0.53 / 0.50 / 0.50 | 0.30 | 0.28 |
| 27B | English (bare, quoted, tokens) | 0.90 / 0.87 / 0.71 | 0.57 | **0.00** |
| 27B | symbolic | **0.31 / 0.29** / 0.75 | 0.25 | 0.30 |

## Conclusion

- 0.8B: chance. Its signal was surface word matching.
- 27B: information passes through the symbols (attack 0.75), but both turns are inverted, ~3 standard
  errors below chance: "enemy left of the crosshair" → "turn right" (camera-pan confusion). So even 27B's
  correct turning in English is at least partly word matching; forced to reason spatially, it gets
  direction backwards.
- 27B with the best English combination reads all three actions and never shoots an empty screen.
- Discussion recorded in blbci: a fully allocentric scene (coordinates + heading) needs trigonometry,
  which a one-pass readout cannot do and jev rules out; it is also not more unbiased, just less processed.
  The line that holds is perception (bearing, size) vs decision; the bias that matters is shared
  vocabulary between scene and actions. Led to 14 (clock positions).
