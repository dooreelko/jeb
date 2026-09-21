# Flappy Bird with the multiple-choice readout

Turn-based Flappy Bird with openjev's physics, played by a plain Qwen3.5 through the multiple-choice
readout (one pass, options `A`/`B`, argmax picks the move). The target is openjev's 27.5-28 pipes; a
perfect rule scores 28 and a lookahead oracle about 25. Run it with `scripts/flappy.sh` (`--watch` draws
the game in the terminal). Decisions and numbers are also in moth `cjd4t`.

## What we learned

1. **The model reads the state fine.** With the numbers it separates "bird below the gap centre" from
   "above" at AUROC 0.94-0.97 (4B). An early claim that it could not read numbers was wrong: the target
   then was a lookahead oracle whose own ceiling is 0.92.
2. **It still plays badly.** The best result is 8.33 pipes (a shortened prompt), against 28 for the
   perfect rule. The failure is per tick: about 5% wrong decisions compound over hundreds of ticks.
   Every death is the bird too high, from a spurious flap while already above the centre. Missed flaps
   below the centre are harmless.
3. **Bigger did not help.** The 9B scored 5.83 against the 4B's 8.33 and flapped wrongly far more often
   when high.
4. **Prompt wording is the largest lever we found**, and it acts mostly on calibration rather than ranking:
   - The signed offset from the gap centre is the essential input. Without it and the position sentence
     the model is at chance (AUROC 0.46).
   - openjev's goal sentence and the final period on the option shift the boundary toward "flap".
     Verbatim prompt: balanced accuracy 0.76, game score 0. Without the goal sentence: 0.89, 2.33 pipes.
   - Scores range from 0 to about 8 across variants; none gets near openjev's.
5. **A fitted threshold does not fix it.** Verbatim text with a threshold gave 1.67; with the goal sentence
   dropped the fitted threshold barely moved (0.495) and scored 1.67.
6. **Hints backfire.** Adding "the centre of the gap is above the bird" collapsed the model to chance
   (it mirrors the option wording), and a words-only state scored 0. Action wording ("flap" / "do
   nothing") fails, as it does for openjev.

## Results (4B unless noted, position wording, 6 episodes, 900 steps)

| state text | reading, balanced acc at 0.5 | game score, mean (max) |
|---|---|---|
| shortened copy of openjev's | 0.96 | 8.33 (18) |
| shortened, Qwen3.5 9B | - | 5.83 (9) |
| openjev's verbatim | 0.76 | 0 (0) |
| openjev's verbatim + fitted threshold | 0.92 | 1.67 (4) |
| verbatim minus goal sentence | 0.89 | 2.33 (5) |
| verbatim minus goal sentence + fitted threshold | - | 1.67 (2) |
| words only (hints, ablation) | 0.86 | 0 (0) |

References: never flap 0, random about 0.17, lookahead oracle about 25, perfect rule 28, openjev 27.5-28.
The reading column is measured on 100 states balanced between "below" and "above" the gap centre.

## Caveats

- 6 episodes per row and 100 states per reading cell: only the big gaps mean anything (0.76 vs 0.89,
  the collapse without the offset), not orderings among rows that score 0-8.
- Plain agreement with an oracle is misleading (the oracle flaps about 10% of ticks, so never flapping
  scores about 0.90); we report balanced accuracy and AUROC against the truth of the statement.
- openjev's own run was played at 15 fps real time; ours is turn-based.
- The shortened prompt is a deviation from openjev's text, so it is an ablation, not the comparison.

## Untried

Action repeat (one decision every 2-3 ticks), a stricter flap threshold (p >= 0.8), few-shot examples,
models beyond 9B.
