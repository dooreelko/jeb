# 15 Doom: best readout in play (27B)

2026-09-23 · moth blbci

## Hypothesis

The readout that separates states best in the harness (12/13: bare instruction, quoted frame, yes/no
tokens, per action) also plays best, beating variant 5 (7.6 kills on these seeds at 27B), once
per-action offsets are calibrated.

## Setup

- Variant 7: per action, the best English combination from 12 (openjev's scene wording, not clock
  positions), argmax after subtracting each action's mean p(yes) fitted on random-play states from
  separate seeds (frozen before play).
- Qwen3.8 27B, seeds 1000-1004, compared per seed with variants 1 and 5 on the same seeds.
- Reproduce: `scripts/doom-experiments.sh --variant 7 --calibrate --model models/Qwen3.8-27B-UD-Q4_K_M.gguf`.

## Data

Fitted offsets: turn left 0.138, turn right 0.122, attack 0.133.

| seed | 1000 | 1001 | 1002 | 1003 | 1004 | mean |
|---|---|---|---|---|---|---|
| variant 1 (joint MC) | 5 | 6 | 6 | 7 | 5 | 5.8 |
| variant 5 (per action, letters) | 8 | 6 | 7 | 12 | 5 | 7.6 |
| variant 7 (per action, best harness readout) | 6 | 4 | 7 | 7 | 5 | 5.8 |

## Conclusion

- The best harness readout does not play better: 5.8, equal to variant 1 and below variant 5 on every
  seed but one tie. Five episodes, so the gap to variant 5 is suggestive, not firm.
- Harness separation predicts play only loosely, as in flappy (05). Likely reasons: per-action AUROC
  measures ranking of states within one action, while play needs the right comparison between actions
  in one state (argmax accuracy 0.57, not better than the rest); the labelled states come from random
  play, not from the states a good policy visits; and the attack label (crosshair inside the box) is
  not the same as "a shot will kill".
- Calibration hardly mattered (offsets nearly equal).
- Open: play with the clock scene (14), which read best in the harness; a harness metric closer to
  play (per-state accuracy on states from the model's own play).
