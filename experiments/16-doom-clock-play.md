# 16 Doom: clock-position scene in play (27B)

2026-09-23 · moth blbci

## Hypothesis

The clock-position scene read best in the harness at 27B (14: AUROC 0.94 / 0.92 / 0.77, correct turn
direction, no shared vocabulary with the actions). In play it should at least match, and hopefully beat,
the English scene (15: 5.8) and per-action letters (10: 7.6).

## Setup

- Variant 7 (per action: bare instruction, quoted frame, yes/no tokens) with the scene in clock positions,
  per-action offsets calibrated on random-play states from separate seeds.
- Scene text identical to 14, including its grammar slip for half-hour bins ("a Demon at between 10 and 11
  o'clock"), kept so play matches the harness.
- Qwen3.8 27B, seeds 1000-1004, per-decision log kept for the action breakdown.
- Reproduce: `scripts/doom-experiments.sh --variant 7 --context clock --calibrate --watch --model models/Qwen3.8-27B-UD-Q4_K_M.gguf`.

## Data

Fitted offsets: turn left 0.110, turn right 0.112, attack 0.125.

| seed | 1000 | 1001 | 1002 | 1003 | 1004 | mean |
|---|---|---|---|---|---|---|
| variant 1 (joint MC, English) | 5 | 6 | 6 | 7 | 5 | 5.8 |
| variant 5 (per action, letters, English) | 8 | 6 | 7 | 12 | 5 | 7.6 |
| variant 7, English (15) | 6 | 4 | 7 | 7 | 5 | 5.8 |
| variant 7, clock | 6 | 6 | 9 | 6 | 4 | 6.2 |

Choices by the nearest enemy's bearing (all 5 episodes, 621 decisions, attack 31%):

| bearing | attack | turn left | turn right |
|---|---|---|---|
| 10-11 o'clock | 18 | 13 | 0 |
| 11 o'clock | 5 | 19 | 0 |
| 11-12 o'clock | 42 | 22 | 0 |
| 12 o'clock | 95 | 0 | 0 |
| 12-1 o'clock | 22 | 0 | 14 |
| 1 o'clock | 10 | 0 | 8 |
| 1-2 o'clock | 0 | 0 | 3 |
| nothing visible | 0 | 212 | 138 |

## Conclusion

- Kills within noise of the other 27B variants (5 episodes each, standard error ≈ 0.7).
- The policy itself is the most sensible yet: no wrong-direction turn at all, always attacks at 12 o'clock,
  never shoots an empty screen (350 of 350 empty-screen decisions turn). Decision-free-ish input produced
  correct, direction-consistent behaviour at 27B.
- The kill count is likely capped by aim resolution, not judgment: it also fires at off-centre bearings
  (10-11 and 11-12 o'clock), and the 12 o'clock bin (±7.5°) is wider than the crosshair covers, so many
  shots miss while the enemy closes in. A property of the clock encoding, not of the model.
- Open: finer bearings without leaving the clock convention (e.g. "12 o'clock, slightly left"), or a
  target-on-crosshair fact; brevity style (17).
