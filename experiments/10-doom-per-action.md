# 10 Doom: per-action yes/no (decomposition)

2026-09-23 · moth blbci

## Hypothesis

A single joint multiple-choice pass asks too much of the model (parse, weigh, rank all options at once).
Decomposing it into one narrow yes/no judgment per action, combined by a dumb argmax, is what openjev's
scoring effectively does (one premise/hypothesis pass per action) and may work where the joint pass
fails, even at 0.8B. A constant bias toward the first letter hits every call alike and cancels.

## Setup

- Variant 5: per action, `Given a context of "<scene> Question: is "<action>" the right move right
  now?" and possible answers of A. Yes B. No …`, p(yes) = p(A); pick the highest p(yes). Scene in openjev's wording.
- Calibration arm: subtract each action's mean p(yes), fitted on random-play states from separate seeds.
- Controls: random, always-attack (no model). Same seeds for everything.
- Reproduce: `scripts/doom-experiments.sh --variant 5 [--calibrate]`, `--control random|attack`.

## Data

0.8B, 20 episodes, seeds 1000-1019:

| config | mean kills |
|---|---|
| random | 1.30 |
| always-attack | 1.50 |
| variant 5 | **2.30** |
| variant 5, calibrated (offsets 0.57 / 0.54 / 0.62) | 2.10 |

Paired with always-attack per seed: variant 5 never worse (+0.8, ~3 standard errors). In the first
5-episode run, turns always went toward the visible enemy (0 wrong-direction turns); it attacked ~87%.

27B, seeds 1000-1004 (run stopped after 7 episodes, the pairing had answered the question):

| seed | 1000 | 1001 | 1002 | 1003 | 1004 | mean |
|---|---|---|---|---|---|---|
| variant 1 | 5 | 6 | 6 | 7 | 5 | 5.8 |
| variant 5 | 8 | 6 | 7 | 12 | 5 | **7.6** |

Readout check (added later): the probability at the answer position sits on the letters, so p(A) is
a real "yes", not leftovers: A+B hold 99.5% on average (min 99.0%) over 120 prompts on 0.8B, and 99.8%
(min 99.5%) over 60 prompts on 27B.

Recording: [`../doom.gif`](../doom.gif), 27B seed 1003 (the 12-kill episode, reproduced exactly on
replay), with a caption bar showing each decision's p(yes) per action. Made with `--gif PATH`.
Visible in it: at point-blank range the model is lost (all three p(yes) near 0).

## Conclusion

- Decomposition lifts 0.8B from worse-than-random to above both controls, and stacks with size (27B +1.8,
  one episode at 12, above openjev's ~11). The first real lever that is not model size.
- Calibration adds nothing: the offsets are too close to reorder choices.
- Caveat found later (12): the scene wording ("left of") nearly names the action ("turn left"), so part
  of this may be word matching. Cost: 3 passes per decision (see moth xaj5x for shared-prefix batching).
