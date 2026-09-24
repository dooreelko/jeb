# 08 Doom: state wording and model size

2026-09-22 · moth blbci

## Hypothesis

Flappy saturated because the prompt stated the decision. In vizdoom, progressively stripping the
bucketed wording (openjev's own → finer buckets → raw numbers) shows how much of the decision the
wording carries; the openjev-style baseline should land near openjev's ~11 kills per episode.

## Setup

- vizdoom `defend_the_center` (openjev's scenario), 3 actions (turn left, turn right, attack),
  one decision per 4 tics, turn-based. Metric: kills per episode, decisions survived.
- State from the game's object labels (enemy type, horizontal offset from the crosshair, apparent
  size) plus ammo and health; self and transient effects (blood, bullet puffs) filtered out.
- Readout: jeb's one-pass MC, all 3 actions as options A-C in one prompt, argmax.
- Variants: 1 = openjev's wording ("a Demon left of the crosshair (far)", ±0.015 of screen width is
  "exactly on"); 2 = finer buckets (7 offset, 5 distance); 3 = raw numbers, no descriptive words.
- Qwen3.5 0.8B (default) and Qwen3.8 27B, seeds 1000-1004.
- Controls (added later, see 10): random 1.0-1.3, always-attack 1.5-1.6 kills.
- Reproduce: `scripts/doom.sh` (variant 1), `scripts/doom-experiments.sh --variant N`.

## Data

| variant | 0.8B kills (steps per episode) | 27B kills |
|---|---|---|
| 1 openjev wording | 0 0 0 0 0 (73 66 82 81 66) | 5 6 6 7 5 → **5.8** |
| 2 finer buckets | 0 0 0 0 0 (identical steps) | - |
| 3 raw numbers | 0 0 0 0 0 (identical steps) | - |

0.8B raw probabilities: "turn left" (option A) 0.60-0.80 in almost every state, attack rarely above 0.13.

## Conclusion

- At 0.8B the wording is not the variable: identical episodes step for step across all three. The joint
  MC readout saturates on the first option, making it worse than random.
- At 27B the same readout reads the state: 5.8 kills (above openjev's first version, 5.2; about half its second, 10.4), longer survival.
  So the position bias is largely a small-model capability issue.
- Led to 09 (reasoning) and 10 (decomposition).
