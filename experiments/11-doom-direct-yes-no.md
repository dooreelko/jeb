# 11 Doom: yes/no read from the yes/no tokens

2026-09-23 · moth blbci

## Hypothesis

In variant 5 the yes/no is wrapped in the MC letter scaffold (answer → A/B mapping), the step where 0.8B
broke before. Asking plainly and reading the yes/no tokens directly removes that step and should sharpen
the signal. "Best move" (comparative) might fix attacking at off-centre enemies.

## Setup

- Variant 6: system "Answer only yes or no.", user `<scene> Is "<action>" the right move right now?`;
  p(yes) from yes/Yes vs no/No logits (~99% of the mass at that position on 0.8B). Argmax across actions.
- `--question right|best` as a separate variable. 0.8B, 20 episodes, seeds as in 10.
- Reproduce: `scripts/doom-experiments.sh --variant 6 --question right|best`.

## Data

| config | mean kills | attack share |
|---|---|---|
| always-attack | 1.50 | 100% |
| variant 5 | 2.30 | ~87% |
| variant 6, "right move" | 1.65 | 95% |
| variant 6, "best move" | 1.45 | 96% |

Wrong-direction turns reappeared (enemy right → turned left, 3×).

## Conclusion

- Worse in play, collapsing toward always-attack. "Best" no better than "right".
- Not a clean ablation: variants 5 and 6 differ in persona, framing and readout at once, so this does
  not show that the letter scaffold is better. That question went to the factorial (12).
