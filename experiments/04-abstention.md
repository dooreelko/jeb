# 04 Abstention as an explicit option

2026-09-20 · moth xuhkj (Cleanup)

## Hypothesis

An explicit "None of the above" option, scored in the same softmax, abstains on out-of-scope inputs
better than thresholding the top probability. Pass criterion set before running: ≥ 80% abstention
recall at ≤ 5% false abstains, costing ≤ ~1 point of in-scope accuracy.

## Setup

- Out-of-scope inputs simulated by hiding 3 of the 14 DBpedia classes from the option list (seed 0:
  Natural place, Village, Plant); 500 examples, 402 in scope, 98 out.
- Arm A: visible classes + abstain option. Arm B: visible classes only, abstain below a threshold.
- Compared at a fixed operating point (5% false abstains) and threshold-free (AUROC).
- Qwen2.5 3B (rerun on the corrected pipeline), Qwen3.5 4B, 9B. Y/N scorer removed from the code.

## Data

| | Qwen2.5 3B | Qwen3.5 4B | Qwen3.5 9B |
|---|---|---|---|
| in-scope accuracy without / with the option | 0.943 / 0.938 | 0.958 / 0.950 | 0.970 / 0.965 |
| recall / false-abstain rate, as is | 0.980 / 0.007 | 0.949 / 0.017 | 0.990 / 0.012 |
| AUROC: p(None) / threshold | 0.998 / 0.950 | 0.997 / 0.899 | 1.000 / 0.963 |
| recall at 5% false abstains: option / threshold | 0.990 / 0.643 | 0.990 / 0.327 | 1.000 / 0.724 |

## Conclusion

- The explicit option passes for all three models (0.99-1.00 recall at 5% false abstains, −0.5 to −0.8
  points in scope); a threshold reaches only 0.33-0.72 and varies a lot by model.
- Qwen2.5 is clearly overconfident (T 3.7-4.9); Qwen3.5 near calibrated. Real, not a tokenization artefact.
- Caveat: one easy draw of hidden classes (distinct from the visible ones), so recall is likely an upper
  bound; lookalike classes and genuinely foreign text untested.
