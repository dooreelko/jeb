# 03 Many classes: DBpedia-14

2026-09-20 · moth m4pwf (and dcjaj, which made MC generic in the number of classes)

## Hypothesis

On a larger, cleaner problem with many options, the MC logit readout still gives correct answers,
honest probabilities and confidence that separates right from wrong; model size might finally matter.

## Setup

- DBpedia-14 (Company, Artist, Athlete, Film, Village, ...), 500 examples, fixed seed. Chance 7%.
- MC generic up to 26 classes (consecutive letters, instruction lists exactly the letters used).
- Qwen3.5 4B and 9B (27B stopped at ~200 of 500: not needed). MC vs greedy generation.
- New metric: usefulness of confidence, AUROC of confidence against correctness and accuracy on the
  most confident subsets (the accept-or-route use of a jev-like classifier).

## Data

| model | MC accuracy | generation | ECE raw | fitted T | MC / generation latency |
|---|---|---|---|---|---|
| 4B | 0.962 | 0.944 | 0.023 | 0.62 | 0.95 s / 0.84 s |
| 9B | 0.972 | 0.972 | 0.011 | 0.91 | 1.65 s / 1.46 s |

| run | AUROC conf → correct | acc on most confident 80% | errors in least confident 20% |
|---|---|---|---|
| ag_news 9B | 0.90 | 0.95 | 14 of 22 |
| ag_news 27B | 0.89 | 0.93 | 21 of 32 |
| DBpedia 4B | 0.90 | 0.99 | 15 of 19 |
| DBpedia 9B | 0.97 | 1.00 | 14 of 14 |

## Conclusion

- 96-97%, sizes indistinguishable; remaining errors are again a label overlap (Artist vs Written work).
- Well calibrated on a clean task (ECE 0.01-0.02, T ≈ 1).
- Confidence is actionable: the least confident 20% hold most errors; the confident 80% is 93-100% right.
- Latency: still no win over terse generation (the best case for generation).
- Verdict: for enum classification/routing, MC logit readout on a small local model behaves
  functionally like the target. MC is the approach going forward.
