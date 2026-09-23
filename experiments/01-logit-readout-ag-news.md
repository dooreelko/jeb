# 01 Logit readout on ag_news

2026-09-20 · moth gdeic (Bootstrap)

## Hypothesis

A plain instruction-tuned LLM used as a classifier by reading the next-token distribution (no
generation) gets close to jev's claims on accuracy, calibration and latency, compared with ordinary
greedy generation. Secondary question: is jev's speed advantage the method or the model size?

## Setup

- Task: ag_news, 4 balanced topics, 200 examples (fixed shuffle), articles truncated.
- Models: Qwen3.5 4B, 9B, Qwen3.8 27B, all 4-bit, GPU, thinking off, model's own chat template.
- Scorers: multiple choice (MC, one pass, letters A-D, softmax over the letter logits) and per-option
  Y/N (one pass per option, full option list in context). Baseline: greedy generation parsed to a class.
- Calibration: one temperature fitted on 100 examples, everything reported on the other 100
  (accuracy, log loss, expected calibration error, ECE).
- Pitfalls found: llama.cpp's score buffer is empty unless full logits are requested (read from the
  context instead); GPU work is asynchronous, so timing must read the logits back.

## Data

Held-out 100 (a few points is noise):

| model | scorer | accuracy | ECE raw → calibrated | fitted T |
|---|---|---|---|---|
| 4B | MC | 0.89 | 0.067 → 0.054 | 1.10 |
| 4B | Y/N gap | 0.82 | 0.098 → 0.070 | 1.78 |
| 4B | generation | 0.84 | n/a | n/a |
| 9B | MC | 0.88 | 0.054 → 0.052 | 1.29 |
| 9B | Y/N gap | 0.86 | 0.069 → 0.061 | 1.37 |
| 9B | generation | 0.86 | n/a | n/a |
| 27B | MC | 0.81 | 0.104 → 0.085 | 2.03 |
| 27B | Y/N gap | 0.84 | 0.126 → 0.048 | 3.51 |
| 27B | generation | 0.81 | n/a | n/a |

Latency per question: MC ≈ generation at every size (0.6 / 1.0 / 3.4 s); Y/N ≈ 3.5× (one pass per option).

## Conclusion

- All methods and sizes land in 0.83-0.89; none clearly wins. A 27B is not more accurate than a 4B.
- The dataset caps accuracy: most 27B errors are the known Business vs Sci/Tech label overlap.
- Modern instruction models are near calibrated (T ≈ 1), but calibration does not transfer across sizes.
- MC is not faster than generation when the generated answer is a few tokens: prefill dominates
  both. Speed wins need long answers, model size or shared-context batching.
- Y/N verdicts are independent and can disagree (two yes, or none) — MC's softmax avoids that.
- Led to: a cleaner, wider dataset (03), and dropping Y/N (04). An early run with mis-tokenized
  control tokens was superseded and must not be used.
