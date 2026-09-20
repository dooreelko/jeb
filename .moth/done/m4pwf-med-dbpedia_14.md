we'll need a bigger boat

----- AI agent updates -------

## Goal
The first dataset (ag_news) was too small a boat: its four classes overlap (Business vs Science
and Technology), which caps accuracy near 88-90% and hides any difference between methods or
model sizes. Move to a larger, cleaner, wider problem, the 14-class DBpedia ontology
(Company, Artist, Athlete, Film, Village, ...), to judge whether the multiple-choice logit
readout holds up with many options and real text, and whether model size matters.

## Decisions taken
- **DBpedia-14 as the second dataset**, with 500 shuffled examples (fixed seed), the article
  text alone, cut to keep prompts short. Datasets are a small registry: adding another is one
  entry (source, split, class names, how to read a row).
- **The multiple-choice scorer becomes generic in the number of classes.** Options are labelled
  by consecutive capital letters, the instruction lists exactly the letters in use, and only
  those letters' logits are read. Up to 26 classes. This is the part of the separate
  "Scale Up" idea needed here; nothing beyond it was done for that task.
- **Only multiple choice and the generation baseline are run on DBpedia.** The per-option Y/N
  scorer needs one pass per class (14 here), which is impractical, and the Y/N direction is
  being dropped anyway. It stays available behind a switch for the small dataset.
- **Model ladder unchanged:** Qwen3.5 4B and 9B, Qwen3.8 27B, all 4-bit, GPU, thinking off.
- **Also measured: how useful the confidence is,** not only how well calibrated. Test: does
  low confidence pick out the wrong answers, and how accurate is the confident subset.
  This is the property a Jev-like classifier is for (accept the confident answers, route the
  rest elsewhere).
- **Chance level is reported next to accuracy** (7% for 14 classes).

## Results (Qwen3.5 on DBpedia-14, 500 examples)
| model | MC accuracy | generation | ECE raw | fitted T | MC / generation latency |
|---|---|---|---|---|---|
| 4B | 0.962 | 0.944 | 0.023 | 0.62 | 0.95 s / 0.84 s |
| 9B | 0.972 | 0.972 | 0.011 | 0.91 | 1.65 s / 1.46 s |

The 27B run was stopped part-way (about 200 of 500 examples) because it was not needed for the
decision: size had not moved accuracy on the first dataset and did not on this one. There is
therefore no 27B result on DBpedia.

Usefulness of the confidence (MC, temperature-invariant ranking):

| run | accuracy | AUROC confidence -> correct | accuracy on most confident 80% | on most confident 50% | errors in least confident 20% |
|---|---|---|---|---|---|
| ag_news 9B | 0.89 | 0.90 | 0.95 | 1.00 | 14 of 22 |
| ag_news 27B | 0.84 | 0.89 | 0.93 | 0.97 | 21 of 32 |
| DBpedia 4B | 0.96 | 0.90 | 0.99 | 1.00 | 15 of 19 |
| DBpedia 9B | 0.97 | 0.97 | 1.00 | 1.00 | 14 of 14 |

## Findings
- Accuracy is 96-97% and the two sizes are statistically indistinguishable (about 1 point
  apart, standard error about 0.8). MC and greedy generation tie at 9B; MC is slightly ahead at
  4B. The remaining errors are concentrated in one confusion (Artist predicted as Written
  work), i.e. a labeling overlap again, so a larger model does not remove them.
- Calibration is good here: raw ECE 0.01-0.02 and fitted temperatures near 1. A clean task
  with strong models gives honest probabilities without much repair.
- Confidence is actionable: low confidence concentrates the errors, and keeping the most
  confident 80% gives 93-100% accuracy across both datasets.
- Latency: MC is not faster than greedy generation on these tasks. The generation baseline is
  the best case for generation (a terse answer of a few tokens), so decoding costs almost
  nothing and a one-pass method has no room to win. The claimed large speedups would have to
  come from long answers (explanations or reasoning), model size, or batching, none of which
  this setup measures. This was consciously left unmeasured.

## Verdict
Qualitatively good enough: for enum classification and routing, multiple choice with logit
readout on a small local model behaves functionally like the target, with correct answers,
usable probabilities and confidence that separates right from wrong. Multiple choice is the
approach going forward; the per-option Y/N scorer is dropped.

## Not established / out of scope
- Abstention: the concept example includes a "none of the above" option, but no dataset run so
  far contains out-of-scope inputs, so it is untested. The largest remaining functional gap.
- Answering several questions in one pass, and non-enum outputs (numbers, free text).
- Latency against a chat-style, long-answer baseline, and shared-context batching.
- Base-vs-instruct calibration.
