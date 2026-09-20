add abstension, remove yn. tidy up

----- AI agent updates -------

## Goal
Finish the multiple-choice approach as the single method: give it a way to abstain (the
"None of the above" of the concept example), drop the per-option Y/N scorer that the earlier
tasks showed to be unnecessary, and leave the code and docs describing only what remains.
Also, replace the invalid early results for the old model (Qwen2.5) with valid ones so the
newer models can be compared with it.

## Decisions taken
- **Y/N is removed entirely**, code and documentation, not kept behind a switch. Reasons: it
  costs one pass per class, its per-class verdicts are independent and can disagree (two options
  yes, or none), it was no more accurate than multiple choice, and the multiple-choice approach
  was already declared the direction. Its results stay in the earlier task specs as the record
  of what was tried; nothing else refers to it.
- **Abstention is an explicit extra option**, appended as the last option of the multiple-choice
  prompt and scored like any other, so the softmax makes it compete with the real classes. It
  is opt-in per call, so ordinary classification is unchanged. This follows the concept
  document and needs no extra pass.
- **How abstention is tested without new data:** hide K classes from the option list. Their
  examples become out of scope with known labels, and the correct answer for them is to abstain.
  The hidden classes are chosen by a fixed seed so runs are repeatable. Chosen setting: 3 of the
  14 DBpedia classes, 500 examples, seed 0 (hidden: Natural place, Village, Plant).
- **Two arms on the same examples:** A = visible classes plus the abstain option; B = visible
  classes only, abstaining when the top probability falls below a threshold. Compared on
  in-scope accuracy, abstention recall and false-abstain rate, threshold-free separation
  (AUROC of p(None) against low top-probability), and recall at a fixed 5% false-abstain rate,
  plus recall per hidden class. The fixed operating point is what makes the two arms comparable.
- **Pass criterion, set before running:** the explicit option is worth keeping if it reaches at
  least 80% recall at no more than 5% false abstains while costing no more than about one point
  of in-scope accuracy.
- **Old-model comparison redone on the corrected pipeline:** Qwen2.5-3B rerun on both datasets
  and the abstention test, so all models are compared under identical prompts and tokenization.
  The invalid earlier scores for it were overwritten.
- **Tidy-up:** the runner is reduced to multiple choice against generation, with the report
  printing separated and the abstention test dispatched by one option; score files are named
  per dataset; the README is rewritten around the single approach with an abstention section.

## Decisions rejected / deferred
- A confidence threshold as the abstention mechanism: measured as clearly worse (below).
- Keeping Y/N as an option "for reference": rejected, it only adds code paths.
- Running the 27B on the new tests: not needed, size had not moved accuracy.

## Results
Abstention, DBpedia-14, 3 classes hidden (402 in scope, 98 out of scope):

| | Qwen2.5 3B | Qwen3.5 4B | Qwen3.5 9B |
|---|---|---|---|
| in-scope accuracy without / with the option | 0.943 / 0.938 | 0.958 / 0.950 | 0.970 / 0.965 |
| abstention recall / false-abstain rate as is | 0.980 / 0.007 | 0.949 / 0.017 | 0.990 / 0.012 |
| AUROC: p(None) / threshold only | 0.998 / 0.950 | 0.997 / 0.899 | 1.000 / 0.963 |
| recall at 5% false abstains: option / threshold | 0.990 / 0.643 | 0.990 / 0.327 | 1.000 / 0.724 |

Model comparison on the corrected pipeline (metrics on the held-out half):

| | Qwen2.5 3B | Qwen3.5 4B | Qwen3.5 9B |
|---|---|---|---|
| ag_news MC accuracy / generation | 0.88 / 0.81 | 0.89 / 0.84 | 0.88 / 0.86 |
| DBpedia MC accuracy / generation | 0.892 / 0.940 | 0.944 / 0.936 | 0.972 / 0.952 |
| raw ECE (ag_news / DBpedia) | 0.117 / 0.101 | 0.067 / 0.023 | 0.054 / 0.011 |
| fitted temperature (ag_news / DBpedia) | 4.9 / 3.7 | 1.1 / 0.6 | 1.3 / 0.9 |
| MC latency, ag_news / DBpedia | 0.4 s / 0.63 s | 0.6 s / 0.95 s | 1.0 s / 1.65 s |

## Findings
- The explicit option meets the pass criterion for all three models: 0.99-1.00 recall at 5%
  false abstains, at a cost of 0.5-0.8 points of in-scope accuracy. It is far better than a
  threshold, which reaches only 0.33-0.72 at the same false-abstain rate. The result does not
  depend on the model, whereas the threshold's quality varies a lot between models.
- The older model is clearly overconfident (fitted temperatures 3.7-4.9), the Qwen3.5 models
  are near calibrated (0.6-1.3). This confirms the overconfidence of the early results was
  real and not an artefact of the tokenization bug found earlier (temperature 4.9 now against
  5.6 then). One temperature repairs it, but it has to be fitted per model.
- On the easy 4-class dataset all models are within a point of each other. On 14 classes the
  old model is 5-8 points behind, and for it generation beats multiple choice (0.940 vs 0.892),
  the only case so far where it does; the newer models tie or favour multiple choice.
- Speed scales with size: the 3B is about 1.5x faster than the 4B, which is about 1.7x faster
  than the 9B. The 4B is the best trade-off; the 9B is best on every quality metric by small,
  mostly within-noise margins.

## Not established
- The abstention result is one draw of hidden classes, and an easy one (the hidden classes are
  distinct from the visible ones), so the recall is probably an upper bound. The hard case, a
  hidden class next to a visible lookalike (for example Film beside Album and Written work),
  and other seeds are untested.
- Out-of-scope inputs are simulated by hiding classes, not genuinely foreign text.
- Smaller sizes of the newer family (0.8B, 2B) are untested.
- Answering several questions in one pass, non-enum outputs, and latency against a long-answer
  baseline remain untested, as recorded in the earlier tasks.
