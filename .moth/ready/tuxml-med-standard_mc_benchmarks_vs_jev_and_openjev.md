Compare our prompted multiple-choice readout against TypeSafe's Jev and Terra and against openjev (a fine-tuned NLI cross-encoder on Qwen3.5) on the standard multiple-choice benchmarks they report.

## Why
Flappy Bird is one task among many. TypeSafe's published radar chart (values read off the image, approximate) and the openjev report give a common scoreboard. Terra is reported as "letter only", which is our approach. openjev is far below Jev on the same tasks, so the useful question is where a plain prompted model with no training lands relative to both.

## Decision
- Suite: MMLU, ARC-Easy, ARC-Challenge, HellaSwag and WinoGrande. All are open datasets.
- Same backbone family as openjev (Qwen3.5 0.8B, 2B, 4B) so that method is separated from model size; larger sizes optional.
- A fixed-seed sample of a few hundred questions per benchmark is enough for the first pass, given the size of the expected gaps.
- Report what they do not: calibration and latency, next to accuracy.
- Reference numbers come from their tables and chart, marked as approximate where read off an image.
- Needs a general "question plus options" interface (each example has its own options), which is also what the "Scale Up" task asked for.

## Out of scope
GPQA, Chess and GSM8K k-choice (separate tasks), fine-tuning, and real-time behaviour.

## Prediction to check
Beating openjev by a wide margin is expected, since our option format is the model's own; landing near Jev/Terra is the open question.
