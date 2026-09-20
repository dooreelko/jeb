the @IDEA.md has the rough concept, here's the initial discussion

So there's Jev, a new kid that is not an LLM. (Have a look if you need).
Assessing feasibility of recreating a model via logit probing.

Not unlikely at all. What you describe is a well-known technique: read the next-token distribution instead of sampling. It's the standard way to get classifier-style probabilities out of an LLM, and it's the baseline Jev has to beat.

What works out of the box

    You don't need to modify the model. One forward pass gives logits over the whole vocab, and you just read the ones you care about.
    Multiple choice is cleaner than yes/no per option. Label the options A/B/C and softmax over just those tokens, which gives a properly normalized distribution. Independent yes/no probabilities won't sum to 1.
    Prefill is cheap. You can cache the shared prompt once and run each question as a short suffix.

Where it gets harder

    Calibration: This is the real gap. Instruction-tuned models are often overconfident and their token probabilities don't match actual accuracy. Base models are better calibrated. You can patch this with temperature or Platt scaling on a held-out set. Fine-tuning with a proper scoring rule (log loss, Brier) fixes it more thoroughly. That is presumably what their RLCD training is aiming at, though I only know what's been publicly reported.
    Bias: Option order and label tokens skew the results, and small prompt changes shift the probabilities.
    Non-enum outputs: Booleans and enums are easy. Numbers and ranges need bucketing or digit-by-digit scoring, and free text is out of scope.
    Speed: Their claimed 100-400x gains probably come from a small specialized model, a serving stack built for it, and answering all questions in a single parallel pass. You can approximate that with attention masking so the question suffixes don't see each other, but it takes some custom work.

I'd expect a fine-tuned 1-8B model with logit readout and a calibration layer to get you most of the way for classification and routing. The open question is how close that gets to Jev's calibration and latency.

My guess is that performance improvement comes from the fact that they only need one prediction pass for a single token, instead of generating many tokens, ne?
Weighing single-pass speedups against overall performance gains.

Partly, yes. Skipping autoregressive decoding is probably the biggest single factor, but it likely doesn't explain the whole 100-400x.

Why one pass helps a lot

    Decoding is memory-bandwidth-bound: every generated token re-reads the model weights. Prefill (processing the input) is compute-bound and batches efficiently, so per-token it's far cheaper. Moving almost all the work into prefill is a big win, and it fits their pricing of cheap input and free output.
    Output length matters too. A typical LLM classification call might emit 20-500 tokens, or thousands if it reasons first. Jev emits effectively none, so latency is roughly one prefill.

Why I doubt that's all of it

    The headline comparisons are against frontier LLMs. Those are probably much larger than whatever Jev is, so part of the gap is likely model size, not method. A small LLM with logit readout would close a good chunk of it.
    Parallel answers matter when there are several questions. One pass for all of them beats N sequential generations, or N separate calls.
    Serving stack: if everything is fixed-shape prefill, batching and scheduling get much simpler and more efficient than with variable-length generation.

If I had to guess the split, it would be roughly: no decoding for the first 10-50x, then model size and serving for the rest. That's my inference from the public claims, not something TypeSafe has broken down. A simple way to test it would be to compare your logit-readout approach on a small model against the same approach on a large one, and see how much of the gap is size versus method.



----- AI agent updates -------

## Goal
A throwaway proof of concept that answers one question: how close does a plain, small
instruction-tuned LLM get to Jev's claims when it is used as a classifier by reading the
next-token distribution instead of generating text? Success is measured on three axes,
against ordinary greedy generation as the baseline: accuracy, calibration (does stated
confidence match actual accuracy), and latency. A second question, whether the speed gap is
due to the method or to model size, is answered by repeating the run across model sizes.

## Decisions taken
- **Unmodified llama.cpp, used through its Python binding.** No fork. A fork is only justified
  later if the output head has to be stripped down to the option tokens, or if this is ever
  exposed as a server endpoint.
- **GPU always on; no CPU/GPU switch.** The machine has an integrated AMD GPU sharing system
  RAM. The ROCm toolchain and libraries come from the project's nix shell, and the GPU is
  presented as the neighbouring architecture for which the vendor kernels are shipped. Why
  keep it on although the measured prefill gain on a 3B model was only ~20% over the CPU
  cores: it frees the CPUs for other work. Decode-bound speedups are not expected on shared
  memory; the workload is prefill-dominated.
- **Compare two scorers instead of choosing one up front.**
  - Multiple choice (MC): one forward pass, options labelled with letters, softmax over the
    label tokens' logits.
  - Per-option Y/N (the IDEA.md approach): one pass per option asking whether it is the most
    likely answer, with the full option list in the context, softmax across options.
- **Prompts of the two scorers share their framing.** Same system message style and same
  article/options introduction; they differ only where the method requires it (letters vs
  bullets, "which is most likely" vs "is X the most likely"). This removes the earlier
  prompt-wording confound.
- **Prompts are rendered with the model's own chat template** (taken from the model file, so
  any model family works), with control tokens parsed as real special tokens, and with
  reasoning mode switched off by prefilling the empty reasoning block, so the very next
  token is the answer. An early version spelled the control tokens out as plain text and
  hard-coded one template; results from it are superseded (see Results).
- **Y/N is scored four ways from the same passes**, to test whether the "N" reading is needed
  (tracked as its own task): logit gap between Y and N; log-probability of Y over just
  {Y, N}; raw logit of Y alone; log-probability of Y over the whole vocabulary. The
  probability mass sitting on {Y, N} is recorded as a sanity check on the token choice.
- **Read logits, never generate.** One forward pass, take the vocabulary logits at the last
  position, index the answer tokens. Pitfall: llama.cpp's per-token score buffer is empty
  unless full logits are requested and silently yields a uniform distribution; logits must
  be taken from the context directly.
- **Timing must force completion.** GPU work is asynchronous, so a pass is only timed
  correctly when its logits are read back inside the timed region; a naive timing produced
  impossible throughput. A warm-up pass precedes measurement so kernel compilation does not
  count as latency.
- **Evaluation data: ag_news** (4 balanced topics, real text, classes map one-to-one to the
  options). Fixed shuffle seed, articles truncated so prompts stay short.
- **Calibration is measured, and repaired only with a single temperature.** Temperature is
  fitted on one half of the examples and everything is reported on the other half: accuracy,
  log loss and expected calibration error, both raw and after scaling.
- **Baseline = greedy generation** of a short answer parsed back to a class, giving accuracy
  and latency only (it has no probabilities).
- **Models: the newest Qwen that fits in RAM, at 4-bit, as a size ladder on one
  architecture family**: Qwen3.5 at 4B and 9B, then the dense Qwen3.8 27B. Base variants are
  wanted for a base-vs-instruct calibration comparison.
- **Parallel answers are approximated with sequence batching, not custom masking.** The shared
  prompt is placed in every sequence and each option's suffix in its own, so all options for
  a question are processed in one decode. This is the "questions don't see each other" trick
  with no kernel work. Planned, not yet done.
- **Every run tees its raw output to a timestamped log file**, so progress is visible while
  it runs and results survive the console.

## Decisions rejected / deferred
- Modifying llama.cpp up front: unnecessary for the first answers.
- banking77: needs hand-grouping into coarse classes, so it was set aside for ag_news.
- Platt scaling, per-class calibration and fine-tuning with a proper scoring rule: out of
  scope for the PoC; a single temperature is enough to show the direction.
- Qwen2.5 as the main model: a generation old. Kept only as an early smoke test.
- Qwen3.8 "Flash-Next" (~180B, new architecture) and the multi-trillion-parameter Qwen3.8:
  neither fits in RAM, and the former is not supported by the bundled llama.cpp.
- A CPU/GPU toggle: not needed, see above.

## Out of scope
Numeric and range outputs, free-text outputs, serving stack (batching/scheduling), fine-tuning,
and reproducing Jev's absolute numbers.

## Results
Metrics are on the held-out 100 of 200 examples (temperature fitted on the other 100), so
differences of a few points are within noise.

**Superseded:** the first run (Qwen2.5-3B, MC 0.90 vs Y/N 0.77, temperatures of 3.5-6.6) used
prompts whose control tokens were mis-tokenized and whose two scorers were worded differently.
Do not rely on it, including its very large fitted temperatures. It has not been rerun under
the corrected setup.

| model | scorer | accuracy | log loss raw | ECE raw -> calibrated | fitted T |
|---|---|---|---|---|---|
| Qwen3.5 4B | MC | 0.89 | 0.34 | 0.067 -> 0.054 | 1.10 |
| | Y/N gap | 0.82 | 0.53 | 0.098 -> 0.070 | 1.78 |
| | Y/N log p(Y) over {Y,N} | 0.82 | 0.48 | 0.049 -> 0.037 | 0.93 |
| | Y/N Y-logit only | 0.84 | 0.44 | 0.081 -> 0.090 | 0.96 |
| | greedy generation | 0.84 | n/a | n/a | n/a |
| Qwen3.5 9B | MC | 0.88 | 0.29 | 0.054 -> 0.052 | 1.29 |
| | Y/N gap | 0.86 | 0.40 | 0.069 -> 0.061 | 1.37 |
| | Y/N log p(Y) over {Y,N} | 0.86 | 0.70 | 0.325 -> 0.123 | 0.29 |
| | Y/N Y-logit only | 0.82 | 0.54 | 0.150 -> 0.049 | 0.57 |
| | greedy generation | 0.86 | n/a | n/a | n/a |

Latency per question (GPU): 4B: MC 0.6 s, Y/N 2.1 s (four passes), generation 0.6 s.
9B: MC 1.0 s, Y/N 3.6 s, generation 1.1 s. Qwen3.8 27B: run in progress.

## Findings so far (tentative)
- MC is at least as accurate as Y/N in every run and better calibrated raw, though the gap
  narrows with size (about 7 points at 4B, 2 at 9B) and is within noise at 100 examples.
- Modern instruction models are far closer to calibrated than the early result suggested
  (fitted temperatures near 1); how much of that is the model versus the corrected prompt
  format is not separated.
- Calibration does not transfer across model sizes: the same Y/N score is overconfident at
  one size and underconfident at another, so each model needs its own fit.
- The "N" reading appears unnecessary for ranking: Y alone ranks about as well as the gap
  on both sizes and the full-vocabulary and {Y,N} versions coincide because nearly all mass
  (>0.99) sits on Y/N. Tentative; decided in its own task.
- One-pass MC costs about the same as greedy generation, because the generated answer is
  only a few tokens and prefill dominates either way. Y/N costs about 3.5x more because it
  re-processes the same context once per option. A latency win over generation is only
  expected once generation would be long or the passes share their context.
- Y/N verdicts are independent, so the answer can be inconsistent: at 4B, 7.5% of articles
  had two options or none saying yes, with much lower accuracy there (0.69 and 0.00 vs 0.85
  when exactly one said yes). MC's softmax forces options to compete, which avoids this.
- The integrated GPU is only slightly faster than the CPU cores for prefill, so speed will
  come from avoiding redundant prefill, not from the device.

## Open
- Finish the 27B run; add a base-model variant; enlarge the example count to tighten the
  error bars.
- Implement the sequence-batching version of the Y/N scorer and measure its latency against
  the sequential one and against generation.
- Rerun the Qwen2.5 3B under the corrected setup only if the size-vs-method comparison needs
  it.
- Decide whether a fork is warranted once latency is known.
