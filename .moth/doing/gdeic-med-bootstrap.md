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


