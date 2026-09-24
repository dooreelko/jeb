# Lessons learned (after experiments 01-17)

A cross-cutting reading of the experiments in this folder. Numbers in brackets are experiment files.
Where the evidence is thin (few episodes, one draw), it says so.

## Summary: what works so far

The hypothesis holds for the mechanism: reading the logits gives jev-like, machine-readable,
normalised output from a vanilla model on plain llama.cpp, with no generation. For classification
and routing it behaves like jev; for decisions it only partly does. The levers, in order of evidence:

1. **A bigger model, for decisions.** The largest single lever: 0 → 5.8 kills in doom with the
   same prompt [08]. Not for classification, where a 4B already matches a 27B [01, 03].
2. **One question per candidate, highest "yes" wins.** Independent per-action yes/no judgments
   beat one joint multiple-choice pass (0.8B above controls, 27B 5.8 → 7.6) [10]. This is the
   decomposition openjev's scoring does. Not yet tested: the fuller quorum (several narrower
   sub-questions combined by a vote), and whether openjev's trained head is what closes the rest
   of the gap (7.6 vs ~11).
3. **A scene that shares no vocabulary with the answers, in a convention the model knows.**
   Clock positions instead of "left of the crosshair" removed word matching; the 27B read them
   correctly (its best reading, cleanest play) [14, 16]. Style and density do not matter: a terse
   radio call equals prose at 27B and hurts 0.8B [17]. Needs the big model; the 0.8B only word-matches.
4. **Pre-computed named buckets** are the most reliable lever in practice (Flappy 28/28 [06], and
   jev's own advice), legitimate when the bucket is a perception, not the decision itself.

Also essential:

- **Speed and cost are not reproduced.** One pass is not faster than a short generated answer, and
  per-candidate questions cost N passes; shared-prefix batching (moth xaj5x) is the prerequisite [01, 03].
- **Readout hygiene:** check the probability mass at the answer position; avoid lexical collisions
  in the question ("the right move" vs "turn right"); change one prompt factor at a time, since
  persona, framing and readout interact [02, 11, 12].
- **For classification:** an explicit "none of the above" option beats confidence thresholds for
  abstention; calibration must be fitted per model [04].
- **Measurement:** controls (random, always-act), balanced metrics, paired seeds; most doom numbers
  are 5 episodes, so differences under ~1.5 kills are noise [05, 10].

## The short answer to jeb's question

How close do plain llama.cpp and a vanilla model get to jev / openjev by reading logits instead of
generating?

- **Classification and routing: close.** Multiple choice (MC) over option letters gives the accuracy of
  generation, near-calibrated probabilities on modern models, confidence that separates right from wrong,
  and clean abstention through an explicit "none of the above" option [01, 03, 04]. That is the jev-like
  core, and a 4B model already does it.
- **Decisions and control: only partly.** The model executes a comparison the prompt states (Flappy 28/28
  [06]), but it does not infer the move from a goal or a game description [06]. Without the pipeline
  pre-digesting the decision, a 0.8B fails [08, 12-14, 17], and a 27B reaches 5.8-7.6 kills in doom
  against openjev's ~11 [08, 10, 15, 16]. openjev has a trained classification head; we use a vanilla
  model. That training is the most likely remaining gap, untested here.

## 1. Classification saturates early; decisions need scale

- For classification, size barely mattered: 4B ≈ 9B ≈ 27B on ag_news and DBpedia, with the remaining errors
  sitting in label overlaps no model removes [01, 03].
- For decisions, size is the biggest single lever: the same doom prompt and readout gives 0 kills at 0.8B
  (worse than random) and 5.8 at 27B [08]. Every later improvement at 0.8B was small by comparison [10].

## 2. Small models match words; big models read, within limits

- The 0.8B's only real signal in doom is surface word matching: "left of the crosshair" → "turn left" [12].
  Remove the shared word (symbols [13], clock positions [14], radio call [17]) and it drops to chance
  or turns the wrong way.
- The 27B reads a perceptual scene in a convention it already knows (clock positions, the right direction,
  its best readout [14]), but inverts direction when the convention must be learned from the prompt
  (symbols [13]). So even its correct English turning is partly word matching.
- Consequence for "decision-free" prompts: they only work with a model big enough to do the mapping itself.

## 3. What goes into the prompt decides more than the readout

- If the prompt states the comparison the decision hinges on, the game saturates (Flappy 28 with named
  buckets [06]); if it gives raw quantities the model has to combine, it fails (heights alone at chance
  [05]). jev's own advice (named buckets, only the needed fields) is in effect pre-computing the decision.
- A workable line between fair and unfair input: perception-level facts (bearing, apparent size, health)
  are fair; naming the decision or the action's vocabulary is not. The bias that matters in practice is
  shared vocabulary between scene and actions [13, 14]. Fully allocentric input (coordinates + heading)
  needs trigonometry a one-pass readout cannot do, and is not more "unbiased", only less processed.
- Wording mostly moves the decision boundary, not the ranking: a goal sentence or a trailing period shifted
  Flappy toward "flap" [05]. Small models react to form ("No contact." → more attacks [17]); the 27B was
  indifferent to prose vs radio-call style [17]. For a small model any wording change is a new experiment.

## 4. The readout: decompose, and watch for collisions

- The joint MC pass (all actions as A/B/C in one prompt) has a strong first-option bias at small scale
  [08]. Asking one narrow yes/no question per action and taking the highest p(yes) lifted the 0.8B above
  both controls and stacked with size at 27B (7.6, the best doom result) [10]. That is what openjev's
  scoring effectively does (one judgment per action), and a first confirmation of the decompose-and-vote idea.
- Readout details interact; changing several at once hides the cause [11 vs 12]. A factorial on fixed
  states found it.
- Lexical collisions in the question poison single actions: "is \"turn right\" the right move right now?"
  flattened the turn-right signal [12].
- Always check where the probability mass sits at the answer position before trusting a readout
  (Y/N > 99% [02]; yes/no ~99%, T/N ~80% [11, 13]).

## 5. Things that did not help

- A reasoning pass before the readout: the small model ignores its own plan; the big one does not need
  it [09]. (Thinking mode proper is still untested, moth bkv0v.)
- Calibration offsets per action [10, 15], a direct yes/no token readout [11], sampling instead of argmax
  [05], a bigger model for classification [01, 03], the NPU [07].

## 6. Measuring

- Plain agreement with an oracle misleads when one action dominates (never flapping agrees ~90%) [05]:
  use balanced accuracy / AUROC against the truth of the statement.
- Controls are not optional: without random and always-attack, the 0.8B's 0 kills looked like "weak",
  not "worse than random" [10].
- Fixed labelled states + AUROC measure a readout in seconds and separate signal from bias [12]; gameplay
  is slow and noisy. But separation on labelled states predicts play only loosely [05, 15]: play needs the
  right choice between actions in one state, on the states the policy itself visits.
- The label definition is part of the result: "exactly on the crosshair" (±1.5% of the width) vs "crosshair
  inside the box" changed what attack could mean [12, 16].
- Paired comparisons on the same seeds, and a pre-stated pass criterion [04], made small samples usable.
  Most doom play numbers are 5 episodes: differences under ~1.5 kills are noise.

## 7. Speed and engineering

- One pass is not faster than generation when the generated answer is a few tokens: prefill dominates both
  [01, 03]. Wins need long answers, shared-context batching or smaller models. Per-action decomposition
  multiplies passes, so shared-prefix batching (moth xaj5x) is the prerequisite for using it at 27B.
- Vanilla llama.cpp on the GPU is enough. The NPU works for isolated large matmuls but is a dead end next
  to a GPU in today's llama.cpp [07].
- Silent pitfalls cost real time: an empty logits buffer, asynchronous GPU timing, spelled-out control
  tokens [01], a background wait loop matching its own command line.

## Open threads

- Close the doom gap to openjev: finer aim within the clock convention [16]; narrower questions per
  judgment (is an enemy left? on target? close?) combined by a dumb vote, the fuller quorum idea.
- Can anything short of training make a small model decide without word matching? A trained head (like
  openjev's) is the obvious untested lever.
- Shared-prefix batching (xaj5x), thinking mode (bkv0v), real-time play (rxor3).
