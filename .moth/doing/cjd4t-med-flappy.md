look, somebody's doing it too but differently https://huggingface.co/AlexWortega/openjev

we should take their flappy bird (https://huggingface.co/AlexWortega/openjev/blob/main/code/flappy.py) and test it against our implementation



----- AI agent updates -------

## Decisions
- Compare against openjev using turn-based Flappy Bird: the game waits for each decision, so latency cannot affect the score. Physics kept identical to theirs (user: "let's keep it close"); 900 steps caps at 28 pipes. Real-time play is out of scope here (follow-up moth rxor3), as are the standard MC benchmarks (tuxml), GPQA (gy63u) and chess/GSM8K (t319t).
- Our interface is the same MC readout as before: state described as text context, two options, softmax over the letters, argmax picks the move. Option 0 always means flap. Watchable as ASCII in the terminal.
- Two option wordings: actions ("flap" / "do nothing") and position ("The bird is below/above the centre of the gap"). openjev found the same split: position works, actions fail. Ours: actions ~chance (0.52), position works.
- Metric: plain agreement with an oracle is misleading (the oracle flaps ~10% of ticks, so never flapping scores ~0.90). Use balanced accuracy and AUROC against the truth of the statement, on states balanced between flap and no-flap, plus game score.
- State rendering is numbers (default, the fair comparison). A words-only rendering with the comparison pre-made is kept only as an ablation and must not go in the "vs openjev" table. A combined numbers+hint rendering was dropped: its mirrored wording ("gap centre is above the bird" vs "bird is above the centre") collapsed the model to chance.

## Results (Qwen3.5, position wording, 6 episodes, 900 steps)
- Reading the state: 4B numbers balanced accuracy 0.96, AUROC 0.94. Words 0.86 / 0.80. Number reading is not the bottleneck (an earlier "weak number reading" claim was wrong: it was measured against a lookahead oracle whose own ceiling is 0.92).
- Game score: 4B mean 8.33 (max 18); 9B mean 5.83 (max 9); words ablation 0 (all die at step 63). Reference: perfect rule 28, lookahead oracle ~25, openjev ~27.5-28. Never-flap 0, random ~0.17.
- Error analysis: every death is the bird too high. Killing errors are spurious flaps while far above the gap centre (4B 5-9% of such ticks, 9B 36% at >=0.2 above); missed flaps near or below the centre are harmless. Per-tick error of a few percent compounds over hundreds of decisions. The bigger model did not help.

## Open / not decided
- Whether to add a fitted probability threshold (flap only above ~0.7) as a labelled calibration row next to the raw result. Not done.
- Other levers not tried: action repeat, few-shot, larger model on the position task beyond 9B.
- openjev's training mix (v1) was not inspected, so why theirs plays ~28 is unexplained. TypeSafe Jev/Terra numbers exist only as a radar chart, so comparisons to them are approximate.

## Correction: prompt parity with openjev (added after review)
- The results above used a shortened copy of openjev's state text and hypotheses. That was an unexamined deviation, not a decision. The state text and the `position` hypotheses are now openjev's own, verbatim (their results JSON shows their 27.5 came from `prompt=base` with `hyp=position`). Their run was played at 15 fps real time; ours is turn-based. The earlier table (4B 8.33, 9B 5.83) is the shortened-text ablation, not the comparison with openjev.
- Identical prompt, raw argmax (4B, 6 episodes): score 0 in all six (bird flies into the ceiling at step 63). With a threshold fitted on separate states (p(flap) >= 0.57, a labelled calibrated row): mean 1.67, max 4. Both far below openjev.

## Finding: the prompt wording moves the result a lot
Measured on 100 balanced states (below/above the gap centre), 4B, position wording. Separation is AUROC, decision quality at the raw 0.5 threshold is balanced accuracy.
- Full openjev text: AUROC 0.94, balanced accuracy 0.76 (says "flap" for 42% of above-centre states).
- Drop the goal sentence ("must fly through the gap..."): 0.96 / 0.89. Drop goal and the "currently above/below the gap" sentence: 0.97 / 0.90. Also drop the final period on the options: 0.94 / 0.91.
- Drop only the position sentence: 0.89 / 0.59. Drop only the signed offset from the gap centre: 0.70 / 0.58.
- Drop position, offset and goal (only raw heights and the gap span left): 0.46 / 0.50, i.e. chance. The model cannot subtract heights itself in one pass.
- Patterns: (1) a signed offset number is the essential signal, and "the bird is currently above/below the gap" partly duplicates it. (2) Extra task framing (the goal sentence) does not help and biases the model toward "flap": the ordering stays good but the 0.5 boundary shifts, so raw argmax fails while a fitted threshold recovers it. (3) A trailing period on the options changes the boundary too. So the text mostly moves calibration, not ranking, except when the offset is removed.
- Even with the best variants and a threshold, game score stays low because a few percent of spurious flaps while high are fatal (all deaths are the bird too high).

## Open / not decided
- Play the game with the best variant (goal dropped) as its own labelled row; report it separately from the verbatim-prompt row.
- Other levers not tried: action repeat, few-shot, models beyond 9B.
- TypeSafe Jev/Terra numbers exist only as a radar chart, so comparisons to them are approximate.


## Result: goal sentence dropped (added after the run)
- 4B, position, 6 episodes: raw argmax mean 2.33 (max 5); threshold fitted on separate states (0.495, so barely moved) mean 1.67 (max 2). Better than the verbatim prompt (raw 0, calibrated 1.67), still far below the perfect rule (28) and openjev (27.5).
- So the balanced-accuracy gain from removing the goal sentence (0.76 to 0.89) turns into a couple of pipes at most. A per-decision error of about 5% still kills the game: every death is the bird just above the gap top, having flapped when it should not.
- Conclusion for now: prompt wording explains most of the swing between our variants (0 to about 8 pipes) but none of them get near openjev's, so the remaining gap is not a prompt-tuning problem alone.


## Write-up and reproducibility note
- Summary of findings lives in flappy.md (linked from the README); this moth stays the decision record.
- The 8.33 row was measured with an earlier, shortened state text (no position sentence, no goal sentence, no "with vertical velocity", options without the final period). The current code uses openjev's verbatim text, and that row is only approximately reproducible with the ablation flags; it has not been rerun in exactly that form. Treat it as a lead.
