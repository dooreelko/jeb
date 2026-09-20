# A naive attmept to implement TypeSafe's Jev

The idea is to use a plain LLM but instead of generating tokens, query probabilities of each option to be the next token.

E.g. given a question "What is the capital of Ukraine?" and options "Kyiv, Kharkiv, Salem and None of the above", construct the prompt and query, then softmax probabilities of each option.

Single pass, no recursive token generation, no thinking.

```
You are a decision maker. You can only answer with the letter of one option, "A", "B", "C" or "D".
Given a context of "What is the capital of Ukraine?" and possible options of 
A. Kyiv
B. Kharkiv
C. Salem 
D. None of the above

Which option is the most likely one?
```

Then take the logits of the tokens `A`, `B`, `C` and `D` at the next position and softmax over just those four. One pass, and the options compete for probability, so the result is a proper distribution.

