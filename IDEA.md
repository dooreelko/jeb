# A naive attmept to implement TypeSafe's Jev

The idea is to use a plain LLM but instead of generating tokens, query probabilities of each option to be the next token.

E.g. given a question "What is the capital of Ukraine?" and options "Kyiv, Kharkiv, Salem and None of the above", construct the prompt and for each option ask the question 

```
You are a decision maker. You can only answer "Y" or "N".
Given a context of "What is the capital of Ukraine?" and possible options of 
- Kyiv
- Kharkiv
- Salem 
- None of the above

Is "Kyiv" the most likely answer?
```

Then see what's the probability of "Y" and "N", collect for each option and softmax the results.

