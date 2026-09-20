"""Step 0: logit readout via one forward pass, softmax over option-label tokens."""
import sys, time
import numpy as np
from llama_cpp import Llama

MODEL = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
llm = Llama(model_path=MODEL, n_ctx=2048, logits_all=False, verbose=False, n_threads=16)

LABELS = ["A", "B", "C"]
label_ids = [llm.tokenize(l.encode(), add_bos=False)[0] for l in LABELS]


def prompt(text, options):
    opts = "\n".join(f"{l}. {o}" for l, o in zip(LABELS, options))
    user = f"{text}\n\n{opts}\n\nAnswer with a single letter."
    return (
        "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n"
    )


def readout(text, options):
    llm.reset()
    llm.eval(llm.tokenize(prompt(text, options).encode(), add_bos=False))
    logits = np.array(llm._ctx.get_logits()[: llm.n_vocab()])[label_ids]
    p = np.exp(logits - logits.max())
    return p / p.sum()


QS = [
    ("My card was charged twice for one order.", ["billing", "shipping", "technical bug"]),
    ("The package has been stuck in transit for two weeks.", ["billing", "shipping", "technical bug"]),
    ("The app crashes whenever I open settings.", ["billing", "shipping", "technical bug"]),
]
for text, opts in QS:
    t = time.perf_counter()
    p = readout(text, opts)
    dt = time.perf_counter() - t
    print(f"{dt*1000:6.0f} ms  p={np.round(p, 3)}  -> {opts[int(p.argmax())]}   | {text}")
