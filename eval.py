"""Compare logit-readout scorers (MC vs README-style Y/N) on ag_news.

usage: uv run python eval.py [model.gguf] [n_examples]
"""
import sys, time, json
import numpy as np
from datasets import load_dataset
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from llama_cpp import Llama

MODEL = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
CLASSES = ["World news", "Sports", "Business", "Science and technology"]
LABELS = ["A", "B", "C", "D"]

llm = Llama(model_path=MODEL, n_ctx=2048, verbose=False, n_threads=16)
V = llm.n_vocab()
tok = lambda s: llm.tokenize(s.encode(), add_bos=False)[0]
label_ids = [tok(l) for l in LABELS]
Y, Nn = tok("Y"), tok("N")


def chat(system, user):
    s = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
    return s + f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def last_logits(text):
    llm.reset()
    llm.eval(llm.tokenize(text.encode(), add_bos=False))
    return np.array(llm._ctx.get_logits()[:V])


def score_mc(article):
    opts = "\n".join(f"{l}. {c}" for l, c in zip(LABELS, CLASSES))
    lg = last_logits(chat(None, f"Article: {article}\n\nWhich topic is this article about?\n{opts}\n\nAnswer with a single letter."))
    return lg[label_ids]


def score_yn(article):
    """One Y/N pass per option. Returns (gap, log p(Y)) per option."""
    opts = "\n".join(f"- {c}" for c in CLASSES)
    sysm = 'You are a decision maker. You can only answer "Y" or "N".'
    gaps, logpy = [], []
    for c in CLASSES:
        lg = last_logits(chat(sysm, f'Given a context of "{article}" and possible topics of\n{opts}\n\nIs "{c}" the most likely topic?'))
        yn = lg[[Y, Nn]]
        gaps.append(yn[0] - yn[1])
        logpy.append(log_softmax(yn)[0])
    return np.array(gaps), np.array(logpy)


def gen_baseline(article):
    opts = ", ".join(CLASSES)
    out = llm.create_chat_completion(
        messages=[{"role": "user", "content": f"Article: {article}\n\nClassify the topic as one of: {opts}. Reply with only the topic name."}],
        max_tokens=8, temperature=0)
    txt = out["choices"][0]["message"]["content"].strip().lower()
    for i, c in enumerate(CLASSES):
        if txt.startswith(c.lower()):
            return i
    return -1


# ---- metrics ----
def ece(p, y, bins=10):
    conf, pred = p.max(1), p.argmax(1)
    acc = (pred == y).astype(float)
    e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e


def nll(s, y, T):
    return -log_softmax(s / T, axis=1)[np.arange(len(y)), y].mean()


def report(name, s, y):
    h = len(y) // 2  # fit temperature on first half, evaluate on second
    T = minimize_scalar(lambda t: nll(s[:h], y[:h], t), bounds=(0.05, 100), method="bounded").x
    ev, yv = s[h:], y[h:]
    raw, cal = softmax(ev, axis=1), softmax(ev / T, axis=1)
    print(f"{name:8s} acc={np.mean(raw.argmax(1) == yv):.3f}  "
          f"raw: nll={nll(ev, yv, 1):.3f} ece={ece(raw, yv):.3f}   "
          f"T={T:5.2f} cal: nll={nll(ev, yv, T):.3f} ece={ece(cal, yv):.3f}")


if __name__ == "__main__":
    ds = load_dataset("fancyzhx/ag_news", split="test").shuffle(seed=0).select(range(N))
    # ag_news: 0 World, 1 Sports, 2 Business, 3 Sci/Tech -> matches CLASSES order
    arts = [t[:600] for t in ds["text"]]
    y = np.array(ds["label"])
    S = {"mc": [], "yn_gap": [], "yn_p": []}
    t_mc = t_yn = t_gen = 0.0
    gen_pred = []
    for i, a in enumerate(arts):
        t = time.perf_counter(); S["mc"].append(score_mc(a)); t_mc += time.perf_counter() - t
        t = time.perf_counter(); g, lp = score_yn(a); t_yn += time.perf_counter() - t
        S["yn_gap"].append(g); S["yn_p"].append(lp)
        t = time.perf_counter(); gen_pred.append(gen_baseline(a)); t_gen += time.perf_counter() - t
        if i % 20 == 0:
            print(f"  {i}/{N}", flush=True)
    S = {k: np.array(v) for k, v in S.items()}
    np.savez(f"scores_{MODEL.split('/')[-1]}.npz", y=y, gen=np.array(gen_pred), **S)
    print(f"\nmodel={MODEL} n={N}  (eval on 2nd half; T fitted on 1st half)")
    for k, v in S.items():
        report(k, v, y)
    print(f"gen      acc={np.mean(np.array(gen_pred)[N // 2:] == y[N // 2:]):.3f}")
    print(f"latency/question: mc={t_mc / N * 1000:.0f}ms  yn(4 passes)={t_yn / N * 1000:.0f}ms  gen={t_gen / N * 1000:.0f}ms")
