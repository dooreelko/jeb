"""Shared model wrapper: loading, chat template, single forward pass -> last-position logits."""
import numpy as np
from llama_cpp import Llama

CLASSES = ["World news", "Sports", "Business", "Science and technology"]  # ag_news label order


class Model:
    def __init__(self, path, n_ctx=2048, n_threads=16):
        self.llm = Llama(model_path=path, n_ctx=n_ctx, verbose=False, n_threads=n_threads)
        self.vocab = self.llm.n_vocab()

    def tok(self, s):
        """Id of the first token of `s` (no BOS)."""
        return self.llm.tokenize(s.encode(), add_bos=False)[0]

    @staticmethod
    def chat(user, system=None):
        """Qwen (ChatML) template, ending at the assistant turn so the next token is the answer."""
        sys_part = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
        return sys_part + f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"

    def last_logits(self, text):
        """One forward pass; logits over the whole vocab at the final position."""
        self.llm.reset()
        self.llm.eval(self.llm.tokenize(text.encode(), add_bos=False))
        # llm.scores stays zero unless logits_all=True, so read from the context directly
        return np.array(self.llm._ctx.get_logits()[: self.vocab])
