"""Shared model wrapper: loading, chat template, single forward pass -> last-position logits."""
from datetime import datetime

import numpy as np
from jinja2.sandbox import ImmutableSandboxedEnvironment
from llama_cpp import Llama

SYSTEM = "You are a decision maker. You can only answer {answer}."


def intro(article):
    """Framing shared by every scorer, so prompts differ only in what the method needs."""
    return f'Given a context of "{article}" and possible topics of\n'


def _raise(msg):
    raise ValueError(msg)


class Model:
    def __init__(self, path, n_ctx=2048, n_threads=16):
        # Always offload everything to the GPU; this also frees the CPU cores.
        self.llm = Llama(model_path=path, n_ctx=n_ctx, n_gpu_layers=-1, verbose=False, n_threads=n_threads)
        self.vocab = self.llm.n_vocab()
        self._template = self.llm.metadata["tokenizer.chat_template"]
        self._tok_str = lambda i: self.llm.detokenize([i]).decode(errors="replace")
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.globals["raise_exception"] = _raise
        env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
        self._jinja = env.from_string(self._template)

    def tok(self, s):
        """Id of the first token of `s` (no BOS)."""
        return self.llm.tokenize(s.encode(), add_bos=False, special=False)[0]

    def chat(self, user, system=None):
        """Render the GGUF's own chat template, ending at the assistant turn so the next token is
        the answer. Thinking is switched off (templates that don't know the flag ignore it)."""
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        return self._jinja.render(
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=False,
            bos_token=self._tok_str(self.llm.token_bos()),
            eos_token=self._tok_str(self.llm.token_eos()),
        )

    def _tokens(self, text):
        # special=True: control tokens like <|im_start|> must map to their ids, not be spelled out
        return self.llm.tokenize(text.encode(), add_bos=False, special=True)

    def last_logits(self, text):
        """One forward pass; logits over the whole vocab at the final position."""
        self.llm.reset()
        self.llm.eval(self._tokens(text))
        # llm.scores stays zero unless logits_all=True, so read from the context directly.
        # Reading also forces the (asynchronous) GPU work to finish, which timing relies on.
        return np.array(self.llm._ctx.get_logits()[: self.vocab])

    def generate(self, text, max_tokens=8):
        """Greedy continuation of a rendered prompt."""
        out = self.llm.create_completion(self._tokens(text), max_tokens=max_tokens, temperature=0)
        return out["choices"][0]["text"]
