"""Decoding guards for long verbatim generations.

``GuardedLogitsProcessor`` applies the JSON Schema grammar (optional) and ends a sequence
with EOS as soon as it enters a degenerate cycle, such as a token or short phrase repeated
until the token budget runs out. Rows of the batch are handled independently.

``close_truncated_json`` turns a generation cut by the guard or by the token budget into
valid JSON. Under grammar-constrained decoding every output is a prefix of a valid object,
so closing the open structure after the last complete list item recovers all finished spans.
"""

from __future__ import annotations

import json
import math

import torch
from transformers import LogitsProcessor


def is_looping(tokens: list[int], min_tokens: int, max_period: int) -> bool:
    """True when the tail of ``tokens`` is a cycle of period ``<= max_period`` spanning ``min_tokens``."""
    for period in range(1, max_period + 1):
        span = period * max(3, math.ceil(min_tokens / period))
        if len(tokens) < span:
            break
        tail = tokens[-span:]
        if all(tail[k] == tail[k - period] for k in range(period, span)):
            return True
    return False


class GuardedLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        compiled_grammar,
        eos_token_id: int,
        prompt_length: int,
        loop_min_tokens: int = 48,
        loop_max_period: int = 64,
        check_every: int = 8,
    ):
        self.compiled_grammar = compiled_grammar
        self.eos_token_id = eos_token_id
        self.prompt_length = prompt_length
        self.loop_min_tokens = loop_min_tokens
        self.loop_max_period = loop_max_period
        self.check_every = check_every
        self.matchers: list | None = None
        self.bitmask = None
        self.stopped: list[bool] = []
        self.first_call = True

    def _init(self, batch_size: int) -> None:
        self.stopped = [False] * batch_size
        if self.compiled_grammar is not None:
            import xgrammar as xgr

            self.matchers = [xgr.GrammarMatcher(self.compiled_grammar) for _ in range(batch_size)]
            self.bitmask = xgr.allocate_token_bitmask(batch_size, self.compiled_grammar.tokenizer_info.vocab_size)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]
        if self.first_call:
            self._init(batch_size)

        if self.matchers is not None and not self.first_call:
            last_tokens = input_ids[:, -1].tolist()
            for i, matcher in enumerate(self.matchers):
                if not self.stopped[i] and not matcher.is_terminated() and not matcher.accept_token(last_tokens[i]):
                    self.stopped[i] = True
        self.first_call = False

        generated = input_ids.shape[1] - self.prompt_length
        if generated >= self.loop_min_tokens and generated % self.check_every == 0:
            window = 3 * self.loop_max_period + self.loop_min_tokens
            tails = input_ids[:, -window:].tolist()
            for i, tail in enumerate(tails):
                if not self.stopped[i] and is_looping(tail, self.loop_min_tokens, self.loop_max_period):
                    self.stopped[i] = True

        if self.matchers is not None:
            import xgrammar as xgr

            for i, matcher in enumerate(self.matchers):
                if not self.stopped[i] and not matcher.is_terminated():
                    matcher.fill_next_token_bitmask(self.bitmask, i)
            xgr.apply_token_bitmask_inplace(scores, self.bitmask.to(scores.device))

        for i, stopped in enumerate(self.stopped):
            if stopped:
                scores[i, :] = -math.inf
                scores[i, self.eos_token_id] = 0.0
        return scores


def close_truncated_json(raw: str) -> str | None:
    """Close a truncated JSON object after its last complete list item, or return None."""
    stack: list[str] = []
    in_string = escaped = False
    best: tuple[int, str] | None = None

    def closers() -> str:
        return "".join("]" if c == "[" else "}" for c in reversed(stack))

    for i, char in enumerate(raw):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
                if stack and stack[-1] == "[":
                    best = (i + 1, closers())
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
            if char == "{" and len(stack) == 1:
                best = (i + 1, "}")
        elif char in "}]":
            if not stack:
                return None
            stack.pop()
            if not stack:
                return raw[: i + 1]
            best = (i + 1, closers())

    if best is None:
        return None
    candidate = raw[: best[0]] + best[1]
    try:
        json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return candidate
