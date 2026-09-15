"""BIO tagging of CUAD spans for the token classification baseline.

BIO tagging allows a single label per token, while about a quarter of CUAD spans overlap
a span of another category. When spans overlap, the shorter span takes precedence.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from contract_extractor.data.cuad import Span
from contract_extractor.schema import FIELD_KEYS

LABELS: tuple[str, ...] = ("O",) + tuple(f"{prefix}-{key}" for key in FIELD_KEYS for prefix in ("B", "I"))
LABEL_TO_ID: dict[str, int] = {label: i for i, label in enumerate(LABELS)}
IGNORE_INDEX = -100


def tokenize_windows(tokenizer, text: str, max_length: int, stride: int) -> list[dict[str, list]]:
    """Split a document into overlapping token windows with character offsets."""
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )
    return [
        {
            "input_ids": encoding["input_ids"][i],
            "attention_mask": encoding["attention_mask"][i],
            "offset_mapping": [tuple(o) for o in encoding["offset_mapping"][i]],
        }
        for i in range(len(encoding["input_ids"]))
    ]


def char_span_ids(text_length: int, spans: Sequence[Span]) -> np.ndarray:
    """Per-character index into ``spans`` (``-1`` outside any span), shorter spans on top."""
    ids = np.full(text_length, -1, dtype=np.int32)
    order = sorted(range(len(spans)), key=lambda i: spans[i].end - spans[i].start, reverse=True)
    for i in order:
        ids[spans[i].start : spans[i].end] = i
    return ids


def token_labels(offsets: Sequence[tuple[int, int]], spans: Sequence[Span], span_ids: np.ndarray) -> list[int]:
    """Label ids for one tokenized window; special tokens get ``IGNORE_INDEX``."""
    labels: list[int] = []
    previous = -1
    for start, end in offsets:
        if start == end:
            labels.append(IGNORE_INDEX)
            continue
        span_index = int(span_ids[start])
        if span_index < 0:
            labels.append(LABEL_TO_ID["O"])
        else:
            prefix = "I" if span_index == previous else "B"
            labels.append(LABEL_TO_ID[f"{prefix}-{spans[span_index].key}"])
        previous = span_index
    return labels


def decode_spans(offsets: Sequence[tuple[int, int]], label_ids: Sequence[int]) -> list[tuple[str, int, int]]:
    """Decode token labels into ``(key, char_start, char_end)`` spans.

    An ``I-`` tag following a different label opens a new span, which is the usual
    lenient BIO decoding.
    """
    spans: list[tuple[str, int, int]] = []
    current: list | None = None
    for (start, end), label_id in zip(offsets, label_ids, strict=True):
        label = LABELS[label_id]
        if label == "O":
            if current is not None:
                spans.append(tuple(current))
                current = None
            continue
        prefix, key = label.split("-", 1)
        if current is not None and current[0] == key and prefix == "I":
            current[2] = end
            continue
        if current is not None:
            spans.append(tuple(current))
        current = [key, start, end]
    if current is not None:
        spans.append(tuple(current))
    return spans
