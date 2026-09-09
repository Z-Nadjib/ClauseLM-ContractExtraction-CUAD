"""Field-level span metrics.

A predicted span counts as a true positive when its word-level Jaccard similarity with a
still unmatched gold span of the same field is at least ``threshold`` (0.5 by default, as
in the CUAD paper). Matching is one-to-one and greedy on similarity.
"""

from __future__ import annotations

import re
import string
from collections.abc import Iterable
from dataclasses import dataclass

from contract_extractor.schema import FIELD_KEYS, Extraction

_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})
_WS = re.compile(r"\s+")


def normalize_tokens(text: str) -> frozenset[str]:
    return frozenset(_WS.split(text.lower().translate(_PUNCT_TABLE).strip())) - {""}


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def count_matches(predicted: list[str], gold: list[str], threshold: float) -> int:
    pred_tokens = [normalize_tokens(p) for p in predicted]
    gold_tokens = [normalize_tokens(g) for g in gold]
    pairs = []
    for i, p in enumerate(pred_tokens):
        for j, g in enumerate(gold_tokens):
            score = jaccard(p, g)
            if score >= threshold:
                pairs.append((score, i, j))
    pairs.sort(reverse=True)
    used_pred: set[int] = set()
    used_gold: set[int] = set()
    for _, i, j in pairs:
        if i not in used_pred and j not in used_gold:
            used_pred.add(i)
            used_gold.add(j)
    return len(used_pred)


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: Counts) -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def support(self) -> int:
        return self.tp + self.fn

    def as_dict(self) -> dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "support": self.support,
        }


def _dedupe(spans: Iterable[str]) -> list[str]:
    seen: set[frozenset[str]] = set()
    unique = []
    for span in spans:
        tokens = normalize_tokens(span)
        if tokens and tokens not in seen:
            seen.add(tokens)
            unique.append(span)
    return unique


def span_counts(predicted: Extraction, gold: Extraction, threshold: float = 0.5) -> dict[str, Counts]:
    result = {}
    for key in FIELD_KEYS:
        pred = _dedupe(predicted.get(key, []))
        ref = _dedupe(gold.get(key, []))
        tp = count_matches(pred, ref, threshold)
        result[key] = Counts(tp=tp, fp=len(pred) - tp, fn=len(ref) - tp)
    return result


def presence_counts(predicted: Extraction, gold: Extraction) -> dict[str, Counts]:
    """Document-level detection of each clause type, independent of span boundaries."""
    result = {}
    for key in FIELD_KEYS:
        has_pred, has_gold = bool(predicted.get(key)), bool(gold.get(key))
        result[key] = Counts(
            tp=int(has_pred and has_gold), fp=int(has_pred and not has_gold), fn=int(has_gold and not has_pred)
        )
    return result


def aggregate(per_document: list[dict[str, Counts]]) -> dict[str, object]:
    """Aggregate per-document counts into per-field, micro and macro scores."""
    totals = {key: Counts() for key in FIELD_KEYS}
    for doc in per_document:
        for key, counts in doc.items():
            totals[key].add(counts)
    micro = Counts()
    for counts in totals.values():
        micro.add(counts)
    supported = [c for c in totals.values() if c.support > 0]
    macro = {
        "precision": sum(c.precision for c in supported) / len(supported) if supported else 0.0,
        "recall": sum(c.recall for c in supported) / len(supported) if supported else 0.0,
        "f1": sum(c.f1 for c in supported) / len(supported) if supported else 0.0,
    }
    return {
        "per_field": {key: c.as_dict() for key, c in totals.items()},
        "micro": micro.as_dict(),
        "macro": macro,
    }
