"""Token classification baseline (BERT-NER) running over full contracts."""

from __future__ import annotations

import time

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from contract_extractor.data.bio import LABELS, decode_spans, tokenize_windows
from contract_extractor.inference.base import DocumentResult
from contract_extractor.schema import empty_extraction


class BertNERExtractor:
    def __init__(self, model_path: str, max_length: int = 512, stride: int = 128, batch_size: int = 32):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path).to(self.device).eval()
        if self.device == "cuda":
            self.model.half()
        self.max_length = max_length
        self.stride = stride
        self.batch_size = batch_size

    @torch.inference_mode()
    def _token_probabilities(self, text: str) -> tuple[list[tuple[int, int]], np.ndarray]:
        windows = tokenize_windows(self.tokenizer, text, self.max_length, self.stride)
        sums: dict[tuple[int, int], np.ndarray] = {}
        counts: dict[tuple[int, int], int] = {}
        for i in range(0, len(windows), self.batch_size):
            batch = windows[i : i + self.batch_size]
            padded = self.tokenizer.pad(
                [{"input_ids": w["input_ids"], "attention_mask": w["attention_mask"]} for w in batch],
                return_tensors="pt",
            ).to(self.device)
            probs = torch.softmax(self.model(**padded).logits.float(), dim=-1).cpu().numpy()
            for window, window_probs in zip(batch, probs, strict=True):
                for position, offset in enumerate(window["offset_mapping"]):
                    if offset[0] == offset[1]:
                        continue
                    if offset in sums:
                        sums[offset] += window_probs[position]
                        counts[offset] += 1
                    else:
                        sums[offset] = window_probs[position].copy()
                        counts[offset] = 1
        offsets = sorted(sums)
        if not offsets:
            return [], np.zeros((0, len(LABELS)))
        return offsets, np.stack([sums[o] / counts[o] for o in offsets])

    def extract(self, text: str) -> DocumentResult:
        start = time.perf_counter()
        offsets, probs = self._token_probabilities(text)
        label_ids = probs.argmax(axis=-1).tolist() if len(offsets) else []
        extraction = empty_extraction()
        for key, span_start, span_end in decode_spans(offsets, label_ids):
            span = text[span_start:span_end].strip()
            if span and span not in extraction[key]:
                extraction[key].append(span)
        latency = time.perf_counter() - start
        n_spans = sum(len(v) for v in extraction.values())
        return DocumentResult(
            extraction=extraction,
            latency_s=latency,
            input_tokens=len(offsets),
            n_spans=n_spans,
            n_grounded_spans=n_spans,
        )
