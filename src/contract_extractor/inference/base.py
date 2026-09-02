"""Document-level extraction pipeline: chunk, extract per chunk, merge into one JSON."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Protocol

from contract_extractor.data.chunking import Chunk, chunk_text
from contract_extractor.schema import FIELD_KEYS, Extraction, empty_extraction


@dataclass
class ChunkResult:
    raw: str
    extraction: Extraction
    valid: bool
    repaired: bool = False
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class DocumentResult:
    extraction: Extraction
    latency_s: float
    n_chunks: int = 0
    n_valid_chunks: int = 0
    n_repaired_chunks: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    n_spans: int = 0
    n_grounded_spans: int = 0
    raw_outputs: list[str] = field(default_factory=list)


class ChunkBackend(Protocol):
    """A model that maps a batch of chunk texts to chunk-level extractions."""

    def extract_batch(self, texts: list[str]) -> list[ChunkResult]: ...


class DocumentExtractor(Protocol):
    def extract(self, text: str) -> DocumentResult: ...


_WS = re.compile(r"\s+")


def locate(span: str, text: str) -> tuple[int, int] | None:
    """Character offsets of ``span`` in ``text``, tolerant to whitespace differences."""
    position = text.find(span)
    if position >= 0:
        return position, position + len(span)
    if not span.strip():
        return None
    words = _WS.split(span.strip())
    pattern = r"\s+".join(re.escape(w) for w in words)
    match = re.search(pattern, text)
    return (match.start(), match.end()) if match else None


def merge_chunk_extractions(
    document: str, chunks: list[Chunk], results: list[ChunkResult]
) -> tuple[Extraction, int, int]:
    """Merge chunk-level spans into document-level spans.

    Spans found in their source chunk are mapped to document offsets, and overlapping
    spans of the same field (typically duplicates produced by the chunk overlap) are
    replaced by their union. Spans that cannot be located are kept once, verbatim.
    Returns the extraction, the number of predicted spans and the number of grounded ones.
    """
    merged = empty_extraction()
    n_spans = n_grounded = 0
    for key in FIELD_KEYS:
        intervals: list[tuple[int, int]] = []
        ungrounded: list[str] = []
        for chunk, result in zip(chunks, results, strict=True):
            for span in result.extraction.get(key, []):
                n_spans += 1
                found = locate(span, chunk.text)
                if found is None:
                    if span not in ungrounded:
                        ungrounded.append(span)
                    continue
                n_grounded += 1
                intervals.append((chunk.start + found[0], chunk.start + found[1]))
        intervals.sort()
        unions: list[list[int]] = []
        for start, end in intervals:
            if unions and start < unions[-1][1]:
                unions[-1][1] = max(unions[-1][1], end)
            else:
                unions.append([start, end])
        merged[key] = [document[s:e].strip() for s, e in unions] + ungrounded
    return merged, n_spans, n_grounded


class ChunkedExtractor:
    """Runs a chunk-level backend over a full contract."""

    def __init__(self, backend: ChunkBackend, chunk_size: int, chunk_overlap: int, batch_size: int):
        self.backend = backend
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size

    def extract(self, text: str) -> DocumentResult:
        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        start = time.perf_counter()
        results: list[ChunkResult] = []
        for i in range(0, len(chunks), self.batch_size):
            results.extend(self.backend.extract_batch([c.text for c in chunks[i : i + self.batch_size]]))
        latency = time.perf_counter() - start
        extraction, n_spans, n_grounded = merge_chunk_extractions(text, chunks, results)
        return DocumentResult(
            extraction=extraction,
            latency_s=latency,
            n_chunks=len(chunks),
            n_valid_chunks=sum(r.valid for r in results),
            n_repaired_chunks=sum(r.repaired for r in results),
            input_tokens=sum(r.input_tokens for r in results),
            output_tokens=sum(r.output_tokens for r in results),
            n_spans=n_spans,
            n_grounded_spans=n_grounded,
            raw_outputs=[r.raw for r in results],
        )
