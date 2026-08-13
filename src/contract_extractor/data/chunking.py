"""Character-based chunking of long contracts into overlapping windows.

Chunk boundaries are tokenizer independent, so every system (fine-tuned models, zero-shot
models and the API baseline) sees exactly the same windows.
"""

from __future__ import annotations

from dataclasses import dataclass

from contract_extractor.data.cuad import Span
from contract_extractor.schema import Extraction, empty_extraction


@dataclass(frozen=True)
class Chunk:
    index: int
    start: int
    end: int
    text: str


def _find_break(text: str, start: int, end: int, search_window: int) -> int:
    """Return a cut position close to ``end``, preferring paragraph then line then word breaks."""
    if end >= len(text):
        return len(text)
    lower = max(start + 1, end - search_window)
    for separator in ("\n\n", "\n", ". ", " "):
        position = text.rfind(separator, lower, end)
        if position != -1:
            return position + len(separator)
    return end


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    search_window = max(1, chunk_size // 6)
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = _find_break(text, start, start + chunk_size, search_window)
        chunks.append(Chunk(index=len(chunks), start=start, end=end, text=text[start:end]))
        if end >= len(text):
            break
        next_start = _find_break(text, start, end - overlap, search_window)
        start = next_start if next_start > start else end
    return chunks


def chunk_targets(chunk: Chunk, spans: list[Span], min_coverage: float, min_chars: int) -> Extraction:
    """Gold extraction for one chunk.

    Spans fully inside the chunk are kept as is. Spans crossing a boundary are clipped and
    kept when the visible part covers at least ``min_coverage`` of the span or at least
    ``min_chars`` characters.
    """
    target = empty_extraction()
    for span in spans:
        lo, hi = max(span.start, chunk.start), min(span.end, chunk.end)
        if hi <= lo:
            continue
        visible = hi - lo
        length = span.end - span.start
        if visible < length and visible / length < min_coverage and visible < min_chars:
            continue
        snippet = chunk.text[lo - chunk.start : hi - chunk.start].strip()
        if snippet and snippet not in target[span.key]:
            target[span.key].append(snippet)
    return target
