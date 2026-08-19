"""Prompt templates shared by the fine-tuned models and the zero-shot baselines."""

from __future__ import annotations

from contract_extractor.schema import FIELDS

_FIELD_LINES = "\n".join(f"- {f.key}: {f.description}" for f in FIELDS)

SYSTEM_PROMPT = f"""You extract clauses from commercial contracts.
Read the contract excerpt and copy, word for word, every passage that belongs to one of the fields below.

Fields:
{_FIELD_LINES}

Output rules:
1. Return a single JSON object. Keys are field names from the list, values are lists of strings.
2. Each string is an exact, contiguous copy of text from the excerpt. Do not paraphrase or normalize.
3. Omit fields with no matching passage, or give them an empty list.
4. Return {{}} when the excerpt contains none of the fields."""


def build_messages(chunk_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Contract excerpt:\n\n{chunk_text}"},
    ]
