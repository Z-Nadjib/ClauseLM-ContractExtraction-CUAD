"""Loading of the CUAD v1 dataset from its SQuAD-style JSON release."""

from __future__ import annotations

import json
import re
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from contract_extractor.schema import CUAD_NAME_TO_KEY, FIELD_KEYS

CUAD_ARCHIVE_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
SPLIT_FILES = {"train": "train_separate_questions.json", "test": "test.json"}

_CATEGORY_PATTERN = re.compile(r'related to "(.+?)"')


@dataclass(frozen=True)
class Span:
    key: str
    start: int
    end: int
    text: str


@dataclass
class Contract:
    title: str
    text: str
    spans: list[Span] = field(default_factory=list)

    def gold(self) -> dict[str, list[str]]:
        """Document-level gold extraction: unique spans per field in reading order."""
        result: dict[str, list[str]] = {key: [] for key in FIELD_KEYS}
        for span in sorted(self.spans, key=lambda s: s.start):
            if span.text not in result[span.key]:
                result[span.key].append(span.text)
        return result


def download_cuad(raw_dir: Path) -> None:
    """Download and extract the official CUAD archive (train 408 / test 102 contracts)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    if all((raw_dir / name).exists() for name in SPLIT_FILES.values()):
        return
    archive = raw_dir / "data.zip"
    if not archive.exists():
        urllib.request.urlretrieve(CUAD_ARCHIVE_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(raw_dir)


def _category(question: str) -> str:
    match = _CATEGORY_PATTERN.search(question)
    if match is None:
        raise ValueError(f"Unexpected CUAD question format: {question[:80]}")
    return match.group(1)


def load_split(raw_dir: Path, split: str) -> list[Contract]:
    """Load one official split as a list of contracts with character-level spans."""
    with open(raw_dir / SPLIT_FILES[split], encoding="utf-8") as f:
        data = json.load(f)["data"]

    contracts: list[Contract] = []
    for doc in data:
        paragraph = doc["paragraphs"][0]
        text = paragraph["context"]
        spans: set[Span] = set()
        for qa in paragraph["qas"]:
            key = CUAD_NAME_TO_KEY[_category(qa["question"])]
            for answer in qa["answers"]:
                start = answer["answer_start"]
                answer_text = answer["text"]
                end = start + len(answer_text)
                if text[start:end] != answer_text:
                    located = text.find(answer_text)
                    if located < 0:
                        continue
                    start, end = located, located + len(answer_text)
                spans.add(Span(key=key, start=start, end=end, text=answer_text))
        contracts.append(Contract(title=doc["title"], text=text, spans=sorted(spans, key=lambda s: (s.start, s.key))))
    return contracts
