"""Conversion of CUAD contracts into chunk-level ``text -> JSON`` supervised examples."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from contract_extractor.config import DataConfig
from contract_extractor.data.chunking import chunk_targets, chunk_text
from contract_extractor.data.cuad import Contract, download_cuad, load_split
from contract_extractor.prompts import build_messages
from contract_extractor.schema import to_compact_json


def split_contracts(contracts: list[Contract], fraction: float, seed: int) -> tuple[list[Contract], list[Contract]]:
    """Hold out a fraction of the official training contracts for validation."""
    titles = sorted(c.title for c in contracts)
    rng = random.Random(seed)
    rng.shuffle(titles)
    n_val = max(1, round(len(titles) * fraction))
    val_titles = set(titles[:n_val])
    train = [c for c in contracts if c.title not in val_titles]
    val = [c for c in contracts if c.title in val_titles]
    return train, val


def contract_examples(contract: Contract, cfg: DataConfig) -> Iterable[dict[str, Any]]:
    for chunk in chunk_text(contract.text, cfg.chunk_size, cfg.chunk_overlap):
        target = chunk_targets(chunk, contract.spans, cfg.min_clip_coverage, cfg.min_clip_chars)
        completion = to_compact_json(target)
        yield {
            "contract": contract.title,
            "chunk": chunk.index,
            "is_empty": completion == "{}",
            "prompt": build_messages(chunk.text),
            "completion": [{"role": "assistant", "content": completion}],
        }


def build_sft_examples(contracts: list[Contract], cfg: DataConfig, downsample_negatives: bool) -> list[dict[str, Any]]:
    rng = random.Random(cfg.seed)
    examples = []
    for contract in contracts:
        for example in contract_examples(contract, cfg):
            if downsample_negatives and example["is_empty"] and rng.random() > cfg.negative_keep_ratio:
                continue
            examples.append(example)
    return examples


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def prepare(cfg: DataConfig) -> dict[str, Any]:
    """Download CUAD, create the contract-level splits and write the SFT datasets."""
    raw_dir, out_dir = Path(cfg.raw_dir), Path(cfg.processed_dir)
    download_cuad(raw_dir)
    train_all = load_split(raw_dir, "train")
    test = load_split(raw_dir, "test")
    train, val = split_contracts(train_all, cfg.validation_fraction, cfg.seed)

    splits = {
        "train": sorted(c.title for c in train),
        "validation": sorted(c.title for c in val),
        "test": sorted(c.title for c in test),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "splits.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    train_examples = build_sft_examples(train, cfg, downsample_negatives=True)
    val_examples = build_sft_examples(val, cfg, downsample_negatives=False)
    stats = {
        "contracts": {name: len(titles) for name, titles in splits.items()},
        "sft_examples": {
            "train": write_jsonl(train_examples, out_dir / "sft_train.jsonl"),
            "validation": write_jsonl(val_examples, out_dir / "sft_validation.jsonl"),
        },
        "empty_fraction": {
            "train": sum(e["is_empty"] for e in train_examples) / max(1, len(train_examples)),
            "validation": sum(e["is_empty"] for e in val_examples) / max(1, len(val_examples)),
        },
        "chunking": {"chunk_size": cfg.chunk_size, "chunk_overlap": cfg.chunk_overlap},
    }
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats
