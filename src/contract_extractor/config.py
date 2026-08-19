"""Typed configuration objects loaded from YAML files in ``configs/``."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def _from_dict(cls: type[T], values: dict[str, Any]) -> T:
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**values)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    chunk_size: int = 6000
    chunk_overlap: int = 600
    min_clip_coverage: float = 0.5
    min_clip_chars: int = 300
    validation_fraction: float = 0.1
    negative_keep_ratio: float = 0.5
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> DataConfig:
        return _from_dict(cls, load_yaml(path))


@dataclass
class LoraConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )


@dataclass
class SFTTrainConfig:
    run_name: str
    base_model: str
    output_dir: str
    max_seq_length: int = 4096
    load_in_4bit: bool = True
    num_train_epochs: float = 2.0
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    eval_steps: int = 100
    logging_steps: int = 10
    seed: int = 42
    lora: LoraConfig = field(default_factory=LoraConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SFTTrainConfig:
        values = load_yaml(path)
        values["lora"] = _from_dict(LoraConfig, values.get("lora", {}))
        return _from_dict(cls, values)


@dataclass
class NERTrainConfig:
    run_name: str
    base_model: str
    output_dir: str
    max_length: int = 512
    stride: int = 128
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 16
    learning_rate: float = 3e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    negative_keep_ratio: float = 0.3
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> NERTrainConfig:
        return _from_dict(cls, load_yaml(path))
