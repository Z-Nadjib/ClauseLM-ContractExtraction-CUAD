"""Fine-tuning of the BERT token classification baseline."""

from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from contract_extractor.config import DataConfig, NERTrainConfig
from contract_extractor.data.bio import LABEL_TO_ID, LABELS, char_span_ids, token_labels, tokenize_windows
from contract_extractor.data.cuad import Contract, load_split


def build_windows(contracts: list[Contract], tokenizer, cfg: NERTrainConfig, downsample_negatives: bool) -> Dataset:
    rng = random.Random(cfg.seed)
    outside = LABEL_TO_ID["O"]
    records: dict[str, list] = {"input_ids": [], "attention_mask": [], "labels": []}
    for contract in contracts:
        span_ids = char_span_ids(len(contract.text), contract.spans)
        for window in tokenize_windows(tokenizer, contract.text, cfg.max_length, cfg.stride):
            labels = token_labels(window["offset_mapping"], contract.spans, span_ids)
            is_negative = all(label in (outside, -100) for label in labels)
            if downsample_negatives and is_negative and rng.random() > cfg.negative_keep_ratio:
                continue
            records["input_ids"].append(window["input_ids"])
            records["attention_mask"].append(window["attention_mask"])
            records["labels"].append(labels)
    return Dataset.from_dict(records)


def train_ner(train_cfg: NERTrainConfig, data_cfg: DataConfig) -> Path:
    with open(Path(data_cfg.processed_dir) / "splits.json", encoding="utf-8") as f:
        splits = json.load(f)
    contracts = load_split(Path(data_cfg.raw_dir), "train")
    train_contracts = [c for c in contracts if c.title in set(splits["train"])]
    val_contracts = [c for c in contracts if c.title in set(splits["validation"])]

    tokenizer = AutoTokenizer.from_pretrained(train_cfg.base_model)
    train_ds = build_windows(train_contracts, tokenizer, train_cfg, downsample_negatives=True)
    val_ds = build_windows(val_contracts, tokenizer, train_cfg, downsample_negatives=False)

    model = AutoModelForTokenClassification.from_pretrained(
        train_cfg.base_model,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id=LABEL_TO_ID,
    )
    args = TrainingArguments(
        output_dir=train_cfg.output_dir,
        run_name=train_cfg.run_name,
        num_train_epochs=train_cfg.num_train_epochs,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=train_cfg.per_device_train_batch_size * 2,
        learning_rate=train_cfg.learning_rate,
        warmup_steps=train_cfg.warmup_ratio,
        weight_decay=train_cfg.weight_decay,
        lr_scheduler_type="linear",
        bf16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        report_to="none",
        seed=train_cfg.seed,
        dataloader_num_workers=2,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        processing_class=tokenizer,
    )
    trainer.train()
    final_dir = Path(train_cfg.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    with open(final_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_windows": len(train_ds),
                "validation_windows": len(val_ds),
                "log_history": trainer.state.log_history,
            },
            f,
            indent=2,
        )
    return final_dir
