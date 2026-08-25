"""QLoRA supervised fine-tuning with Unsloth and TRL.

The loss is computed on the assistant completion only (the JSON), not on the prompt.
``unsloth`` must be imported before ``transformers`` and ``trl`` so its patches apply,
which is why the imports live inside :func:`train_sft`.
"""

from __future__ import annotations

import json
from pathlib import Path

from contract_extractor.config import DataConfig, SFTTrainConfig


def train_sft(cfg: SFTTrainConfig, data_cfg: DataConfig, max_eval_samples: int = 200, resume: bool = False) -> Path:
    from unsloth import FastLanguageModel  # noqa: I001

    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )

    processed = Path(data_cfg.processed_dir)
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(processed / "sft_train.jsonl"),
            "validation": str(processed / "sft_validation.jsonl"),
        },
    )
    dataset = dataset.select_columns(["prompt", "completion"])

    def fits(example: dict) -> bool:
        ids = tokenizer.apply_chat_template(example["prompt"] + example["completion"], tokenize=True)
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        return len(ids) <= cfg.max_seq_length

    n_before = {split: len(ds) for split, ds in dataset.items()}
    dataset = dataset.filter(fits, num_proc=4)
    validation = dataset["validation"].shuffle(seed=cfg.seed)
    validation = validation.select(range(min(max_eval_samples, len(validation))))

    args = SFTConfig(
        output_dir=cfg.output_dir,
        run_name=cfg.run_name,
        max_length=cfg.max_seq_length,
        completion_only_loss=True,
        packing=False,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        optim="adamw_8bit",
        bf16=True,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=cfg.seed,
        dataset_num_proc=4,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=validation,
        args=args,
    )
    trainer.train(resume_from_checkpoint=resume or None)

    adapter_dir = Path(cfg.output_dir) / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    summary = {
        "base_model": cfg.base_model,
        "examples_before_length_filter": n_before,
        "examples_after_length_filter": {split: len(ds) for split, ds in dataset.items()},
        "eval_samples": len(validation),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": trainer.state.best_metric,
        "log_history": trainer.state.log_history,
    }
    with open(Path(cfg.output_dir) / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return adapter_dir
