"""Fine-tune a small instruction model with QLoRA on the CUAD extraction task."""

from __future__ import annotations

import argparse

from contract_extractor.config import DataConfig, SFTTrainConfig
from contract_extractor.training.sft import train_sft


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Training config, e.g. configs/train/qwen2.5-1.5b.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from the last checkpoint in output_dir")
    args = parser.parse_args()
    adapter_dir = train_sft(
        SFTTrainConfig.from_yaml(args.config), DataConfig.from_yaml(args.data_config), resume=args.resume
    )
    print(f"Adapter saved to {adapter_dir}")


if __name__ == "__main__":
    main()
