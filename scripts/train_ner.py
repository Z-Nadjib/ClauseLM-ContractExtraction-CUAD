"""Fine-tune the BERT token classification baseline on CUAD."""

from __future__ import annotations

import argparse

from contract_extractor.config import DataConfig, NERTrainConfig
from contract_extractor.training.ner import train_ner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train/bert-ner.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    args = parser.parse_args()
    model_dir = train_ner(NERTrainConfig.from_yaml(args.config), DataConfig.from_yaml(args.data_config))
    print(f"Model saved to {model_dir}")


if __name__ == "__main__":
    main()
