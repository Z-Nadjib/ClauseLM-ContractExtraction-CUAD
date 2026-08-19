"""Download CUAD and build the chunk-level SFT datasets."""

from __future__ import annotations

import argparse
import json

from contract_extractor.config import DataConfig
from contract_extractor.data.build import prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    stats = prepare(DataConfig.from_yaml(args.config))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
