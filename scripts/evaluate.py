"""Run one or more systems on the CUAD test contracts and write their metrics."""

from __future__ import annotations

import argparse

from contract_extractor.config import DataConfig, load_yaml
from contract_extractor.evaluation.runner import evaluate_system


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--systems", nargs="+", required=True, help="System names defined in the eval config")
    parser.add_argument("--max-documents", type=int, default=None)
    args = parser.parse_args()

    eval_cfg = load_yaml(args.config)
    if args.max_documents is not None:
        eval_cfg["max_documents"] = args.max_documents
    data_cfg = DataConfig.from_yaml(args.data_config)
    for name in args.systems:
        metrics = evaluate_system(name, eval_cfg, data_cfg)
        span, presence = metrics["span"], metrics["presence"]
        print(
            f"{name}: span micro F1={span['micro']['f1']:.3f} macro F1={span['macro']['f1']:.3f} "
            f"presence micro F1={presence['micro']['f1']:.3f} json_valid={metrics['json_valid_rate']} "
            f"latency/doc={metrics['latency_s']['mean']:.2f}s cost/doc=${metrics['cost_usd']['per_document']:.5f}"
        )


if __name__ == "__main__":
    main()
