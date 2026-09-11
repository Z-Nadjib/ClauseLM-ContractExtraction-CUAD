"""Evaluation of one system on the CUAD test contracts.

Predictions are written incrementally to ``predictions/<system>.jsonl`` so an interrupted
run (for instance a rate-limited API run) resumes where it stopped. Metrics are then
computed from that file.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from tqdm import tqdm

from contract_extractor.config import DataConfig
from contract_extractor.data.cuad import Contract, load_split
from contract_extractor.evaluation.metrics import aggregate, presence_counts, span_counts
from contract_extractor.inference.base import ChunkedExtractor, DocumentExtractor

WARMUP_TEXT = "This Supply Agreement is entered into as of January 1, 2020 by Acme Corp. and Beta LLC."


def build_extractor(system: dict[str, Any], data_cfg: DataConfig) -> DocumentExtractor:
    kind = system["type"]
    if kind == "bert":
        from contract_extractor.inference.bert_extractor import BertNERExtractor

        return BertNERExtractor(system["model"], batch_size=system.get("batch_size", 32))
    if kind == "hf":
        from contract_extractor.inference.hf_backend import HFBackend

        backend = HFBackend(
            system["model"],
            adapter_path=system.get("adapter"),
            constrained=system.get("constrained", True),
            max_new_tokens=system.get("max_new_tokens", 2048),
        )
    else:
        raise ValueError(f"Unknown system type: {kind}")
    return ChunkedExtractor(backend, data_cfg.chunk_size, data_cfg.chunk_overlap, system.get("batch_size", 16))


def _load_done(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return {r["title"]: r for r in records}


def predict(name: str, system: dict[str, Any], contracts: list[Contract], data_cfg: DataConfig, out_dir: Path) -> Path:
    path = out_dir / "predictions" / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(path)
    todo = [c for c in contracts if c.title not in done]
    if not todo:
        return path

    extractor = build_extractor(system, data_cfg)
    extractor.extract(WARMUP_TEXT)
    with open(path, "a", encoding="utf-8") as f:
        for contract in tqdm(todo, desc=name):
            result = extractor.extract(contract.text)
            record = {
                "title": contract.title,
                "n_chars": len(contract.text),
                "extraction": result.extraction,
                "latency_s": result.latency_s,
                "n_chunks": result.n_chunks,
                "n_valid_chunks": result.n_valid_chunks,
                "n_repaired_chunks": result.n_repaired_chunks,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "n_spans": result.n_spans,
                "n_grounded_spans": result.n_grounded_spans,
                "raw_outputs": result.raw_outputs,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
    return path


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def score(
    name: str,
    system: dict[str, Any],
    contracts: list[Contract],
    predictions_path: Path,
    cost_cfg: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    records = _load_done(predictions_path)
    missing = [c.title for c in contracts if c.title not in records]
    if missing:
        raise RuntimeError(f"{len(missing)} contracts have no prediction for {name}")
    ordered = [records[c.title] for c in contracts]

    span_docs = [span_counts(r["extraction"], c.gold(), threshold) for r, c in zip(ordered, contracts, strict=True)]
    presence_docs = [presence_counts(r["extraction"], c.gold()) for r, c in zip(ordered, contracts, strict=True)]
    latencies = [r["latency_s"] for r in ordered]
    n_chunks = sum(r["n_chunks"] for r in ordered)
    n_spans = sum(r["n_spans"] for r in ordered)
    input_tokens = sum(r["input_tokens"] for r in ordered)
    output_tokens = sum(r["output_tokens"] for r in ordered)

    hours = sum(latencies) / 3600
    total_cost = hours * cost_cfg["local_gpu_hourly_usd"]
    energy_cost = hours * cost_cfg["local_gpu_power_w"] / 1000 * cost_cfg["electricity_usd_per_kwh"]

    return {
        "system": name,
        "type": system["type"],
        "constrained": system.get("constrained"),
        "n_documents": len(ordered),
        "span": aggregate(span_docs),
        "presence": aggregate(presence_docs),
        "json_valid_rate": (sum(r["n_valid_chunks"] for r in ordered) / n_chunks) if n_chunks else None,
        "json_usable_rate": (
            sum(r["n_valid_chunks"] + r.get("n_repaired_chunks", 0) for r in ordered) / n_chunks if n_chunks else None
        ),
        "grounded_span_rate": (sum(r["n_grounded_spans"] for r in ordered) / n_spans) if n_spans else None,
        "latency_s": {
            "mean": statistics.mean(latencies),
            "p50": _percentile(latencies, 0.5),
            "p95": _percentile(latencies, 0.95),
            "total": sum(latencies),
        },
        "tokens": {"input": input_tokens, "output": output_tokens},
        "cost_usd": {
            "total": total_cost,
            "per_document": total_cost / len(ordered),
            "energy_per_document": energy_cost / len(ordered),
        },
    }


def evaluate_system(name: str, eval_cfg: dict[str, Any], data_cfg: DataConfig) -> dict[str, Any]:
    system = eval_cfg["systems"][name]
    contracts = load_split(Path(data_cfg.raw_dir), eval_cfg.get("split", "test"))
    if eval_cfg.get("max_documents"):
        contracts = contracts[: eval_cfg["max_documents"]]
    out_dir = Path(eval_cfg["output_dir"])
    predictions_path = predict(name, system, contracts, data_cfg, out_dir)
    metrics = score(name, system, contracts, predictions_path, eval_cfg["cost"], eval_cfg.get("match_threshold", 0.5))
    metrics_path = out_dir / "metrics" / f"{name}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics
