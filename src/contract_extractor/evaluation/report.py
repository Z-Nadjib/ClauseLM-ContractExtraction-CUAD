"""Result tables (Markdown and CSV) and figures built from ``reports/results/metrics``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from contract_extractor.data.cuad import load_split
from contract_extractor.evaluation import figures
from contract_extractor.evaluation.metrics import aggregate, presence_counts, span_counts
from contract_extractor.schema import FIELD_KEYS


def family(metrics: dict[str, Any]) -> str:
    if metrics["type"] == "bert":
        return "baseline"
    return "zero-shot" if "zero-shot" in metrics["system"] else "fine-tuned"


def load_metrics(results_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split metrics into full test set runs and partial runs (ablations)."""
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((results_dir / "metrics").glob("*.json"))]
    n_full = max((m["n_documents"] for m in runs), default=0)
    systems = sorted((m for m in runs if m["n_documents"] == n_full), key=lambda m: -m["span"]["micro"]["f1"])
    partial = [m for m in runs if m["n_documents"] < n_full]
    return systems, partial


def summary_table(systems: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for m in systems:
        rows.append(
            {
                "system": m["system"],
                "family": family(m),
                "span_micro_p": m["span"]["micro"]["precision"],
                "span_micro_r": m["span"]["micro"]["recall"],
                "span_micro_f1": m["span"]["micro"]["f1"],
                "span_macro_f1": m["span"]["macro"]["f1"],
                "presence_micro_f1": m["presence"]["micro"]["f1"],
                "json_valid_rate": m["json_valid_rate"],
                "grounded_span_rate": m["grounded_span_rate"],
                "latency_mean_s": m["latency_s"]["mean"],
                "latency_p95_s": m["latency_s"]["p95"],
                "cost_per_doc_usd": m["cost_usd"]["per_document"],
                "energy_cost_per_doc_usd": m["cost_usd"].get("energy_per_document"),
            }
        )
    return pd.DataFrame(rows)


def per_field_table(systems: list[dict[str, Any]]) -> pd.DataFrame:
    table = {m["system"]: {k: m["span"]["per_field"][k]["f1"] for k in FIELD_KEYS} for m in systems}
    frame = pd.DataFrame(table)
    if systems:
        frame.insert(0, "support", [systems[0]["span"]["per_field"][k]["support"] for k in FIELD_KEYS])
    return frame


def _fmt(value: Any, kind: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if kind == "pct":
        return f"{100 * value:.1f}"
    if kind == "s":
        return f"{value:.1f}"
    if kind == "usd":
        return f"{value:.5f}"
    return str(value)


def markdown_summary(frame: pd.DataFrame) -> str:
    header = (
        "| System | Span P | Span R | Span F1 (micro) | Span F1 (macro) | Presence F1 | Valid JSON % "
        "| Grounded % | Latency / doc (s) | Cost / doc (USD) | Energy / doc (USD) |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for _, row in frame.iterrows():
        cost = _fmt(row.get("cost_per_doc_usd"), "usd")
        lines.append(
            f"| {row['system']} | {_fmt(row.get('span_micro_p'), 'pct')} | {_fmt(row.get('span_micro_r'), 'pct')} "
            f"| {_fmt(row.get('span_micro_f1'), 'pct')} | {_fmt(row.get('span_macro_f1'), 'pct')} "
            f"| {_fmt(row.get('presence_micro_f1'), 'pct')} | {_fmt(row.get('json_valid_rate'), 'pct')} "
            f"| {_fmt(row.get('grounded_span_rate'), 'pct')} | {_fmt(row.get('latency_mean_s'), 's')} | {cost} "
            f"| {_fmt(row.get('energy_cost_per_doc_usd'), 'usd')} |"
        )
    return header + "\n".join(lines) + "\n"


def _subset_scores(predictions_path: Path, titles: list[str], contracts: dict[str, Any]) -> dict[str, float]:
    with open(predictions_path, encoding="utf-8") as f:
        records = {r["title"]: r for r in map(json.loads, f) if r["title"] in titles}
    ordered = [records[t] for t in titles]
    golds = [contracts[t].gold() for t in titles]
    span = aggregate([span_counts(r["extraction"], g) for r, g in zip(ordered, golds, strict=True)])
    presence = aggregate([presence_counts(r["extraction"], g) for r, g in zip(ordered, golds, strict=True)])
    n_chunks = sum(r["n_chunks"] for r in ordered)
    return {
        "documents": len(ordered),
        "chunks": n_chunks,
        "json_valid_rate": sum(r["n_valid_chunks"] for r in ordered) / n_chunks,
        "json_usable_rate": sum(r["n_valid_chunks"] + r.get("n_repaired_chunks", 0) for r in ordered) / n_chunks,
        "span_micro_f1": span["micro"]["f1"],
        "presence_micro_f1": presence["micro"]["f1"],
        "latency_mean_s": sum(r["latency_s"] for r in ordered) / len(ordered),
        "output_tokens_per_doc": sum(r["output_tokens"] for r in ordered) / len(ordered),
    }


def ablation_table(partial: list[dict[str, Any]], results_dir: Path, raw_dir: Path) -> pd.DataFrame:
    """Compare each ``<name>-unconstrained`` run with ``<name>`` on the same contracts."""
    predictions = results_dir / "predictions"
    rows = []
    contracts = None
    for m in partial:
        reference = m["system"].removesuffix("-unconstrained")
        if reference == m["system"] or not (predictions / f"{reference}.jsonl").exists():
            continue
        if contracts is None:
            contracts = {c.title: c for c in load_split(raw_dir, "test")}
        with open(predictions / f"{m['system']}.jsonl", encoding="utf-8") as f:
            titles = [json.loads(line)["title"] for line in f]
        for name, decoding in ((reference, "constrained"), (m["system"], "unconstrained")):
            rows.append(
                {
                    "system": name,
                    "decoding": decoding,
                    **_subset_scores(predictions / f"{name}.jsonl", titles, contracts),
                }
            )
    return pd.DataFrame(rows)


def markdown_ablation(frame: pd.DataFrame) -> str:
    header = (
        "| System | Decoding | Contracts | Valid JSON % | Usable JSON % | Span F1 (micro) | Presence F1 "
        "| Latency / doc (s) |\n|---|---|---|---|---|---|---|---|\n"
    )
    lines = [
        f"| {r['system']} | {r['decoding']} | {r['documents']} | {_fmt(r['json_valid_rate'], 'pct')} "
        f"| {_fmt(r['json_usable_rate'], 'pct')} | {_fmt(r['span_micro_f1'], 'pct')} "
        f"| {_fmt(r['presence_micro_f1'], 'pct')} | {_fmt(r['latency_mean_s'], 's')} |"
        for _, r in frame.iterrows()
    ]
    return header + "\n".join(lines) + "\n"


def collect_training_summaries(outputs_dir: Path, results_dir: Path) -> dict[str, dict[str, Any]]:
    """Copy SFT training summaries into the versioned results folder and load them."""
    target = results_dir / "training"
    target.mkdir(parents=True, exist_ok=True)
    for source in outputs_dir.glob("*/train_summary.json"):
        data = json.loads(source.read_text(encoding="utf-8"))
        if "base_model" in data:
            (target / f"{source.parent.name}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(target.glob("*.json"))}


def build_report(
    results_dir: Path, figures_dir: Path, outputs_dir: Path | None = None, raw_dir: Path = Path("data/raw")
) -> dict[str, Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    systems, partial = load_metrics(results_dir)
    summary = summary_table(systems)
    per_field = per_field_table(systems)

    outputs = {
        "summary_csv": results_dir / "summary.csv",
        "per_field_csv": results_dir / "per_field_f1.csv",
        "summary_md": results_dir / "summary.md",
    }
    summary.to_csv(outputs["summary_csv"], index=False)
    per_field.to_csv(outputs["per_field_csv"])
    outputs["summary_md"].write_text(markdown_summary(summary), encoding="utf-8")

    ablation = ablation_table(partial, results_dir, raw_dir)
    if not ablation.empty:
        outputs["ablation_md"] = results_dir / "ablation_constrained_decoding.md"
        outputs["ablation_csv"] = results_dir / "ablation_constrained_decoding.csv"
        ablation.to_csv(outputs["ablation_csv"], index=False)
        outputs["ablation_md"].write_text(markdown_ablation(ablation), encoding="utf-8")

    n_docs = systems[0]["n_documents"] if systems else 0
    drawers = {
        "span_f1": figures.ranked_bars(
            summary,
            "span_micro_f1",
            "Span F1 by system",
            f"Micro F1 over 41 clause types, {n_docs} CUAD test contracts. "
            "A span is correct when it shares at least half of its words with a gold span.",
            lambda v: f"{v:.1f}",
            scale=100,
        ),
        "latency": figures.ranked_bars(
            summary,
            "latency_mean_s",
            "Mean latency per contract",
            "Wall-clock seconds for a full contract on one RTX 5080, all windows of the contract in one batch.",
            lambda v: f"{v:.2f} s" if v < 1 else f"{v:.1f} s",
        ),
        "cost_vs_f1": figures.cost_quality(summary),
        "per_field_f1": figures.per_field_heatmap(per_field),
    }
    gain = figures.finetuning_gain(summary)
    if gain is not None:
        drawers["finetuning_gain"] = gain
    if outputs_dir is not None:
        summaries = collect_training_summaries(outputs_dir, results_dir)
        if summaries:
            drawers["validation_loss"] = figures.validation_loss(summaries)
    for stem, draw in drawers.items():
        outputs[stem] = figures.render(draw, figures_dir, stem)
    return outputs
