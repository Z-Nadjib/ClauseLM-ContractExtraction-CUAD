"""Local Gradio demo: paste a contract, get the extracted clauses as JSON.

Runs on a GPU when available and falls back to CPU. The model is read from the ``MODEL_ID``
environment variable and defaults to the fine-tuned Qwen2.5-1.5B published on the Hub.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import gradio as gr
import torch

from contract_extractor.inference.base import ChunkedExtractor
from contract_extractor.inference.hf_backend import HFBackend
from contract_extractor.schema import KEY_TO_FIELD

MODEL_ID = os.environ.get("MODEL_ID", "Z-Nadjib/ClauseLM-Qwen2.5-1.5B")
CHUNK_SIZE = 6000
CHUNK_OVERLAP = 600
MAX_CHARS_CPU = 30_000
MAX_CHARS_GPU = 400_000
EXAMPLES_DIR = Path(__file__).parent / "examples"

backend = HFBackend(
    MODEL_ID, constrained=True, max_new_tokens=2048, device="cuda" if torch.cuda.is_available() else None
)
extractor = ChunkedExtractor(backend, CHUNK_SIZE, CHUNK_OVERLAP, batch_size=8)


def run(contract: str, file_path: str | None):
    if file_path:
        contract = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    contract = (contract or "").strip()
    if not contract:
        raise gr.Error("Paste a contract or upload a .txt file.")
    limit = MAX_CHARS_GPU if torch.cuda.is_available() else MAX_CHARS_CPU
    truncated = len(contract) > limit
    contract = contract[:limit]

    start = time.perf_counter()
    result = extractor.extract(contract)
    elapsed = time.perf_counter() - start

    found = {key: spans for key, spans in result.extraction.items() if spans}
    rows = [[KEY_TO_FIELD[key].cuad_name, span] for key, spans in found.items() for span in spans]
    stats = (
        f"{len(found)} clause types found, {result.n_chunks} chunks, "
        f"{result.n_valid_chunks}/{result.n_chunks} valid JSON outputs, {elapsed:.1f} s"
    )
    if truncated:
        stats += f". Input truncated to {limit:,} characters."
    return found, rows, stats


def load_example(name: str) -> str:
    return (EXAMPLES_DIR / name).read_text(encoding="utf-8")


examples = sorted(p.name for p in EXAMPLES_DIR.glob("*.txt"))

with gr.Blocks(title="Contract Clause Extractor") as demo:
    gr.Markdown(
        "# Contract Clause Extractor\n"
        f"Model: `{MODEL_ID}`, a small language model fine-tuned on "
        "[CUAD](https://www.atticusprojectai.org/cuad) to extract 41 clause types as verbatim spans. "
        "Decoding is constrained by a JSON Schema, so the output is always valid JSON."
    )
    with gr.Row():
        with gr.Column():
            contract_box = gr.Textbox(label="Contract text", lines=22, max_lines=40)
            file_box = gr.File(label="Or upload a .txt file", file_types=[".txt"], type="filepath")
            if examples:
                example_choice = gr.Dropdown(examples, label="Load an example contract")
                example_choice.change(load_example, example_choice, contract_box)
            button = gr.Button("Extract", variant="primary")
        with gr.Column():
            stats_box = gr.Markdown()
            json_box = gr.JSON(label="Extraction")
            table_box = gr.Dataframe(headers=["Clause type", "Text"], wrap=True, label="Extracted spans")
    button.click(run, [contract_box, file_box], [json_box, table_box, stats_box])

if __name__ == "__main__":
    demo.launch()
