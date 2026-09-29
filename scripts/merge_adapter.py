"""Merge a trained LoRA adapter into its 16-bit base model for standalone inference.

The generation config of the merged model is set to greedy decoding without repetition
penalty, the setting used for evaluation: outputs are verbatim copies of the input.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    generation = model.generation_config
    generation.do_sample = False
    generation.temperature = None
    generation.top_p = None
    generation.top_k = None
    generation.repetition_penalty = 1.0

    output = Path(args.output)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    print(f"Merged model saved to {output}")


if __name__ == "__main__":
    main()
