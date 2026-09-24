"""Local generation backend: transformers + optional grammar-constrained decoding.

Decoding is greedy with no repetition penalty, since outputs are copies of the input text.
When ``constrained`` is enabled, a grammar compiled from the extraction JSON Schema masks
every token that would break the schema, so each completed generation is a valid JSON
object with known keys and list-of-string values. Generations stopped early (degenerate
loop or token budget) are closed after their last complete span.
"""

from __future__ import annotations

import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from contract_extractor.inference.base import ChunkResult
from contract_extractor.inference.guards import GuardedLogitsProcessor, close_truncated_json
from contract_extractor.prompts import build_messages
from contract_extractor.schema import JSON_SEPARATORS, json_schema, parse_output


class HFBackend:
    def __init__(
        self,
        model_path: str,
        adapter_path: str | None = None,
        constrained: bool = True,
        max_new_tokens: int = 2048,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_path)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, attn_implementation="sdpa")
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
        self.model = model.to(self.device).eval()
        self.max_new_tokens = max_new_tokens

        self.compiled_grammar = None
        if constrained:
            import xgrammar as xgr

            tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                self.tokenizer,
                vocab_size=self.model.config.vocab_size,
                stop_token_ids=self._eos_token_ids(),
            )
            compiler = xgr.GrammarCompiler(tokenizer_info)
            self.compiled_grammar = compiler.compile_json_schema(
                json.dumps(json_schema()), any_whitespace=False, separators=JSON_SEPARATORS
            )

    def _prompts(self, texts: list[str]) -> list[str]:
        return [
            self.tokenizer.apply_chat_template(build_messages(t), tokenize=False, add_generation_prompt=True)
            for t in texts
        ]

    @torch.inference_mode()
    def extract_batch(self, texts: list[str]) -> list[ChunkResult]:
        encoded = self.tokenizer(self._prompts(texts), return_tensors="pt", padding=True, add_special_tokens=False).to(
            self.device
        )
        guard = GuardedLogitsProcessor(
            self.compiled_grammar,
            eos_token_id=self._eos_token_ids()[0],
            prompt_length=encoded["input_ids"].shape[1],
        )
        generated = self.model.generate(
            **encoded,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=1.0,
            logits_processor=[guard],
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_tokens = generated[:, encoded["input_ids"].shape[1] :]
        stop_ids = set(self._stop_token_ids())
        results = []
        for row, mask in zip(new_tokens.tolist(), encoded["attention_mask"], strict=True):
            length = next((i for i, token in enumerate(row) if token in stop_ids), len(row))
            raw = self.tokenizer.decode(row[:length], skip_special_tokens=True).strip()
            extraction, valid = parse_output(raw)
            repaired = False
            if not valid and self.compiled_grammar is not None:
                closed = close_truncated_json(raw)
                if closed is not None:
                    extraction, repaired = parse_output(closed)
            results.append(
                ChunkResult(
                    raw=raw,
                    extraction=extraction,
                    valid=valid,
                    repaired=repaired,
                    input_tokens=int(mask.sum()),
                    output_tokens=length,
                )
            )
        return results

    def _eos_token_ids(self) -> list[int]:
        eos = self.model.generation_config.eos_token_id
        ids = eos if isinstance(eos, list) else [eos]
        return [i for i in ids if i is not None]

    def _stop_token_ids(self) -> list[int]:
        return [*self._eos_token_ids(), self.tokenizer.pad_token_id]
