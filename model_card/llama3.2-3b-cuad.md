---
language:
- en
license: llama3.2
library_name: transformers
pipeline_tag: text-generation
base_model: meta-llama/Llama-3.2-3B-Instruct
datasets:
- theatticusproject/cuad
tags:
- legal
- contracts
- information-extraction
- structured-output
- json
- qlora
- unsloth
---

# Llama-3.2-3B CUAD Clause Extractor

A 3B parameter model that reads a commercial contract and returns a JSON object with the
passages belonging to each of the 41 clause categories of
[CUAD](https://www.atticusprojectai.org/cuad): parties, dates, renewal terms, governing law,
termination rights, restrictive covenants, licences, liability caps, insurance, audit rights
and more. Every value is a verbatim span of the contract.

It is [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) fine-tuned
with QLoRA on the CUAD training contracts, with the LoRA weights merged into the 16-bit base.
Built with Llama.

* Code, training and evaluation: <!-- GITHUB_URL -->
* Demo: <!-- SPACE_URL -->

<!-- RESULTS:START -->
## Results

102 contracts of the official CUAD test split, document level. A span is correct when its
word-level Jaccard similarity with a gold span of the same field is at least 0.5.

| System | Span F1 (micro) | Span F1 (macro) | Presence F1 | Grounded spans | Usable JSON | Latency / contract |
|---|---|---|---|---|---|---|
| **This model** | **61.7** | **50.4** | **83.8** | 97.4% | 100% | 27.5 s |
| Llama-3.2-3B-Instruct, zero-shot | 9.6 | 5.0 | 49.5 | 20.9% | 100% | 40.5 s |
| Qwen2.5-1.5B + QLoRA (same recipe) | 57.2 | 43.2 | 80.9 | 98.6% | 100% | 18.0 s |
| Legal-BERT NER (token classification) | 42.7 | 30.2 | 80.8 | 100% | n/a | 0.08 s |

Latency is measured on one RTX 5080 (16 GB) with transformers in bf16, all windows of a
contract in one batch. At an on-demand rental price of USD 0.40 per GPU hour this is
USD 0.0031 per contract; the electricity cost at 360 W and USD 0.25 per kWh is USD 0.00069.

Best fields: governing law (92 F1), document name (88), agreement date (82), expiration date (82).
Weakest fields: affiliate license of the licensor (0), IP ownership assignment (17),
post-termination services (18), volume restriction (24).
<!-- RESULTS:END -->

## Output format

Given a contract excerpt of up to about 6,000 characters, the model returns a compact JSON
object containing only the fields it found. Values are lists of strings.

```json
{"document_name": ["SUPPLY CONTRACT"], "insurance": ["To be covered by the Seller for 110% invoice value against All Risks and War Risk."]}
```

Field keys: `document_name`, `parties`, `agreement_date`, `effective_date`, `expiration_date`,
`renewal_term`, `notice_period_to_terminate_renewal`, `governing_law`, `most_favored_nation`,
`non_compete`, `exclusivity`, `no_solicit_of_customers`, `competitive_restriction_exception`,
`no_solicit_of_employees`, `non_disparagement`, `termination_for_convenience`, `rofr_rofo_rofn`,
`change_of_control`, `anti_assignment`, `revenue_profit_sharing`, `price_restrictions`,
`minimum_commitment`, `volume_restriction`, `ip_ownership_assignment`, `joint_ip_ownership`,
`license_grant`, `non_transferable_license`, `affiliate_license_licensor`,
`affiliate_license_licensee`, `unlimited_all_you_can_eat_license`,
`irrevocable_or_perpetual_license`, `source_code_escrow`, `post_termination_services`,
`audit_rights`, `uncapped_liability`, `cap_on_liability`, `liquidated_damages`,
`warranty_duration`, `insurance`, `covenant_not_to_sue`, `third_party_beneficiary`.

## How to use

The model was trained with a fixed system prompt that lists the field definitions. The
simplest way to get identical inputs, long-document chunking, grammar-constrained decoding
and merging is the project package:

```python
from contract_extractor.inference.base import ChunkedExtractor
from contract_extractor.inference.hf_backend import HFBackend

backend = HFBackend("<this-repo-id>", constrained=True)
extractor = ChunkedExtractor(backend, chunk_size=6000, chunk_overlap=600, batch_size=8)
result = extractor.extract(open("contract.txt").read())
print(result.extraction)
```

With `transformers` only, for one excerpt:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from contract_extractor.prompts import build_messages  # or copy SYSTEM_PROMPT from the repository

tokenizer = AutoTokenizer.from_pretrained("<this-repo-id>")
model = AutoModelForCausalLM.from_pretrained("<this-repo-id>", dtype="auto", device_map="auto")
prompt = tokenizer.apply_chat_template(build_messages(excerpt), tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
output = model.generate(**inputs, max_new_tokens=2048, do_sample=False, repetition_penalty=1.0)
print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Use greedy decoding without repetition penalty, as set in the bundled generation config: the
outputs copy the input text, and a repetition penalty degrades copying.

## Training

| | |
|---|---|
| Data | CUAD v1, 367 training contracts, 41 held out for validation (official train split) |
| Examples | 3,402 windows of 6,000 characters with 600 characters of overlap, 21% without any clause |
| Method | QLoRA (4-bit NF4 base), LoRA r=32, alpha=32, all linear projections |
| Framework | Unsloth 2026.10.9, TRL 1.15.0, transformers 5.17.0 |
| Optimisation | AdamW 8-bit, learning rate 2e-4, cosine schedule, 3% warmup, effective batch 16, 2 epochs scheduled |
| Selected checkpoint | step 180 of 426 (0.85 epoch), lowest validation loss |
| Loss | completion (JSON) tokens only |
| Sequence length | 4,096 tokens |
| Hardware | one NVIDIA RTX 5080 (16 GB), about 50 minutes |

## Evaluation

The full protocol, per-field scores and the constrained decoding ablation are in the project
repository.

## Limitations

* Trained on English contracts filed with the US SEC. Other languages, jurisdictions and
  document types are out of distribution.
* Spans are verbatim and not normalized (dates are returned as written).
* Clause categories with few training examples (for example *Source Code Escrow* or
  *Unlimited/All-You-Can-Eat License*) are less reliable.
* The output is an aid for contract review, not legal advice.

## License

Llama 3.2 Community License, inherited from Llama-3.2-3B-Instruct
([license](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/blob/main/LICENSE.txt),
[acceptable use policy](https://www.llama.com/llama3_2/use-policy)). Training data: CUAD,
CC BY 4.0, The Atticus Project.

## Citation

```bibtex
@article{hendrycks2021cuad,
  title   = {CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review},
  author  = {Hendrycks, Dan and Burns, Collin and Chen, Anya and Ball, Spencer},
  journal = {NeurIPS Datasets and Benchmarks},
  year    = {2021}
}
```
