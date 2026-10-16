PYTHON ?= .venv/bin/python
export PYTHONUNBUFFERED := 1
LOGS := outputs/logs

.PHONY: setup data train-qwen train-llama train-bert merge eval-all report test lint demo

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e .

data:
	$(PYTHON) scripts/prepare_data.py --config configs/data.yaml

train-qwen:
	@mkdir -p $(LOGS)
	$(PYTHON) scripts/train_sft.py --config configs/train/qwen2.5-1.5b.yaml 2>&1 | tee $(LOGS)/train_qwen2.5-1.5b.log

train-llama:
	@mkdir -p $(LOGS)
	$(PYTHON) scripts/train_sft.py --config configs/train/llama3.2-3b.yaml 2>&1 | tee $(LOGS)/train_llama3.2-3b.log

train-bert:
	@mkdir -p $(LOGS)
	$(PYTHON) scripts/train_ner.py --config configs/train/bert-ner.yaml 2>&1 | tee $(LOGS)/train_bert_ner.log

merge:
	$(PYTHON) scripts/merge_adapter.py --base-model unsloth/Qwen2.5-1.5B-Instruct \
		--adapter outputs/qwen2.5-1.5b-cuad/adapter --output outputs/qwen2.5-1.5b-cuad/merged
	$(PYTHON) scripts/merge_adapter.py --base-model unsloth/Llama-3.2-3B-Instruct \
		--adapter outputs/llama3.2-3b-cuad/adapter --output outputs/llama3.2-3b-cuad/merged

eval-all:
	$(PYTHON) scripts/evaluate.py --systems qwen2.5-1.5b-cuad legal-bert-ner llama3.2-3b-cuad \
		qwen2.5-1.5b-zero-shot llama3.2-3b-zero-shot
	$(PYTHON) scripts/evaluate.py --systems qwen2.5-1.5b-cuad-unconstrained --max-documents 30

report:
	$(PYTHON) scripts/make_report.py

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src scripts tests app
	$(PYTHON) -m ruff format --check src scripts tests app

demo:
	PYTHONPATH=src $(PYTHON) app/app.py
