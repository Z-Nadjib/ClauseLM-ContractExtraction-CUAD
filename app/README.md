---
title: Contract Clause Extractor
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Extract 41 CUAD clause types from contracts as JSON
---

# Contract Clause Extractor

Paste a commercial contract and get a JSON object with the passages that match each of the
41 [CUAD](https://www.atticusprojectai.org/cuad) clause categories (parties, dates, governing
law, termination, liability caps, licences, and so on).

The model is a Qwen2.5-1.5B-Instruct checkpoint fine-tuned with QLoRA on CUAD. Long contracts
are split into overlapping windows of 6,000 characters, each window is decoded under a JSON
Schema grammar (xgrammar), and the window results are merged into one document-level JSON.

Set the `MODEL_ID` Space variable to the Hub id of the merged model.
