"""Publish the merged model and the Gradio demo to the Hugging Face Hub.

Requires a write token in ``HF_TOKEN`` (or a prior ``hf auth login``).

    python scripts/push_to_hub.py model --model-dir outputs/qwen2.5-1.5b-cuad/merged \
        --repo-id <user>/qwen2.5-1.5b-cuad-extractor --card model_card/qwen2.5-1.5b-cuad.md
    python scripts/push_to_hub.py space --repo-id <user>/contract-clause-extractor \
        --model-id <user>/qwen2.5-1.5b-cuad-extractor
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def push_model(api: HfApi, model_dir: Path, repo_id: str, card: Path, private: bool) -> None:
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        shutil.copytree(model_dir, staging, dirs_exist_ok=True)
        shutil.copy(card, staging / "README.md")
        api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=staging, commit_message="Upload model")
    print(f"https://huggingface.co/{repo_id}")


def push_space(api: HfApi, repo_id: str, model_id: str, hardware: str | None, private: bool) -> None:
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", private=private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        shutil.copytree(ROOT / "app", staging, dirs_exist_ok=True)
        shutil.copytree(
            ROOT / "src" / "contract_extractor",
            staging / "contract_extractor",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=staging, commit_message="Upload demo")
    api.add_space_variable(repo_id, "MODEL_ID", model_id)
    if hardware:
        api.request_space_hardware(repo_id, hardware)
    print(f"https://huggingface.co/spaces/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="target", required=True)

    model = sub.add_parser("model")
    model.add_argument("--model-dir", required=True, type=Path)
    model.add_argument("--repo-id", required=True)
    model.add_argument("--card", required=True, type=Path, help="Model card, e.g. model_card/qwen2.5-1.5b-cuad.md")
    model.add_argument("--private", action="store_true")

    space = sub.add_parser("space")
    space.add_argument("--repo-id", required=True)
    space.add_argument("--model-id", required=True)
    space.add_argument("--hardware", default=None, help="Optional Space hardware, e.g. zero-a10g or t4-small")
    space.add_argument("--private", action="store_true")

    args = parser.parse_args()
    api = HfApi()
    if args.target == "model":
        push_model(api, args.model_dir, args.repo_id, args.card, args.private)
    else:
        push_space(api, args.repo_id, args.model_id, args.hardware, args.private)


if __name__ == "__main__":
    main()
