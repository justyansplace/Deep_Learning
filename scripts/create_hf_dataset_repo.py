"""Optional helper for instructors.

Usage:
    HF_TOKEN=... python scripts/create_hf_dataset_repo.py --repo-id ORG/vlm-math-hw-assets

This script intentionally does not upload hidden labels by default.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="e.g. my-course/vlm-math-hw-assets")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--local-dir", default="assets/toy_math_vqa")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, upload_folder
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub first: pip install huggingface_hub") from exc

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN with write access")

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=args.local_dir,
        path_in_repo="toy_math_vqa",
        token=token,
    )
    print(f"Uploaded {args.local_dir} to dataset repo {args.repo_id}")


if __name__ == "__main__":
    main()
