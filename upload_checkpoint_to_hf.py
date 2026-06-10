#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, login


# Local checkpoint folder to upload. Override with --folder-path or the
# HF_UPLOAD_FOLDER environment variable.
DEFAULT_CHECKPOINT_DIR = os.environ.get("HF_UPLOAD_FOLDER", "./checkpoint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a local checkpoint directory to the Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target Hub repo in the form 'username_or_org/repo_name'.",
    )
    parser.add_argument(
        "--folder-path",
        default=DEFAULT_CHECKPOINT_DIR,
        help=f"Local checkpoint folder to upload. Defaults to: {DEFAULT_CHECKPOINT_DIR}",
    )
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=["model", "dataset", "space"],
        help="Hub repo type.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token. Defaults to HF_TOKEN if set.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private.",
    )
    parser.add_argument(
        "--single-commit",
        action="store_true",
        help="Use upload_folder instead of upload_large_folder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folder = Path(args.folder_path).expanduser().resolve()

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder}")

    if args.token:
        login(token=args.token)
    else:
        login()

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
    )

    if args.single_commit:
        api.upload_folder(
            repo_id=args.repo_id,
            folder_path=str(folder),
            repo_type=args.repo_type,
        )
    else:
        api.upload_large_folder(
            repo_id=args.repo_id,
            folder_path=str(folder),
            repo_type=args.repo_type,
        )

    print(f"Upload completed: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
