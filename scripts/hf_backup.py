#!/usr/bin/env python3
"""Back up everything that is NOT in git to a private Hugging Face dataset repo.

Usage:
    hf auth login                                   # once, token needs write access
    python scripts/hf_backup.py --repo USER/mtp-backup            # everything
    python scripts/hf_backup.py --repo USER/mtp-backup --only checkpoints   # re-run during training

Groups: raw (8 image datasets), data (data/ folds + metadata), weights, checkpoints (outputs/).
See RESTORE.md for how to put the files back.
"""
import argparse
import shutil
import tarfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT.parent / "hf_staging"

RAW = [
    "AGDD",
    "Aircraft-AI-Dataset-4",
    "Aircraft-Defect-Detection-6",
    "Aircraft-Defect-Detection-closeup",
    "Aircraft-Defect-Detection-fuselage",
    "Aircraft-Defect-Detection-korean-aerospace",
    "Aircraft-Defect-Detection-sydney",
    "DATASET/RFDETR_DATASET",
]


def tar_dir(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dst, "w") as tf:  # images are already compressed -> plain tar
        tf.add(src, arcname=str(src.relative_to(ROOT)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. USER/mtp-backup (created private)")
    ap.add_argument("--only", choices=["raw", "data", "weights", "checkpoints"])
    args = ap.parse_args()
    want = lambda g: args.only in (None, g)

    shutil.rmtree(STAGE, ignore_errors=True)
    STAGE.mkdir(parents=True)

    if want("raw"):
        for d in RAW:
            tar_dir(ROOT / d, STAGE / "raw" / (d.replace("/", "__") + ".tar"))
    if want("data"):
        tar_dir(ROOT / "data", STAGE / "data.tar")
    if want("weights"):
        (STAGE / "weights").mkdir()
        shutil.copy2(ROOT / "weights/groundingdino_swint_ogc.pth", STAGE / "weights")
    if want("checkpoints"):
        # a plain copy is fine; the trainer writes checkpoints atomically (tmp + rename)
        shutil.copytree(ROOT / "outputs", STAGE / "outputs",
                        ignore=shutil.ignore_patterns(".training.lock", ".training.pid"))

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(folder_path=str(STAGE), repo_id=args.repo, repo_type="dataset",
                      commit_message=f"backup ({args.only or 'all'})")
    shutil.rmtree(STAGE)
    print(f"Done: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
