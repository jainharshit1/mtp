#!/usr/bin/env python3
"""Compute the DINOv2 embeddings used by grouping and thinning, and cache them by file path.

Optional step before 01_ingest_and_unify.py (pass the cache with --embed-cache). Use it where
computing embeddings inside the pipeline is slow or unstable (it crashed on a Windows CPU machine).
Works in chunks and resumes from a partial cache, so an interrupted run loses at most one chunk.

    python scripts/00_embed_images.py --config configs/datasets.yaml --out data/processed/embed_cache.npz
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dedup import EMBED_MODEL, compute_embeddings  # noqa: E402
from src.data.ingest import ingest_all  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def load_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=False)
    return dict(zip(data["paths"].tolist(), data["emb"]))


def save_cache(path: Path, cache: dict):
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, paths=np.array(list(cache)), emb=np.stack(list(cache.values())).astype(np.float32))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/datasets.yaml")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--embed-model", default=EMBED_MODEL)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    paths = sorted({img["file_name"] for ds in ingest_all(str(Path(args.config).resolve())) for img in ds["images"]})
    cache = load_cache(args.out)
    todo = [p for p in paths if p not in cache]
    logger.info(f"{len(paths)} images, {len(cache)} cached, {len(todo)} to embed")

    for start in range(0, len(todo), args.chunk):
        chunk = todo[start:start + args.chunk]
        emb, valid = compute_embeddings(chunk, args.embed_model, args.device, workers=args.workers)
        bad = [p for p, ok in zip(chunk, valid) if not ok]
        if bad:
            sys.exit(f"{len(bad)} unreadable images (first: {bad[0]})")
        cache.update(zip(chunk, emb.numpy()))
        save_cache(args.out, cache)
        logger.info(f"  {min(start + args.chunk, len(todo))}/{len(todo)} embedded")

    logger.info(f"Cache: {args.out} ({len(cache)} images)")


if __name__ == "__main__":
    main()
