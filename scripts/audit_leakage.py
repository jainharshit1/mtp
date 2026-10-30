#!/usr/bin/env python3
"""Independent leakage audit of finished LODO folds.

Reads only the fold files (train.jsonl, val.jsonl, test_annotations.json), never the
group ids, and searches for near-duplicates across splits from scratch:
  - pHash under the best of 8 flips/90-degree rotations (distance <= --phash-threshold)
  - DINOv2 cosine similarity at several thresholds
  - same Roboflow source name

For every val/test image the nearest train image is found. A contact sheet of the most
similar val->train and test->train pairs is written for each fold: look at it, since
counts above a threshold include look-alike surfaces as well as true copies.

    python scripts/audit_leakage.py --odvg-dir data/odvg --folds 0 1 --out outputs/leakage_audit
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dedup import (  # noqa: E402
    EMBED_MODEL, PHASH_THRESHOLD, compute_embeddings, compute_phashes, roboflow_source,
)


def load_fold(fold_dir: Path) -> dict[str, list[str]]:
    # Relative paths (portable build) are relative to the folder holding fold_*/.
    root = fold_dir.resolve().parent
    resolve = lambda f: f if Path(f).is_absolute() else str(root / f)
    splits = {}
    for name in ("train", "val"):
        with open(fold_dir / f"{name}.jsonl") as f:
            splits[name] = [resolve(json.loads(line)["filename"]) for line in f if line.strip()]
    splits["test"] = [resolve(img["file_name"]) for img in json.load(open(fold_dir / "test_annotations.json"))["images"]]
    return splits


def contact_sheet(pairs, path: Path, thumb: int = 200):
    rows = len(pairs)
    sheet = Image.new("RGB", (2 * thumb + 10, rows * (thumb + 8)), "white")
    draw = ImageDraw.Draw(sheet)
    for r, (query, match, cos, ph) in enumerate(pairs):
        for c, p in enumerate((query, match)):
            try:
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    im.thumbnail((thumb, thumb))
                    sheet.paste(im, (c * (thumb + 10), r * (thumb + 8)))
            except Exception:
                pass
        draw.rectangle((0, r * (thumb + 8), 150, r * (thumb + 8) + 14), fill="black")
        draw.text((3, r * (thumb + 8) + 2), f"cos {cos:.3f} pHash {ph}", fill="yellow")
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--odvg-dir", type=Path, default=ROOT / "data" / "odvg")
    ap.add_argument("--folds", type=int, nargs="+", default=None, help="default: every fold_* in --odvg-dir")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "leakage_audit")
    ap.add_argument("--embed-model", default=EMBED_MODEL)
    ap.add_argument("--cos-thresholds", type=float, nargs="+", default=[0.85, 0.90, 0.95])
    ap.add_argument("--phash-threshold", type=int, default=PHASH_THRESHOLD)
    ap.add_argument("--sheet-pairs", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--embed-cache", type=Path, default=None,
                    help="Embeddings from scripts/00_embed_images.py, keyed by the paths in the fold files")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    fold_ids = args.folds or sorted(int(d.name[5:]) for d in args.odvg_dir.glob("fold_*") if d.name[5:].isdigit())
    folds = {fid: load_fold(args.odvg_dir / f"fold_{fid}") for fid in fold_ids}
    paths = sorted({p for splits in folds.values() for files in splits.values() for p in files})
    index = {p: i for i, p in enumerate(paths)}
    print(f"Fingerprinting {len(paths)} unique images...")
    hashes, hvalid = compute_phashes(paths, args.workers)
    if args.embed_cache:
        data = np.load(args.embed_cache, allow_pickle=False)
        cache = dict(zip(data["paths"].tolist(), data["emb"]))
        evalid = np.array([p in cache for p in paths])
        emb = np.stack([cache[p] if p in cache else np.zeros(data["emb"].shape[1], np.float32) for p in paths])
    else:
        emb, evalid = compute_embeddings(paths, args.embed_model, args.device, workers=args.workers)
        emb = emb.numpy()
    unreadable = [p for p, a, b in zip(paths, hvalid, evalid) if not (a and b)]
    # An unreadable image matches nothing and would look clean, so refuse to report.
    if unreadable:
        sys.exit(f"{len(unreadable)} images could not be read (first: {unreadable[0]}). "
                 "Run from the directory the fold paths are relative to.")
    sources = [roboflow_source(p) for p in paths]

    summary = {}
    for fid, splits in folds.items():
        tr = np.array([index[p] for p in splits["train"]])
        tr_sources = {sources[i] for i in tr if sources[i]}
        fold_summary = {"n_train": len(tr)}
        for split in ("val", "test"):
            q = np.array([index[p] for p in splits[split]])
            if len(q) == 0:
                continue
            sim = emb[q] @ emb[tr].T
            best = sim.argmax(axis=1)
            best_cos = sim[np.arange(len(q)), best]
            ph = np.array([
                int(np.bitwise_count(hashes[i, 0] ^ hashes[tr]).min(axis=1).min()) if hvalid[i] else 64
                for i in q
            ])
            stats = {
                "n": len(q),
                f"phash<={args.phash_threshold}": int((ph <= args.phash_threshold).sum()),
                "same_roboflow_name": sum(1 for i in q if sources[i] and sources[i] in tr_sources),
            }
            stats.update({f"cos>={t}": int((best_cos >= t).sum()) for t in args.cos_thresholds})
            fold_summary[split] = stats

            order = np.argsort(-best_cos)
            with open(args.out / f"fold_{fid}_{split}_nearest_train.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([f"{split}_image", "nearest_train_image", "cosine", "min_dihedral_phash"])
                for k in order:
                    w.writerow([paths[q[k]], paths[tr[best[k]]], f"{best_cos[k]:.4f}", ph[k]])
            top = [(paths[q[k]], paths[tr[best[k]]], best_cos[k], ph[k]) for k in order[:args.sheet_pairs]]
            contact_sheet(top, args.out / f"fold_{fid}_{split}_top_pairs.jpg")

        summary[fid] = fold_summary
        print(f"fold {fid}: {json.dumps({k: v for k, v in fold_summary.items()})}")

    json.dump(summary, open(args.out / "summary.json", "w"), indent=2)
    print(f"Wrote CSVs, contact sheets and summary.json to {args.out}")


if __name__ == "__main__":
    main()
