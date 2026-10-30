#!/usr/bin/env python3
"""Sampling check that no val image has a copy or near-copy in its fold's training set.

Val is a slice of the training datasets, so leakage between train and val makes val mAP (best
checkpoint, early stopping) reward memorising. This reads only the fold files (no group ids, no
build state) and, per fold:
  - samples --sample val images; for each, its --k most similar training images of ANY dataset
    (DINOv2 cosine), and photo-matches every pair (SIFT + RANSAC, mirrored too): >= --min-inliers
    inliers = same photo
  - over ALL val images: the highest DINOv2 cosine to any training image (the build drops
    training images >= 0.80, so nothing should be there) and near-duplicates inside val (>= 0.95)
Writes per fold a sheet of the sampled pairs with the most inliers and of the most similar val ->
train pairs (look at them: look-alike surfaces also match), and a CSV of every matched pair.
Exit code 1 if any sampled val image has a same-photo match in training.

    python scripts/check_val_leakage.py --odvg-dir build/odvg_abs --embed-cache build/embed_cache.npz \
        --out build/val_check --known build/fold_geometric_pairs.csv build/geometric_links.csv
"""
import argparse
import csv
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))


def match(pair):
    from geometric_match import inliers
    a, b, c = pair
    return a, b, c, inliers(a, b)


def files(path: Path) -> list[str]:
    return [json.loads(line)["filename"] for line in open(path) if line.strip()]


def load_known(paths) -> dict:
    known = {}
    for p in paths or []:
        if Path(p).exists():
            for r in csv.DictReader(open(p)):
                known[frozenset((r["path_a"], r["path_b"]))] = int(r["inliers"])
    return known


def sheet(pairs, title, path, t=260):
    if not pairs:
        return
    im = Image.new("RGB", (2 * t + 10, len(pairs) * (t + 18) + 22), "white")
    d = ImageDraw.Draw(im)
    d.text((4, 4), title, fill="black")
    for r, (a, b, c, n) in enumerate(pairs):
        for col, p in enumerate((a, b)):
            x = Image.open(p).convert("RGB")
            x.thumbnail((t, t))
            im.paste(x, (col * (t + 10), 22 + r * (t + 18)))
        d.text((2, 22 + r * (t + 18) + t + 2),
               f"val {Path(a).parent.parent.name}/{Path(a).name[:28]}  |  train {Path(b).parent.parent.name}/{Path(b).name[:28]}"
               f"  cos {c:.3f}  inliers {n}", fill="black")
    im.save(path, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--odvg-dir", required=True, type=Path, help="fold_*/train.jsonl, val.jsonl (absolute paths)")
    ap.add_argument("--embed-cache", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--folds", type=int, nargs="*")
    ap.add_argument("--sample", type=int, default=200, help="val images photo-matched per fold")
    ap.add_argument("--k", type=int, default=10, help="most similar training images matched per val image")
    ap.add_argument("--min-cos", type=float, default=0.30)
    ap.add_argument("--min-inliers", type=int, default=25)
    ap.add_argument("--similar-inliers", type=int, default=None,
                    help="fewer than --min-inliers but at least this many also counts as the same photo when "
                         "the DINOv2 cosine is >= --similar-cos (same rule as fold_geometric_filter.py)")
    ap.add_argument("--similar-cos", type=float, default=0.60)
    ap.add_argument("--known", nargs="*", help="CSV(s) of pairs already matched (path_a, path_b, inliers)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    cache = np.load(args.embed_cache)
    row = {p: k for k, p in enumerate(cache["paths"].tolist())}
    emb = cache["emb"]
    known = load_known(args.known)
    fold_dirs = sorted(args.odvg_dir.glob("fold_*"), key=lambda d: int(d.name[5:]))
    if args.folds:
        fold_dirs = [d for d in fold_dirs if int(d.name[5:]) in args.folds]

    failed = False
    with open(args.out / "val_train_pairs.csv", "w", newline="") as fcsv, Pool(args.workers) as pool:
        w = csv.writer(fcsv)
        w.writerow(["fold", "val", "train", "cosine", "inliers"])
        for fd in fold_dirs:
            fid = int(fd.name[5:])
            train, val = files(fd / "train.jsonl"), files(fd / "val.jsonl")
            et, ev = emb[[row[f] for f in train]], emb[[row[f] for f in val]]

            # All val images: nearest training image, and near-duplicates inside val.
            nearest = np.concatenate([(ev[s:s + 1024] @ et.T).max(axis=1) for s in range(0, len(val), 1024)])
            vv = ev @ ev.T
            np.fill_diagonal(vv, -1)
            dup_val = int((np.triu(vv) >= 0.95).sum())

            # Sample: photo-match each sampled val image against its k most similar training images.
            rng = random.Random(args.seed + fid)
            sample = rng.sample(range(len(val)), min(args.sample, len(val)))
            pairs = []
            for q in sample:
                sim = et @ ev[q]
                for j in np.argsort(-sim)[:args.k]:
                    if sim[j] >= args.min_cos:
                        pairs.append((val[q], train[j], float(sim[j])))
            todo = [p for p in pairs if frozenset(p[:2]) not in known]
            done = {frozenset((a, b)): n for a, b, _, n in pool.imap(match, todo, chunksize=8)}
            results = [(a, b, c, known.get(frozenset((a, b)), done.get(frozenset((a, b))))) for a, b, c in pairs]
            for a, b, c, n in results:
                w.writerow([fid, a, b, f"{c:.4f}", n])

            same = [r for r in results if r[3] >= args.min_inliers or (
                args.similar_inliers is not None and r[3] >= args.similar_inliers and r[2] >= args.similar_cos)]
            hit = len({r[0] for r in same})
            failed |= bool(same)
            print(f"fold {fid}: val {len(val)}, train {len(train)} | all val: nearest-train cosine max "
                  f"{nearest.max():.3f}, >= 0.80: {int((nearest >= 0.80).sum())}, >= 0.70: {int((nearest >= 0.70).sum())}; "
                  f"val-val pairs >= 0.95: {dup_val} | sample {len(sample)} val x {args.k} nearest train "
                  f"({len(pairs)} pairs, {len(todo)} newly matched): same photo (>= {args.min_inliers} inliers"
                  f"{f', or >= {args.similar_inliers} at cosine >= {args.similar_cos}' if args.similar_inliers else ''}) "
                  f"for {hit} val images ({100 * hit / max(1, len(sample)):.1f}%)", flush=True)
            sheet(sorted(results, key=lambda r: -r[3])[:12],
                  f"fold {fid}: sampled val -> train pairs with the most inliers (same photo >= {args.min_inliers})",
                  args.out / f"fold_{fid}_most_inliers.jpg")
            top = np.argsort(-nearest)[:12]
            sheet([(val[i], train[int(np.argmax(et @ ev[i]))], float(nearest[i]),
                    known.get(frozenset((val[i], train[int(np.argmax(et @ ev[i]))])), -1)) for i in top],
                  f"fold {fid}: all val images, the 12 most similar val -> train pairs (inliers -1 = not matched)",
                  args.out / f"fold_{fid}_most_similar.jpg")
    print("VAL CHECK FAILED: same-photo matches between val and train" if failed else "VAL CHECK PASSED")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
