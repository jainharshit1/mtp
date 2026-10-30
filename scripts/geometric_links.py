#!/usr/bin/env python3
"""Find copies across datasets that no other rule links, by local-feature matching (geometric_match.py).

Candidates: for every image of the query datasets, its K most similar images (DINOv2 cosine in
[--min-cos, --max-cos)) in the target datasets that are not already in its scene group. Each
candidate pair gets a SIFT/RANSAC inlier count; results are appended to a CSV as they finish, so an
interrupted run resumes where it stopped. 01_ingest_and_unify.py --geometric-links <csv> links pairs
with at least --geometric-min-inliers (25) inliers.

Pair sets used for lodo_v3 (from a 3,406-pair sample: 95% of copies are RFDETR re-exports of sydney
and korean-aerospace photos; Defect-Detection-6, closeup and fuselage share a few more):

    python scripts/geometric_links.py --processed <build>/processed --embed-cache <build>/embed_cache.npz \
        --out <build>/geometric_links.csv --workers 2 \
        --pairs "sydney,korean-aerospace:RFDETR_DATASET" \
        --pairs "Aircraft-Defect-Detection-6,fuselage:closeup,Aircraft-Defect-Detection-6,fuselage"
"""
import argparse
import csv
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def work(pair):
    from geometric_match import inliers
    a, b, c = pair
    return a, b, c, inliers(a, b)


def candidates(imgs, emb, queries, targets, k, lo, hi):
    ds = np.array([i["source_dataset"] for i in imgs])
    grp = np.array([i["group_id"] for i in imgs])
    t_idx = np.flatnonzero(np.isin(ds, list(targets)))
    out = []
    for q in np.flatnonzero(np.isin(ds, list(queries))):
        sim = emb[t_idx] @ emb[q]
        sim[(ds[t_idx] == ds[q]) | (grp[t_idx] == grp[q])] = -1
        for j in np.argsort(-sim)[:k]:
            if lo <= sim[j] < hi:
                out.append((int(q), int(t_idx[j]), float(sim[j])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True, type=Path)
    ap.add_argument("--embed-cache", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--pairs", action="append", required=True,
                    help="'query_ds1,query_ds2:target_ds1,target_ds2' (repeatable)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--min-cos", type=float, default=0.60)
    ap.add_argument("--max-cos", type=float, default=0.90)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    imgs = json.load(open(args.processed / "unified_annotations.json"))["images"]
    cache = np.load(args.embed_cache)
    row = {p: k for k, p in enumerate(cache["paths"].tolist())}
    emb = cache["emb"][[row[i["file_name"]] for i in imgs]]

    pairs, seen = [], set()
    for spec in args.pairs:
        q, t = spec.split(":")
        for a, b, c in candidates(imgs, emb, q.split(","), t.split(","), args.k, args.min_cos, args.max_cos):
            key = (min(a, b), max(a, b))
            if key not in seen:
                seen.add(key)
                pairs.append((imgs[a]["file_name"], imgs[b]["file_name"], c))
    pairs.sort()  # by query path, so each query's features are reused from the cache

    done = set()
    if args.out.exists():
        with open(args.out) as f:
            done = {(r["path_a"], r["path_b"]) for r in csv.DictReader(f)}
    todo = [p for p in pairs if (p[0], p[1]) not in done]
    print(f"{len(pairs)} candidate pairs, {len(done)} already done, {len(todo)} to match", flush=True)

    new = not args.out.exists()
    start = time.time()
    with open(args.out, "a", newline="") as f, Pool(args.workers) as pool:
        w = csv.writer(f)
        if new:
            w.writerow(["path_a", "path_b", "cosine", "inliers"])
        for n, (a, b, c, k) in enumerate(pool.imap(work, todo, chunksize=12), 1):
            w.writerow([a, b, f"{c:.4f}", k])
            if n % 500 == 0 or n == len(todo):
                f.flush()
                rate = n / (time.time() - start)
                print(f"  {n}/{len(todo)} matched, {rate:.2f} pairs/s, ~{(len(todo) - n) / rate / 60:.0f} min left", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
