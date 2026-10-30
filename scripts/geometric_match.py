#!/usr/bin/env python3
"""Same-photo test by local features: SIFT keypoints + ratio test + RANSAC homography inliers.

Robust to recolouring (grayscale + CLAHE), rotation, cropping and mirroring (the second image is also
tried flipped); look-alike textures rarely produce a consistent homography. Used to estimate how many
copies remain between test/val and train after a build (they score DINOv2 below the 0.80 filter when
recoloured heavily).

    python scripts/geometric_match.py --calibrate --processed <build>/processed
    python scripts/geometric_match.py --audit <build>/leakage_audit --out <dir>
"""
import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.dedup import roboflow_source  # noqa: E402

SIFT = cv2.SIFT_create(nfeatures=1500)
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_cache = {}


def features(path: str, flip: bool = False):
    key = (path, flip)
    if key not in _cache:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if flip:
            img = cv2.flip(img, 1)
        img = CLAHE.apply(img)
        _cache[key] = SIFT.detectAndCompute(img, None)
        if len(_cache) > 300:  # ~1500 SIFT descriptors per image; keep memory per process small
            _cache.pop(next(iter(_cache)))
    return _cache[key]


def inliers(a: str, b: str) -> int:
    """Best RANSAC homography inlier count of a vs b and a vs mirrored b."""
    ka, da = features(a)
    if da is None or len(ka) < 8:
        return 0
    best = 0
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    for flip in (False, True):
        kb, db = features(b, flip)
        if db is None or len(kb) < 8:
            continue
        good = [m for m, n in (p for p in matcher.knnMatch(da, db, k=2) if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < 8:
            continue
        src = np.float32([ka[m.queryIdx].pt for m in good])
        dst = np.float32([kb[m.trainIdx].pt for m in good])
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if mask is not None:
            best = max(best, int(mask.sum()))
    return best


def calibrate(processed: Path, n: int):
    imgs = json.load(open(processed / "unified_annotations.json"))["images"]
    by = defaultdict(lambda: defaultdict(list))
    for i in imgs:
        s = roboflow_source(i["file_name"])
        if s:
            by[i["source_dataset"]][s].append(i["file_name"])
    rng = random.Random(0)

    def pairs(a, b):
        common = sorted(set(by[a]) & set(by[b]))
        return [(by[a][s][0], by[b][s][0]) for s in rng.sample(common, min(n, len(common)))]

    for label, ps in [("same photo (Defect-Detection-6 / closeup, same name)", pairs("Aircraft-Defect-Detection-6", "closeup")),
                      ("different photos (sydney / closeup, same number)", pairs("sydney", "closeup"))]:
        v = np.array([inliers(a, b) for a, b in ps])
        print(f"{label}: n={len(v)}  inliers p10/p50/p90 = {np.percentile(v, [10, 50, 90]).round(0).tolist()}  "
              f">=15: {(v >= 15).mean():.0%}  >=25: {(v >= 25).mean():.0%}  >=40: {(v >= 40).mean():.0%}")


def audit(audit_dir: Path, out: Path, lo: float, hi: float, per_fold: int, min_inliers: int):
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    rows_out = []
    for csv_path in sorted(audit_dir.glob("fold_*_nearest_train.csv")):
        rows = [r for r in csv.DictReader(open(csv_path)) if lo <= float(r["cosine"]) < hi]
        sample = rng.sample(rows, min(per_fold, len(rows)))
        hits = 0
        for r in sample:
            q = r.get("test_image") or r.get("val_image")
            k = inliers(q, r["nearest_train_image"])
            hits += k >= min_inliers
            rows_out.append([csv_path.stem, q, r["nearest_train_image"], r["cosine"], k])
        print(f"{csv_path.stem}: {len(rows)} in [{lo}, {hi}), sampled {len(sample)}, "
              f"same photo (>= {min_inliers} inliers): {hits} ({hits / max(len(sample), 1):.0%})")
    with open(out / "geometric_matches.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "query", "nearest_train", "cosine", "inliers"])
        w.writerows(rows_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--processed", type=Path)
    ap.add_argument("--audit", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--lo", type=float, default=0.60)
    ap.add_argument("--hi", type=float, default=0.80)
    ap.add_argument("--per-fold", type=int, default=100)
    ap.add_argument("--min-inliers", type=int, default=25)
    args = ap.parse_args()
    if args.calibrate:
        calibrate(args.processed, args.n)
    if args.audit:
        audit(args.audit, args.out, args.lo, args.hi, args.per_fold, args.min_inliers)


if __name__ == "__main__":
    main()
