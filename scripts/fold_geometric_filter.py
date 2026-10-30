#!/usr/bin/env python3
"""Per-fold photo-matching filter: drop training images that are copies of a test or val image.

Grouping links each sydney/korean-aerospace photo to its 3 most similar RFDETR images only, but
RFDETR holds families of 20+ augmented re-exports of one photo, so some copies stay unlinked and
reach training (fold 4 audit: 11% of sampled sydney test images). This checks, per fold, only the
pairs that can leak: each test/val image of the query datasets against its K most similar training
images of the target datasets in that fold (DINOv2 cosine in [--min-cos, --max-cos)). A training
image with >= --min-inliers SIFT/RANSAC inliers to any of them is dropped from that fold's training
set, together with the rest of its sub-scene (other copies of the same photo). Nothing is linked,
so groups cannot chain.

Pair sets used for lodo_v3:
    --pairs "sydney,korean-aerospace:RFDETR_DATASET:20"   folds 1, 2, 4, 5 (originals held out)
    --pairs "RFDETR_DATASET:sydney,korean-aerospace:5"    fold 0 (RFDETR held out; one original each)
    --pairs "*:*:K"    (2026-10-29) every test/val image against its K most similar training images
                       in each dataset, its own included (a sydney test image had an inverted copy
                       in korean; val is a slice of the training datasets)
Run it on folds built without --fold-drops: the list is computed for all pair sets at once. Test
images are checked against train and val: a copy is dropped with its sub-scene ("drops"). Val
images are checked against train: a val image with a copy (>= --val-min-inliers) is listed in
"val_to_train" and its scene group moves to train. Moved groups can hold new copies, so run it again with --merge on the folds
built with the list, until nothing new is found.

Results are appended to --out-pairs as they finish (resumable); pairs already in --known
(geometric_links.csv) are reused. The drop list (--out-drops) is read by
01_ingest_and_unify.py --fold-drops; the folds are deterministic, so a rebuild reproduces the same
train sets before the drop.
"""
import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from geometric_links import work  # noqa: E402


def candidates(fold, imgs, emb, queries, targets, k, lo, hi, splits=("test", "val")):
    """'*' as queries: every test/val image; '*' as targets: the k most similar training images of
    each dataset, separately per dataset. That includes the query's own dataset: val is a slice of
    the training datasets, so a val image's likeliest copies (other frames, re-exports in another
    scene group) are in its own dataset. A test image's own dataset is never in training.
    '=' as targets: the k most similar training images of the query's own dataset only.
    '+' as targets: the k most similar training images of all datasets together (one pool)."""
    every_q, own, pooled = queries == {"*"}, targets == {"="}, targets == {"+"}
    every_t = targets == {"*"} or pooled
    out = []
    # A test image is checked against train and val (a val group can move to train); a val image
    # against train.
    for split, others in (("test", ("train", "val")), ("val", ("train",))):
        if split not in splits:
            continue
        held = [i for i in fold[f"{split}_image_ids"] if every_q or imgs[i]["source_dataset"] in queries]
        train = np.array([i for s in others for i in fold[f"{s}_image_ids"]
                          if every_t or own or imgs[i]["source_dataset"] in targets])
        if not held or not len(train):
            continue
        train_ds = np.array([imgs[i]["source_dataset"] for i in train])
        pools = ({ds: np.flatnonzero(train_ds == ds) for ds in np.unique(train_ds)} if (every_t or own) and not pooled
                 else {None: np.arange(len(train))})
        for q in held:
            sim = emb[train] @ emb[q]
            for ds, rows in pools.items():
                if own and ds != imgs[q]["source_dataset"]:
                    continue
                for j in rows[np.argsort(-sim[rows])[:k]]:
                    if lo <= sim[j] < hi:
                        out.append((int(q), int(train[j]), float(sim[j])))
    return out


def load_done(path):
    done = {}
    if path and Path(path).exists():
        with open(path) as f:
            for r in csv.DictReader(f):
                done[frozenset((r["path_a"], r["path_b"]))] = int(r["inliers"])
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True, type=Path)
    ap.add_argument("--embed-cache", required=True, type=Path)
    ap.add_argument("--pairs", action="append", required=True,
                    help="'query_ds1,query_ds2:train_ds1,train_ds2:K[:MIN_COS[:MAX_COS]]' (repeatable); the band "
                         "overrides --min-cos/--max-cos for this spec. '*:+:10:0:0.45' = among the 10 most similar "
                         "training images of any dataset, those below cosine 0.45 (heavily filtered copies, e.g. "
                         "embossed, score 0.36-0.44: val check 2026-10-30). Not above 0.45: K=10 there matched mostly "
                         "look-alike fuselages at >= 25 inliers (visual review 2026-10-30)")
    ap.add_argument("--val-similar-inliers", type=int, default=None,
                    help="a val pair with fewer than --val-min-inliers but at least this many inliers is also "
                         "a copy if its DINOv2 cosine is >= --val-similar-cos (review 2026-10-29: 40-69 "
                         "inliers were mostly the same photo at cosine >= 0.60, mostly different below)")
    ap.add_argument("--val-similar-cos", type=float, default=0.60)
    ap.add_argument("--val-pairs", action="append", default=[],
                    help="like --pairs, for val images only; '*:=:20' = the 20 most similar training images "
                         "of the val image's own dataset (RFDETR re-export families: copies ranked 6-10 "
                         "were missed with K=5)")
    ap.add_argument("--known", type=Path, default=None, help="geometric_links.csv: pairs already matched")
    ap.add_argument("--out-pairs", required=True, type=Path)
    ap.add_argument("--out-drops", required=True, type=Path)
    ap.add_argument("--min-cos", type=float, default=0.55)
    ap.add_argument("--max-cos", type=float, default=0.80, help="training images >= 0.80 are already dropped")
    ap.add_argument("--min-inliers", type=int, default=25)
    ap.add_argument("--val-min-inliers", type=int, default=None,
                    help="same-photo threshold for val -> train matches (default --min-inliers); val pairs "
                         "with 25-70 inliers were look-alike surfaces in a visual review (2026-10-29)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true", help="only count the pairs")
    ap.add_argument("--merge", action="store_true",
                    help="union with the existing --out-drops (a second pass on folds built with it)")
    args = ap.parse_args()

    imgs = json.load(open(args.processed / "unified_annotations.json"))["images"]
    assert all(img["id"] == n for n, img in enumerate(imgs))
    folds = json.load(open(args.processed / "lodo_folds.json"))
    cache = np.load(args.embed_cache)
    row = {p: k for k, p in enumerate(cache["paths"].tolist())}
    emb = cache["emb"][[row[i["file_name"]] for i in imgs]]

    fold_pairs = defaultdict(list)  # fold_id -> [(held_id, train_id)]
    unique = {}
    specs = [(s, ("test", "val")) for s in args.pairs] + [(s, ("val",)) for s in args.val_pairs]
    for spec, splits in specs:
        q, t, k, *band = spec.split(":")
        lo = float(band[0]) if band else args.min_cos
        hi = float(band[1]) if len(band) > 1 else args.max_cos
        for fold in folds:
            found = candidates(fold, imgs, emb, set(q.split(",")), set(t.split(",")), int(k),
                               lo, hi, splits)
            fold_pairs[fold["fold_id"]].extend((a, b) for a, b, _ in found)
            for a, b, c in found:
                unique.setdefault(frozenset((imgs[a]["file_name"], imgs[b]["file_name"])),
                                  (imgs[a]["file_name"], imgs[b]["file_name"], c))
    for fid in sorted(fold_pairs):
        print(f"fold {fid}: {len(fold_pairs[fid])} candidate pairs", flush=True)

    known = load_done(args.known)
    done = load_done(args.out_pairs)
    todo = sorted(v for key, v in unique.items() if key not in known and key not in done)
    print(f"{len(unique)} unique pairs: {sum(k in known for k in unique)} already in --known, "
          f"{sum(k in done for k in unique)} already in --out-pairs, {len(todo)} to match", flush=True)
    if args.dry_run:
        return

    new = not args.out_pairs.exists()
    start = time.time()
    with open(args.out_pairs, "a", newline="") as f, Pool(args.workers) as pool:
        w = csv.writer(f)
        if new:
            w.writerow(["path_a", "path_b", "cosine", "inliers"])
        for n, (a, b, c, k) in enumerate(pool.imap(work, todo, chunksize=12), 1):
            w.writerow([a, b, f"{c:.4f}", k])
            done[frozenset((a, b))] = k
            if n % 500 == 0 or n == len(todo):
                f.flush()
                rate = n / (time.time() - start)
                print(f"  {n}/{len(todo)} matched, {rate:.2f} pairs/s, ~{(len(todo) - n) / rate / 60:.0f} min left",
                      flush=True)

    # Copies of a test image: the copy (train or val) is dropped with the rest of its sub-scene.
    # A val image with a copy in train: listed in val_to_train; its scene group moves to train.
    inl = {**known, **done}
    val_min = args.val_min_inliers or args.min_inliers

    def val_copy(a, b):
        n = inl[frozenset((imgs[a]["file_name"], imgs[b]["file_name"]))]
        return n >= val_min or (args.val_similar_inliers is not None and n >= args.val_similar_inliers
                                and float(emb[a] @ emb[b]) >= args.val_similar_cos)
    drops, val_to_train, summary = {}, {}, {}
    for fold in folds:
        fid = fold["fold_id"]
        test = set(fold["test_image_ids"])
        hits = [(a, b) for a, b in fold_pairs[fid]
                if inl[frozenset((imgs[a]["file_name"], imgs[b]["file_name"]))] >= args.min_inliers]
        test_hit = {a for a, b in hits if a in test}
        matched = {b for a, b in hits if a in test}
        val_hit = {a for a, b in fold_pairs[fid] if a not in test and val_copy(a, b)}
        subs = {imgs[i]["subscene_id"] for i in matched}
        dropped = sorted(i for i in fold["train_image_ids"] + fold["val_image_ids"] if imgs[i]["subscene_id"] in subs)
        drops[str(fid)] = [imgs[i]["file_name"] for i in dropped]
        val_to_train[str(fid)] = sorted(imgs[i]["file_name"] for i in val_hit)
        n_test = len({a for a, _ in fold_pairs[fid] if a in test})
        n_val = len({a for a, _ in fold_pairs[fid] if a not in test})
        summary[fid] = {"test_images_checked": n_test, "test_images_with_copy": len(test_hit),
                        "copies_matched": len(matched), "dropped_with_subscene": len(dropped),
                        "val_images_checked": n_val, "val_images_moved_to_train": len(val_hit)}
        print(f"fold {fid}: {len(test_hit)}/{n_test} test images had a copy in train/val; {len(matched)} copies, "
              f"{len(dropped)} dropped with their sub-scenes; {len(val_hit)}/{n_val} val images had a copy in "
              f"train (their groups move to train)", flush=True)

    if args.merge and args.out_drops.exists():
        # Folds rebuilt with an earlier list no longer contain what it removed: keep it.
        old = json.load(open(args.out_drops))
        for key, new in (("drops", drops), ("val_to_train", val_to_train)):
            for fid, names in old.get(key, {}).items():
                new[fid] = sorted(set(new.get(fid, [])) | set(names))
        print("merged with the existing list: " + ", ".join(
            f"fold {fid}: {len(drops[fid])} dropped, {len(val_to_train[fid])} val moved" for fid in sorted(drops)),
            flush=True)
    with open(args.out_drops, "w") as f:
        json.dump({"min_inliers": args.min_inliers, "val_min_inliers": val_min, "val_similar_inliers": args.val_similar_inliers,
                   "val_similar_cos": args.val_similar_cos, "pairs": args.pairs, "val_pairs": args.val_pairs, "summary": summary, "drops": drops,
                   "val_to_train": val_to_train}, f, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
