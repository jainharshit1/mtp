#!/usr/bin/env python3
"""Rebuild train/val ODVG files so that val shares no pHash group with train.

Test sets are untouched. For each fold the pool (old train + old val) is re-split
by whole groups. Fold 0 can instead use --clean-val-only: train stays EXACTLY as it
is (the run in progress has already seen it) and val becomes the old val images whose
group never appears in train.

    python scripts/resplit_val_by_group.py --out data/odvg_v2 --folds 1 2 3 4 5 6 7
    python scripts/resplit_val_by_group.py --out data/odvg_v2 --folds 0 --clean-val-only
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.coco_to_odvg import convert_fold_to_odvg  # noqa: E402
from src.data.lodo_splits import split_val_by_group  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clean-val-only", action="store_true")
    args = ap.parse_args()

    proc = ROOT / "data" / "processed"
    uni = json.load(open(proc / "unified_annotations.json"))
    images, anns = uni["images"], uni["annotations"]
    ac = {int(k): v for k, v in json.load(open(proc / "annotated_classes.json")).items()}
    folds = json.load(open(proc / "lodo_folds.json"))
    g = {i["id"]: i["group_id"] for i in images}

    for fold in folds:
        fid = fold["fold_id"]
        if fid not in args.folds:
            continue
        new = dict(fold)
        if args.clean_val_only:
            train_groups = {g[i] for i in fold["train_image_ids"]}
            new["val_image_ids"] = [i for i in fold["val_image_ids"] if g[i] not in train_groups]
        else:
            pool = fold["train_image_ids"] + fold["val_image_ids"]
            rng = random.Random(args.seed + fid)
            new["train_image_ids"], new["val_image_ids"] = split_val_by_group(
                pool, g, args.val_fraction, rng)

        tg = {g[i] for i in new["train_image_ids"]}
        leak = sum(g[i] in tg for i in new["val_image_ids"])
        assert leak == 0, f"fold {fid}: {leak} val images share a group with train"
        assert not set(new["test_image_ids"]) & (set(new["train_image_ids"]) | set(new["val_image_ids"]))
        print(f"fold {fid} ({fold['held_out_dataset']}): train={len(new['train_image_ids'])} "
              f"val={len(new['val_image_ids'])} test={len(new['test_image_ids'])} leak=0")
        convert_fold_to_odvg(new, images, anns, ac, str(Path(args.out) / f"fold_{fid}"))


if __name__ == "__main__":
    main()
