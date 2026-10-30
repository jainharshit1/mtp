#!/usr/bin/env python3
"""Check a portable fold build (scripts/build_portable_folds.py) before training on it.

Checks, failing loudly on any violation:
  - every referenced image exists and matches the SHA256 in MANIFEST.json
  - test holds only the held-out dataset, val only the fold's val dataset, train neither
  - no scene group in two splits; at most --max-per-subscene / --max-test-per-subscene per sub-scene
  - every box lies inside its image, has positive size and a valid label
Writes checks/verify_report.txt (counts per fold, split, dataset and class) and
checks/overlay_<dataset>.jpg and checks/fold_<k>_<split>.jpg (random images with their boxes and
class names drawn) for a visual check.

    python scripts/verify_folds.py --folds-dir D:/MTP/mtp-data/lodo_v3 --processed D:/MTP/mtp-data/lodo_v3/build/processed \
        --raw-root D:/MTP/mtp-data/raw
"""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.taxonomy import ID_TO_CLASS, NUM_CLASSES  # noqa: E402


def font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds-dir", required=True, type=Path)
    ap.add_argument("--processed", required=True, type=Path, help="--output-dir of 01_ingest_and_unify.py")
    ap.add_argument("--raw-root", required=True, type=Path)
    ap.add_argument("--max-per-subscene", type=int, default=5)
    ap.add_argument("--max-test-per-subscene", type=int, default=2)
    ap.add_argument("--per-dataset-caps", default="",
                    help="comma-separated datasets whose copies of one photo are capped per dataset "
                         "(lodo.keep_copies_together)")
    ap.add_argument("--overlays-per-dataset", type=int, default=12)
    ap.add_argument("--samples-per-split", type=int, default=12)
    args = ap.parse_args()
    out = args.folds_dir / "checks"
    out.mkdir(exist_ok=True)
    raw_root = args.raw_root.resolve()
    errors, lines = [], []
    p = lines.append

    manifest = json.load(open(args.folds_dir / "MANIFEST.json"))
    reg = {}
    for img in json.load(open(args.processed / "unified_annotations.json"))["images"]:
        rel = "images/" + Path(img["file_name"]).resolve().relative_to(raw_root).as_posix()
        reg[rel] = img
    fold_meta = {f["fold_id"]: f for f in manifest.get("folds", [])}

    # 1. images exist and match their recorded SHA256
    bad = [r for r, h in manifest["images_sha256"].items()
           if not (args.folds_dir / r).exists()
           or hashlib.sha256((args.folds_dir / r).read_bytes()).hexdigest() != h]
    if bad:
        errors.append(f"{len(bad)} images missing or changed (first: {bad[0]})")
    p(f"Images: {len(manifest['images_sha256'])} in images/, {len(bad)} missing or changed")

    by_dataset = defaultdict(list)  # dataset -> [(rel, instances)] for overlays
    fold_samples = {}  # (fold, split) -> [(rel, instances)] for per-fold sample sheets
    rng = random.Random(0)
    for fold_dir in sorted(args.folds_dir.glob("fold_*"), key=lambda d: int(d.name[5:])):
        fid = int(fold_dir.name[5:])
        meta = fold_meta.get(fid, {})
        held, val_ds = meta.get("held_out_dataset"), meta.get("val_dataset")
        splits = {}
        for name in ("train", "val"):
            splits[name] = [json.loads(l) for l in open(fold_dir / f"{name}.jsonl") if l.strip()]
        test = json.load(open(fold_dir / "test_annotations.json"))
        anns = defaultdict(list)
        for a in test["annotations"]:
            x, y, w, h = a["bbox"]
            anns[a["image_id"]].append({"bbox": [x, y, x + w, y + h], "label": a["category_id"]})
        splits["test"] = [{"filename": i["file_name"], "width": i["width"], "height": i["height"],
                           "detection": {"instances": anns[i["id"]]}} for i in test["images"]]

        p(f"\nFold {fid}: test = {held}, val = {val_ds or 'slice of the training datasets'}")
        groups = {}
        for name, recs in splits.items():
            missing = [r["filename"] for r in recs if r["filename"] not in reg]
            if missing:
                errors.append(f"fold {fid} {name}: {len(missing)} images not in the registry")
                continue
            imgs = [reg[r["filename"]] for r in recs]
            ds = Counter(i["source_dataset"] for i in imgs)
            groups[name] = {i["group_id"] for i in imgs}
            per_ds = set(filter(None, args.per_dataset_caps.split(",")))
            subs = Counter((i["subscene_id"], i["source_dataset"]) if i["source_dataset"] in per_ds
                           else i["subscene_id"] for i in imgs)
            # Test and val are both selected with the test cap (val: whole dataset or group slice).
            held_out_like = name in ("test", "val")
            cap = args.max_test_per_subscene if held_out_like else args.max_per_subscene
            if subs and max(subs.values()) > cap:
                errors.append(f"fold {fid} {name}: {max(subs.values())} images from one sub-scene (cap {cap})")
            allowed = {"test": {held}, "val": {val_ds} if val_ds else None, "train": None}[name]
            if allowed is not None and set(ds) - allowed:
                errors.append(f"fold {fid} {name}: unexpected datasets {sorted(set(ds) - allowed)}")
            excluded = set(meta.get("excluded_from_train") or [held, val_ds])
            if name == "train" and (excluded & set(ds)):
                errors.append(f"fold {fid} train: contains held-out dataset {sorted(excluded & set(ds))}")
            if name == "val" and not val_ds and (excluded & set(ds)):
                errors.append(f"fold {fid} val: contains held-out dataset {sorted(excluded & set(ds))}")
            if name == "train" and any(not r["detection"]["instances"] for r in recs):
                errors.append(f"fold {fid} train: {sum(not r['detection']['instances'] for r in recs)} images without boxes")
            boxes = Counter()
            for r in recs:
                for inst in r["detection"]["instances"]:
                    x1, y1, x2, y2 = inst["bbox"]
                    lab = inst["label"]
                    if not (0 <= lab < NUM_CLASSES) or x2 <= x1 or y2 <= y1 or x1 < -1 or y1 < -1 \
                            or x2 > r["width"] + 1 or y2 > r["height"] + 1:
                        errors.append(f"fold {fid} {name}: bad box {inst['bbox']} label {lab} in {r['filename']}")
                    boxes[ID_TO_CLASS.get(lab, lab)] += 1
                by_dataset[reg[r["filename"]]["source_dataset"]].append((r["filename"], r["detection"]["instances"]))
            boxed = [(r["filename"], r["detection"]["instances"]) for r in recs if r["detection"]["instances"]]
            fold_samples[(fid, name)] = rng.sample(boxed, min(args.samples_per_split, len(boxed)))
            p(f"  {name:5s} {len(recs):6d} images  {sum(boxes.values()):6d} boxes  datasets {dict(sorted(ds.items()))}")
            p(f"        boxes per class {dict(sorted(boxes.items()))}")
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            if a in groups and b in groups and groups[a] & groups[b]:
                errors.append(f"fold {fid}: {len(groups[a] & groups[b])} groups in both {a} and {b}")

    # Overlays: random images per dataset, and per fold and split, with their boxes
    colours = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6"]
    def draw_sheet(items, title, path, t=300, cols=4):
        sheet = Image.new("RGB", (cols * (t + 6), 40 + ((len(items) + cols - 1) // cols) * (t + 26)), "white")
        d = ImageDraw.Draw(sheet)
        d.text((6, 10), title, fill="black", font=font(14))
        for k, (f, insts) in enumerate(items):
            x0, y0 = (k % cols) * (t + 6), 40 + (k // cols) * (t + 26)
            with Image.open(args.folds_dir / f) as im:
                im = im.convert("RGB")
                s = t / max(im.size)
                im = im.resize((round(im.width * s), round(im.height * s)))
                dr = ImageDraw.Draw(im)
                for inst in insts:
                    x1, y1, x2, y2 = (v * s for v in inst["bbox"])
                    c = colours[inst["label"] % len(colours)]
                    dr.rectangle((x1, y1, x2, y2), outline=c, width=2)
                    dr.text((x1 + 2, y1 + 1), ID_TO_CLASS.get(inst["label"], str(inst["label"])), fill=c, font=font(11))
                sheet.paste(im, (x0, y0))
            d.text((x0, y0 + t + 4), f.split("/")[1][:28] + " | " + Path(f).name[:22], fill="#444444", font=font(11))
        sheet.save(path, quality=88)

    for ds, items in sorted(by_dataset.items()):
        uniq = {f: inst for f, inst in items if inst}
        pick = rng.sample(sorted(uniq), min(args.overlays_per_dataset, len(uniq)))
        draw_sheet([(f, uniq[f]) for f in pick], f"{ds}: random images with the boxes stored in the fold files",
                   out / f"overlay_{ds}.jpg")
    for (fid, name), items in sorted(fold_samples.items()):
        meta = fold_meta.get(fid, {})
        draw_sheet(items, f"fold {fid} {name} (test = {meta.get('held_out_dataset')}, val = {meta.get('val_dataset') or 'training-data slice'})",
                   out / f"fold_{fid}_{name}.jpg")

    p("\n" + ("ALL CHECKS PASSED" if not errors else f"{len(errors)} PROBLEMS:\n  " + "\n  ".join(errors[:50])))
    report = "\n".join(lines)
    (out / "verify_report.txt").write_text(report, encoding="utf-8")
    print(report)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
