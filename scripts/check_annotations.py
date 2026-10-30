#!/usr/bin/env python3
"""Check that every image in a portable fold build carries exactly the boxes of its raw annotation file.

Reads the raw annotation files itself (independently of src/data/ingest.py), applies the dataset's
class_map, converts to [x1, y1, x2, y2] pixels (boxes cropped to the image), and compares each image's
boxes in every fold file (train/val .jsonl and test COCO) with them as a multiset. Differences are split into:
  - expected: a raw box dropped because another box of the same class on the same image overlaps it
    with IoU > 0.9 (the pipeline merges those)
  - errors: anything else (missing, extra, moved or relabelled boxes)
Also reports raw boxes whose class is not in class_map (dropped on purpose) per dataset and class.

    python scripts/check_annotations.py --folds-dir D:/MTP/mtp-data/lodo_v3 \
        --config D:/MTP/mtp-data/lodo_v3/build/datasets_local.yaml --raw-root D:/MTP/mtp-data/raw
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.taxonomy import CLASS_TO_ID, ID_TO_CLASS  # noqa: E402

TOL = 0.05  # pixels; boxes are compared after rounding to this grid
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def key(p) -> str:
    return str(Path(p).resolve()).lower()


def crop(x1, y1, x2, y2, w, h):
    return max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)


def read_raw(cfg) -> tuple[dict, Counter, dict]:
    """Returns ({image key: [(class, x1, y1, x2, y2)]}, Counter of dropped (dataset, raw class), {image key: dataset})."""
    boxes, dropped, dataset_of = defaultdict(list), Counter(), {}
    for ds in cfg["datasets"]:
        name, base, cmap = ds["name"], Path(ds["base_dir"]), ds["class_map"]
        if ds["format"] == "coco_roboflow":
            for split in ds["splits"]:
                ann_file = base / split / "_annotations.coco.json"
                if not ann_file.exists():
                    continue
                coco = json.load(open(ann_file))
                cats = {c["id"]: c["name"] for c in coco["categories"]}
                imgs = {i["id"]: base / split / i["file_name"] for i in coco["images"]}
                for p in imgs.values():
                    dataset_of[key(p)] = name
                for a in coco["annotations"]:
                    cname = cats[a["category_id"]]
                    if cname not in cmap:
                        dropped[(name, cname)] += 1
                        continue
                    x, y, w, h = a["bbox"]
                    if w <= 0 or h <= 0:
                        dropped[(name, f"{cname} (zero-size box)")] += 1
                        continue
                    boxes[key(imgs[a["image_id"]])].append((CLASS_TO_ID[cmap[cname]], x, y, x + w, y + h))
        elif ds["format"] == "yolo":
            names = {int(k): v for k, v in ds["yolo_classes"].items()}
            for split in ds["splits"]:
                for img in sorted((base / "images" / split).iterdir()):
                    lbl = base / "labels_rect" / split / (img.stem + ".txt")
                    if img.suffix.lower() not in IMAGE_EXT or not lbl.exists():
                        continue
                    dataset_of[key(img)] = name
                    with Image.open(img) as im:
                        W, H = im.size
                    for line in open(lbl):
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        cname = names.get(int(parts[0]), parts[0])
                        if cname not in cmap:
                            dropped[(name, cname)] += 1
                            continue
                        cx, cy, bw, bh = (float(v) for v in parts[1:5])
                        b = crop((cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H, W, H)
                        if b[2] <= b[0] or b[3] <= b[1]:
                            dropped[(name, f"{cname} (outside image)")] += 1
                            continue
                        boxes[key(img)].append((CLASS_TO_ID[cmap[cname]], *b))
        elif ds["format"] == "fuselage_json":
            for ann_file in sorted((base / "annotations").glob("*.json")):
                recs = json.load(open(ann_file))
                if not recs:
                    continue
                img = base / "images" / recs[0]["image"]
                if not img.exists():
                    continue
                dataset_of[key(img)] = name
                with Image.open(img) as im:
                    W, H = im.size
                for a in recs[0].get("annotations", []):
                    if a["label"] not in cmap:
                        dropped[(name, a["label"])] += 1
                        continue
                    c = a["coordinates"]
                    b = crop(c["x"] - c["width"] / 2, c["y"] - c["height"] / 2,
                             c["x"] + c["width"] / 2, c["y"] + c["height"] / 2, W, H)
                    if b[2] <= b[0] or b[3] <= b[1]:
                        dropped[(name, f"{a['label']} (outside image)")] += 1
                        continue
                    boxes[key(img)].append((CLASS_TO_ID[cmap[a["label"]]], *b))
    return boxes, dropped, dataset_of


def rounded(b):
    return (int(b[0]),) + tuple(round(v / TOL) * TOL for v in b[1:])


def iou(a, b):
    ix = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    iy = max(0.0, min(a[4], b[4]) - max(a[2], b[2]))
    inter = ix * iy
    union = (a[3] - a[1]) * (a[4] - a[2]) + (b[3] - b[1]) * (b[4] - b[2]) - inter
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds-dir", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--raw-root", required=True, type=Path)
    ap.add_argument("--min-box-side", type=float, default=1.0,
                    help="Same rule as 01_ingest_and_unify.py: raw boxes thinner than this are expected to be gone")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    annotated = {d["name"]: sorted(CLASS_TO_ID[c] for c in d["annotated_classes"]) for d in cfg["datasets"]}
    raw, dropped, dataset_of = read_raw(cfg)

    stats = defaultdict(Counter)
    errors, examples = [], defaultdict(list)
    for fold_dir in sorted(args.folds_dir.glob("fold_*"), key=lambda d: int(d.name[5:])):
        entries = []  # (split, rel path, [(cls, x1, y1, x2, y2)], annotated_classes or None)
        for split in ("train", "val"):
            for line in open(fold_dir / f"{split}.jsonl"):
                if line.strip():
                    r = json.loads(line)
                    entries.append((split, r["filename"],
                                    [(i["label"], *i["bbox"]) for i in r["detection"]["instances"]],
                                    r.get("annotated_classes")))
        test = json.load(open(fold_dir / "test_annotations.json"))
        per = defaultdict(list)
        for a in test["annotations"]:
            x, y, w, h = a["bbox"]
            per[a["image_id"]].append((a["category_id"], x, y, x + w, y + h))
        entries += [("test", i["file_name"], per[i["id"]], None) for i in test["images"]]

        for split, rel, found, ann_cls in entries:
            src = key(args.raw_root / rel.removeprefix("images/"))
            ds = dataset_of.get(src)
            where = f"{fold_dir.name}/{split} {rel}"
            if ds is None:
                errors.append(f"{where}: no raw annotation file lists this image")
                continue
            st = stats[ds]
            st["image entries"] += 1
            expected = [b for b in raw.get(src, []) if min(b[3] - b[1], b[4] - b[2]) >= args.min_box_side]
            st["raw boxes under min side (removed by rule)"] += len(raw.get(src, [])) - len(expected)
            st["raw boxes"] += len(expected)
            st["boxes in folds"] += len(found)
            if ann_cls is not None and sorted(ann_cls) != annotated[ds]:
                errors.append(f"{where}: annotated_classes {ann_cls} != dataset's {annotated[ds]}")
            e, f = Counter(map(rounded, expected)), Counter(map(rounded, found))
            if e == f:
                st["images exact"] += 1
                continue
            missing, extra = list((e - f).elements()), list((f - e).elements())
            explained = [m for m in missing if any(k[0] == m[0] and iou(k, m) > 0.9 for k in f)]
            if not extra and len(explained) == len(missing):
                st["images with merged overlapping boxes"] += 1
                st["overlapping boxes merged"] += len(missing)
                continue
            st["images with errors"] += 1
            errors.append(f"{where}: {len(missing)} raw boxes missing, {len(extra)} unexpected")
            if len(examples[ds]) < 5:
                examples[ds].append({"entry": where, "missing": missing[:4], "unexpected": extra[:4]})

    lines = ["Annotation check: every image in every fold file vs its raw annotation file\n"]
    for ds in sorted(stats):
        st = stats[ds]
        lines.append(f"{ds}: {st['image entries']} image entries, {st['images exact']} exact, "
                     f"{st['images with merged overlapping boxes']} with overlapping same-class boxes merged "
                     f"({st['overlapping boxes merged']} boxes), {st['images with errors']} with errors; "
                     f"boxes raw {st['raw boxes']} / in folds {st['boxes in folds']}; "
                     f"{st['raw boxes under min side (removed by rule)']} raw boxes under {args.min_box_side} px removed by rule")
        for ex in examples[ds]:
            lines.append(f"    e.g. {ex['entry']}\n         missing {ex['missing']}\n         unexpected {ex['unexpected']}")
    lines.append("\nRaw boxes dropped on purpose (class not in class_map, or empty):")
    for (ds, c), n in sorted(dropped.items()):
        lines.append(f"  {ds:30s} {c:30s} {n}")
    lines.append("\n" + ("ALL ANNOTATIONS PRESERVED" if not errors else f"{len(errors)} ENTRIES WITH ERRORS"))
    report = "\n".join(lines)
    out = args.folds_dir / "checks"
    out.mkdir(exist_ok=True)
    (out / "annotation_check.txt").write_text(report + "\n\nAll errors:\n" + "\n".join(errors), encoding="utf-8")
    print(report)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
