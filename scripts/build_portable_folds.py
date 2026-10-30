#!/usr/bin/env python3
"""Make a self-contained, portable copy of the folds built by 01_ingest_and_unify.py.

Every image that any fold uses is copied once (original size) to <out>/images/, keeping its path
below --raw-root, and the fold files are rewritten to use paths relative to <out>. Training and
evaluation resolve relative paths against the folder that holds fold_*/, so <out> can be copied to
any machine and used as data.odvg_dir as it is.

    python scripts/build_portable_folds.py --odvg-dir <01 --odvg-dir> --raw-root D:/MTP/mtp-data/raw \
        --out D:/MTP/mtp-data/lodo_v3
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

FOLD_FILES_KEPT = ("label_map.json", "annotated_classes.json")


def portable(path: str, raw_root: Path) -> str:
    try:
        rel = Path(path).resolve().relative_to(raw_root)
    except ValueError:
        sys.exit(f"{path} is not under --raw-root {raw_root}")
    return "images/" + rel.as_posix()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--odvg-dir", required=True, type=Path)
    ap.add_argument("--raw-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    raw_root = args.raw_root.resolve()
    fold_dirs = sorted(d for d in args.odvg_dir.glob("fold_*") if d.is_dir())
    if not fold_dirs:
        sys.exit(f"no fold_* in {args.odvg_dir}")
    if any((args.out / d.name).exists() for d in fold_dirs) or (args.out / "images").exists():
        sys.exit(f"{args.out} already holds folds or images; choose a new --out (builds are not overwritten)")

    to_copy = {}  # relative path -> source path
    for fold_dir in fold_dirs:
        dst = args.out / fold_dir.name
        dst.mkdir(parents=True)
        # One line per image with all its boxes; lines sorted by path, boxes by class then position.
        for name in ("train.jsonl", "val.jsonl"):
            recs = []
            with open(fold_dir / name) as f_in:
                for line in f_in:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    rel = portable(rec["filename"], raw_root)
                    to_copy[rel] = rec["filename"]
                    rec["filename"] = rel
                    rec["detection"]["instances"].sort(key=lambda b: (b["label"], b["bbox"][1], b["bbox"][0]))
                    recs.append(rec)
            with open(dst / name, "w") as f_out:
                for rec in sorted(recs, key=lambda r: r["filename"]):
                    f_out.write(json.dumps(rec) + "\n")
        # Test: images sorted by path; annotations grouped per image in that order, then by class
        # and position, and renumbered.
        test = json.load(open(fold_dir / "test_annotations.json"))
        for img in test["images"]:
            rel = portable(img["file_name"], raw_root)
            to_copy[rel] = img["file_name"]
            img["file_name"] = rel
        test["images"].sort(key=lambda i: i["file_name"])
        order = {img["id"]: k for k, img in enumerate(test["images"])}
        test["annotations"].sort(key=lambda a: (order[a["image_id"]], a["category_id"], a["bbox"][1], a["bbox"][0]))
        for k, a in enumerate(test["annotations"]):
            a["id"] = k
        json.dump(test, open(dst / "test_annotations.json", "w"))
        for name in FOLD_FILES_KEPT:
            shutil.copy2(fold_dir / name, dst / name)

    print(f"copying {len(to_copy)} images...")
    image_sha = {}
    for rel, src in sorted(to_copy.items()):
        target = args.out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        data = target.read_bytes()
        if data != Path(src).read_bytes():
            sys.exit(f"copy mismatch: {src}")
        image_sha[rel] = hashlib.sha256(data).hexdigest()

    build_manifest = args.odvg_dir / "MANIFEST.json"
    manifest = json.load(open(build_manifest)) if build_manifest.exists() else {}
    manifest.pop("files_sha256", None)
    manifest["portable_copy"] = {"raw_root": str(raw_root), "n_images": len(image_sha)}
    manifest["files_sha256"] = {
        p.relative_to(args.out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for d in sorted(args.out.glob("fold_*")) for p in sorted(d.iterdir())
    }
    manifest["images_sha256"] = image_sha
    with open(args.out / "MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=1, default=str)
    size = sum((args.out / r).stat().st_size for r in image_sha) / 1e9
    print(f"{len(fold_dirs)} folds, {len(image_sha)} images ({size:.2f} GB) -> {args.out}")


if __name__ == "__main__":
    main()
