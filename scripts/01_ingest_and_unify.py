#!/usr/bin/env python3
"""
End-to-end data processing pipeline: ingest → dedup → merge → LODO splits → ODVG.

Usage:
    python scripts/01_ingest_and_unify.py [--config configs/datasets.yaml] [--output-dir data/processed] [--no-phash]
        [--embed-model facebook/dinov2-small|none] [--embed-threshold 0.90] [--max-per-group 5] [--max-test-per-group 2]
        [--thin-threshold 0.95]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ingest import ingest_all, load_config
from src.data.dedup import EMBED_MODEL, EMBED_THRESHOLD, PHASH_MIN_COSINE, PHASH_THRESHOLD, build_image_registry
from src.data.merge import merge_annotations
from src.data.lodo_splits import PROXIMITY_FILTER, THIN_THRESHOLD, generate_lodo_folds
from src.data.coco_to_odvg import convert_fold_to_odvg
from src.data.taxonomy import ID_TO_CLASS, NUM_CLASSES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Aircraft defect data pipeline")
    parser.add_argument("--config", default="configs/datasets.yaml")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--odvg-dir", default="data/odvg")
    parser.add_argument("--no-phash", action="store_true", help="Skip perceptual grouping (faster)")
    parser.add_argument("--phash-threshold", type=int, default=PHASH_THRESHOLD)
    parser.add_argument("--phash-min-cosine", type=float, default=PHASH_MIN_COSINE,
                        help="A pHash link also needs this DINOv2 cosine")
    parser.add_argument("--embed-model", default=EMBED_MODEL, help="'none' disables the crop-catching embedding rule")
    parser.add_argument("--embed-threshold", type=float, default=EMBED_THRESHOLD)
    parser.add_argument("--cross-dataset-embed-threshold", type=float, default=None,
                        help="Lower DINOv2 threshold for links between images of different datasets")
    parser.add_argument("--embed-cache", default=None, help="Embeddings from scripts/00_embed_images.py")
    parser.add_argument("--geometric-links", default=None, help="CSV from scripts/geometric_links.py")
    parser.add_argument("--geometric-min-inliers", type=int, default=25,
                        help="Link a geometric pair with at least this many RANSAC inliers")
    parser.add_argument("--max-per-group", type=int, default=5, help="Images kept per sub-scene in train/val")
    parser.add_argument("--max-test-per-group", type=int, default=2, help="Images kept per sub-scene in test")
    parser.add_argument("--thin-threshold", type=float, default=THIN_THRESHOLD,
                        help="Drop near-repeats within a group at this DINOv2 cosine; 0 disables")
    parser.add_argument("--proximity-filter", type=float, default=PROXIMITY_FILTER,
                        help="Drop training images this close (DINOv2 cosine) to a test/val image of the fold; 0 disables")
    parser.add_argument("--fold-drops", default=None,
                        help="JSON from scripts/fold_geometric_filter.py: training images to drop per fold")
    parser.add_argument("--min-box-side", type=float, default=1.0,
                        help="Drop boxes with a side shorter than this (original pixels); 0 keeps all")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config_path = str(Path(args.config).resolve())
    output_dir = Path(args.output_dir)
    odvg_dir = Path(args.odvg_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("STEP 1: Ingestion + Class Unification")
    logger.info("=" * 60)
    ingested = ingest_all(config_path)

    for ds in ingested:
        logger.info(f"  {ds['name']}: {len(ds['images'])} images, {len(ds['annotations'])} annotations")
        if ds["issues"]:
            for issue in ds["issues"][:5]:
                logger.warning(f"    ISSUE: {issue}")
            if len(ds["issues"]) > 5:
                logger.warning(f"    ... and {len(ds['issues']) - 5} more issues")

    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: Image Dedup + Cross-Dataset Grouping")
    logger.info("=" * 60)
    dedup_result = build_image_registry(
        ingested,
        phash_dedup=not args.no_phash,
        phash_threshold=args.phash_threshold,
        embed_model=None if args.embed_model.lower() == "none" else args.embed_model,
        embed_threshold=args.embed_threshold,
        workers=args.workers,
        embed_cache=_load_embed_cache(args.embed_cache) if args.embed_cache else None,
        phash_min_cosine=args.phash_min_cosine,
        cross_dataset_embed_threshold=args.cross_dataset_embed_threshold,
        shared_name_groups=(load_config(config_path).get("lodo") or {}).get("shared_source_names"),
        source_name_patterns=(load_config(config_path).get("lodo") or {}).get("source_name_photo"),
        geometric_links=_load_geometric_links(args.geometric_links, args.geometric_min_inliers)
        if args.geometric_links else None,
    )

    with open(output_dir / "dedup_stats.json", "w") as f:
        json.dump(dedup_result["stats"], f, indent=2)
    if dedup_result["embeddings"] is not None:
        np.save(output_dir / "embeddings.npy", dedup_result["embeddings"])
    logger.info(f"Stats: {json.dumps(dedup_result['stats'], indent=2)}")

    groups_serializable = {
        str(k): {"source_datasets": sorted(v["source_datasets"]), "n_images": len(v["image_ids"])}
        for k, v in dedup_result["groups"].items()
    }
    with open(output_dir / "image_groups.json", "w") as f:
        json.dump(groups_serializable, f)

    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: Annotation Merging")
    logger.info("=" * 60)
    merge_result = merge_annotations(ingested, dedup_result)
    if args.min_box_side:
        # Boxes thinner than this (in original pixels) are lines or points, not defects.
        n = len(merge_result["annotations"])
        merge_result["annotations"] = [a for a in merge_result["annotations"]
                                       if min(a["bbox"][2], a["bbox"][3]) >= args.min_box_side]
        logger.info(f"Removed {n - len(merge_result['annotations'])} boxes with a side under "
                    f"{args.min_box_side} px")

    unified_coco = {
        "images": dedup_result["images"],
        "annotations": merge_result["annotations"],
        "categories": [
            {"id": i, "name": name, "supercategory": "defect"}
            for i, name in ID_TO_CLASS.items()
        ],
    }
    with open(output_dir / "unified_annotations.json", "w") as f:
        json.dump(unified_coco, f)

    with open(output_dir / "annotated_classes.json", "w") as f:
        ac_serializable = {str(k): v for k, v in merge_result["annotated_classes"].items()}
        json.dump(ac_serializable, f)

    logger.info(f"Stats: {json.dumps(merge_result['stats'], indent=2)}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4: LODO Split Generation")
    logger.info("=" * 60)
    cfg = load_config(config_path)
    dataset_names = [ds["name"] for ds in cfg["datasets"]]
    lodo_cfg = cfg.get("lodo") or {}
    test_datasets = [d for d in lodo_cfg.get("test_datasets", dataset_names) if d in dataset_names]
    val_dataset_for = {k: v for k, v in (lodo_cfg.get("val_dataset_for") or {}).items()
                       if k in test_datasets and v in dataset_names}
    logger.info(f"Test folds: {test_datasets}; trained on only: "
                f"{[d for d in dataset_names if d not in test_datasets]}")

    folds = generate_lodo_folds(
        unified_images=dedup_result["images"],
        merged_annotations=merge_result["annotations"],
        annotated_classes=merge_result["annotated_classes"],
        groups=dedup_result["groups"],
        dataset_names=dataset_names,
        val_fraction=args.val_fraction,
        seed=args.seed,
        max_per_group=args.max_per_group,
        max_test_per_group=args.max_test_per_group,
        embeddings=dedup_result["embeddings"],
        thin_threshold=args.thin_threshold or None,
        test_datasets=test_datasets,
        val_dataset_for=val_dataset_for,
        proximity_filter=args.proximity_filter or None,
        train_drop_files=_load_fold_drops(args.fold_drops) if args.fold_drops else None,
        hold_out_together=lodo_cfg.get("hold_out_together"),
        keep_copies_together=lodo_cfg.get("keep_copies_together"),
        val_to_train_files=_load_fold_drops(args.fold_drops, "val_to_train") if args.fold_drops else None,
    )

    with open(output_dir / "lodo_folds.json", "w") as f:
        json.dump(folds, f, indent=2)

    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 5: ODVG Conversion")
    logger.info("=" * 60)
    for fold in folds:
        fold_dir = odvg_dir / f"fold_{fold['fold_id']}"
        logger.info(f"Fold {fold['fold_id']} ({fold['held_out_dataset']}):")
        convert_fold_to_odvg(
            fold=fold,
            unified_images=dedup_result["images"],
            merged_annotations=merge_result["annotations"],
            annotated_classes=merge_result["annotated_classes"],
            output_dir=str(fold_dir),
        )

    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)

    total_images = len(dedup_result["images"])
    total_anns = len(merge_result["annotations"])
    logger.info(f"Unified dataset: {total_images} images, {total_anns} annotations")
    logger.info(f"Classes ({NUM_CLASSES}): {', '.join(ID_TO_CLASS.values())}")
    logger.info(f"LODO folds: {len(folds)}")
    logger.info(f"Outputs saved to: {output_dir} and {odvg_dir}")

    _write_manifest(odvg_dir, folds, dedup_result["stats"], vars(args))

    _print_class_distribution(merge_result["annotations"])
    _print_dataset_contribution(dedup_result["images"], dataset_names)


def _load_geometric_links(path: str, min_inliers: int) -> list[tuple[str, str]]:
    import csv
    with open(path) as f:
        return [(r["path_a"], r["path_b"]) for r in csv.DictReader(f) if int(r["inliers"]) >= min_inliers]


def _load_fold_drops(path: str, key: str = "drops") -> dict[int, set[str]]:
    with open(path) as f:
        return {int(k): set(v) for k, v in json.load(f).get(key, {}).items()}


def _load_embed_cache(path: str) -> dict:
    data = np.load(path, allow_pickle=False)
    return dict(zip(data["paths"].tolist(), data["emb"]))


def _write_manifest(odvg_dir: Path, folds: list[dict], dedup_stats: dict, args: dict):
    """Record the rules, per-fold counts and a SHA256 of every fold file, so the folds can be frozen."""
    import hashlib
    from datetime import datetime

    files = {
        str(p.relative_to(odvg_dir)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(odvg_dir.rglob("*")) if p.is_file() and p.name != "MANIFEST.json"
    }
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "args": args,
        "dedup_stats": dedup_stats,
        "folds": [
            {"fold_id": f["fold_id"], "held_out_dataset": f["held_out_dataset"],
             "val_dataset": f.get("val_dataset"),
             "train_dropped_near_test_or_val": f.get("train_dropped_near_test_or_val"),
             "train_dropped_photo_match": f.get("train_dropped_photo_match"),
             "val_groups_moved_to_train": f.get("val_groups_moved_to_train"),
             "val_groups_in_multi_group_clusters": f.get("val_groups_in_multi_group_clusters"),
             "train_dropped_cross_dataset_copies": f.get("train_dropped_cross_dataset_copies"),
             "excluded_from_train": f.get("excluded_from_train"),
             **{s: len(f[f"{s}_image_ids"]) for s in ("train", "val", "test")}}
            for f in folds
        ],
        "files_sha256": files,
    }
    with open(odvg_dir / "MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info(f"Manifest: {odvg_dir / 'MANIFEST.json'} ({len(files)} files)")


def _print_class_distribution(annotations):
    from collections import Counter
    counts = Counter(ann["category_id"] for ann in annotations)
    logger.info("\nClass distribution:")
    for cls_id in sorted(counts.keys()):
        logger.info(f"  {ID_TO_CLASS[cls_id]:20s}: {counts[cls_id]:6d}")


def _print_dataset_contribution(images, dataset_names):
    from collections import Counter
    ds_counts = Counter(img["source_dataset"] for img in images)
    logger.info("\nDataset contributions (images):")
    for name in dataset_names:
        logger.info(f"  {name:35s}: {ds_counts.get(name, 0):6d}")


if __name__ == "__main__":
    main()
