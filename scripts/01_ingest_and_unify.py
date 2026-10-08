#!/usr/bin/env python3
"""
End-to-end data processing pipeline: ingest → dedup → merge → LODO splits → ODVG.

Usage:
    python scripts/01_ingest_and_unify.py [--config configs/datasets.yaml] [--output-dir data/processed] [--no-phash]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ingest import ingest_all, load_config
from src.data.dedup import build_image_registry
from src.data.merge import merge_annotations
from src.data.lodo_splits import generate_lodo_folds
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
    dedup_result = build_image_registry(ingested, phash_dedup=not args.no_phash)

    with open(output_dir / "dedup_stats.json", "w") as f:
        json.dump(dedup_result["stats"], f, indent=2)
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

    folds = generate_lodo_folds(
        unified_images=dedup_result["images"],
        merged_annotations=merge_result["annotations"],
        annotated_classes=merge_result["annotated_classes"],
        groups=dedup_result["groups"],
        dataset_names=dataset_names,
        val_fraction=args.val_fraction,
        seed=args.seed,
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

    _print_class_distribution(merge_result["annotations"])
    _print_dataset_contribution(dedup_result["images"], dataset_names)


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
