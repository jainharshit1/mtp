"""
Leave-One-Dataset-Out (LODO) split generation.

Uses pHash groups to determine cross-dataset image sharing:
  - An image group is "shared" with dataset k if ANY image in the group
    came from dataset k (including augmented versions)
  - Test set: images whose GROUP is exclusive to the held-out dataset
  - Train: all other images
  - Val: ~10% of train, split by pHash group (no scene appears in both)

This prevents data leakage from Roboflow augmentations — even if an
augmented copy has a different filename, it's caught by pHash grouping.
"""

import logging
import random
from collections import defaultdict

logger = logging.getLogger(__name__)


def split_val_by_group(
    pool_ids: list,
    img_to_group: dict,
    val_fraction: float,
    rng: random.Random,
) -> tuple[list, list]:
    """Split pool_ids into (train, val) so that no pHash group spans both sides.

    Whole groups are moved to val until it holds ~val_fraction of the images.
    """
    by_group = defaultdict(list)
    for img_id in pool_ids:
        by_group[img_to_group[img_id]].append(img_id)

    group_ids = sorted(by_group)
    rng.shuffle(group_ids)

    target = max(1, int(len(pool_ids) * val_fraction))
    val_ids, train_ids = [], []
    for gid in group_ids:
        if len(val_ids) < target:
            val_ids.extend(by_group[gid])
        else:
            train_ids.extend(by_group[gid])
    return train_ids, val_ids


def generate_lodo_folds(
    unified_images: list[dict],
    merged_annotations: list[dict],
    annotated_classes: dict[int, list[int]],
    groups: dict[int, dict],
    dataset_names: list[str],
    val_fraction: float = 0.10,
    seed: int = 42,
) -> list[dict]:
    """
    Generate LODO folds using group-level dataset membership.

    Args:
        unified_images: all images with group_id
        merged_annotations: merged annotation list
        annotated_classes: per-image annotated class sets
        groups: {group_id: {"source_datasets": set, "image_ids": list}}
        dataset_names: list of dataset names for fold ordering
        val_fraction: fraction of train to use as validation
        seed: random seed for reproducibility
    """
    rng = random.Random(seed)

    img_to_group = {img["id"]: img["group_id"] for img in unified_images}

    anns_by_image = defaultdict(list)
    for ann in merged_annotations:
        anns_by_image[ann["image_id"]].append(ann)

    folds = []

    for fold_id, held_out in enumerate(dataset_names):
        test_ids = []
        train_pool = []

        for img in unified_images:
            img_id = img["id"]
            group_id = img_to_group[img_id]
            group_datasets = groups[group_id]["source_datasets"]

            if held_out in group_datasets:
                if group_datasets == {held_out}:
                    if img_id in anns_by_image:
                        test_ids.append(img_id)
                else:
                    train_pool.append(img_id)
            else:
                train_pool.append(img_id)

        # Group-level split: copies of one scene must not straddle train/val.
        train_ids, val_ids = split_val_by_group(
            train_pool, img_to_group, val_fraction, rng,
        )

        if len(test_ids) < 10:
            logger.warning(
                f"Fold {fold_id} ({held_out}): only {len(test_ids)} exclusive "
                f"test images — results may be unreliable"
            )

        folds.append({
            "fold_id": fold_id,
            "held_out_dataset": held_out,
            "train_image_ids": train_ids,
            "val_image_ids": val_ids,
            "test_image_ids": test_ids,
        })

        logger.info(
            f"Fold {fold_id} ({held_out}): "
            f"train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}"
        )

    return folds
