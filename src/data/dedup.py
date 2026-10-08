"""
Image deduplication and cross-dataset overlap detection.

Two modes:
  1. Identity grouping (pHash): groups perceptually similar images across datasets.
     Used for LODO splitting — determines which datasets an image "belongs to."
  2. Exact dedup (SHA256): removes byte-identical copies. These are truly redundant.

Roboflow augmentations (.rf.<hash> suffix) are similar enough to match via pHash
but are intentionally augmented for training diversity — so we keep them as separate
training samples while tracking them as belonging to the same "image group" for LODO.
"""

import hashlib
import logging
from collections import defaultdict
from pathlib import Path

import imagehash
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

PHASH_THRESHOLD = 8


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _phash(path: str) -> imagehash.ImageHash:
    with PILImage.open(path) as img:
        return imagehash.phash(img)


def build_image_registry(
    ingested_datasets: list[dict],
    phash_dedup: bool = True,
) -> dict:
    """
    Build image registry with cross-dataset grouping.

    Returns:
        {
            "images": [all images with group_id and contiguous new IDs],
            "old_to_new_id": {(dataset_name, old_img_id): new_img_id},
            "groups": {group_id: {"source_datasets": set, "image_ids": [new_ids]}},
            "stats": {...}
        }
    """
    logger.info("Pass 1: Computing SHA256 hashes for exact dedup...")
    sha_groups = defaultdict(list)
    for ds in ingested_datasets:
        for img in ds["images"]:
            sha = _sha256(img["file_name"])
            sha_groups[sha].append((ds["name"], img))

    total_input = sum(len(ds["images"]) for ds in ingested_datasets)
    exact_dups = total_input - len(sha_groups)
    logger.info(f"  {total_input} images → {len(sha_groups)} unique by SHA256 ({exact_dups} exact duplicates)")

    sha_to_phash = {}
    if phash_dedup:
        logger.info("Pass 2: Computing pHash for perceptual grouping...")
        for sha, entries in sha_groups.items():
            path = entries[0][1]["file_name"]
            try:
                sha_to_phash[sha] = _phash(path)
            except Exception as e:
                logger.warning(f"pHash failed for {path}: {e}")
                sha_to_phash[sha] = None

        phash_groups = defaultdict(list)
        assigned = set()
        sha_list = list(sha_groups.keys())

        for i, sha_i in enumerate(sha_list):
            if sha_i in assigned:
                continue
            ph_i = sha_to_phash.get(sha_i)
            if ph_i is None:
                phash_groups[sha_i] = [sha_i]
                assigned.add(sha_i)
                continue

            group = [sha_i]
            assigned.add(sha_i)
            for j in range(i + 1, len(sha_list)):
                sha_j = sha_list[j]
                if sha_j in assigned:
                    continue
                ph_j = sha_to_phash.get(sha_j)
                if ph_j is None:
                    continue
                if ph_i - ph_j <= PHASH_THRESHOLD:
                    group.append(sha_j)
                    assigned.add(sha_j)
            phash_groups[sha_i] = group

        logger.info(f"  {len(sha_groups)} SHA-unique → {len(phash_groups)} perceptual groups")
    else:
        phash_groups = {sha: [sha] for sha in sha_groups}

    old_to_new_id = {}
    all_images = []
    groups = {}
    new_img_id = 0

    for group_id, (canonical_sha, sha_members) in enumerate(phash_groups.items()):
        group_datasets = set()
        group_image_ids = []

        for sha in sha_members:
            seen_paths = set()
            for ds_name, img in sha_groups[sha]:
                if img["file_name"] in seen_paths:
                    old_to_new_id[(ds_name, img["id"])] = None
                    continue
                seen_paths.add(img["file_name"])

                old_to_new_id[(ds_name, img["id"])] = new_img_id
                group_datasets.add(ds_name)
                group_image_ids.append(new_img_id)

                all_images.append({
                    "id": new_img_id,
                    "file_name": img["file_name"],
                    "width": img["width"],
                    "height": img["height"],
                    "source_dataset": ds_name,
                    "group_id": group_id,
                })
                new_img_id += 1

        groups[group_id] = {
            "source_datasets": group_datasets,
            "image_ids": group_image_ids,
        }

    none_count = sum(1 for v in old_to_new_id.values() if v is None)
    if none_count:
        logger.info(f"  {none_count} exact-duplicate entries mapped to None (same path in same dataset)")

    stats = {
        "total_input_images": total_input,
        "unique_sha256": len(sha_groups),
        "perceptual_groups": len(phash_groups),
        "final_images": len(all_images),
        "exact_duplicates_removed": none_count,
        "cross_dataset_groups": sum(
            1 for g in groups.values() if len(g["source_datasets"]) > 1
        ),
    }
    logger.info(
        f"Registry: {len(all_images)} images in {len(groups)} groups "
        f"({stats['cross_dataset_groups']} cross-dataset groups)"
    )

    return {
        "images": all_images,
        "old_to_new_id": old_to_new_id,
        "groups": groups,
        "stats": stats,
    }
