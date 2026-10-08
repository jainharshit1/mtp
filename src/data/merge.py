"""
Merge annotations using the deduplicated image registry.

Each image keeps its own annotations (no cross-image merging), but
we do bbox dedup when the SAME image has duplicate annotations from
multiple datasets (exact-duplicate images that appeared in multiple sources).
"""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

IOU_DEDUP_THRESHOLD = 0.9


def _bbox_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[0] + b1[2], b2[0] + b2[2])
    y2 = min(b1[1] + b1[3], b2[1] + b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = b1[2] * b1[3]
    area2 = b2[2] * b2[3]
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def merge_annotations(
    ingested_datasets: list[dict],
    dedup_result: dict,
) -> dict:
    """
    Remap annotations to new image IDs and deduplicate overlapping bboxes.

    For exact-duplicate images (same SHA256) that appeared in multiple datasets,
    collect all annotations on the deduplicated image and remove high-IoU
    same-class duplicates.

    Returns:
        {
            "annotations": [annotation dicts with contiguous IDs],
            "annotated_classes": {image_id: sorted list of unified class IDs},
            "stats": {...}
        }
    """
    old_to_new = dedup_result["old_to_new_id"]

    anns_by_image = defaultdict(list)
    image_annotated_classes = defaultdict(set)

    for ds in ingested_datasets:
        ds_name = ds["name"]
        ds_annotated = set(ds["annotated_classes"])

        for img in ds["images"]:
            key = (ds_name, img["id"])
            new_id = old_to_new.get(key)
            if new_id is None:
                continue
            image_annotated_classes[new_id].update(ds_annotated)

        for ann in ds["annotations"]:
            old_img_id = ann["image_id"]
            key = (ds_name, old_img_id)
            new_id = old_to_new.get(key)
            if new_id is None:
                continue
            anns_by_image[new_id].append({
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "area": ann["area"],
                "iscrowd": ann["iscrowd"],
                "source_dataset": ds_name,
            })

    merged_anns = []
    ann_id = 0
    total_before = 0
    total_after = 0
    bbox_dedup_count = 0

    for img_id in sorted(anns_by_image.keys()):
        anns = anns_by_image[img_id]
        total_before += len(anns)

        kept = []
        for ann in anns:
            is_dup = False
            for existing in kept:
                if (existing["category_id"] == ann["category_id"] and
                        _bbox_iou(existing["bbox"], ann["bbox"]) > IOU_DEDUP_THRESHOLD):
                    is_dup = True
                    break
            if not is_dup:
                kept.append(ann)
            else:
                bbox_dedup_count += 1

        for ann in kept:
            merged_anns.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "area": ann["area"],
                "iscrowd": ann["iscrowd"],
            })
            ann_id += 1
        total_after += len(kept)

    annotated_classes = {
        img_id: sorted(classes)
        for img_id, classes in image_annotated_classes.items()
    }

    stats = {
        "total_annotations_before": total_before,
        "total_annotations_after": total_after,
        "bbox_duplicates_removed": bbox_dedup_count,
        "images_with_annotations": len(anns_by_image),
        "images_with_annotated_classes": len(annotated_classes),
    }

    logger.info(
        f"Merge: {total_before} → {total_after} annotations "
        f"({bbox_dedup_count} bbox duplicates removed)"
    )

    return {
        "annotations": merged_anns,
        "annotated_classes": annotated_classes,
        "stats": stats,
    }
