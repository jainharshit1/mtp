"""
Convert unified COCO-format data to ODVG JSONL for Grounding DINO training.

Each line in the ODVG file:
{
  "filename": "path/to/image.jpg",
  "height": 1024,
  "width": 1024,
  "detection": {
    "instances": [
      {"bbox": [x1, y1, x2, y2], "label": 0, "category": "crack"}
    ]
  },
  "annotated_classes": [0, 1, 3]  // for loss masking
}
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

from src.data.taxonomy import ID_TO_CLASS

logger = logging.getLogger(__name__)


def convert_fold_to_odvg(
    fold: dict,
    unified_images: list[dict],
    merged_annotations: list[dict],
    annotated_classes: dict[int, list[int]],
    output_dir: str,
) -> dict:
    """
    Convert one LODO fold to ODVG format.

    Writes:
      - train.jsonl
      - val.jsonl
      - test_annotations.json (COCO format for evaluation)
      - label_map.json
      - annotated_classes.json
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    img_by_id = {img["id"]: img for img in unified_images}
    anns_by_image = defaultdict(list)
    for ann in merged_annotations:
        anns_by_image[ann["image_id"]].append(ann)

    label_map = {str(i): name for i, name in ID_TO_CLASS.items()}
    with open(out / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    ac_out = {}
    for img_id, classes in annotated_classes.items():
        ac_out[str(img_id)] = classes
    with open(out / "annotated_classes.json", "w") as f:
        json.dump(ac_out, f)

    stats = {}
    for split_name, img_ids in [
        ("train", fold["train_image_ids"]),
        ("val", fold["val_image_ids"]),
    ]:
        odvg_path = out / f"{split_name}.jsonl"
        count = 0
        with open(odvg_path, "w") as f:
            for img_id in img_ids:
                img = img_by_id.get(img_id)
                if img is None:
                    continue
                # Images without boxes are kept as negatives: annotated_classes tells the
                # loss which classes this dataset labels, so only those are penalised.
                anns = anns_by_image.get(img_id, [])

                instances = []
                for ann in anns:
                    x, y, w, h = ann["bbox"]
                    instances.append({
                        "bbox": [x, y, x + w, y + h],
                        "label": ann["category_id"],
                        "category": ID_TO_CLASS[ann["category_id"]],
                    })

                ac = annotated_classes.get(img_id, [])

                line = {
                    "filename": img["file_name"],
                    "height": img["height"],
                    "width": img["width"],
                    "detection": {"instances": instances},
                    "annotated_classes": ac,
                }
                f.write(json.dumps(line) + "\n")
                count += 1

        stats[split_name] = count
        logger.info(f"  {split_name}: {count} images written to {odvg_path}")

    test_img_ids = set(fold["test_image_ids"])
    test_coco = _build_coco_test(test_img_ids, img_by_id, anns_by_image)
    with open(out / "test_annotations.json", "w") as f:
        json.dump(test_coco, f)
    stats["test"] = len(test_coco["images"])
    logger.info(f"  test: {stats['test']} images, {len(test_coco['annotations'])} annotations")

    return stats


def _build_coco_test(
    test_img_ids: set,
    img_by_id: dict,
    anns_by_image: dict,
) -> dict:
    categories = [
        {"id": i, "name": name, "supercategory": "defect"}
        for i, name in ID_TO_CLASS.items()
    ]

    images = []
    annotations = []
    ann_id = 0

    for img_id in sorted(test_img_ids):
        img = img_by_id.get(img_id)
        if img is None:
            continue
        images.append({
            "id": img_id,
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
        })
        for ann in anns_by_image.get(img_id, []):
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "area": ann["area"],
                "iscrowd": ann["iscrowd"],
            })
            ann_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
