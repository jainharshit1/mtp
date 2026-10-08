"""
Ingest datasets from multiple formats into a unified internal representation.

Supported formats:
  - coco_roboflow: Roboflow-exported COCO JSON with train/valid/test splits
  - yolo: YOLO txt labels with images in parallel directory
  - fuselage_json: Per-image JSON from Aircraft_Fuselage_DET2023
"""

import json
import logging
from pathlib import Path
from typing import Any

from PIL import Image as PILImage

from src.data.taxonomy import CLASS_TO_ID

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_config(config_path: str) -> dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def ingest_dataset(ds_cfg: dict) -> dict:
    fmt = ds_cfg["format"]
    if fmt == "coco_roboflow":
        return _ingest_coco_roboflow(ds_cfg)
    elif fmt == "yolo":
        return _ingest_yolo(ds_cfg)
    elif fmt == "fuselage_json":
        return _ingest_fuselage(ds_cfg)
    else:
        raise ValueError(f"Unknown format: {fmt}")


def _ingest_coco_roboflow(ds_cfg: dict) -> dict:
    """Load Roboflow COCO JSON splits, remap categories, validate."""
    base = Path(ds_cfg["base_dir"])
    class_map = ds_cfg["class_map"]
    name = ds_cfg["name"]

    all_images = []
    all_annotations = []
    img_id_offset = 0
    ann_id_counter = 0
    issues = []

    for split in ds_cfg["splits"]:
        split_dir = base / split
        ann_path = split_dir / "_annotations.coco.json"
        if not ann_path.exists():
            issues.append(f"{name}/{split}: annotation file missing")
            continue

        with open(ann_path) as f:
            coco = json.load(f)

        local_cats = {c["id"]: c["name"] for c in coco["categories"]}
        local_id_to_unified = {}
        for local_id, local_name in local_cats.items():
            if local_name in class_map:
                unified_name = class_map[local_name]
                local_id_to_unified[local_id] = CLASS_TO_ID[unified_name]

        old_to_new_img = {}
        for img in coco["images"]:
            old_id = img["id"]
            img_path = split_dir / img["file_name"]
            if not img_path.exists():
                issues.append(f"{name}/{split}: missing image {img['file_name']}")
                continue
            new_id = img_id_offset + old_id
            old_to_new_img[old_id] = new_id
            all_images.append({
                "id": new_id,
                "file_name": str(img_path),
                "width": img["width"],
                "height": img["height"],
                "source_dataset": name,
                "source_split": split,
            })

        for ann in coco["annotations"]:
            if ann["image_id"] not in old_to_new_img:
                continue
            if ann["category_id"] not in local_id_to_unified:
                continue
            bbox = ann["bbox"]
            if bbox[2] <= 0 or bbox[3] <= 0:
                issues.append(f"{name}: zero-area bbox ann_id={ann['id']}")
                continue
            all_annotations.append({
                "id": ann_id_counter,
                "image_id": old_to_new_img[ann["image_id"]],
                "category_id": local_id_to_unified[ann["category_id"]],
                "bbox": bbox,
                "area": ann.get("area", bbox[2] * bbox[3]),
                "iscrowd": ann.get("iscrowd", 0),
            })
            ann_id_counter += 1

        if coco["images"]:
            img_id_offset += max(img["id"] for img in coco["images"]) + 1

    if issues:
        for issue in issues:
            logger.warning(issue)

    return {
        "name": name,
        "images": all_images,
        "annotations": all_annotations,
        "annotated_classes": [CLASS_TO_ID[c] for c in ds_cfg["annotated_classes"]],
        "issues": issues,
    }


def _ingest_yolo(ds_cfg: dict) -> dict:
    """Load YOLO-format dataset (txt labels, parallel image dirs)."""
    base = Path(ds_cfg["base_dir"])
    class_map = ds_cfg["class_map"]
    yolo_classes = ds_cfg["yolo_classes"]
    name = ds_cfg["name"]

    yolo_id_to_unified = {}
    for yolo_id, yolo_name in yolo_classes.items():
        yolo_id_int = int(yolo_id)
        if yolo_name in class_map:
            unified_name = class_map[yolo_name]
            yolo_id_to_unified[yolo_id_int] = CLASS_TO_ID[unified_name]

    all_images = []
    all_annotations = []
    img_id = 0
    ann_id = 0
    issues = []

    for split in ds_cfg["splits"]:
        img_dir = base / "images" / split
        lbl_dir = base / "labels_rect" / split

        if not img_dir.exists():
            issues.append(f"{name}/{split}: image dir missing")
            continue

        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue

            try:
                with PILImage.open(img_path) as pil_img:
                    w, h = pil_img.size
            except Exception as e:
                issues.append(f"{name}: can't open {img_path}: {e}")
                continue

            all_images.append({
                "id": img_id,
                "file_name": str(img_path),
                "width": w,
                "height": h,
                "source_dataset": name,
                "source_split": split,
            })

            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    if cls_id not in yolo_id_to_unified:
                        continue
                    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    x1 = (cx - bw / 2) * w
                    y1 = (cy - bh / 2) * h
                    box_w = bw * w
                    box_h = bh * h
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    box_w = min(box_w, w - x1)
                    box_h = min(box_h, h - y1)
                    if box_w <= 0 or box_h <= 0:
                        continue
                    all_annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": yolo_id_to_unified[cls_id],
                        "bbox": [x1, y1, box_w, box_h],
                        "area": box_w * box_h,
                        "iscrowd": 0,
                    })
                    ann_id += 1

            img_id += 1

    if issues:
        for issue in issues:
            logger.warning(issue)

    return {
        "name": name,
        "images": all_images,
        "annotations": all_annotations,
        "annotated_classes": [CLASS_TO_ID[c] for c in ds_cfg["annotated_classes"]],
        "issues": issues,
    }


def _ingest_fuselage(ds_cfg: dict) -> dict:
    """Load Aircraft_Fuselage_DET2023 per-image JSON annotations."""
    base = Path(ds_cfg["base_dir"])
    class_map = ds_cfg["class_map"]
    name = ds_cfg["name"]

    img_dir = base / "images"
    ann_dir = base / "annotations"

    all_images = []
    all_annotations = []
    img_id = 0
    ann_id = 0
    issues = []

    for ann_path in sorted(ann_dir.glob("*.json")):
        with open(ann_path) as f:
            records = json.load(f)

        if not records:
            continue

        record = records[0]
        img_filename = record["image"]
        img_path = img_dir / img_filename

        if not img_path.exists():
            issues.append(f"{name}: missing image {img_filename}")
            continue

        try:
            with PILImage.open(img_path) as pil_img:
                w, h = pil_img.size
        except Exception as e:
            issues.append(f"{name}: can't open {img_path}: {e}")
            continue

        all_images.append({
            "id": img_id,
            "file_name": str(img_path),
            "width": w,
            "height": h,
            "source_dataset": name,
            "source_split": "all",
        })

        for ann in record.get("annotations", []):
            label = ann["label"]
            if label not in class_map:
                issues.append(f"{name}: unknown label '{label}' in {ann_path.name}")
                continue
            unified_name = class_map[label]
            coords = ann["coordinates"]
            cx, cy = coords["x"], coords["y"]
            bw, bh = coords["width"], coords["height"]
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x1 = max(0, x1)
            y1 = max(0, y1)
            bw = min(bw, w - x1)
            bh = min(bh, h - y1)
            if bw <= 0 or bh <= 0:
                continue
            all_annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": CLASS_TO_ID[unified_name],
                "bbox": [x1, y1, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
            })
            ann_id += 1

        img_id += 1

    if issues:
        for issue in issues:
            logger.warning(issue)

    return {
        "name": name,
        "images": all_images,
        "annotations": all_annotations,
        "annotated_classes": [CLASS_TO_ID[c] for c in ds_cfg["annotated_classes"]],
        "issues": issues,
    }


def ingest_all(config_path: str) -> list[dict]:
    cfg = load_config(config_path)
    results = []
    for ds_cfg in cfg["datasets"]:
        logger.info(f"Ingesting {ds_cfg['name']}...")
        result = ingest_dataset(ds_cfg)
        logger.info(
            f"  {ds_cfg['name']}: {len(result['images'])} images, "
            f"{len(result['annotations'])} annotations, "
            f"{len(result['issues'])} issues"
        )
        results.append(result)
    return results
