"""Visualization of predictions and ground truth bboxes."""

import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.data.taxonomy import UNIFIED_CLASSES

logger = logging.getLogger(__name__)

COLORS = [
    (255, 0, 0),     # crack - red
    (0, 0, 255),     # dent - blue
    (255, 165, 0),   # corrosion - orange
    (0, 255, 0),     # scratch - green
    (255, 255, 0),   # paint-off - yellow
    (128, 0, 128),   # missing-head - purple
    (0, 255, 255),   # rupture - cyan
    (255, 0, 255),   # fastener-damage - magenta
]


def draw_boxes(
    image_path: str,
    gt_boxes: list = None,
    pred_boxes: list = None,
    output_path: str = None,
) -> Image.Image:
    """Draw GT (solid) and predicted (dashed) bounding boxes on an image.

    gt_boxes: list of {"bbox": [x,y,w,h], "category_id": int}
    pred_boxes: list of {"bbox": [x,y,w,h], "category_id": int, "score": float}
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    if gt_boxes:
        for box in gt_boxes:
            x, y, w, h = box["bbox"]
            cat_id = box["category_id"]
            color = COLORS[cat_id % len(COLORS)]
            label = UNIFIED_CLASSES[cat_id] if cat_id < len(UNIFIED_CLASSES) else str(cat_id)
            draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
            draw.text((x, max(0, y - 12)), f"GT:{label}", fill=color)

    if pred_boxes:
        for box in pred_boxes:
            x, y, w, h = box["bbox"]
            cat_id = box["category_id"]
            score = box.get("score", 0.0)
            color = COLORS[cat_id % len(COLORS)]
            label = UNIFIED_CLASSES[cat_id] if cat_id < len(UNIFIED_CLASSES) else str(cat_id)
            # Dashed effect via dotted rectangle
            draw.rectangle([x, y, x + w, y + h], outline=color, width=1)
            draw.text((x, y + h + 2), f"{label}:{score:.2f}", fill=color)

    if output_path:
        img.save(output_path)
        logger.info(f"Saved visualization: {output_path}")

    return img
