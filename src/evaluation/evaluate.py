"""COCO mAP evaluation for Grounding DINO predictions."""

import json
import logging
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np

from src.data.taxonomy import UNIFIED_CLASSES, NUM_CLASSES
from src.training.odvg_dataset import box_cxcywh_to_xyxy

logger = logging.getLogger(__name__)


def evaluate_fold(
    model,
    test_annotations_path: str,
    prompt_builder,
    config: dict,
    device: torch.device = None,
) -> dict:
    """Run evaluation on a COCO-format test set.

    Returns dict with mAP, mAP_50, mAP_75, per-class AP.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from PIL import Image
    import torchvision.transforms.functional as F

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_cfg = config.get("evaluation", {})
    box_threshold = eval_cfg.get("box_threshold", 0.3)

    coco_gt = COCO(test_annotations_path)
    image_ids = coco_gt.getImgIds()

    caption, cat_list = prompt_builder.build_caption_and_cat_list()

    model.eval()
    results = []

    from src.training.loss_masking import compute_class_token_spans
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.get("model", {}).get("bert_model", "bert-base-uncased")
    )
    class_to_spans = compute_class_token_spans(caption, cat_list, tokenizer)

    for img_id in image_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = img_info["file_name"]
        img = Image.open(img_path).convert("RGB")
        w_orig, h_orig = img.size

        # Preprocess
        img_tensor = F.to_tensor(img)
        img_tensor = F.normalize(img_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            outputs = model(img_tensor, captions=[caption])

        pred_logits = outputs["pred_logits"].sigmoid()[0]  # [num_queries, max_text_len]
        pred_boxes = outputs["pred_boxes"][0]  # [num_queries, 4] cxcywh normalized

        # Map predictions to classes via token spans
        for query_idx in range(pred_logits.shape[0]):
            for class_id, (tok_start, tok_end) in class_to_spans.items():
                score = pred_logits[query_idx, tok_start:tok_end].max().item()
                if score < box_threshold:
                    continue

                # Convert box: normalized cxcywh -> absolute xyxy -> COCO xywh
                box = pred_boxes[query_idx]
                box_abs = box_cxcywh_to_xyxy(box.unsqueeze(0))[0]
                box_abs = box_abs * torch.tensor(
                    [w_orig, h_orig, w_orig, h_orig],
                    device=box_abs.device, dtype=box_abs.dtype,
                )
                x1, y1, x2, y2 = box_abs.tolist()
                coco_box = [x1, y1, x2 - x1, y2 - y1]

                results.append({
                    "image_id": img_id,
                    "category_id": class_id,
                    "bbox": coco_box,
                    "score": score,
                })

    if not results:
        logger.warning("No predictions generated")
        return {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0, "per_class": {}}

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Per-class AP
    per_class = {}
    cat_ids = coco_gt.getCatIds()
    for cat_id in cat_ids:
        coco_eval_cls = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval_cls.params.catIds = [cat_id]
        coco_eval_cls.evaluate()
        coco_eval_cls.accumulate()
        coco_eval_cls.summarize()

        cat_name = UNIFIED_CLASSES[cat_id] if cat_id < NUM_CLASSES else f"class_{cat_id}"
        n_anns = len(coco_gt.getAnnIds(catIds=[cat_id]))
        per_class[cat_name] = {
            "AP": coco_eval_cls.stats[0],
            "AP_50": coco_eval_cls.stats[1],
            "support": n_anns,
        }

    return {
        "mAP": coco_eval.stats[0],
        "mAP_50": coco_eval.stats[1],
        "mAP_75": coco_eval.stats[2],
        "per_class": per_class,
    }


def validate_coco_eval_synthetic() -> bool:
    """Synthetic test: perfect predictions should give mAP=1.0.

    Creates a tiny COCO dataset with category IDs 0..7,
    generates perfect predictions, and validates COCOeval works.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import tempfile, os

    gt = {
        "images": [
            {"id": 1, "width": 100, "height": 100, "file_name": "test.jpg"},
            {"id": 2, "width": 100, "height": 100, "file_name": "test2.jpg"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0,
             "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 1,
             "bbox": [50, 50, 30, 30], "area": 900, "iscrowd": 0},
            {"id": 3, "image_id": 2, "category_id": 3,
             "bbox": [5, 5, 40, 40], "area": 1600, "iscrowd": 0},
        ],
        "categories": [
            {"id": i, "name": UNIFIED_CLASSES[i]} for i in range(NUM_CLASSES)
        ],
    }

    # Perfect predictions (matching GT exactly)
    preds = [
        {"image_id": 1, "category_id": 0,
         "bbox": [10, 10, 20, 20], "score": 0.99},
        {"image_id": 1, "category_id": 1,
         "bbox": [50, 50, 30, 30], "score": 0.99},
        {"image_id": 2, "category_id": 3,
         "bbox": [5, 5, 40, 40], "score": 0.99},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(gt, f)
        gt_path = f.name

    try:
        coco_gt = COCO(gt_path)
        coco_dt = coco_gt.loadRes(preds)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        ap = coco_eval.stats[0]
        logger.info(f"Synthetic COCO eval: mAP={ap:.4f} (expected 1.0)")
        success = ap > 0.99

        # Test empty predictions (pycocotools crashes on empty list, handle gracefully)
        try:
            empty_dt = coco_gt.loadRes([])
            empty_eval = COCOeval(coco_gt, empty_dt, "bbox")
            empty_eval.evaluate()
            empty_eval.accumulate()
            empty_eval.summarize()
            logger.info(f"Empty predictions: mAP={empty_eval.stats[0]:.4f} (expected ~0)")
        except (IndexError, Exception):
            logger.info("Empty predictions: pycocotools cannot handle empty list (expected, handled in evaluate_fold)")

        return success
    finally:
        os.unlink(gt_path)
