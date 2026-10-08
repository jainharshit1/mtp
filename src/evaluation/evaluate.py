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
    annotated_classes_path: str = None,
    max_images: int = None,
) -> dict:
    """Run evaluation on a COCO-format test set.

    Matches the training setup: images are resized exactly like the val transform
    (short side = data.image_size), scores are kept down to `score_threshold`
    (default 0.001, top `max_dets` per image) so the PR curve isn't truncated, and
    predictions for classes the held-out dataset does not annotate are dropped
    (training masks those classes too), using annotated_classes.json next to the
    test file.

    Returns dict with mAP, mAP_50, mAP_75, per-class AP.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from PIL import Image
    import torchvision.transforms.functional as F
    from src.training.odvg_dataset import resize

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_cfg = config.get("evaluation", {})
    score_threshold = eval_cfg.get("score_threshold", 0.001)
    max_dets = eval_cfg.get("max_dets_per_image", 100)
    image_size = config.get("data", {}).get("image_size", 800)

    if annotated_classes_path is None:
        annotated_classes_path = str(Path(test_annotations_path).parent / "annotated_classes.json")
    annotated = None
    if Path(annotated_classes_path).exists():
        with open(annotated_classes_path) as f:
            annotated = {int(k): set(v) for k, v in json.load(f).items()}
    else:
        logger.warning(f"{annotated_classes_path} not found: not masking unannotated classes")

    coco_gt = COCO(test_annotations_path)
    image_ids = coco_gt.getImgIds()
    if max_images and max_images < len(image_ids):
        import random
        image_ids = random.Random(0).sample(image_ids, max_images)  # fixed seed: comparable across checkpoints

    caption, cat_list = prompt_builder.build_caption_and_cat_list()

    model.eval()
    results = []

    from src.training.loss_masking import compute_class_token_spans
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.get("model", {}).get("bert_model", "bert-base-uncased")
    )
    class_to_spans = compute_class_token_spans(caption, cat_list, tokenizer)
    class_ids = sorted(class_to_spans)

    for img_id in image_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        img = Image.open(img_info["file_name"]).convert("RGB")
        w_orig, h_orig = img.size

        # Same preprocessing as training/val: resize short side to image_size, then normalise.
        img, _ = resize(img, {}, image_size, 1333)
        img_tensor = F.normalize(F.to_tensor(img), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            outputs = model(img_tensor, captions=[caption])

        pred_logits = outputs["pred_logits"].sigmoid()[0].float()  # [Q, max_text_len]
        pred_boxes = outputs["pred_boxes"][0].float()              # [Q, 4] cxcywh normalised

        # Score of a class for a query = max over that class's tokens.
        scores = torch.stack(
            [pred_logits[:, class_to_spans[c][0]:class_to_spans[c][1]].max(dim=1).values
             for c in class_ids], dim=1)                            # [Q, C]

        if annotated is not None and img_id in annotated:
            keep = torch.tensor([c in annotated[img_id] for c in class_ids], device=scores.device)
            scores = scores.masked_fill(~keep.unsqueeze(0), 0.0)

        top_scores, top_idx = scores.flatten().topk(min(max_dets, scores.numel()))
        sel = top_scores >= score_threshold
        top_scores, top_idx = top_scores[sel], top_idx[sel]
        if top_idx.numel() == 0:
            continue
        q_idx = top_idx // len(class_ids)
        c_idx = top_idx % len(class_ids)

        boxes = box_cxcywh_to_xyxy(pred_boxes[q_idx]) * torch.tensor(
            [w_orig, h_orig, w_orig, h_orig], device=pred_boxes.device)
        for box, c, sc in zip(boxes.tolist(), c_idx.tolist(), top_scores.tolist()):
            x1, y1, x2, y2 = box
            results.append({
                "image_id": img_id,
                "category_id": class_ids[c],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": sc,
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
