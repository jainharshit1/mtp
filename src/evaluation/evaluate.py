"""COCO mAP evaluation for Grounding DINO predictions."""

import contextlib
import io
import json
import logging
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np

from src.data.taxonomy import UNIFIED_CLASSES, NUM_CLASSES
from src.training.odvg_dataset import box_cxcywh_to_xyxy, box_frame, make_nested

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
    (data.fixed_square: padded to a square and resized to image_size x image_size; otherwise
    short side = data.image_size), scores are kept down to `score_threshold`
    (default 0.001, top `max_dets` per image) so the PR curve isn't truncated, and
    predictions for classes the held-out dataset does not annotate are dropped
    (training masks those classes too), using annotated_classes.json next to the
    test file.

    Returns dict with mAP, mAP_50, mAP_75, per-class AP.
    """
    from pycocotools.coco import COCO

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_cfg = config.get("evaluation", {})
    score_threshold = eval_cfg.get("score_threshold", 0.001)
    max_dets = eval_cfg.get("max_dets_per_image", 100)
    image_size = config.get("data", {}).get("image_size", 800)
    fixed_square = config.get("data", {}).get("fixed_square", False)
    pad_mask = fixed_square and config.get("data", {}).get("pad_mask", False)

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
    # Relative image paths are relative to the folder holding fold_*/ (a self-contained build).
    image_root = Path(test_annotations_path).resolve().parent.parent
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

    # With fixed_square every image is image_size x image_size, so images can be batched; with
    # varying shapes they run one at a time (batching would pad them and change the results).
    batch_size = eval_cfg.get("batch_size", 16) if fixed_square else 1
    files = [(img_id, str(image_root / coco_gt.loadImgs(img_id)[0]["file_name"])) for img_id in image_ids]
    loader = torch.utils.data.DataLoader(
        _EvalImages(files, image_size, fixed_square, pad_mask), batch_size=batch_size, shuffle=False,
        num_workers=eval_cfg.get("num_workers", config.get("data", {}).get("num_workers", 0)),
        collate_fn=_collate_eval, pin_memory=device.type == "cuda",
    )

    for ids, sizes, valid, batch in loader:
        # Mask = True on padding (the square padding too, with pad_mask), as in training.
        samples = make_nested(list(batch), valid).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model(samples, captions=[caption] * len(ids))

        for k, (img_id, (w_orig, h_orig)) in enumerate(zip(ids, sizes)):
            results.extend(detections_from_outputs(
                outputs["pred_logits"][k], outputs["pred_boxes"][k], class_to_spans, img_id,
                w_orig, h_orig, fixed_square=fixed_square, pad_mask=pad_mask,
                annotated=annotated.get(img_id) if annotated is not None else None,
                score_threshold=score_threshold, max_dets=max_dets,
            ))

    return coco_map(coco_gt, results, per_class=True)


class _EvalImages(torch.utils.data.Dataset):
    """Test images preprocessed exactly like the val transform.

    Returns (id, (w, h), [h, w] of the real image inside the input or None, tensor).
    """

    def __init__(self, files: list, image_size: int, fixed_square: bool, pad_mask: bool = False):
        self.files, self.image_size = files, image_size
        self.fixed_square, self.pad_mask = fixed_square, pad_mask

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        from PIL import Image
        import torchvision.transforms.functional as F
        from src.training.odvg_dataset import prepare_eval_image, square_valid_size

        img_id, path = self.files[i]
        img = Image.open(path).convert("RGB")
        size = img.size
        valid = square_valid_size(*size, self.image_size) if self.pad_mask else None
        img = prepare_eval_image(img, self.image_size, self.fixed_square)
        return img_id, size, valid, F.normalize(F.to_tensor(img), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def _collate_eval(items):
    ids, sizes, valid, tensors = zip(*items)
    return list(ids), list(sizes), list(valid), torch.stack(tensors)


def detections_from_outputs(pred_logits, pred_boxes, class_to_spans: dict, image_id: int,
                            width: int, height: int, fixed_square: bool = False, annotated=None,
                            score_threshold: float = 0.001, max_dets: int = 100,
                            pad_mask: bool = False) -> list[dict]:
    """COCO detections for one image from the model's raw outputs ([Q, text_len], [Q, 4]).

    Boxes are normalised to the network input: with fixed_square and no pad mask that is the image
    padded on the right/bottom to a square of side max(width, height), so scaling by that side
    gives original pixel coordinates; with a pad mask (or without fixed_square) it is the real
    image area, so scaling by (width, height) does. Classes not in `annotated` score 0.
    """
    class_ids = sorted(class_to_spans)
    logits = pred_logits.sigmoid().float()
    boxes_n = pred_boxes.float()

    # Score of a class for a query = max over that class's tokens.
    scores = torch.stack(
        [logits[:, class_to_spans[c][0]:class_to_spans[c][1]].max(dim=1).values
         for c in class_ids], dim=1)                                # [Q, C]

    if annotated is not None:
        keep = torch.tensor([c in annotated for c in class_ids], device=scores.device)
        scores = scores.masked_fill(~keep.unsqueeze(0), 0.0)

    top_scores, top_idx = scores.flatten().topk(min(max_dets, scores.numel()))
    sel = top_scores >= score_threshold
    top_scores, top_idx = top_scores[sel], top_idx[sel]
    if top_idx.numel() == 0:
        return []
    q_idx = top_idx // len(class_ids)
    c_idx = top_idx % len(class_ids)

    sx, sy = box_frame(width, height, fixed_square, pad_mask)
    boxes = box_cxcywh_to_xyxy(boxes_n[q_idx]) * torch.tensor([sx, sy, sx, sy], device=boxes_n.device)
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height)
    return [
        {"image_id": image_id, "category_id": class_ids[c],
         "bbox": [x1, y1, x2 - x1, y2 - y1], "score": sc}
        for (x1, y1, x2, y2), c, sc in zip(boxes.tolist(), c_idx.tolist(), top_scores.tolist())
    ]


def coco_from_odvg_lines(lines: list[str]):
    """COCO ground truth from ODVG jsonl lines; image id = line index (as in MaskedODVGDataset)."""
    from pycocotools.coco import COCO

    images, annotations = [], []
    for idx, line in enumerate(lines):
        data = json.loads(line)
        images.append({"id": idx, "width": data["width"], "height": data["height"],
                       "file_name": data["filename"]})
        for inst in data["detection"]["instances"]:
            if inst["label"] >= NUM_CLASSES:
                continue
            x1, y1, x2, y2 = inst["bbox"]
            annotations.append({"id": len(annotations) + 1, "image_id": idx, "category_id": inst["label"],
                                "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1),
                                "iscrowd": 0})
    coco = COCO()
    coco.dataset = {"images": images, "annotations": annotations,
                    "categories": [{"id": i, "name": n} for i, n in enumerate(UNIFIED_CLASSES)]}
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def coco_map(coco_gt, results: list[dict], per_class: bool = False, quiet: bool = False) -> dict:
    """COCO bbox mAP of `results` against `coco_gt`; per-class AP if asked."""
    from pycocotools.cocoeval import COCOeval

    if not results:
        logger.warning("No predictions generated")
        return {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0, "per_class": {}}

    out = io.StringIO() if quiet else None
    with contextlib.redirect_stdout(out) if quiet else contextlib.nullcontext():
        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    if not per_class:
        return {"mAP": coco_eval.stats[0], "mAP_50": coco_eval.stats[1], "mAP_75": coco_eval.stats[2]}

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
