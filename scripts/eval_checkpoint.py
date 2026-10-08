#!/usr/bin/env python3
"""Evaluate any saved checkpoint on a fold's held-out test set.

    python scripts/eval_checkpoint.py --fold 0                       # outputs/fold_0/best.pt (EMA weights)
    python scripts/eval_checkpoint.py --fold 0 --ckpt outputs/fold_0/epoch_14.pt
    python scripts/eval_checkpoint.py --fold 0 --max-images 50       # quick smoke test

Writes outputs/fold_N/eval_<ckpt-name>.json next to the checkpoint.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import os
os.chdir(ROOT)

import torch
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--ckpt", default=None, help="default: outputs/fold_N/best.pt")
    ap.add_argument("--config", default="configs/base_config.yaml")
    ap.add_argument("--no-ema", action="store_true", help="use raw LoRA weights instead of EMA")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--zero-shot", action="store_true",
                    help="evaluate the pretrained Grounding DINO with no fine-tuning (baseline)")
    ap.add_argument("--on-train", type=int, default=None, metavar="N",
                    help="evaluate on N random annotated TRAIN images instead of the held-out test set")
    args = ap.parse_args()

    from src.data.prompt_builder import PromptBuilder
    from src.evaluation.evaluate import evaluate_fold
    from src.training.loss_masking import compute_class_token_spans
    from src.training.model_builder import build_model_and_criterion
    from transformers import AutoTokenizer
    from peft import set_peft_model_state_dict

    cfg = yaml.safe_load(open(args.config))
    ckpt_path = Path(args.ckpt or f"outputs/fold_{args.fold}/best.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pb = PromptBuilder(strategy_name=cfg.get("prompt_strategy", "bare_class_names"))
    caption, cat_list = pb.build_caption_and_cat_list()
    tok = AutoTokenizer.from_pretrained(cfg["model"].get("bert_model", "bert-base-uncased"))
    spans = compute_class_token_spans(caption, cat_list, tok)
    model, _, _ = build_model_and_criterion(cfg, device, class_to_token_spans=spans)

    if args.zero_shot:
        ckpt, ckpt_path = {"epoch": None}, Path(f"outputs/fold_{args.fold}/zero_shot.pt")
    else:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        set_peft_model_state_dict(model, ckpt["lora_state_dict"])
    if args.zero_shot:
        weights = "pretrained (no LoRA)"
    elif not args.no_ema and "ema_state_dict" in ckpt:
        shadow = ckpt["ema_state_dict"]["shadow"]
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and n in shadow:
                    p.data.copy_(shadow[n])
        weights = "EMA"
    else:
        weights = "raw"
    print(f"Evaluating {ckpt_path} (epoch {ckpt.get('epoch')}, {weights} weights)")

    fold_dir = Path(cfg["data"]["odvg_dir"]) / f"fold_{args.fold}"
    if args.on_train:
        import random
        from src.data.taxonomy import ID_TO_CLASS
        lines = [json.loads(l) for l in open(fold_dir / "train.jsonl")]
        random.Random(1).shuffle(lines)
        lines = [d for d in lines if d["detection"]["instances"]][:args.on_train]
        images, anns = [], []
        for i, d in enumerate(lines):
            images.append({"id": i, "file_name": d["filename"], "width": d["width"], "height": d["height"]})
            for b in d["detection"]["instances"]:
                x1, y1, x2, y2 = b["bbox"]
                anns.append({"id": len(anns), "image_id": i, "category_id": b["label"],
                             "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
        import tempfile
        tmp = Path(tempfile.mkdtemp()) / "train_coco.json"   # no annotated_classes.json next to it -> no class masking
        json.dump({"images": images, "annotations": anns,
                   "categories": [{"id": i, "name": n} for i, n in ID_TO_CLASS.items()]}, open(tmp, "w"))
        ann_path, split = str(tmp), "train"
    else:
        ann_path, split = str(fold_dir / "test_annotations.json"), "test"
    res = evaluate_fold(model, ann_path, pb, cfg, device, max_images=args.max_images)

    out = ckpt_path.parent / f"eval_{ckpt_path.stem}_{split}.json"
    res.update({"split": split, "n_images": args.on_train or args.max_images, "checkpoint": str(ckpt_path), "epoch": ckpt.get("epoch"), "weights": weights})
    json.dump(res, open(out, "w"), indent=2, default=float)
    print({k: round(v, 4) for k, v in res.items() if k.startswith("mAP")})
    print("saved", out)


if __name__ == "__main__":
    main()
