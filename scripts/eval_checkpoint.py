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

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    set_peft_model_state_dict(model, ckpt["lora_state_dict"])
    if not args.no_ema and "ema_state_dict" in ckpt:
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
    res = evaluate_fold(model, str(fold_dir / "test_annotations.json"), pb, cfg, device,
                        max_images=args.max_images)

    out = ckpt_path.parent / f"eval_{ckpt_path.stem}.json"
    res.update({"checkpoint": str(ckpt_path), "epoch": ckpt.get("epoch"), "weights": weights})
    json.dump(res, open(out, "w"), indent=2, default=float)
    print({k: round(v, 4) for k, v in res.items() if k.startswith("mAP")})
    print("saved", out)


if __name__ == "__main__":
    main()
