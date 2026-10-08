"""LODO (Leave-One-Dataset-Out) runner: trains and evaluates all folds."""

import gc
import json
import logging
import platform
import sys
from pathlib import Path

import torch
import yaml
import numpy as np

from src.data.prompt_builder import PromptBuilder
from src.data.taxonomy import UNIFIED_CLASSES, NUM_CLASSES
from src.training.odvg_dataset import MaskedODVGDataset, make_odvg_transforms, collate_fn
from src.training.model_builder import build_model_and_criterion, build_optimizer, build_scheduler
from src.training.trainer import Trainer
from src.training.loss_masking import compute_class_token_spans
from src.evaluation.evaluate import evaluate_fold

logger = logging.getLogger(__name__)


class LODORunner:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.config_path = config_path
        self.odvg_dir = Path(self.config.get("data", {}).get("odvg_dir", "data/odvg"))
        self.output_base = Path(self.config.get("output", {}).get("base_dir", "outputs"))

    def run_fold(self, fold_id: int, resume: bool = True) -> dict:
        """Train and evaluate a single LODO fold."""
        logger.info(f"\n{'='*60}\nStarting fold {fold_id}\n{'='*60}")

        fold_dir = self.odvg_dir / f"fold_{fold_id}"
        output_dir = self.output_base / f"fold_{fold_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True

        # Set seed
        seed = self.config.get("seed", 42) + fold_id
        _set_seed(seed)

        # Prompt builder
        strategy = self.config.get("prompt_strategy", "bare_class_names")
        prompt_builder = PromptBuilder(strategy_name=strategy)

        # Datasets
        image_size = self.config.get("data", {}).get("image_size", 800)
        train_transforms = make_odvg_transforms("train", image_size)
        val_transforms = make_odvg_transforms("val", image_size)

        train_ds = MaskedODVGDataset(
            str(fold_dir / "train.jsonl"), prompt_builder, transforms=train_transforms,
        )
        val_ds = MaskedODVGDataset(
            str(fold_dir / "val.jsonl"), prompt_builder, transforms=val_transforms,
        )

        data_cfg = self.config.get("data", {})
        batch_size = self.config.get("training", {}).get("batch_size", 4)

        num_workers = data_cfg.get("num_workers", 4)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers,
            pin_memory=data_cfg.get("pin_memory", True),
            collate_fn=collate_fn, drop_last=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers,
            pin_memory=data_cfg.get("pin_memory", True),
            collate_fn=collate_fn,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
        )

        # Token spans (must be computed before building model)
        caption, cat_list = prompt_builder.build_caption_and_cat_list()
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.get("model", {}).get("bert_model", "bert-base-uncased")
        )
        class_to_spans = compute_class_token_spans(caption, cat_list, tokenizer)

        # Model (with MaskedSetCriterion wrapping)
        model, criterion, postprocessors = build_model_and_criterion(
            self.config, device, class_to_token_spans=class_to_spans
        )
        optimizer = build_optimizer(model, self.config)
        steps_per_epoch = len(train_loader) // self.config.get("training", {}).get(
            "gradient_accumulation_steps", 4
        )
        scheduler = build_scheduler(optimizer, self.config, steps_per_epoch)

        # Trainer
        trainer = Trainer(
            model=model, criterion=criterion, optimizer=optimizer,
            scheduler=scheduler, train_loader=train_loader,
            val_loader=val_loader, config=self.config,
            output_dir=str(output_dir), class_to_token_spans=class_to_spans,
            prompt_builder=prompt_builder, device=device,
        )

        # Resume from checkpoint
        checkpoint_path = output_dir / "latest.pt"
        if resume and checkpoint_path.exists():
            trainer.load_checkpoint(str(checkpoint_path))

        # Train
        train_results = trainer.train()

        # Load best checkpoint for evaluation
        best_path = output_dir / "best.pt"
        if best_path.exists():
            logger.info(f"Loading best checkpoint for evaluation: {best_path}")
            best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
            if "lora_state_dict" in best_ckpt:
                from peft import set_peft_model_state_dict
                set_peft_model_state_dict(model, best_ckpt["lora_state_dict"])
            if trainer.ema is not None and "ema_state_dict" in best_ckpt:
                trainer.ema.load_state_dict(best_ckpt["ema_state_dict"])
                trainer.ema.apply_shadow(model)

        # Evaluate
        test_ann_path = str(fold_dir / "test_annotations.json")
        eval_results = evaluate_fold(model, test_ann_path, prompt_builder, self.config, device)

        # Save results
        all_results = {
            "fold": fold_id,
            "train": train_results,
            **eval_results,
        }
        with open(output_dir / "eval_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        # Save reproducibility metadata
        _save_metadata(output_dir, self.config, fold_id, seed, train_ds, val_ds)

        # Cleanup GPU
        del model, criterion, optimizer, scheduler, trainer
        del train_loader, val_loader, train_ds, val_ds
        gc.collect()
        torch.cuda.empty_cache()

        logger.info(f"Fold {fold_id} complete: mAP={eval_results.get('mAP', 0):.4f}")
        return all_results

    def run_all(self, fold_ids: list = None, resume: bool = True) -> dict:
        """Run all LODO folds sequentially."""
        if fold_ids is None:
            fold_ids = list(range(8))

        results = {}
        for fold_id in fold_ids:
            results[fold_id] = self.run_fold(fold_id, resume=resume)

        # Aggregate
        from src.evaluation.aggregate import aggregate_results, print_report
        summary = aggregate_results(str(self.output_base))
        print_report(summary)

        with open(self.output_base / "lodo_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        return summary


def _set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_metadata(output_dir: Path, config: dict, fold_id: int, seed: int,
                   train_ds, val_ds):
    """Save reproducibility metadata."""
    meta = {
        "fold_id": fold_id,
        "seed": seed,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "num_classes": NUM_CLASSES,
        "classes": UNIFIED_CLASSES,
        "config": config,
    }

    try:
        import transformers, peft
        meta["transformers_version"] = transformers.__version__
        meta["peft_version"] = peft.__version__
    except ImportError:
        pass

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
