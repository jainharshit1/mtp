#!/usr/bin/env python3
"""Phase 2 setup verification: environment, imports, forward pass, token spans."""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def check_environment():
    """Verify all required packages are installed."""
    logger.info("=== Step 1: Environment Check ===")
    checks = {}

    try:
        import torch
        checks["torch"] = torch.__version__
        checks["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            checks["cuda_version"] = torch.version.cuda
            checks["gpu_name"] = torch.cuda.get_device_name(0)
            checks["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_mem / 1e9
    except ImportError:
        logger.error("PyTorch not installed")
        return False

    for pkg_name in ["transformers", "peft", "pycocotools"]:
        try:
            mod = __import__(pkg_name)
            checks[pkg_name] = getattr(mod, "__version__", "installed")
        except ImportError:
            logger.error(f"{pkg_name} not installed")
            return False

    for k, v in checks.items():
        logger.info(f"  {k}: {v}")

    if not checks.get("cuda_available"):
        logger.error("CUDA not available")
        return False

    return True


def check_data():
    """Verify ODVG data and taxonomy."""
    logger.info("\n=== Step 2: Data Check ===")
    from pathlib import Path
    import json

    from src.data.taxonomy import UNIFIED_CLASSES, NUM_CLASSES
    logger.info(f"  Taxonomy: {NUM_CLASSES} classes: {UNIFIED_CLASSES}")

    odvg_dir = Path(PROJECT_ROOT) / "data" / "odvg"
    for fold_id in range(8):
        fold_dir = odvg_dir / f"fold_{fold_id}"
        for fname in ["train.jsonl", "val.jsonl", "test_annotations.json", "label_map.json"]:
            fpath = fold_dir / fname
            if not fpath.exists():
                logger.error(f"Missing: {fpath}")
                return False

        # Check label_map
        with open(fold_dir / "label_map.json") as f:
            lm = json.load(f)
        if len(lm) != NUM_CLASSES:
            logger.error(f"Fold {fold_id} label_map has {len(lm)} classes, expected {NUM_CLASSES}")
            return False

    logger.info(f"  All 8 folds present with {NUM_CLASSES}-class label maps")
    return True


def check_prompt_builder():
    """Verify prompt builder and token spans."""
    logger.info("\n=== Step 3: Prompt Builder & Token Spans ===")
    from src.data.prompt_builder import PromptBuilder
    from transformers import AutoTokenizer
    from src.training.loss_masking import compute_class_token_spans

    pb = PromptBuilder(strategy_name="bare_class_names")
    caption, cat_list = pb.build_caption_and_cat_list()
    logger.info(f"  Caption: {caption}")
    logger.info(f"  Categories: {cat_list}")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    spans = compute_class_token_spans(caption, cat_list, tokenizer)

    logger.info(f"  Token spans computed for {len(spans)} classes")
    for cid in sorted(spans):
        s, e = spans[cid]
        tokens = tokenizer.decode(
            tokenizer(caption, return_offsets_mapping=True)["input_ids"][s:e]
        )
        logger.info(f"    class {cid} ({cat_list[cid]}): tokens [{s},{e}) = '{tokens}'")

    return True


def check_dataset_loading():
    """Test dataset loading with a few samples."""
    logger.info("\n=== Step 4: Dataset Loading ===")
    from src.data.prompt_builder import PromptBuilder
    from src.training.odvg_dataset import MaskedODVGDataset, make_odvg_transforms, collate_fn
    import torch

    pb = PromptBuilder()
    transforms = make_odvg_transforms("train", 800)
    ds = MaskedODVGDataset(
        str(Path(PROJECT_ROOT) / "data" / "odvg" / "fold_0" / "train.jsonl"),
        pb, transforms=transforms,
    )
    logger.info(f"  Dataset size: {len(ds)}")

    # Load a few samples
    loader = torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=collate_fn)
    images, targets = next(iter(loader))

    logger.info(f"  Batch: {len(images)} images")
    for i, (img, tgt) in enumerate(zip(images, targets)):
        logger.info(
            f"    img[{i}]: shape={img.shape}, "
            f"boxes={tgt['boxes'].shape}, labels={tgt['labels'].tolist()}, "
            f"annotated={tgt['annotated_classes'].sum().item()}/{tgt['annotated_classes'].shape[0]} classes"
        )

    return True


def check_coco_eval():
    """Run synthetic COCOeval validation."""
    logger.info("\n=== Step 5: COCO Eval Validation ===")
    from src.evaluation.evaluate import validate_coco_eval_synthetic
    result = validate_coco_eval_synthetic()
    if result:
        logger.info("  Synthetic COCOeval: PASSED")
    else:
        logger.error("  Synthetic COCOeval: FAILED")
    return result


def main():
    from pathlib import Path
    os.chdir(PROJECT_ROOT)

    results = {}
    for name, check_fn in [
        ("environment", check_environment),
        ("data", check_data),
        ("prompt_builder", check_prompt_builder),
        ("dataset_loading", check_dataset_loading),
        ("coco_eval", check_coco_eval),
    ]:
        try:
            results[name] = check_fn()
        except Exception as e:
            logger.error(f"{name} check failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    logger.info("\n=== SUMMARY ===")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"  {name}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        logger.info("\nAll checks passed. Ready for training.")
    else:
        logger.error("\nSome checks failed. Fix issues before training.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
