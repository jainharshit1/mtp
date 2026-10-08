#!/usr/bin/env python3
"""Launch LODO training with comprehensive file logging."""
import sys
import os
import logging
import time
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "vendor" / "Open-GroundingDino"))
os.chdir(str(PROJECT_ROOT))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "training.log"
    status_file = output_dir / "status.json"

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    return log_file, status_file


def write_status(status_file: Path, **kwargs):
    data = {}
    if status_file.exists():
        try:
            data = json.loads(status_file.read_text())
        except Exception:
            pass
    for k, v in kwargs.items():
        if v is None:
            data.pop(k, None)
        else:
            data[k] = v
    data["updated_at"] = datetime.now().isoformat()
    status_file.write_text(json.dumps(data, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(description="Run LODO training")
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--folds", type=str, default=None,
                        help="Comma-separated fold IDs, e.g. '0' or '0,1,2'. Default: all 8")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config.get("output", {}).get("base_dir", "outputs"))
    log_file, status_file = setup_logging(output_dir)

    logger = logging.getLogger("run_training")
    logger.info("=" * 70)
    logger.info("LODO TRAINING STARTED")
    logger.info("=" * 70)
    logger.info(f"Config: {args.config}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Status file: {status_file}")

    import torch
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Free VRAM: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")

    fold_ids = list(range(8))
    if args.folds:
        fold_ids = [int(x.strip()) for x in args.folds.split(",")]
    logger.info(f"Folds to train: {fold_ids}")

    write_status(status_file,
                 state="started",
                 folds=fold_ids,
                 current_fold=None,
                 started_at=datetime.now().isoformat(),
                 pid=os.getpid())

    from src.training.lodo_runner import LODORunner
    runner = LODORunner(args.config)

    # Monkey-patch the trainer to write status updates
    from src.training.trainer import Trainer

    _orig_load_checkpoint = Trainer.load_checkpoint

    def _patched_load_checkpoint(self, path):
        resume_epoch = _orig_load_checkpoint(self, path)
        write_status(status_file,
                     current_epoch=resume_epoch,
                     max_epochs=self.max_epochs,
                     state="training")
        return resume_epoch

    Trainer.load_checkpoint = _patched_load_checkpoint
    _orig_train_one_epoch = Trainer.train_one_epoch

    def _patched_train_one_epoch(self, epoch):
        write_status(status_file,
                     current_epoch=epoch,
                     max_epochs=self.max_epochs,
                     state="training")
        result = _orig_train_one_epoch(self, epoch)
        write_status(status_file,
                     last_train_loss=result["loss"],
                     last_cls_loss=result["cls_loss"],
                     last_box_loss=result["box_loss"],
                     vram_gb=round(torch.cuda.memory_reserved() / 1e9, 2))
        return result

    _orig_validate = Trainer.validate

    def _patched_validate(self, epoch):
        write_status(status_file, state="validating")
        result = _orig_validate(self, epoch)
        write_status(status_file,
                     last_val_loss=result["loss"],
                     state="training")
        return result

    Trainer.train_one_epoch = _patched_train_one_epoch
    Trainer.validate = _patched_validate

    start_time = time.time()
    completed_folds = []
    all_results = {}

    for i, fold_id in enumerate(fold_ids):
        fold_start = time.time()
        logger.info(f"\n{'#' * 70}")
        logger.info(f"FOLD {fold_id} ({i + 1}/{len(fold_ids)})")
        logger.info(f"{'#' * 70}")

        write_status(status_file,
                     current_fold=fold_id,
                     fold_progress=f"{i + 1}/{len(fold_ids)}",
                     current_epoch=0,
                     state="training",
                     fold_started_at=datetime.now().isoformat(),
                     error=None)

        try:
            result = runner.run_fold(fold_id, resume=not args.no_resume)
            all_results[fold_id] = result
            completed_folds.append(fold_id)

            fold_elapsed = time.time() - fold_start
            total_elapsed = time.time() - start_time
            remaining_folds = len(fold_ids) - (i + 1)
            avg_fold_time = total_elapsed / (i + 1)
            eta = timedelta(seconds=int(avg_fold_time * remaining_folds))

            logger.info(f"Fold {fold_id} done in {timedelta(seconds=int(fold_elapsed))}")
            logger.info(f"  mAP={result.get('mAP', 0):.4f}")
            logger.info(f"  Elapsed: {timedelta(seconds=int(total_elapsed))}, ETA: {eta}")

            write_status(status_file,
                         state="fold_complete",
                         completed_folds=completed_folds,
                         last_mAP=result.get("mAP", 0),
                         elapsed=str(timedelta(seconds=int(total_elapsed))),
                         eta=str(eta))

        except Exception as e:
            logger.error(f"Fold {fold_id} FAILED: {e}", exc_info=True)
            write_status(status_file,
                         state="error",
                         error=str(e),
                         failed_fold=fold_id)
            continue

    total_time = time.time() - start_time
    logger.info(f"\n{'=' * 70}")
    logger.info("LODO TRAINING COMPLETE")
    logger.info(f"{'=' * 70}")
    logger.info(f"Total time: {timedelta(seconds=int(total_time))}")
    logger.info(f"Completed folds: {completed_folds}")

    if all_results:
        maps = [r.get("mAP", 0) for r in all_results.values()]
        logger.info(f"Mean mAP: {sum(maps) / len(maps):.4f} ± {(max(maps) - min(maps)) / 2:.4f}")

    write_status(status_file,
                 state="complete",
                 completed_folds=completed_folds,
                 total_time=str(timedelta(seconds=int(total_time))),
                 current_fold=None,
                 current_epoch=None)


if __name__ == "__main__":
    main()
