"""Training loop with loss masking, EMA, mixed precision, and crash-resilient checkpoints."""

import gc
import hashlib
import json
import logging
import pickle
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.training.ema import LoRAModelEma
from src.training.loss_masking import build_annotated_token_mask

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        output_dir: str,
        class_to_token_spans: dict,
        prompt_builder,
        device: torch.device = None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.class_to_token_spans = class_to_token_spans
        self.prompt_builder = prompt_builder
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_cfg = config.get("training", {})
        self.max_epochs = train_cfg.get("max_epochs", 30)
        self.grad_accum_steps = train_cfg.get("gradient_accumulation_steps", 4)
        self.grad_clip_max_norm = train_cfg.get("grad_clip_max_norm", 0.1)
        self.patience = train_cfg.get("early_stopping_patience", 7)
        # "mAP" (val mAP, higher is better) or "loss" (val loss, lower is better). Picks best.pt
        # and drives early stopping; the test set is never looked at during training.
        self.monitor = train_cfg.get("early_stopping_metric", "mAP")
        assert self.monitor in ("mAP", "loss"), f"early_stopping_metric must be mAP or loss, got {self.monitor}"
        self.save_every_n = config.get("output", {}).get("save_every_n_epochs", 5)
        self.fixed_square = config.get("data", {}).get("fixed_square", False)
        self.pad_mask = self.fixed_square and config.get("data", {}).get("pad_mask", False)
        eval_cfg = config.get("evaluation", {})
        self.score_threshold = eval_cfg.get("score_threshold", 0.001)
        self.max_dets = eval_cfg.get("max_dets_per_image", 100)
        self.val_gt = None
        if self.monitor == "mAP":
            from src.evaluation.evaluate import coco_from_odvg_lines
            self.val_gt = coco_from_odvg_lines(val_loader.dataset.lines)

        # Mixed precision
        mp_cfg = config.get("mixed_precision", {})
        self.use_amp = mp_cfg.get("enabled", True)
        self.scaler = GradScaler(enabled=self.use_amp)

        # EMA
        ema_cfg = config.get("ema", {})
        self.ema = None
        if ema_cfg.get("enabled", True):
            self.ema = LoRAModelEma(model, decay=ema_cfg.get("decay", 0.9997))

        # State. best_metric is a score (val mAP, or -val loss): higher is better, None = no epoch yet.
        self.start_epoch = 0
        self.resume_batches = 0      # batches of start_epoch already trained (interrupted epoch)
        self.global_step = 0
        self.best_metric = None
        self.epochs_without_improvement = 0
        self.finished = False        # training ran to the end or stopped early; resume does nothing
        self._interrupted = False
        self._batches_done = 0

        # Fingerprint of the validation set, so a changed val split on resume
        # resets best-checkpoint tracking (val losses on different sets aren't comparable).
        self.val_fingerprint = hashlib.md5(
            "\n".join(getattr(getattr(val_loader, "dataset", None), "lines", [])).encode()
        ).hexdigest()[:12]

        # Fingerprint of the training data (train + val files and the prompt): a checkpoint from other
        # data (e.g. an old run in the same output dir) must not be resumed.
        self.data_fingerprint = hashlib.md5("\n".join(
            getattr(getattr(train_loader, "dataset", None), "lines", [])
            + getattr(getattr(val_loader, "dataset", None), "lines", [])
            + [str(getattr(getattr(train_loader, "dataset", None), "caption", ""))]
        ).encode()).hexdigest()[:12]

        # Config hash for resume verification
        self.config_hash = hashlib.md5(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()[:12]

    def _handle_signal(self, signum, frame):
        # Only set a flag: the loop stops at the next optimizer-step boundary and saves a
        # consistent checkpoint there. A second signal exits at once.
        signame = signal.Signals(signum).name
        if self._interrupted:
            raise KeyboardInterrupt(f"second {signame}")
        logger.warning(f"Received {signame}: stopping at the next optimizer step and saving a checkpoint")
        self._interrupted = True

    def train(self) -> dict:
        """Full training loop with early stopping on the validation set."""
        if self.finished or self.start_epoch >= self.max_epochs:
            logger.info(f"Training already complete (last epoch {self.start_epoch - 1}); nothing to resume")
            return {"best_metric": self.best_metric, "monitor": self.monitor,
                    "final_epoch": self.start_epoch - 1, "all_metrics": [], "interrupted": False}

        logger.info(f"Starting training for {self.max_epochs} epochs "
                    f"(best checkpoint and early stopping on val {self.monitor}, patience {self.patience})")
        logger.info(f"Output: {self.output_dir}")

        prev_sigterm = signal.signal(signal.SIGTERM, self._handle_signal)
        prev_sigint = signal.signal(signal.SIGINT, self._handle_signal)

        all_metrics = []
        epoch = self.start_epoch - 1

        try:
            for epoch in range(self.start_epoch, self.max_epochs):
                if self._interrupted:
                    break
                train_metrics = self.train_one_epoch(epoch)

                if self._interrupted and self._batches_done < len(self.train_loader):
                    # Mid-epoch stop: save where this epoch got to; resume finishes it.
                    self._save_checkpoint(epoch, {"interrupted_at_batch": self._batches_done},
                                          is_best=False, label="latest", epoch_complete=False)
                    logger.info(f"Interrupted in epoch {epoch} after {self._batches_done}/"
                                f"{len(self.train_loader)} batches; checkpoint saved, resume continues this epoch")
                    break

                val_metrics = self.validate(epoch)
                metrics = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
                all_metrics.append(metrics)

                logger.info(
                    f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f} "
                    f"val_loss={val_metrics['loss']:.4f} "
                    + (f"val_mAP={val_metrics['mAP']:.4f} val_mAP50={val_metrics['mAP_50']:.4f} "
                       if "mAP" in val_metrics else "")
                    + f"lr={self.optimizer.param_groups[0]['lr']:.2e} "
                    f"peak_mem={train_metrics['peak_mem_gb']:.1f}GB"
                )

                score = val_metrics["mAP"] if self.monitor == "mAP" else -val_metrics["loss"]
                is_best = self.best_metric is None or score > self.best_metric
                if is_best:
                    self.best_metric = score
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1

                stop = self.epochs_without_improvement >= self.patience
                if stop:
                    logger.info(f"Early stopping at epoch {epoch} (no val {self.monitor} improvement "
                                f"for {self.patience} epochs)")
                self.finished = stop or epoch + 1 >= self.max_epochs

                # Save checkpoints
                self._save_checkpoint(epoch, metrics, is_best=is_best, label="latest")
                if is_best:
                    self._save_checkpoint(epoch, metrics, is_best=True, label="best")
                if (epoch + 1) % self.save_every_n == 0:
                    self._save_checkpoint(epoch, metrics, is_best=False, label=f"epoch_{epoch}")
                if stop:
                    break
        finally:
            signal.signal(signal.SIGTERM, prev_sigterm)
            signal.signal(signal.SIGINT, prev_sigint)

        if self._interrupted and not self.finished:
            logger.info("Training interrupted, stopping gracefully.")
        return {
            "best_metric": self.best_metric,
            "monitor": self.monitor,
            "final_epoch": epoch,
            "all_metrics": all_metrics,
            "interrupted": self._interrupted and not self.finished,
        }

    def train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        self.criterion.train()

        total_loss = 0.0
        total_cls_loss = 0.0
        total_box_loss = 0.0
        total_giou_loss = 0.0
        num_batches = 0
        self.optimizer.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        oom_count = 0

        # Resuming an interrupted epoch: train only the batches it had left. The shuffle order
        # differs from the interrupted run, so this is the right number of steps, not the same images.
        skip, self.resume_batches = self.resume_batches, 0
        n_batches = len(self.train_loader)
        self._batches_done = n_batches
        if skip:
            logger.info(f"  Resuming epoch {epoch} at batch {skip}/{n_batches}")

        for i, (images, targets) in enumerate(self.train_loader):
            batch_idx = skip + i
            if batch_idx >= n_batches:
                break
            if self._interrupted and batch_idx % self.grad_accum_steps == 0:
                # At an optimizer-step boundary, so no half-accumulated gradients are lost.
                self._batches_done = batch_idx
                break

            try:
                # Move to device
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in t.items()} for t in targets]

                # Build NestedTensor-style input
                samples = self._nest_images(images, targets)

                # Extract captions
                captions = [t["caption"] for t in targets]

                with autocast(enabled=self.use_amp):
                    outputs = self.model(samples, captions=captions)
                    loss_dict = self._compute_losses(outputs, targets)
                    loss = sum(loss_dict.values())
                    loss = loss / self.grad_accum_steps

                if not torch.isfinite(loss):
                    logger.warning(f"  [{epoch}][{batch_idx}] NaN/Inf loss detected, skipping batch")
                    self.optimizer.zero_grad()
                    continue

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.grad_clip_max_norm,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    self.global_step += 1

                    if self.ema is not None:
                        self.ema.update(self.model)

            except torch.cuda.OutOfMemoryError:
                oom_count += 1
                logger.warning(f"  [{epoch}][{batch_idx}] CUDA OOM (#{oom_count}), skipping batch")
                self.optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                gc.collect()
                if oom_count > 10:
                    raise RuntimeError(f"CUDA out of memory: too many OOM errors ({oom_count}) in epoch {epoch}")
                continue

            total_loss += loss.item() * self.grad_accum_steps
            total_cls_loss += loss_dict.get("loss_ce", torch.tensor(0.0)).item()
            total_box_loss += loss_dict.get("loss_bbox", torch.tensor(0.0)).item()
            total_giou_loss += loss_dict.get("loss_giou", torch.tensor(0.0)).item()
            num_batches += 1

            if batch_idx % 50 == 0:
                logger.info(
                    f"  [{epoch}][{batch_idx}/{n_batches}] "
                    f"loss={loss.item() * self.grad_accum_steps:.4f} "
                    f"{_gpu_mem()}"
                )

        return {
            "loss": total_loss / max(num_batches, 1),
            "cls_loss": total_cls_loss / max(num_batches, 1),
            "box_loss": total_box_loss / max(num_batches, 1),
            "giou_loss": total_giou_loss / max(num_batches, 1),
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
        }

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """Val loss, and val COCO mAP (same scoring as the test evaluation) when monitoring mAP."""
        from src.evaluation.evaluate import coco_map, detections_from_outputs

        if self.ema is not None:
            self.ema.apply_shadow(self.model)

        self.model.eval()
        self.criterion.eval()

        total_loss = 0.0
        num_batches = 0
        detections = []

        for images, targets in self.val_loader:
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in t.items()} for t in targets]

            samples = self._nest_images(images, targets)
            captions = [t["caption"] for t in targets]

            with autocast(enabled=self.use_amp):
                outputs = self.model(samples, captions=captions)
                loss_dict = self._compute_losses(outputs, targets)
                loss = sum(loss_dict.values())

            total_loss += loss.item()
            num_batches += 1

            if self.val_gt is not None:
                for k, t in enumerate(targets):
                    h, w = t["orig_size"].tolist()
                    detections.extend(detections_from_outputs(
                        outputs["pred_logits"][k], outputs["pred_boxes"][k], self.class_to_token_spans,
                        int(t["image_id"]), w, h, fixed_square=self.fixed_square, pad_mask=self.pad_mask,
                        annotated=set(t["annotated_classes"].nonzero().flatten().tolist()),
                        score_threshold=self.score_threshold, max_dets=self.max_dets,
                    ))

        if self.ema is not None:
            self.ema.restore(self.model)

        result = {"loss": total_loss / max(num_batches, 1)}
        if self.val_gt is not None:
            stats = coco_map(self.val_gt, detections, quiet=True)
            result.update(mAP=float(stats["mAP"]), mAP_50=float(stats["mAP_50"]))
        return result

    def _compute_losses(self, outputs, targets):
        """Compute losses via the vendor SetCriterion (wrapped with masking)."""
        caption = targets[0]["caption"]
        cat_list = targets[0]["cat_list"]
        captions = [caption] * len(targets)
        cat_lists = [cat_list] * len(targets)

        loss_dict = self.criterion(outputs, targets, cat_lists, captions)

        # Apply weight_dict
        weight_dict = getattr(self.criterion, 'weight_dict', None)
        if weight_dict is None and hasattr(self.criterion, 'criterion'):
            weight_dict = getattr(self.criterion.criterion, 'weight_dict', None)

        if weight_dict:
            weighted = {}
            for k, v in loss_dict.items():
                if k in weight_dict:
                    weighted[k] = v * weight_dict[k]
                else:
                    weighted[k] = v
            return weighted
        return loss_dict

    def _nest_images(self, images: list[torch.Tensor], targets: list[dict] = None):
        """NestedTensor of the batch; the mask covers batch padding and, with data.pad_mask, the
        square padding inside each image (target["valid_size"])."""
        from src.training.odvg_dataset import make_nested
        return make_nested(images, [t.get("valid_size") for t in targets] if targets else None)

    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool, label: str,
                         epoch_complete: bool = True):
        path = self.output_dir / f"{label}.pt"

        # Gather random states for exact resume
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state()

        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_metric": self.best_metric,
            "monitor": self.monitor,
            "epochs_without_improvement": self.epochs_without_improvement,
            "epoch_complete": epoch_complete,
            "batches_done": len(self.train_loader) if epoch_complete else self._batches_done,
            "finished": self.finished,
            "config_hash": self.config_hash,
            "val_fingerprint": self.val_fingerprint,
            "data_fingerprint": self.data_fingerprint,
            "metrics": metrics,
            "rng_state": rng_state,
        }

        # Save LoRA-specific state for compact checkpoint
        try:
            from peft import get_peft_model_state_dict
            checkpoint["lora_state_dict"] = {
                k: v.cpu() for k, v in get_peft_model_state_dict(self.model).items()
            }
        except Exception:
            pass

        if self.ema is not None:
            checkpoint["ema_state_dict"] = self.ema.state_dict()

        tmp_path = path.with_suffix(".pt.tmp")
        torch.save(checkpoint, tmp_path)
        tmp_path.replace(path)  # atomic overwrite (rename fails on Windows if path exists)
        logger.info(f"Saved checkpoint: {path} (epoch={epoch}, is_best={is_best})")

    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint and return the epoch to resume from."""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except (RuntimeError, EOFError, pickle.UnpicklingError) as e:
            logger.error(f"Checkpoint {path} is corrupted: {e}")
            best_path = Path(path).parent / "best.pt"
            if best_path.exists() and str(best_path) != path:
                logger.info(f"Falling back to {best_path}")
                checkpoint = torch.load(str(best_path), map_location=self.device, weights_only=False)
            else:
                raise

        if checkpoint.get("data_fingerprint") != self.data_fingerprint:
            raise ValueError(
                f"{path} was trained on other data (data fingerprint {checkpoint.get('data_fingerprint')}, "
                f"current {self.data_fingerprint}; checkpoints from before 2026-10-29 have none). Use a new "
                f"output.base_dir, or --no-resume to start this fold from scratch.")

        # Verify config match
        saved_hash = checkpoint.get("config_hash", "")
        if saved_hash and saved_hash != self.config_hash:
            logger.warning(
                f"Config hash mismatch: checkpoint={saved_hash}, current={self.config_hash}. "
                f"Proceeding anyway."
            )

        # Restore model
        if "lora_state_dict" in checkpoint:
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(self.model, checkpoint["lora_state_dict"])
        elif "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.best_metric = checkpoint.get("best_metric")
        self.epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.finished = checkpoint.get("finished", False)

        if checkpoint.get("val_fingerprint") != self.val_fingerprint or checkpoint.get("monitor") != self.monitor:
            logger.warning(
                "Validation set or monitored metric differs from the checkpoint "
                f"({checkpoint.get('val_fingerprint')}/{checkpoint.get('monitor')} -> "
                f"{self.val_fingerprint}/{self.monitor}): resetting best_metric and early-stopping counter."
            )
            self.best_metric = None
            self.epochs_without_improvement = 0
            self.finished = False

        if self.ema is not None and "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])

        # Restore RNG state (best-effort — never block resume)
        rng = checkpoint.get("rng_state", {})
        try:
            if "python" in rng:
                random.setstate(rng["python"])
            if "numpy" in rng:
                np.random.set_state(rng["numpy"])
            if "torch" in rng:
                state = rng["torch"]
                if isinstance(state, torch.Tensor):
                    state = state.cpu().to(torch.uint8)
                else:
                    state = torch.tensor(state, dtype=torch.uint8)
                torch.random.set_rng_state(state)
            if "cuda" in rng and torch.cuda.is_available():
                state = rng["cuda"]
                if isinstance(state, torch.Tensor):
                    state = state.cpu().to(torch.uint8)
                else:
                    state = torch.tensor(state, dtype=torch.uint8)
                torch.cuda.set_rng_state(state)
        except Exception as e:
            logger.warning(f"Could not restore RNG state, training will continue without exact reproducibility: {e}")

        if checkpoint.get("epoch_complete", True):
            resume_epoch = checkpoint["epoch"] + 1
            self.resume_batches = 0
        else:
            # Saved mid-epoch: finish that epoch rather than skipping the rest of it.
            resume_epoch = checkpoint["epoch"]
            self.resume_batches = checkpoint.get("batches_done", 0)
        self.start_epoch = resume_epoch
        logger.info(f"Resumed from {path}: epoch={checkpoint['epoch']}, resuming at {resume_epoch}"
                    + (f" batch {self.resume_batches}" if self.resume_batches else "")
                    + (" (training already finished)" if self.finished else ""))
        return resume_epoch


def _gpu_mem() -> str:
    """Peak memory this process allocated (what limits batch size) and what it holds reserved."""
    if not torch.cuda.is_available():
        return ""
    return (f"peak_alloc={torch.cuda.max_memory_allocated() / 1e9:.1f}GB "
            f"reserved={torch.cuda.memory_reserved() / 1e9:.1f}GB")
