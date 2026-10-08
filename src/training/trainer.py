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
        self.save_every_n = config.get("output", {}).get("save_every_n_epochs", 5)

        # Mixed precision
        mp_cfg = config.get("mixed_precision", {})
        self.use_amp = mp_cfg.get("enabled", True)
        self.scaler = GradScaler(enabled=self.use_amp)

        # EMA
        ema_cfg = config.get("ema", {})
        self.ema = None
        if ema_cfg.get("enabled", True):
            self.ema = LoRAModelEma(model, decay=ema_cfg.get("decay", 0.9997))

        # State
        self.start_epoch = 0
        self.global_step = 0
        self.best_metric = -1.0
        self.epochs_without_improvement = 0
        self._interrupted = False
        self._current_epoch = 0

        # Config hash for resume verification
        self.config_hash = hashlib.md5(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()[:12]

    def _handle_signal(self, signum, frame):
        signame = signal.Signals(signum).name
        logger.warning(f"Received {signame} — saving emergency checkpoint before exit")
        self._interrupted = True
        try:
            self._save_checkpoint(
                self._current_epoch,
                {"signal": signame},
                is_best=False,
                label="latest",
            )
            logger.info("Emergency checkpoint saved.")
        except Exception as e:
            logger.error(f"Failed to save emergency checkpoint: {e}")

    def train(self) -> dict:
        """Full training loop with early stopping."""
        logger.info(f"Starting training for {self.max_epochs} epochs")
        logger.info(f"Output: {self.output_dir}")

        prev_sigterm = signal.signal(signal.SIGTERM, self._handle_signal)
        prev_sigint = signal.signal(signal.SIGINT, self._handle_signal)

        all_metrics = []

        for epoch in range(self.start_epoch, self.max_epochs):
            if self._interrupted:
                logger.info("Training interrupted, stopping gracefully.")
                break
            self._current_epoch = epoch
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate(epoch)

            metrics = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
            all_metrics.append(metrics)

            # Log
            logger.info(
                f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
            )

            # Check improvement
            val_loss = val_metrics["loss"]
            is_best = False
            if self.best_metric < 0 or val_loss < self.best_metric:
                self.best_metric = val_loss
                self.epochs_without_improvement = 0
                is_best = True
            else:
                self.epochs_without_improvement += 1

            # Save checkpoints
            self._save_checkpoint(epoch, metrics, is_best=is_best, label="latest")
            if is_best:
                self._save_checkpoint(epoch, metrics, is_best=True, label="best")
            if (epoch + 1) % self.save_every_n == 0:
                self._save_checkpoint(epoch, metrics, is_best=False, label=f"epoch_{epoch}")

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {self.patience} epochs)"
                )
                break

        signal.signal(signal.SIGTERM, prev_sigterm)
        signal.signal(signal.SIGINT, prev_sigint)

        return {
            "best_metric": self.best_metric,
            "final_epoch": epoch,
            "all_metrics": all_metrics,
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

        oom_count = 0

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            if self._interrupted:
                break

            try:
                # Move to device
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in t.items()} for t in targets]

                # Build NestedTensor-style input
                samples = self._nest_images(images)

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
                    raise RuntimeError(f"Too many OOM errors ({oom_count}) in epoch {epoch}")
                continue

            total_loss += loss.item() * self.grad_accum_steps
            total_cls_loss += loss_dict.get("loss_ce", torch.tensor(0.0)).item()
            total_box_loss += loss_dict.get("loss_bbox", torch.tensor(0.0)).item()
            total_giou_loss += loss_dict.get("loss_giou", torch.tensor(0.0)).item()
            num_batches += 1

            if batch_idx % 50 == 0:
                logger.info(
                    f"  [{epoch}][{batch_idx}/{len(self.train_loader)}] "
                    f"loss={loss.item() * self.grad_accum_steps:.4f} "
                    f"mem={torch.cuda.memory_reserved() / 1e9:.1f}GB"
                )

        return {
            "loss": total_loss / max(num_batches, 1),
            "cls_loss": total_cls_loss / max(num_batches, 1),
            "box_loss": total_box_loss / max(num_batches, 1),
            "giou_loss": total_giou_loss / max(num_batches, 1),
        }

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        if self.ema is not None:
            self.ema.apply_shadow(self.model)

        self.model.eval()
        self.criterion.eval()

        total_loss = 0.0
        num_batches = 0

        for images, targets in self.val_loader:
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in t.items()} for t in targets]

            samples = self._nest_images(images)
            captions = [t["caption"] for t in targets]

            with autocast(enabled=self.use_amp):
                outputs = self.model(samples, captions=captions)
                loss_dict = self._compute_losses(outputs, targets)
                loss = sum(loss_dict.values())

            total_loss += loss.item()
            num_batches += 1

        if self.ema is not None:
            self.ema.restore(self.model)

        return {"loss": total_loss / max(num_batches, 1)}

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

    def _nest_images(self, images: list[torch.Tensor]):
        """Create a NestedTensor from a list of images (pad to max size)."""
        try:
            from util.misc import NestedTensor, nested_tensor_from_tensor_list
            return nested_tensor_from_tensor_list(images)
        except ImportError:
            max_h = max(img.shape[1] for img in images)
            max_w = max(img.shape[2] for img in images)
            batch = torch.zeros(len(images), 3, max_h, max_w, device=images[0].device)
            mask = torch.ones(len(images), max_h, max_w, dtype=torch.bool, device=images[0].device)
            for i, img in enumerate(images):
                batch[i, :, :img.shape[1], :img.shape[2]] = img
                mask[i, :img.shape[1], :img.shape[2]] = False
            # Simple wrapper
            class _NT:
                def __init__(self, tensors, m):
                    self.tensors = tensors
                    self.mask = m
                    self.device = tensors.device
                def to(self, device):
                    self.tensors = self.tensors.to(device)
                    self.mask = self.mask.to(device)
                    self.device = self.tensors.device
                    return self
                def decompose(self):
                    return self.tensors, self.mask
            return _NT(batch, mask)

    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool, label: str):
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
            "epochs_without_improvement": self.epochs_without_improvement,
            "config_hash": self.config_hash,
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
        tmp_path.rename(path)
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
        self.best_metric = checkpoint.get("best_metric", -1.0)
        self.epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
        self.global_step = checkpoint.get("global_step", 0)

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

        resume_epoch = checkpoint["epoch"] + 1
        self.start_epoch = resume_epoch
        logger.info(f"Resumed from {path}: epoch={checkpoint['epoch']}, resuming at {resume_epoch}")
        return resume_epoch
