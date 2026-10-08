"""Build Grounding DINO model with LoRA adapters."""

import sys
import logging
from pathlib import Path

import torch
import torch.nn as nn
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VENDOR_DIR = PROJECT_ROOT / "vendor" / "Open-GroundingDino"


def _ensure_vendor_on_path():
    vendor_str = str(VENDOR_DIR)
    if vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)


def load_gdino_config(config_path: str = None):
    """Load GroundingDINO model config from the vendor cfg_odvg.py."""
    _ensure_vendor_on_path()
    if config_path is None:
        config_path = str(VENDOR_DIR / "config" / "cfg_odvg.py")

    from util.slconfig import SLConfig
    cfg = SLConfig.fromfile(config_path)
    return cfg


def build_model_and_criterion(
    config: dict,
    device: torch.device = None,
    class_to_token_spans: dict = None,
):
    """Build GroundingDINO + LoRA + MaskedSetCriterion.

    If class_to_token_spans is provided, wraps the criterion with
    MaskedSetCriterion for partial-annotation training.

    Returns (model, criterion, postprocessors) all on device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _ensure_vendor_on_path()

    # 1. Load model config
    model_cfg = load_gdino_config(config.get("model", {}).get("config_path"))

    # Override text encoder
    model_cfg.text_encoder_type = config.get("model", {}).get(
        "bert_model", "bert-base-uncased"
    )
    # Set required attributes for vendor build function
    model_cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg.use_coco_eval = False
    from src.data.taxonomy import UNIFIED_CLASSES
    model_cfg.label_list = list(UNIFIED_CLASSES)
    if not hasattr(model_cfg, 'nms_iou_threshold'):
        model_cfg.nms_iou_threshold = -1
    if not hasattr(model_cfg, 'num_select'):
        model_cfg.num_select = 300

    # 2. Build model
    from models.GroundingDINO.groundingdino import build_groundingdino
    model, criterion, postprocessors = build_groundingdino(model_cfg)

    # 3. Load pretrained weights
    weights_path = config.get("model", {}).get("pretrained_weights")
    if weights_path and Path(weights_path).exists():
        checkpoint = torch.load(weights_path, map_location="cpu")
        if "model" in checkpoint:
            state = checkpoint["model"]
        else:
            state = checkpoint
        # Strip 'module.' prefix if present (from DataParallel checkpoints)
        cleaned = {}
        for k, v in state.items():
            cleaned[k.removeprefix("module.")] = v
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        logger.info(
            f"Loaded pretrained weights: {len(missing)} missing, {len(unexpected)} unexpected"
        )
        if missing:
            logger.debug(f"Missing keys sample: {missing[:3]}")
    else:
        logger.warning(f"Pretrained weights not found at {weights_path}")

    # 4. Freeze all params before applying LoRA
    for param in model.parameters():
        param.requires_grad = False

    # 5. Apply LoRA
    lora_cfg = config.get("lora", {})
    model = _apply_lora(model, lora_cfg)

    # 6. Wrap criterion with masking
    if class_to_token_spans is not None:
        from src.training.loss_masking import MaskedSetCriterion
        criterion = MaskedSetCriterion(criterion, class_to_token_spans)

    # 7. Move to device
    model = model.to(device)
    criterion = criterion.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Model params: {total:,} total, {trainable:,} trainable "
        f"({100 * trainable / total:.2f}%)"
    )

    return model, criterion, postprocessors


def _apply_lora(model: nn.Module, lora_cfg: dict) -> nn.Module:
    """Apply LoRA adapters using PEFT."""
    from peft import LoraConfig, get_peft_model, TaskType

    target_modules = lora_cfg.get("target_modules", [])
    modules_to_save = lora_cfg.get("modules_to_save", [])
    rank = lora_cfg.get("rank", 16)
    alpha = lora_cfg.get("alpha", 32)
    dropout = lora_cfg.get("dropout", 0.05)

    if not target_modules:
        logger.warning("No LoRA target modules specified, using defaults")
        target_modules = [
            "cross_attn.sampling_offsets",
            "cross_attn.attention_weights",
            "cross_attn.value_proj",
            "cross_attn.output_proj",
            "self_attn.out_proj",
            "linear1",
            "linear2",
        ]

    peft_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        modules_to_save=modules_to_save if modules_to_save else None,
        bias="none",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Build optimizer with separate LR for backbone LoRA params."""
    opt_cfg = config.get("optimizer", {})
    lr = opt_cfg.get("lr", 5e-5)
    backbone_lr = opt_cfg.get("backbone_lr", 1e-5)
    weight_decay = opt_cfg.get("weight_decay", 0.01)
    betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))

    backbone_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if other_params:
        param_groups.append({"params": other_params, "lr": lr})

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
    )

    logger.info(
        f"Optimizer: {len(backbone_params)} backbone params (lr={backbone_lr}), "
        f"{len(other_params)} other params (lr={lr})"
    )
    return optimizer


def build_scheduler(optimizer, config: dict, steps_per_epoch: int):
    """Cosine annealing with linear warmup."""
    sched_cfg = config.get("scheduler", {})
    warmup_epochs = sched_cfg.get("warmup_epochs", 3)
    max_epochs = config.get("training", {}).get("max_epochs", 30)
    min_lr = sched_cfg.get("min_lr", 1e-6)

    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = max_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        import math
        return max(min_lr / optimizer.defaults["lr"],
                   0.5 * (1 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
