# Aircraft Defect Detection — Phase 1 Documentation + Phase 2 Plan

## Context

We are fine-tuning **Grounding DINO (Swin-T backbone)** with **LoRA** (rank 16, standard — NOT QLoRA, since the RTX A5000's 24 GB gives ~12 GB headroom) for aircraft defect detection targeting **edge deployment** for crack detection on aircraft.

The evaluation protocol is **Leave-One-Dataset-Out (LODO)** — train on N-1 datasets, test on the held-out one, rotate across all 8 datasets. Two evaluation protocols are reported per fold: **strict-exclusive** (test only images whose pHash group belongs exclusively to the held-out dataset) and **all-held-out** (all images from the held-out dataset, noting that shared groups may have near-duplicates in training).

The core technical challenge is **loss masking**: each dataset only annotates a subset of the **8 defect classes** (spot removed — see Taxonomy Decisions). When the model sees all 8 class names in its text prompt but an image only has labels for {crack, dent}, predictions of "corrosion" must NOT be penalized — we don't know if corrosion is absent or just unlabeled.

**Phase 1 (data pipeline) is complete.** This document covers what was built and provides a detailed implementation plan for Phase 2 (training infrastructure).

### Key Decisions (from Phase 2 review)

1. **Spot class removed** — only 215 annotations (0.28%) from one dataset (AGDD). Semantically ambiguous, irrelevant for crack/defect detection, too few samples to learn. Taxonomy is now 8 classes.
2. **Standard LoRA, not QLoRA** — 24 GB VRAM is sufficient for LoRA rank 16 + batch 4 + fp16 (~10-12 GB usage). QLoRA would save ~350 MB but introduces quantization noise.
3. **Prompt strategy: `bare_class_names`** — simplest, most robust, lowest latency for edge deployment. Template/VLM descriptions are optional ablations after baseline.
4. **Two evaluation protocols** — strict-exclusive and all-held-out, reported separately per fold.
5. **Incremental build order** — validate each component before scaling to all 8 folds.
6. **Crash-resilient checkpoints** — full state saved (LoRA, optimizer, scheduler, scaler, EMA, epoch, step, metric).
7. **SSH-independent training** — run inside `tmux` with session lock for protection.

---

# PART 1: Phase 1 Documentation (Complete)

## What Was Built

A 5-step data processing pipeline that reads 8 datasets in 3 different annotation formats, unifies them into a single taxonomy of 8 classes (spot removed), detects cross-dataset image overlap via perceptual hashing, and produces training-ready ODVG files for 8 LODO folds.

### Pipeline Script

**Run command:**
```bash
cd /DATA/air_force_object_detection
source .venv/bin/activate
python -I scripts/01_ingest_and_unify.py --config configs/datasets.yaml
```

**Script:** `/DATA/air_force_object_detection/scripts/01_ingest_and_unify.py`
**Config:** `/DATA/air_force_object_detection/configs/datasets.yaml`

### Source Modules (all in `/DATA/air_force_object_detection/src/data/`)

| File | Purpose |
|------|---------|
| `taxonomy.py` | 9 unified classes: `UNIFIED_CLASSES`, `CLASS_TO_ID`, `ID_TO_CLASS`, `NUM_CLASSES` |
| `ingest.py` | Reads 3 formats (COCO Roboflow, YOLO, per-image JSON), remaps class labels, validates |
| `dedup.py` | SHA256 exact dedup + pHash perceptual grouping (threshold=8) for cross-dataset overlap |
| `merge.py` | Remaps annotations to new image IDs, bbox dedup (IoU > 0.9), tracks `annotated_classes` per image |
| `lodo_splits.py` | Generates 8 LODO folds using pHash groups to prevent data leakage |
| `coco_to_odvg.py` | Converts each fold to ODVG JSONL (train/val) + COCO JSON (test) |

### The 5 Pipeline Steps

1. **Ingestion + Class Unification** — Reads all 8 datasets, remaps every annotation to 9 unified class IDs
2. **Image Dedup + Cross-Dataset Grouping** — SHA256 finds 864 byte-identical duplicates; pHash groups 31,715 unique images into 12,387 perceptual groups; identifies 2,252 cross-dataset groups
3. **Annotation Merging** — Collects annotations per deduplicated image, removes duplicate bboxes, builds per-image `annotated_classes` (union of source datasets' class sets)
4. **LODO Split Generation** — For each fold, test = images whose pHash group is exclusive to the held-out dataset; train/val = everything else (10% val split)
5. **ODVG Conversion** — Writes `train.jsonl`, `val.jsonl`, `test_annotations.json`, `label_map.json`, `annotated_classes.json` per fold

### 8 Datasets

| # | Dataset | Path | Images | Format | Annotated Classes |
|---|---------|------|--------|--------|-----------------|
| 1 | RFDETR_DATASET | `DATASET/RFDETR_DATASET` | 14,753 | COCO | crack, dent, corrosion, scratch, paint-off, missing-head |
| 2 | korean-aerospace | `Aircraft-Defect-Detection-korean-aerospace` | 3,362 | COCO | crack, dent |
| 3 | Aircraft-AI-Dataset-4 | `Aircraft-AI-Dataset-4` | 983 | COCO | corrosion, scratch, missing-head |
| 4 | Aircraft-Defect-Detection-6 | `Aircraft-Defect-Detection-6` | 1,937 | COCO | paint-off, scratch, corrosion, missing-head |
| 5 | sydney | `Aircraft-Defect-Detection-sydney` | 6,803 | COCO | dent, fastener-damage, rupture |
| 6 | closeup | `Aircraft-Defect-Detection-closeup` | 4,133 | COCO | crack, dent, rupture, corrosion, scratch, paint-off, missing-head |
| 7 | AGDD | `AGDD/data` | 219 | YOLO | scratch, crack ~~, spot~~ |
| 8 | fuselage | `Aircraft-Defect-Detection-fuselage/.../aircraft_fuselage_coco` | 389 | Custom JSON | scratch, paint-off, corrosion, missing-head |

All paths relative to `/DATA/air_force_object_detection/`.

### Unified Taxonomy (8 classes — spot removed)

| ID | Unified Class | Original Labels Mapped |
|----|--------------|----------------------|
| 0 | crack | crack, Crack |
| 1 | dent | dent, Dent |
| 2 | corrosion | corrosion, rust, Rust |
| 3 | scratch | scratch, Scratch, scractch, contusion, scratches |
| 4 | paint-off | paint-peel-off, paint_peel |
| 5 | missing-head | missing-head, missing-rivet, rivet_damage |
| 6 | rupture | Rupture, Hole |
| 7 | fastener-damage | Fastener Damage |

**Removed:** `spot` (ID 8) — only 215 annotations from AGDD dataset (0.28% of total). Semantically overlaps with corrosion/paint-off, too few samples for reliable learning, irrelevant for edge crack detection.

**Taxonomy decisions documented:**
- `contusion` → `scratch`: AGDD uses "contusion" for surface abrasion marks; functionally identical to scratch in aircraft inspection context.
- `scratches` → `scratch`: plural form, same defect type.
- These mappings are appropriate for the super-class taxonomy; original fine-grained distinctions are out of scope for this 8-class detector.

### Output Files

**Processed data** at `/DATA/air_force_object_detection/data/processed/`:
- `unified_annotations.json` (17.2 MB) — merged COCO-format annotations
- `annotated_classes.json` (820 KB) — per-image annotated class sets
- `image_groups.json` (808 KB) — pHash group summaries
- `lodo_folds.json` (3.3 MB) — 8 fold definitions
- `dedup_stats.json` — dedup statistics

**ODVG folds** at `/DATA/air_force_object_detection/data/odvg/fold_{0..7}/`:
- `train.jsonl` — ODVG training data
- `val.jsonl` — ODVG validation data
- `test_annotations.json` — COCO format for evaluation
- `label_map.json` — `{class_id_str: class_name}`
- `annotated_classes.json` — per-image class sets

### ODVG Line Format

```json
{
  "filename": "/DATA/air_force_object_detection/DATASET/RFDETR_DATASET/train/img.jpg",
  "height": 640, "width": 640,
  "detection": {
    "instances": [{"bbox": [x1, y1, x2, y2], "label": 0, "category": "crack"}]
  },
  "annotated_classes": [0, 1, 2, 3, 4, 5]
}
```

Bboxes are `[x1, y1, x2, y2]` absolute coordinates. `annotated_classes` lists the unified class IDs this image was labeled for.

### LODO Fold Summary

| Fold | Held-out Dataset | Train | Val | Test |
|------|-----------------|-------|-----|------|
| 0 | RFDETR_DATASET | 23,649 | 2,627 | 6,201 |
| 1 | korean-aerospace | 28,341 | 3,148 | 1,090 |
| 2 | Aircraft-AI-Dataset-4 | 28,440 | 3,159 | 980 |
| 3 | Aircraft-Defect-Detection-6 | 27,996 | 3,110 | 1,265 |
| 4 | sydney | 26,037 | 2,892 | 3,650 |
| 5 | closeup | 25,948 | 2,883 | 3,686 |
| 6 | AGDD | 29,124 | 3,236 | 219 |
| 7 | fuselage | 29,016 | 3,224 | 339 |

### Class Distribution (76,524 total annotations)

| Class | Annotations |
|-------|------------|
| crack | 13,974 |
| dent | 20,123 |
| corrosion | 7,007 |
| scratch | 10,004 |
| paint-off | 5,056 |
| missing-head | 12,092 |
| rupture | 4,217 |
| fastener-damage | 3,836 |
| ~~spot~~ | ~~215~~ (removed) |

### Cross-Dataset Overlap and Duplicate Images

**Yes, byte-identical images exist across datasets.** 864 images share SHA256 hashes (exact copies in different datasets). Additionally, 2,252 perceptual hash groups spanning 14,777 images cross dataset boundaries (near-duplicates from different compression/resolution).

Top overlaps:
- RFDETR_DATASET ↔ sydney: 1,210 shared groups
- RFDETR_DATASET ↔ korean-aerospace: 841 shared groups
- Aircraft-Defect-Detection-6 ↔ closeup: 168 shared groups
- AGDD, fuselage, Aircraft-AI-Dataset-4: nearly zero overlap with others

**How this is handled:** The pipeline groups all duplicates/near-duplicates by pHash. During LODO split generation, the entire group is assigned to training or test — never split. This prevents data leakage. The pHash threshold (Hamming distance ≤ 8) should be manually validated with visual inspection before training (see Verification Plan).

**Before Phase 2:** Export 20-30 representative pairs at the pHash threshold boundary and visually confirm grouping quality. Document the threshold justification.

### Python Environment

- **venv:** `/DATA/air_force_object_detection/.venv` (Python 3.10.12)
- **Installed:** numpy, Pillow, opencv-python-headless, scipy, matplotlib, PyYAML, imagehash, roboflow, tqdm
- **NOT installed (needed for Phase 2):** PyTorch, transformers, PEFT, timm, pycocotools

### Hardware

- **GPU:** NVIDIA RTX A5000, 24GB VRAM, ~23.8GB free
- **CUDA:** 13.0 (driver 580.178.04)
- **Local wheels available:** `/DATA/air_force_object_detection/wheels/` contains `torch-2.6.0`, `transformers-5.17.0`, `huggingface_hub`, `tokenizers`, `safetensors`, `accelerate`

---

# PART 2: Phase 2 Plan — Training Infrastructure

## Overview

Build the training pipeline: load Grounding DINO + LoRA, implement loss masking, train all 8 LODO folds, evaluate via COCO mAP. **Build incrementally** — validate each component before scaling.

## Step 0: Environment Setup

### 0.1 Clone Repositories

```bash
cd /DATA/air_force_object_detection
mkdir -p vendor
git clone https://github.com/longzw1997/Open-GroundingDino.git vendor/Open-GroundingDino
git clone https://github.com/Asad-Ismail/Grounding-Dino-FineTuning.git vendor/Grounding-Dino-FineTuning
```

- **Open-GroundingDino** — base model with native ODVG support. Key files: `datasets/odvg.py` (ODVGDataset), `models/GroundingDINO/groundingdino.py` (model + SetCriterion), `util/utils.py` (ModelEma), `config/cfg_odvg.py`
- **Grounding-Dino-FineTuning** — reference for LoRA integration. Key file: `groundingdino/util/lora.py` (LoRA target modules, optimizer param groups)

### 0.2 Install Packages

```bash
source /DATA/air_force_object_detection/.venv/bin/activate

# PyTorch from local wheel
pip install /DATA/air_force_object_detection/wheels/torch-2.6.0-cp310-cp310-manylinux1_x86_64.whl
pip install torchvision==0.21.0

# Transformers + deps from local wheels
pip install /DATA/air_force_object_detection/wheels/transformers-5.17.0-py3-none-any.whl
pip install /DATA/air_force_object_detection/wheels/huggingface_hub-1.33.0-py3-none-any.whl
pip install /DATA/air_force_object_detection/wheels/tokenizers-0.23.2-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip install /DATA/air_force_object_detection/wheels/safetensors-0.8.0-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip install /DATA/air_force_object_detection/wheels/accelerate-1.15.0-py3-none-any.whl

# LoRA + remaining deps
pip install "peft>=0.13.0"
pip install timm pycocotools jsonlines addict "yapf==0.40.1" "supervision==0.6.0" colorlog
```

### 0.3 Build Custom CUDA Ops (Deformable Attention)

```bash
cd /DATA/air_force_object_detection/vendor/Open-GroundingDino/models/GroundingDINO/ops
python setup.py build install
cd /DATA/air_force_object_detection
```

If build fails with CUDA version mismatch, try: `TORCH_CUDA_ARCH_LIST="8.6" python setup.py build install`

### 0.4 Download Pretrained Weights

```bash
mkdir -p /DATA/air_force_object_detection/weights
wget -O weights/groundingdino_swint_ogc.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
# Also pre-download BERT tokenizer
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('bert-base-uncased')"
```

**Weight file:** `weights/groundingdino_swint_ogc.pth` (~694MB)

### 0.5 PYTHONPATH

```bash
export PYTHONPATH="/DATA/air_force_object_detection:/DATA/air_force_object_detection/vendor/Open-GroundingDino:$PYTHONPATH"
```

---

## Step 1: Config Files

### `configs/base_config.yaml`

All training hyperparameters. Key settings:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| LoRA rank | 16 | ~2% params trainable, fits 24GB |
| Batch size | 4 | Fits in VRAM with fp16 |
| Gradient accumulation | 4 | Effective batch = 16 |
| LR (LoRA layers) | 5e-5 | Standard for DINO fine-tuning |
| LR (backbone LoRA) | 1e-5 | Lower for pretrained backbone |
| Max epochs | 30 | Early stopping at patience=7 |
| EMA decay | 0.9997 | Retains pretrained knowledge |
| Mixed precision | fp16 | Essential for A5000 |
| Warmup | 3 epochs | Cosine schedule after |
| Grad clip | max_norm=0.1 | Stability |

LoRA target modules (from Grounding-Dino-FineTuning reference):
- `cross_attn.sampling_offsets`, `cross_attn.attention_weights`, `cross_attn.value_proj`, `cross_attn.output_proj`
- `ca_text.out_proj`, `self_attn.out_proj`
- `linear1`, `linear2`
- `bbox_embed.0.layers.0`, `bbox_embed.0.layers.1`
- `modules_to_save: bbox_embed.0.layers.2` (fully trainable, not LoRA)

Also include: model config path (`vendor/Open-GroundingDino/config/cfg_odvg.py`), weights path, data paths, evaluation thresholds.

### `configs/prompt_strategies.yaml`

Three text prompt strategies for ablation. **Start with (a) only; (b) and (c) are optional future ablations:**

- **(a) bare_class_names** (default, use this): `"crack"`, `"dent"`, etc. — simplest, most robust token mapping, lowest latency for edge deployment.
- **(b) template_descriptions** (future ablation): `"a crack defect on aircraft surface showing linear fracture pattern"`, etc. — unproven benefit for fine-tuning, adds tokenizer complexity.
- **(c) vlm_generated** (future ablation): populated later by running a VLM on example crops — requires extra pipeline step.

---

## Step 2: File-by-File Implementation

### Implementation order (respects dependencies):

```
2a. src/data/prompt_builder.py          ← standalone
2b. src/training/__init__.py            ← empty
2c. src/training/odvg_dataset.py        ← depends on prompt_builder
2d. src/training/loss_masking.py        ← depends on vendor/Open-GroundingDino
2e. src/training/model_builder.py       ← depends on loss_masking, PEFT
2f. src/training/ema.py                 ← standalone
2g. src/training/trainer.py             ← depends on all above
2h. src/evaluation/__init__.py          ← empty
2i. src/evaluation/evaluate.py          ← depends on vendor/Open-GroundingDino, pycocotools
2j. src/evaluation/aggregate.py         ← standalone
2k. src/evaluation/visualize.py         ← standalone
2l. src/training/lodo_runner.py         ← depends on trainer, evaluate
```

### 2a. `src/data/prompt_builder.py` (new)

Builds text prompts from the class taxonomy using the configured strategy.

```python
class PromptBuilder:
    def __init__(self, strategy_name="bare_class_names", config_path="configs/prompt_strategies.yaml")
    def get_class_prompts(self) -> dict[int, str]
    def build_caption(self) -> str
        # Returns: "crack . dent . corrosion . scratch . paint-off . missing-head . rupture . fastener-damage ."
        # ALWAYS all 8 classes, ALWAYS in fixed order (class_id 0..7)
        # Fixed order is critical — makes token-to-class mapping consistent across the batch
    def build_caption_and_cat_list(self) -> tuple[str, list[str]]
        # Returns (caption_string, ordered_prompt_list)
```

**Key design decision:** The caption is FIXED and IDENTICAL for every image. All 8 classes are always present. This differs from Open-GroundingDino's default (which randomly samples negative classes). Fixed caption is required so that loss masking can consistently map token positions to classes.

### 2b–2c. `src/training/odvg_dataset.py` (new)

Custom ODVG dataset extending Open-GroundingDino's approach.

```python
class MaskedODVGDataset(torch.utils.data.Dataset):
    def __init__(self, odvg_path, prompt_builder, transforms=None, max_text_len=256)
    def __getitem__(self, index) -> tuple[Image, dict]:
        # Returns target with:
        # - "boxes": Tensor[N, 4] xyxy
        # - "labels": Tensor[N] class IDs (0..7)
        # - "caption": str (fixed, all 8 classes)
        # - "cap_list": list[str] (8 prompt texts in order)
        # - "annotated_classes": BoolTensor[8] (True for labeled classes)
        # - "size": Tensor[2]
```

**Data source:** Reads `annotated_classes` inline from each ODVG line (not from the separate JSON file). Image paths are absolute in the ODVG data, so no root directory needed.

**Transforms:** Reuse `make_coco_transforms` from `vendor/Open-GroundingDino/datasets/odvg.py` (random resize/crop/flip for train, resize-only for val).

### 2d. `src/training/loss_masking.py` (new) — THE CORE PIECE

**PREREQUISITE: Validate GDINO loss internals before implementing.** This is the highest-risk component. Before writing any masking code, inspect and document the exact Open-GroundingDino checkout:

1. The shape and meaning of `pred_logits` (is it `[B, num_queries, max_text_len]`?)
2. How captions are tokenized and how `positive_map` is constructed
3. How target labels map to token positions in the classification loss
4. Whether classification loss is computed over all token positions
5. Which classification terms the Hungarian matcher uses
6. Whether auxiliary decoder losses use the same classification path
7. How padding and special tokens (`[CLS]`, `[SEP]`) are handled

Do not assume that masking the final focal BCE tensor automatically masks all classification influence in the matcher or auxiliary losses.

Subclasses `SetCriterion` from Open-GroundingDino to mask focal classification loss.

```python
def compute_class_token_spans(caption, cat_list, tokenizer) -> dict[int, tuple[int, int]]:
    # Pre-computes which BERT token positions correspond to each class
    # Uses tokenizer.char_to_token() to map character positions to token positions
    # Called ONCE at setup, result cached
    # MUST be recomputed for each prompt strategy (never reuse across strategies)
    # MUST verify: decode each span back to text and assert it matches the class name

class MaskedSetCriterion(SetCriterion):
    def __init__(self, matcher, weight_dict, focal_alpha, focal_gamma, losses,
                 class_to_token_spans: dict[int, tuple[int, int]])
    
    def _build_annotated_token_mask(self, targets, max_text_len) -> Tensor[batch, max_text_len]:
        # For each image, set True at token positions of annotated classes
    
    # Override the classification loss method:
    # 1. Build annotated_token_mask from targets[b]["annotated_classes"]
    # 2. Combine with text_mask (padding mask)
    # 3. Expand to [batch, num_queries, max_text_len]
    # 4. Apply mask BEFORE computing focal BCE
    # 5. Normalize by num_positive_matches (same as original)
    # 6. Apply same masking to auxiliary decoder losses if they exist
```

**What gets masked:** Only the classification focal loss. Box losses (L1, GIoU) need NO masking — they only fire on matched GT-prediction pairs, which are always for annotated classes.

**Hungarian matcher is NOT masked:** The matcher uses both classification cost and box cost. Since GT boxes only have labels for annotated classes (by construction), the matcher naturally only matches to annotated-class boxes. **Verify this assumption** by checking that the matcher's classification cost term only references annotated GT labels.

**Token span example** (approximate BERT tokenization of the caption — THESE ARE EXAMPLES ONLY):
```
"crack . dent . corrosion . scratch . paint-off . missing-head . rupture . fastener-damage ."

class 0 (crack):           tokens [1, 2)
class 1 (dent):            tokens [3, 4)
class 2 (corrosion):       tokens [5, 8)     — "cor", "##ros", "##ion"
class 3 (scratch):         tokens [9, 10)
class 4 (paint-off):       tokens [11, 14)   — "paint", "-", "off"
class 5 (missing-head):    tokens [15, 18)   — "missing", "-", "head"
class 6 (rupture):         tokens [19, 21)   — "rupt", "##ure"
class 7 (fastener-damage): tokens [22, 26)   — "fast", "##ener", "-", "damage"
```

**Exact positions MUST be verified by running the tokenizer.** `compute_class_token_spans()` does this. The above are only illustrative — they change with Transformers version and tokenizer settings.

**Mandatory synthetic gradient test (implement before any training):**
- Image A: `annotated_classes = [0, 1]` (crack, dent only)
- Image B: all 8 classes annotated
- Assert: unannotated token positions for image A have zero classification gradient
- Assert: annotated positions for image A have nonzero gradients
- Assert: image B has gradients for all classes
- Assert: auxiliary decoder losses obey the same masking

### 2e. `src/training/model_builder.py` (new)

```python
def build_model(config) -> tuple[nn.Module, MaskedSetCriterion, dict]:
    # 1. Load Open-GroundingDino config (cfg_odvg.py)
    # 2. Build model via build_groundingdino(args)
    # 3. Load pretrained weights from weights/groundingdino_swint_ogc.pth
    # 4. Apply LoRA via PEFT (rank=16, target_modules from config)
    # 5. Replace criterion with MaskedSetCriterion (same matcher + weights, plus class_to_token_spans)
    # 6. Return (model, criterion, postprocessors)

def build_optimizer(model, config) -> Optimizer:
    # Two param groups:
    # - Backbone LoRA params (name contains 'backbone'): lr=1e-5
    # - All other LoRA params: lr=5e-5
    # Only requires_grad=True params included

def build_scheduler(optimizer, config, steps_per_epoch):
    # Cosine annealing with linear warmup over 3 epochs
```

**PEFT concern:** After `get_peft_model()`, model attributes move under `model.base_model.model.*`. The training loop must access outputs via the PEFT wrapper (which delegates forward correctly) but may need the unwrapped model for EMA or checkpoint saving.

### 2f. `src/training/ema.py` (new)

```python
class ModelEma(nn.Module):
    def __init__(self, model, decay=0.9997)
    def update(self, model)  # EMA update step
    def set(self, model)     # Hard copy
```

Reuse pattern from `vendor/Open-GroundingDino/util/utils.py`. **EMA tracks only LoRA (trainable) parameters** — not the full frozen model — to minimize VRAM cost (~50MB vs ~700MB). The checkpoint must store EMA state alongside LoRA state. Test both: loading for inference and loading for training resume.

### 2g. `src/training/trainer.py` (new)

```python
class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, ema_model,
                 train_loader, val_loader, config, output_dir)
    
    def train_one_epoch(self, epoch) -> dict:
        # For each batch:
        # 1. Forward under autocast(fp16)
        # 2. MaskedSetCriterion computes loss
        # 3. loss / grad_accum_steps → backward with GradScaler
        # 4. Every 4 steps: clip grads (max_norm=0.1), scaler.step(), scaler.update()
        # 5. EMA update
    
    def validate(self, epoch) -> dict:
        # Uses EMA model for inference
        # Computes val loss + COCO mAP
    
    def train(self) -> dict:
        # Full loop: train_one_epoch → validate → early stopping → checkpoint
    
    def _save_checkpoint(self, epoch, metrics, is_best):
        # Saves FULL state for crash recovery:
        # - LoRA state dict via peft.get_peft_model_state_dict()
        # - EMA state dict
        # - Optimizer state dict
        # - Scheduler state dict
        # - GradScaler state dict (for fp16 continuity)
        # - Current epoch + step within epoch
        # - Best metric value
        # - Config hash (to verify correct experiment on resume)
        # - Random states: python, numpy, torch, cuda
        # Saves both latest.pt (always overwritten) and best.pt (when is_best)
    
    def _load_checkpoint(self, checkpoint_path):
        # Restores all state from checkpoint
        # Verifies config hash matches current config
        # Resumes from exact point of interruption
```

**Checkpoint files per fold:** `outputs/fold_{id}/latest.pt`, `outputs/fold_{id}/best.pt`

**Data flow through training loop:**
1. DataLoader yields `(NestedTensor_images, list_of_target_dicts)`
2. Extract `captions = [t["caption"] for t in targets]` (all identical)
3. `outputs = model(samples, captions=captions)` → `pred_logits [B,900,256]`, `pred_boxes [B,900,4]`
4. `losses = criterion(outputs, targets, cat_lists, captions)` — masked focal loss applied here
5. Standard backward + accumulate + step + EMA

### 2i. `src/evaluation/evaluate.py` (new)

```python
def evaluate_fold(model, test_annotations_path, prompt_builder, config) -> dict:
    # 1. Load test_annotations.json (COCO format, [x,y,w,h] bboxes)
    # 2. For each test image: forward pass → PostProcess → COCO result format
    # 3. Run COCOeval (pycocotools)
    # 4. Return: mAP, mAP_50, mAP_75, per-class AP, per-size AP
    # 5. Report per-class support counts; flag classes with < 50 test annotations

def evaluate_fold_dual_protocol(model, fold_id, prompt_builder, config) -> dict:
    # Run both evaluation protocols:
    # (a) strict-exclusive: test only images whose pHash group is exclusive to the held-out dataset
    # (b) all-held-out: all images from the held-out dataset
    # Report both results separately; never describe (b) as "zero-shot" unless
    # the training and test groups are demonstrably disjoint
```

**Bbox format:** Model outputs normalized `[cx,cy,w,h]` → PostProcess converts to absolute `[x1,y1,x2,y2]` → evaluation code converts to COCO `[x,y,w,h]`.

**Category ID validation (MANDATORY before training):** Create a tiny synthetic COCO dataset with category IDs `0..7`, generate perfect predictions, run COCOeval, and confirm perfect mAP. Also test empty predictions and classes absent from a fold. Our data uses 0-indexed IDs — COCOeval supports this but must be verified with the exact version installed.

### 2j–2k. `src/evaluation/aggregate.py` and `visualize.py` (new)

- **aggregate.py**: Mean mAP ± std across folds, per-class transfer analysis, per-fold breakdown. Report strict-exclusive and all-held-out results separately. Include per-class support counts and flag low-support classes. Include per-fold: number of test images, exclusive groups, shared groups, test annotations by class.
- **visualize.py**: Draw predicted + ground-truth bboxes on sample test images

### 2l. `src/training/lodo_runner.py` (new)

```python
class LODORunner:
    def run_fold(self, fold_id, resume=True) -> dict:
        # 1. Build datasets from data/odvg/fold_{fold_id}/
        # 2. Build fresh model + LoRA + criterion + optimizer + EMA
        # 3. If resume=True and outputs/fold_{fold_id}/latest.pt exists, load checkpoint
        # 4. Train via Trainer.train()
        # 5. Evaluate using dual protocol (strict-exclusive + all-held-out)
        # 6. Save results + LoRA checkpoint
        # 7. GPU cleanup: del everything, gc.collect(), torch.cuda.empty_cache()
    
    def run_all(self, fold_ids=None, resume=True) -> dict:
        # Run all folds sequentially, auto-resume crashed folds, aggregate results
```

---

## Step 3: Scripts

### `scripts/02_setup_training.py`
Verify environment: torch+CUDA, Open-GroundingDino imports, weights exist, pre-compute token spans, run 1-image forward pass, print GPU memory.

### `scripts/03_train_single_fold.py`
```bash
python scripts/03_train_single_fold.py --fold 0 --config configs/base_config.yaml
python scripts/03_train_single_fold.py --fold 0 --smoke-test  # 5 epochs, 100 images
python scripts/03_train_single_fold.py --fold 0 --resume  # resume from latest checkpoint
```

### `scripts/04_run_all_lodo.py`
```bash
python scripts/04_run_all_lodo.py --config configs/base_config.yaml --prompt-strategy bare_class_names
python scripts/04_run_all_lodo.py --folds 0 1 2  # subset
python scripts/04_run_all_lodo.py --resume  # resume crashed folds from latest checkpoints
```

### Running Training (SSH-independent, password-protected)

```bash
# 1. Start a tmux session
tmux new-session -s training

# 2. Inside tmux, start training
cd /DATA/air_force_object_detection
source .venv/bin/activate
export PYTHONPATH="/DATA/air_force_object_detection:/DATA/air_force_object_detection/vendor/Open-GroundingDino:$PYTHONPATH"
python -I scripts/04_run_all_lodo.py --config configs/base_config.yaml --resume

# 3. Detach from tmux (training continues after SSH disconnect)
#    Press: Ctrl+B, then D

# 4. Lock tmux session (prevents others from attaching/killing)
#    Press: Ctrl+B, then :lock-session
#    Or before detaching: tmux lock-session -t training

# 5. Reconnect later
tmux attach -t training
```

**Crash recovery:** If SSH drops, power flickers, or the process crashes — just re-run the same command with `--resume`. It picks up from the latest checkpoint (epoch + step + optimizer state). No training progress is lost beyond the current batch.

### `scripts/05_evaluate_and_report.py`
```bash
python scripts/05_evaluate_and_report.py --results-dir outputs/lodo_bare --output reports/report.md
```

Note: `02_generate_lodo_splits.py` from the original plan is NOT needed — splits are already generated by `01_ingest_and_unify.py`.

---

## Step 4: Verification Plan (Incremental Build Order)

**Build each component and verify before moving to the next. Do NOT jump to full training.**

| Step | What | How to verify | Gate |
|------|------|---------------|------|
| 0 | **Phase 1 re-validation** | Re-run `01_ingest_and_unify.py`, verify all 8 datasets ingest, every ODVG image path exists, every bbox is valid, train/val/test IDs are disjoint per fold, `annotated_classes` matches config. Inspect 20-30 pHash pairs at threshold boundary. | All checks pass |
| 1 | **Environment** | `import torch; torch.cuda.is_available()` → True. Import transformers, peft, timm, pycocotools. Record exact versions. Build deformable attention CUDA ops. | All imports succeed, ops build |
| 2 | **COCO eval validation** | Synthetic dataset with cat IDs 0..7, perfect predictions → COCOeval gives mAP=1.0. Test empty predictions, missing classes. | Perfect score confirmed |
| 3 | **Data loading** | Create MaskedODVGDataset for fold_0, iterate 5 batches, check shapes and types. Verify `annotated_classes` BoolTensor is correct per image. | Shapes correct |
| 4 | **Token spans** | Run `compute_class_token_spans()` for `bare_class_names`, decode every span back to text, verify each of 8 classes maps correctly. Assert no overlap, no empty spans. | All 8 classes verified |
| 5 | **Forward pass** | Load model + LoRA, forward 1 batch (4 images), verify output shapes `[4, 900, max_text_len]`. Check `pred_logits` and `pred_boxes` shapes. | Shapes correct |
| 6 | **GDINO loss internals** | Instrument `SetCriterion`: log `positive_map` shape, matcher costs, auxiliary loss paths. Document all findings. | Documented |
| 7 | **Loss masking (CRITICAL)** | Synthetic batch: img_A has `annotated_classes=[0,1]`, img_B has all 8. After backward: (a) gradients at unannotated token positions for img_A are exactly zero, (b) annotated positions for img_A have nonzero gradients, (c) img_B has gradients everywhere, (d) auxiliary losses obey masking | All 4 assertions pass |
| 8 | **Checkpoint save/load** | Save checkpoint, load it, verify all states match. Run 2 more steps, compare outputs to a fresh run from checkpoint. | Outputs match |
| 9 | **Smoke test** | Train fold 0 for 5 epochs (100 images). All losses are finite. Expected params have nonzero gradients. GPU memory stays within budget. Loss decreases over epochs. | Loss decreases, no OOM |
| 10 | **One fold to convergence** | Train fold 0 fully (30 epochs / early stop). Evaluate with dual protocol. Inspect predictions visually. | mAP > random baseline |
| 11 | **Full LODO** | All 8 folds. GPU memory properly cleaned between folds (no leak). Report mean mAP ± std for both protocols. Include per-class support counts. | All folds complete |

### Reproducibility: save with every fold

Save the following in every fold output directory (`outputs/fold_{id}/`):

- Resolved YAML configuration
- Command-line arguments
- Python, PyTorch, CUDA, Transformers, PEFT, timm, pycocotools versions
- Git commit IDs for vendor repositories
- Dataset config hash + ODVG file hashes
- Random seeds (Python, NumPy, PyTorch, CUDA)
- Tokenizer name and revision
- GPU name and driver version
- Train/val/test split counts
- Checkpoint metadata

---

## Step 5: Risk Areas

1. **CUDA ops build failure** — Deformable attention ops may fail to compile. Mitigation: try `TORCH_CUDA_ARCH_LIST="8.6"`, or use pure-Python fallback. Record compiler version, CUDA arch, and build output.
2. **PEFT wrapping** — After `get_peft_model()`, attribute access changes (`model.class_embed` → `model.base_model.model.class_embed`). Verify training loop works through PEFT wrapper.
3. **Tokenizer drift** — Different transformers versions may tokenize "paint-off" differently. Always verify token spans empirically. Pin transformers version after successful setup.
4. **EMA with PEFT** — `deepcopy` of PEFT-wrapped model might fail. Fallback: parameter-dict-based EMA. **Decision required:** EMA should track only trainable LoRA parameters (not full model) to save memory. Test both checkpoint loading for inference and for training resume.
5. **0-indexed categories** — Validated by synthetic COCOeval test (Step 4, item 2). Must be confirmed before any real training.
6. **Auxiliary decoder losses** — Grounding DINO may have intermediate decoder layers that compute classification loss independently. Loss masking must apply to ALL of them, not just the final layer. Verify during Step 4, item 6.
7. **pHash false negatives** — A false negative in dedup causes data leakage (same image in train and test). Visual inspection of threshold boundary pairs is mandatory (Step 4, item 0).

---

## Step 6: Acceptance Criteria

Phase 2 is NOT complete until:

1. Phase 1 has been re-validated from the current configuration (spot removed, 8 classes)
2. Environment is set up with pinned dependency versions in a lockfile
3. Grounding DINO forward and backward passes succeed
4. Token-span mapping is tested for `bare_class_names`
5. Loss masking passes the synthetic gradient test (Step 4, item 7)
6. COCO evaluation passes a perfect-prediction fixture
7. Strict-exclusive and all-held-out evaluation protocols report separately
8. One complete fold trains, evaluates, saves, and reloads successfully
9. Results are reproducible from the recorded configuration
10. All 8 folds run without GPU memory leakage between folds
11. Final report includes per-class support, confidence limitations, and leakage protocol details

---

## Key File Paths Reference

### Existing (Phase 1)
| File | Path |
|------|------|
| Pipeline script | `/DATA/air_force_object_detection/scripts/01_ingest_and_unify.py` |
| Dataset config | `/DATA/air_force_object_detection/configs/datasets.yaml` |
| Taxonomy | `/DATA/air_force_object_detection/src/data/taxonomy.py` |
| Ingestion | `/DATA/air_force_object_detection/src/data/ingest.py` |
| Dedup | `/DATA/air_force_object_detection/src/data/dedup.py` |
| Merge | `/DATA/air_force_object_detection/src/data/merge.py` |
| LODO splits | `/DATA/air_force_object_detection/src/data/lodo_splits.py` |
| ODVG converter | `/DATA/air_force_object_detection/src/data/coco_to_odvg.py` |
| Processed data | `/DATA/air_force_object_detection/data/processed/` |
| ODVG folds | `/DATA/air_force_object_detection/data/odvg/fold_{0..7}/` |
| Python venv | `/DATA/air_force_object_detection/.venv/` |
| Local wheels | `/DATA/air_force_object_detection/wheels/` |
| Original plan | `/DATA/air_force_object_detection/create-a-plan-for-mellow-planet.md` |

### To Create (Phase 2)
| File | Path |
|------|------|
| Base config | `/DATA/air_force_object_detection/configs/base_config.yaml` |
| Prompt strategies | `/DATA/air_force_object_detection/configs/prompt_strategies.yaml` |
| Prompt builder | `/DATA/air_force_object_detection/src/data/prompt_builder.py` |
| ODVG dataset | `/DATA/air_force_object_detection/src/training/odvg_dataset.py` |
| Loss masking | `/DATA/air_force_object_detection/src/training/loss_masking.py` |
| Model builder | `/DATA/air_force_object_detection/src/training/model_builder.py` |
| EMA | `/DATA/air_force_object_detection/src/training/ema.py` |
| Trainer | `/DATA/air_force_object_detection/src/training/trainer.py` |
| LODO runner | `/DATA/air_force_object_detection/src/training/lodo_runner.py` |
| Evaluate | `/DATA/air_force_object_detection/src/evaluation/evaluate.py` |
| Aggregate | `/DATA/air_force_object_detection/src/evaluation/aggregate.py` |
| Visualize | `/DATA/air_force_object_detection/src/evaluation/visualize.py` |
| Setup script | `/DATA/air_force_object_detection/scripts/02_setup_training.py` |
| Train fold | `/DATA/air_force_object_detection/scripts/03_train_single_fold.py` |
| Run LODO | `/DATA/air_force_object_detection/scripts/04_run_all_lodo.py` |
| Report | `/DATA/air_force_object_detection/scripts/05_evaluate_and_report.py` |
| Pretrained weights | `/DATA/air_force_object_detection/weights/groundingdino_swint_ogc.pth` |
| Open-GroundingDino | `/DATA/air_force_object_detection/vendor/Open-GroundingDino/` |
| FineTuning reference | `/DATA/air_force_object_detection/vendor/Grounding-Dino-FineTuning/` |
| Outputs | `/DATA/air_force_object_detection/outputs/` |
