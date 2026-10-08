# Aircraft Defect Detection — Training Pipeline Plan

## Context

Fine-tuning Grounding DINO for aircraft defect detection across multiple aircraft types. The core objective is **zero-shot cross-domain generalization** — proving the model detects defects on aircraft it has never seen during training. The guide has directed focus on this generalization aspect.

**Setup:** 5-6 datasets (COCO JSON), 7 super-classes (crack, dent, corrosion, scratch, paint-off, missing-head, delamination), each dataset annotates a subset. Overlapping images across datasets with different classes annotated. Single A5000 GPU (24GB).

**Evaluation protocol:** Leave-One-Dataset-Out (LODO) — train on N-1 datasets, zero-shot test on the held-out one, rotate, report average mAP.

---

## 1. Project Structure

```
aircraft-defect-detection/
├── configs/
│   ├── datasets.yaml              # Dataset registry: paths, class mappings, annotated classes
│   ├── base_config.yaml           # Training hyperparameters
│   ├── prompt_strategies.yaml     # 3 text prompt strategies
│   └── lodo_experiments.yaml      # LODO fold definitions (auto-generated)
├── data/
│   ├── raw/                       # Original COCO JSON datasets (gitignored)
│   │   ├── dataset_a/
│   │   │   ├── images/
│   │   │   └── annotations.json
│   │   └── ...
│   ├── processed/                 # Unified, deduplicated, merged
│   │   ├── unified_annotations.json
│   │   ├── image_registry.json    # Dedup map: hash → canonical path
│   │   └── annotated_classes.json # Per-image: which classes are labeled
│   └── odvg/                      # Per LODO fold, training-format outputs
│       ├── fold_0/
│       │   ├── train.jsonl
│       │   ├── val.jsonl
│       │   ├── test_annotations.json
│       │   ├── label_map.json
│       │   └── annotated_classes.json
│       └── ...
├── src/
│   ├── data/
│   │   ├── ingest.py              # Load + validate COCO JSONs
│   │   ├── taxonomy.py            # Class name → super-class mapping
│   │   ├── dedup.py               # Perceptual hash + SHA256 dedup
│   │   ├── merge.py               # Merge annotations for shared images
│   │   ├── lodo_splits.py         # Generate LODO folds
│   │   ├── coco_to_odvg.py        # Convert to ODVG JSONL
│   │   └── prompt_builder.py      # Build text prompts (3 strategies)
│   ├── training/
│   │   ├── model_builder.py       # Load Grounding DINO + LoRA
│   │   ├── loss_masking.py        # MaskedSetCriterion
│   │   ├── ema.py                 # EMA model wrapper
│   │   ├── trainer.py             # Training loop + early stopping
│   │   └── lodo_runner.py         # Orchestrate all LODO folds
│   └── evaluation/
│       ├── evaluate.py            # COCO mAP on held-out set
│       ├── aggregate.py           # Aggregate LODO results
│       └── visualize.py           # Prediction visualization
├── scripts/
│   ├── 01_ingest_and_unify.py     # End-to-end data processing
│   ├── 02_generate_lodo_splits.py # Generate all LODO folds
│   ├── 03_train_single_fold.py    # Train one fold
│   ├── 04_run_all_lodo.py         # Run all folds sequentially
│   └── 05_evaluate_and_report.py  # Final report generation
├── docs/
│   └── methodology.md             # Method explanation document
└── requirements.txt
```

---

## 2. Data Processing Pipeline

### Step 1: Ingestion + Class Unification

- Read each dataset's COCO JSON
- Validate: check for missing image files, invalid bboxes, orphan annotations
- Remap each dataset's local category IDs to the unified 7 super-classes using `datasets.yaml`:

```yaml
# Example entry in datasets.yaml
- name: "boeing_737"
  images_dir: "/data/raw/boeing_737/images"
  annotations: "/data/raw/boeing_737/annotations.json"
  annotated_classes: [crack, dent, corrosion]  # only these 3 are labeled
  class_name_map:
    "surface_crack": "crack"
    "hairline_crack": "crack"
    "body_dent": "dent"
    "oxidation": "corrosion"
```

### Step 2: Image Deduplication

- **Pass 1**: SHA256 hash for exact byte-level duplicates (fast)
- **Pass 2**: Perceptual hash (pHash via `imagehash`) for resized/recompressed duplicates, Hamming distance threshold = 8
- Output: `image_registry.json` mapping hash → canonical path + list of source datasets

### Step 3: Annotation Merging (Critical)

For images appearing in multiple datasets:
1. Collect all annotations from all source datasets
2. All are already remapped to unified super-class IDs from Step 1
3. **Bbox dedup**: If two annotations from different datasets have IoU > 0.9 AND same super-class, keep one
4. **Track `annotated_classes` per image**: Union of all source datasets' `annotated_classes` for that image
   - E.g., image in Dataset A (annotates {crack, dent}) and Dataset B (annotates {corrosion}) → image's `annotated_classes = {crack, dent, corrosion}`
   - Unannotated classes (scratch, paint-off, etc.) are **unknown**, not absent — this drives loss masking

Output: `unified_annotations.json` (single merged COCO JSON) + `annotated_classes.json` (per-image class sets)

### Step 4: LODO Split Generation

For each fold k (held-out dataset k):
- **Test set**: Images that appear **only** in dataset k (not shared with any training dataset) — prevents data leakage
- **Train set**: All other images (including shared images that also appear in dataset k)
- **Val set**: 10% stratified random split from train set
- If a dataset has very few exclusive images, log a warning

### Step 5: ODVG Conversion

Convert each fold's train/val splits to ODVG JSONL format:
```json
{"filename": "img_001.jpg", "height": 1024, "width": 1024, "detection": {"instances": [{"bbox": [120, 340, 280, 410], "label": 0, "category": "crack"}]}}
```

Also produce `label_map.json` and carry forward `annotated_classes.json` for loss masking.

### Text Prompt Strategy

Three strategies, configurable via `prompt_strategies.yaml`, compared as an ablation:

| Strategy | Example for "crack" | When to use |
|----------|-------------------|-------------|
| **(a) Class name** | `"crack"` | Baseline, simplest, no train-test mismatch |
| **(b) Template description** | `"a crack defect on aircraft surface showing linear fracture pattern"` | Likely best for domain-specific adaptation |
| **(c) Model-generated** | Generated by VLM from example crops | Richest, but noisiest; pre-compute once |

**Recommendation**: Start with (a), run full LODO. Then run (b) as the primary improvement. Run (c) only if time permits. Report all three as a prompt ablation table.

---

## 3. Training Pipeline

### Model: Grounding DINO Swin-T + LoRA

**Base repo**: [Open-GroundingDino](https://github.com/longzw1997/Open-GroundingDino) (native ODVG + mixed dataset support)
**Port from** [Asad-Ismail/Grounding-Dino-FineTuning](https://github.com/Asad-Ismail/Grounding-Dino-FineTuning): LoRA integration (PEFT), EMA wrapper, LoRA-only checkpointing

### Key Hyperparameters (A5000 budget)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| LoRA rank | 16 | Saves VRAM vs rank-32; <2% params trainable |
| Batch size | 4 | Fits in 24GB with fp16 + LoRA |
| Gradient accumulation | 4 | Effective batch = 16 |
| LR (LoRA layers) | 5e-5 | Standard for DINO fine-tuning |
| LR (backbone LoRA) | 1e-5 | Lower for pretrained backbone |
| Epochs | 30 max | Early stopping at patience=7 |
| EMA decay | 0.9997 | Retains pretrained knowledge |
| Mixed precision | fp16 | Essential for A5000 |
| Warmup | 3 epochs | Cosine schedule after warmup |

**Memory estimate**: ~7-8 GB model + ~4 GB activations/gradients + ~3-4 GB EMA ≈ 16-18 GB. Headroom of ~6 GB.

### Loss Masking (Core Technical Piece)

**Problem**: When the model sees all 7 class names in the text prompt but an image only has annotations for {crack, dent}, detections of "corrosion" should NOT be penalized — we don't know if corrosion is absent or just unlabeled.

**Implementation**: Subclass `SetCriterion` → `MaskedSetCriterion`:

1. Each image carries its `annotated_classes` set in the target dict
2. Build a `valid_token_mask` of shape `(batch, num_text_tokens)` — 1 for tokens of annotated classes, 0 for unannotated
3. Multiply the focal classification loss by this mask (broadcast across queries)
4. Adjust normalization denominator to count only unmasked entries
5. **Box losses (L1, GIoU) need NO masking** — they only fire on matched pairs, which are always annotated classes by construction

**Verification**: Unit test with a synthetic batch: one image with `annotated_classes={0,1}`, another with all 7. Confirm gradients for classes 2-6 are zero in the first image, nonzero in the second.

### Training Loop

For each epoch:
1. Forward pass with `torch.cuda.amp.autocast` (fp16)
2. Compute masked loss via `MaskedSetCriterion`
3. Backward + gradient accumulation (step every 4 batches)
4. Clip gradients (max_norm=0.1)
5. Optimizer step + scheduler step
6. Update EMA model
7. End of epoch: validate on val set using EMA model → check early stopping

### LODO Runner

Sequentially run all N folds:
1. Load fold k's ODVG data
2. Build fresh model + LoRA + optimizer + EMA
3. Train to convergence or early stop
4. Evaluate on held-out test set (COCO mAP)
5. **Free GPU memory** (`del model; torch.cuda.empty_cache()`) before next fold
6. Aggregate all fold results

---

## 4. Evaluation + Reporting

### Metrics per fold
- mAP, mAP@50, mAP@75
- Per-class AP (7 classes)
- Per-size AP (small/medium/large defects)

### Aggregated reporting
- **Headline**: Mean mAP ± std across all folds
- **Per-class transfer**: Which defect types generalize best/worst across aircraft
- **Per-fold breakdown**: Which aircraft type is hardest to generalize to
- **Prompt ablation**: Strategy (a) vs (b) vs (c) comparison table

---

## 5. Method Explanation Document (`docs/methodology.md`)

Sections:
1. **Problem Statement** — Why zero-shot cross-domain matters for aircraft inspection
2. **Model Architecture** — Grounding DINO: DETR + text grounding + why it suits cross-domain transfer
3. **LoRA Adaptation** — Which layers, rank, parameter efficiency
4. **Data Pipeline** — Class taxonomy, dedup, annotation merging
5. **Loss Masking** — The "unannotated ≠ absent" problem, token-level masking math, why box loss is unaffected
6. **Text Prompt Strategies** — Three approaches and rationale
7. **LODO Protocol** — Why it proves generalization, handling overlapping images in splits
8. **Training Details** — Hyperparameters, EMA, early stopping, memory optimization
9. **Results Template** — Tables and charts to fill in

---

## 6. Implementation Order

| Phase | Work | Est. Time |
|-------|------|-----------|
| **Phase 1: Data pipeline** | `ingest.py`, `taxonomy.py`, `dedup.py`, `merge.py`, `lodo_splits.py`, `coco_to_odvg.py`, `prompt_builder.py`, configs | 2-3 days |
| **Phase 2: Training infra** | Port LoRA + EMA, implement `MaskedSetCriterion`, `trainer.py`, `model_builder.py` | 3-4 days |
| **Phase 3: LODO automation** | `lodo_runner.py`, `evaluate.py`, `aggregate.py`, end-to-end test on 1 fold | 2 days |
| **Phase 4: Ablation + docs** | Run all prompt strategies, write `methodology.md`, generate final report | 1-2 days |

---

## 7. Verification Plan

1. **Data pipeline**: Manually inspect 5-10 ODVG lines per fold — verify bboxes, class labels, text prompts are correct
2. **Dedup**: Visualize matched image pairs to confirm they're true duplicates
3. **Loss masking unit test**: Synthetic batch with known annotated_classes, verify gradient masking
4. **Single fold smoke test**: Train fold 0 for 5 epochs, verify loss decreases, val mAP > 0
5. **Memory check**: Monitor GPU memory during training — must stay under 24 GB
6. **Full LODO run**: All folds, verify results are reproducible (set seeds)
7. **Evaluation sanity**: Visualize predictions on held-out images — detections should make visual sense
