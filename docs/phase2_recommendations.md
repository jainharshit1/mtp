# Phase 2 Recommendations

## Scope

These recommendations review the proposed Grounding DINO, LoRA, loss-masking,
and Leave-One-Dataset-Out (LODO) training plan in
[`phase1_docs_and_phase2_plan.md`](./phase1_docs_and_phase2_plan.md).

The existing plan is a strong architectural draft, but it should not be
implemented as written until the data split, Grounding DINO loss behavior, and
evaluation path have been verified.

## Priority summary

| Priority | Area | Recommendation |
|---|---|---|
| Critical | Loss masking | Validate token-level logits, positive maps, matcher costs, and `SetCriterion` behavior against the exact Grounding DINO checkout before implementing `MaskedSetCriterion`. |
| High | LODO evaluation | Separate strictly image-disjoint held-out evaluation from evaluation containing perceptually shared images. |
| High | Data leakage | Manually inspect pHash groups and test the grouping threshold before relying on LODO results. |
| High | Evaluation | Keep category IDs consistent between ground truth and predictions; validate zero-based IDs with a synthetic COCOeval test. |
| High | Smoke tests | Replace “validation mAP > 0” with finite-loss, gradient, checkpoint, and reproducibility checks. |
| Medium | Reproducibility | Record seeds, configuration, dependency versions, repository commits, tokenizer version, and data hashes for every fold. |
| Medium | Dependencies | Pin the training environment after a successful installation. |
| Medium | Scope | Establish a minimal baseline before adding LoRA, EMA, prompt ablations, and all eight folds. |

## 1. Freeze and validate Phase 1 first

The current processed files exist, including:

```text
data/processed/unified_annotations.json
data/processed/lodo_folds.json
data/odvg/fold_0/train.jsonl
```

However, the dataset configuration and processed outputs have been modified
recently. Re-run the ingestion pipeline from the current configuration before
starting Phase 2:

```bash
cd /DATA/air_force_object_detection
source .venv/bin/activate
python -I scripts/01_ingest_and_unify.py \
  --config configs/datasets.yaml
```

Then verify:

- All eight configured datasets appear in the generated metadata.
- Every ODVG image path exists.
- Every bounding box has valid coordinates.
- Train, validation, and test image IDs are disjoint within each fold.
- The held-out dataset and fold counts match the current data.
- `annotated_classes` agrees with the classes configured for each source
  dataset.

Do not rely on the numbers in the plan until they have been regenerated and
checked against the current dataset contents.

## 2. Clarify the LODO claim and prevent leakage

The current split logic places images from perceptually shared groups into the
training pool. Consequently, an image from the held-out dataset can have a
near-identical image from another dataset in training.

This is not strict image-disjoint zero-shot evaluation.

Use two explicitly named evaluation protocols:

### Strict exclusive-held-out protocol

Test only images whose perceptual group belongs exclusively to the held-out
dataset. This measures transfer to images not detected in other datasets, but
it may omit difficult shared samples.

### All-held-out protocol

Test all images associated with the held-out dataset. Report this separately
because shared groups may have related images in training.

For every fold, report:

- Number of test images.
- Number of exclusive test groups.
- Number of shared groups excluded or included.
- Number of test annotations by class.
- Whether any exact or perceptual duplicate exists in training.

The final report should avoid describing the all-held-out result as fully
zero-shot unless the training and test groups are demonstrably disjoint.

## 3. Validate perceptual deduplication

The pHash threshold is a high-impact experimental parameter. A false negative
causes leakage, while a false positive removes valid data from training or
testing.

Before training:

1. Export representative pairs from groups at and near the pHash threshold.
2. Inspect exact duplicates, augmentations, and visually unrelated images.
3. Test the SHA-256 exact-duplicate path separately from pHash grouping.
4. Record the selected threshold and its justification.
5. Save the grouping statistics with every experiment.

Add automated tests for:

- Identical files.
- Resized or recompressed copies.
- Mildly augmented copies.
- Visually different aircraft images.

## 4. Verify Grounding DINO loss behavior before coding the mask

Loss masking is the highest-risk part of Phase 2. The proposed implementation
assumes that class names map directly to token positions in the classification
loss. This must be confirmed against the exact Open-GroundingDino source being
used.

Inspect and document:

- The shape and meaning of `pred_logits`.
- How captions are tokenized.
- How `positive_map` is constructed.
- How target labels map to token positions.
- Whether classification loss is computed over all token positions.
- Which classification terms are used by the Hungarian matcher.
- Whether auxiliary decoder losses use the same classification path.
- How padding and special tokens are handled.

Do not assume that masking the final focal BCE tensor automatically masks all
classification influence in the matcher or auxiliary losses.

The implementation should include a synthetic test with two images:

- Image A has `annotated_classes = [0, 1]`.
- Image B has all classes annotated.

Verify that:

- Unannotated token positions for image A contribute zero classification loss.
- Unannotated token positions for image A have zero classification gradient.
- Annotated positions for image A retain nonzero gradients.
- Image B retains gradients for all annotated classes.
- Auxiliary decoder losses obey the same masking behavior.
- The matcher does not create invalid matches for unannotated classes.

## 5. Compute token spans from the actual tokenizer

The token positions in the plan are only examples. They can change with the
Transformers version, tokenizer settings, punctuation, or prompt strategy.

For every prompt strategy:

1. Build the exact caption used during training.
2. Tokenize it with the same tokenizer and settings used by the model.
3. Compute character spans from the exact class strings.
4. Map spans to token indices.
5. Assert that every class has a nonempty span.
6. Assert that class spans do not overlap unexpectedly.
7. Decode each span and compare it with the expected class text.

Never reuse token spans across prompt strategies.

## 6. Make taxonomy decisions explicit

The current taxonomy maps both `contusion` and `scratches` to `scratch`.
This may be appropriate for the intended super-class taxonomy, but it changes
the original AGDD semantics.

Document:

- Why `contusion` is considered equivalent to `scratch`.
- Why `scratches` is considered equivalent to `scratch`.
- Whether the distinction is impossible or simply out of scope.
- How this affects per-class evaluation.

Also call out that `spot` has very few annotations relative to the other
classes. Its AP will have high variance and should not be interpreted as
equivalent evidence of performance.

Every report should include per-class support counts and should flag classes
with low support.

## 7. Validate COCO evaluation independently

COCO category IDs do not need to begin at one. They must be consistent between
the ground-truth JSON and prediction results.

Before model training:

1. Create a tiny synthetic COCO dataset with category IDs `0..8`.
2. Create a perfect prediction for the synthetic annotations.
3. Run COCOeval.
4. Confirm that the perfect prediction produces the expected score.
5. Confirm bbox conversion through the complete pipeline:
   - normalized `cx, cy, w, h`
   - absolute `x1, y1, x2, y2`
   - COCO `x, y, width, height`

Also test empty predictions, images with no detections, and classes absent from
a fold.

## 8. Use stronger smoke-test criteria

“Validation mAP > 0” is not a sufficient smoke-test requirement. A random
model can occasionally produce a nonzero score.

A smoke test should pass only when all of the following are true:

- The model imports successfully.
- The deformable-attention extension imports successfully, if required.
- One batch completes forward propagation.
- One batch completes backward propagation.
- All losses are finite.
- Expected trainable parameters have nonzero gradients.
- Masked token gradients are zero where expected.
- GPU memory remains within the configured limit.
- A checkpoint can be saved and loaded.
- A fixed tiny fixture produces reproducible outputs within tolerance.

Use mAP only as an additional diagnostic at this stage.

## 9. Start with a minimal baseline

The proposed scope is large: LoRA, EMA, loss masking, prompt ablations, eight
folds, visualization, and reporting. Implement these incrementally.

Recommended sequence:

1. One fold, one fixed prompt strategy, pretrained model inference only.
2. Evaluation and visualization on that fold.
3. A short full-parameter or existing-reference fine-tuning baseline.
4. Loss masking with synthetic gradient tests.
5. LoRA fine-tuning.
6. EMA and checkpoint resume.
7. One complete fold to convergence.
8. Remaining folds.
9. Prompt ablations and optional VLM-generated prompts.

This ordering makes failures attributable and prevents expensive eight-fold runs
from hiding a broken component.

## 10. Reproducibility requirements

Save the following in every fold output directory:

- Resolved YAML configuration.
- Command-line arguments.
- Python version.
- PyTorch, CUDA, Transformers, PEFT, timm, and pycocotools versions.
- Git commit IDs for vendor repositories.
- Dataset configuration hash.
- Processed annotation and ODVG file hashes.
- Random seeds for Python, NumPy, and PyTorch.
- Tokenizer name, revision, and configuration.
- GPU name and driver version.
- Training and validation split counts.
- Checkpoint metadata.

Use deterministic settings where practical, and document any nondeterministic
CUDA operations that remain enabled.

## 11. Resolve environment and dependency risks

The training prerequisites listed in the plan are not currently present:

```text
/DATA/air_force_object_detection/vendor/Open-GroundingDino/
/DATA/air_force_object_detection/weights/groundingdino_swint_ogc.pth
```

Install and validate the environment before implementing the trainer. Avoid
unbounded installation of latest package versions. After a successful setup,
record exact versions in a locked requirements file.

Validate all of the following:

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import transformers, peft, timm, pycocotools"
```

For custom CUDA operations, record:

- `torch.version.cuda`
- NVIDIA driver version
- compiler version
- GPU compute capability
- extension build output
- successful import test

Do not begin a full training run until the setup script passes.

## 12. Define EMA and checkpoint semantics

PEFT and EMA need a precise state contract. Decide and test whether EMA tracks:

- The complete model parameters.
- Only trainable LoRA parameters.
- A merged model representation.

The checkpoint must specify:

- Base model identifier and revision.
- LoRA configuration.
- LoRA state.
- EMA state, if present.
- Optimizer and scheduler state.
- Epoch and best metric.
- Taxonomy and prompt strategy.

Test both checkpoint loading for inference and checkpoint loading for training
resume before starting a long run.

## 13. Recommended acceptance criteria

Phase 2 should not be considered complete until:

- Phase 1 has been regenerated and validated from the current configuration.
- Strict and shared-image evaluation protocols are reported separately.
- Grounding DINO forward and backward passes succeed.
- Token-span mapping is tested for every prompt strategy.
- Loss masking has synthetic gradient coverage.
- COCO evaluation passes a perfect-prediction fixture.
- One complete fold trains, evaluates, saves, and reloads successfully.
- Results are reproducible from the recorded configuration and commit IDs.
- All eight folds run without GPU-memory leakage between folds.
- The final report includes per-class support, confidence limitations, and
  leakage protocol details.

## Final recommendation

Treat the existing plan as an architecture document. First stabilize and
validate the data and evaluation protocols, then prove the loss-masking behavior
with a minimal synthetic test, and only afterward scale to LoRA training and
all eight LODO folds.
