# Findings: validation leakage, evaluation bugs, and a train/held-out gap (fold 0)

Date: 2026-10-08. Setup: Grounding DINO Swin-T + LoRA, leave-one-dataset-out (LODO).
Fold 0 holds out `RFDETR_DATASET`. All numbers below are from fold 0 only.

## One-paragraph summary

During fold 0 training the validation loss looked healthy (train 15.96, val 15.71, both still falling).
Checking it showed that **83% of the validation images had a near-duplicate scene in the training set**, so
validation loss could not reveal overfitting. Re-evaluating saved checkpoints on the held-out dataset, with
three evaluation bugs fixed, gave **mAP ≈ 0.006 on held-out images versus 0.16 on training images**, and held-out
mAP did not improve from epoch 4 to epoch 19 while training mAP rose roughly 4x. The model is learning the
training scenes, not defects that transfer to an unseen dataset.

## 1. Validation leakage

**What we saw.** Val loss sat slightly below train loss from epoch ~9 on. This is not alarming by itself
(EMA weights are used for validation, train loss is an average over the epoch, LoRA dropout is active only in
training), but it prompted a check.

**Root cause.** The dataset has 32,579 images but only **12,387 perceptual-hash groups** (~2.6 images per scene;
Roboflow-style flips/crops/tints of the same photo; one group has 24 copies). The LODO *test* split was built by
group and is clean. The train/val split was not: `generate_lodo_folds` shuffled individual images and took 10%,
so copies of one scene landed on both sides.

**Measured.** Fold 0: 2,182 of 2,627 val images (83.1%) shared a group with a training image.

**What it does and does not affect.**
- It does not change the gradients (only the train loader feeds the optimiser; the LR schedule is step-based).
- It does bias *model selection*: `best.pt`, early stopping, and which checkpoint is tested are all chosen by val
  loss, and a leaky val keeps improving while the model memorises.
- The final test mAP was never affected, because test images come from groups exclusive to the held-out dataset.

**Fix.** `split_val_by_group` moves whole groups into val. Applied to folds 1-7 (0 val images share a group with
train, asserted on every fold). Fold 0 could not be re-split mid-run without throwing away training, so at epoch 20
it was restarted from `latest.pt` with a clean val of the 436 old-val images whose groups never appear in train
(train unchanged). The trainer now fingerprints the val set and resets best-checkpoint tracking when it changes.
Test sets and label maps are byte-identical before and after.

## 2. Evaluation bugs (found by code review)

| # | Bug | Effect |
|---|---|---|
| 1 | Test images were fed at original size; training resizes the short side to 800 | scale mismatch, lower mAP |
| 2 | Detections below score 0.3 were discarded | truncated precision/recall curve, lower mAP |
| 3 | Predictions for classes the held-out dataset does not label counted as false positives (training masks those classes) | lower precision |

All three are fixed in `src/evaluation/evaluate.py` (resize like training; score floor 0.001 with top-100 per image;
unannotated classes masked). `scripts/eval_checkpoint.py` evaluates any saved checkpoint.

## 3. Images without boxes

560 images (1.7%) have no boxes and were silently dropped by the converter (`if not anns: continue`).
They are valid negatives: each carries its dataset's `annotated_classes`, so the loss penalises false positives
only on the classes that dataset labels. We verified that all-empty and mixed batches train without error and now
keep them (about 300-500 per fold). They are group-split like every other image. Fold 0 still trains without them.

## 4. Results: fold 0, fixed evaluator

Held-out = 600 random RFDETR-exclusive test images (fixed seed). Train = 400 random annotated training images.
mAP is COCO mAP@[.5:.95]; mAP50 in brackets.

| Weights | Held-out mAP (mAP50) | Train mAP (mAP50) |
|---|---|---|
| Pretrained, no fine-tuning | 0.0018 (0.0026) | 0.0081 (0.0175) |
| Epoch 4 | 0.0048 (0.0071) | 0.0425 (0.0771) |
| Epoch 9 | 0.0050 (0.0090) | 0.1177 (0.2160) |
| Epoch 14 | 0.0057 (0.0117) | 0.1437 (0.2741) |
| Epoch 19 | 0.0057 (0.0114) | 0.1594 (0.3012) |

Per-class AP at epoch 19, held-out: crack 0.012, dent 0.006, scratch 0.007, paint-off 0.006, corrosion 0.002,
missing-head 0.001 (rupture and fastener-damage have no held-out ground truth in this fold).

Reading the table:
- Fine-tuning helps a lot on the training scenes (0.008 -> 0.16) and barely at all on the held-out dataset
  (0.002 -> 0.006), and the held-out number is flat after epoch 4.
- Training loss and val loss kept falling over the same epochs. Loss alone would have suggested steady progress.
- Even the training-set mAP is modest, consistent with the high loss (~16).
- On probed images the model localises roughly correctly but with low confidence (0.2-0.45), and scores `crack`
  and `rupture` almost identically on the same box.

## 5. What is not established

- **The cause of the gap.** Candidates: a genuine domain shift between RFDETR-exclusive images and the other
  datasets; too little capacity (LoRA rank 16, 0.85% of parameters trainable); weak bare class-name prompts;
  few distinct scenes (~12k). We have not isolated which one.
- **Generality.** Only fold 0 was examined. Folds 6 and 7 have very small test sets (193 and 339 images).
- **Precision of the numbers.** 600/400-image samples give rough estimates; the train-set evaluation does not apply
  class masking. Differences of ~0.001 mAP between checkpoints are noise.

## 6. Suggested next steps

1. Do not run all 8 folds until the gap is understood.
2. Run one controlled change on a short schedule and compare held-out mAP against the table above: higher LoRA
   rank, a richer prompt (class descriptions), stronger augmentation, or sampling one image per scene group per epoch.
3. Evaluate another fold's checkpoint to see whether the held-out collapse is specific to RFDETR.
4. Report test mAP only, selected with the group-split val set.

## Files

`src/data/lodo_splits.py` (`split_val_by_group`), `scripts/resplit_val_by_group.py`,
`src/training/trainer.py` (val fingerprint), `src/evaluation/evaluate.py`, `scripts/eval_checkpoint.py`,
`outputs/fold_0/eval_*_{test,train}.json` (raw results), `data/odvg_leaky_backup/` (original splits, not in git).
