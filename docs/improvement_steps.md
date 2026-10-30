# Improvement steps: data splits and training time

Started 2026-10-27. Work through the steps in order; each has the evidence, the change, and how to
check it. Status: **done** (code changed and tested), **open** (not started), **decide** (needs a choice first).

## A. Data and splits

### A1. Held-out images leaked into training in the old folds (open: regenerate folds)
- **Evidence.** In `data/processed/lodo_folds.json` (2026-10-06), groups containing images from the
  held-out dataset *and* another dataset were sent to train. Held-out images in train/val:
  fold 0 RFDETR 8,450 · fold 4 sydney 3,153 · fold 1 korean-aerospace 2,272 · fold 3 Defect-Detection-6 464 ·
  fold 5 closeup 385 · fold 7 fuselage 50 · fold 2 AI-Dataset-4 3 · fold 6 AGDD 0.
  None of them shares a scene with a test image, so test photos were never trained on, but the
  "unseen" dataset was partly seen. Results from these folds are not valid LODO results.
- **Change (done in code).** `generate_lodo_folds` sends every group that touches the held-out dataset to
  test (held-out images only) and drops its copies from other datasets; `_check_fold` stops on any held-out
  image in train/val.
- **To do.** Rerun `scripts/01_ingest_and_unify.py` on the server, then `scripts/audit_leakage.py` on the new folds.

### A2. Copies per scene: caps (done)
- Train/val keep at most 5 images per sub-scene (`--max-per-group 5`), test at most 2
  (`--max-test-per-group 2`). Byte-identical files count once toward a cap (SHA256).
- Old folds kept every copy (about 2.6 per scene), which also doubled epoch length (see B2).

### A3. DINOv2 chains whole videos into one group (done: thinning + per-sub-scene caps)
- **Decision (2026-10-27).** The group still decides the split side (whole video on one side). Within a
  group, near-repeat frames are dropped (`--thin-threshold 0.95`: an image is skipped if its cosine to an
  image already kept from the group is >= 0.95), and the caps count per sub-scene, not per group.
- **Measured on AI-Dataset-4 + Defect-Detection-6 (2,920 images, cap 5):** cut-off 0.97 keeps 1,710,
  0.95 keeps 1,372, 0.93 keeps 1,184; the 629-image group keeps 385 / 261 / 179. Visual check
  (sheets deleted 2026-10-28): 0.97 still keeps near-identical neighbours;
  0.95 keeps about one frame in 3-4 and every distinct view.
- Evidence before the decision:
  one group holds 629 images (626 from AI-Dataset-4, 64% of that dataset); without DINOv2 links it
  falls apart into 390 sub-scenes. (Sheets deleted 2026-10-28.)
  The first sheets show consecutive frames of a walk-around video of one aircraft, linked frame to frame at
  cosine 0.92-0.99. With a cap of 5 per group, the 629 images would have shrunk to 5.
- **Plan for building the folds:** `fold_regeneration_plan.md`.

### A3b. Chance pHash matches chain unrelated images (done)
- A whole-aircraft photo (AI-Dataset-4 `sceneadd502241`) matched a rotated rivet close-up
  (Defect-Detection-6 `6812`) at pHash distance 8 (DINOv2 cosine -0.01). That one link put the
  626-frame video in the same group as Defect-Detection-6, so the whole video was excluded from the
  fold that holds out Defect-Detection-6.
- Of 2,366 pHash links on the two datasets, 7 have cosine <= 0.33, all false; the next is 0.64.
- **Change.** A pHash link also needs DINOv2 cosine >= 0.7 (`PHASH_MIN_COSINE` in `dedup.py`,
  `--phash-min-cosine`). 0.5 was enough on two datasets but not on all eight: there it chained 10,420
  images into one group (see `fold_regeneration_plan.md`, rule 2). With 0.7 the largest group is 1,806.
- **Remaining false merges (accepted, leakage-safe):** in that 1,806-image group, 193 DINOv2 links join
  plain grey closeup and fuselage panels that only look alike; RFDETR-sydney links there are real copies.

### A3c. One large group could become most of val (done)
- Val took whole groups until it reached 10%; the thinned 261-image video landed in val (val 266 vs train 214).
- **Change.** A group that would push val past 10% goes to train (`split_val_by_group`).

### A4. Roboflow name collisions over-merge (decide)
- In sydney (and a few in AI-Dataset-4), files with the same Roboflow source name are different photos
  (for example `IMG_0389`, `IMG_0313`, `4.jpg`). Safe for leakage, but merged photos are capped away.
- (Example sheets deleted 2026-10-28; numbers in `D:\MTP\dinov2_validation\report.txt`.)
- A cosine floor does not separate them cleanly: sydney same-name pairs at 0.30-0.40 are different photos,
  rotated genuine copies sit at 0.58+, and a few are at 0.48-0.53. Left unchanged: the name rule only links
  within one dataset, so a wrong merge costs a few images, not a chain.

### A5b. RFDETR re-exports other datasets' photos under new names (done: per-fold near-copy filter)
- Rotated a few degrees with black corners and recoloured; DINOv2 0.80-0.90, pHash 10-16; no grouping
  rule links them. 393 test images had a training copy at >= 0.85 (168 in fold 4).
- **Change.** In each fold, drop training images with DINOv2 cosine >= 0.80 to any test or val image
  (`--proximity-filter 0.80`, about 2.4% of training images). See `fold_regeneration_plan.md`, rule 6b.

### A5. Known detection gap (open, record only)
- Heavy crops and rotations of copies stored under different names (RFDETR, fuselage, AGDD, cross-dataset)
  are missed by every rule. Synthetic test: DINOv2 catches 66% of 80% crops, 20% of 15-degree rotations,
  10% of 60% corner crops (`D:\MTP\dinov2_validation\report.txt`).

### A6. Exact duplicates (done locally; server open)
- Local `mtp-data`: 371 copies of labelled fuselage images in `unlabel_aircraft_fuselage` were deleted
  (list: `D:\MTP\mtp-data\removed_unlabel_duplicates.txt`). No exact duplicates remain in the data the pipeline reads.
- **To do.** Check the server copy (`/DATA/air_force_object_detection`) the same way.

## B. Training time

Measured from `training.log` (RTX A5000): fold 0 had 2,905 iterations/epoch (23,241 images, batch 8) at
1.2-1.45 s/iteration, so 60-70 min/epoch plus 3-6 min validation, about 31-37 h for 30 epochs.
The rest of the 3-4 days per fold was lost to crashes and gaps (B3-B5).
Folds are now 6 instead of 8 (AGDD and fuselage are trained on only), see `fold_regeneration_plan.md`.

### B1. Stop training when held-out performance stops improving (done 2026-10-27, not yet run on GPU)
- **Evidence.** Early stopping watched a leaky val loss that kept falling, so all 30 epochs always ran.
  Held-out mAP was flat from epoch 4 to 19 (`findings_val_leakage_and_overfitting.md`). About 80% of compute bought nothing.
- **Change.** Use the clean group-split val, select on val mAP (not loss), lower `max_epochs` (about 10-12) and patience (about 3).
- **Done.** `training.early_stopping_metric: mAP`: each epoch computes val COCO mAP with the same scoring as the
  test evaluation (`evaluate.detections_from_outputs`, EMA weights); best.pt and early stopping use it.
  `max_epochs` 12, patience 3, warmup 1 epoch, EMA decay 0.999 (0.9997 averaged over ~6 of 12 epochs on the
  new, smaller folds). The test set is evaluated once, after training, with best.pt; never used to select.
- **Check.** Fold 0 retrain: epoch at which early stopping fires, held-out mAP vs the old epoch-19 result.

### B2. Fewer near-identical images per epoch (done in code via A2)
- Caps reduce fold 0 train from 23,241 to roughly 9-11k images (old groups), so epochs take less than half the time.

### B3. Resume crash at the last epoch (done 2026-10-27)
- `Trainer.train()` raises `UnboundLocalError: 'epoch'` when resuming a finished fold
  (`src/training/trainer.py:167`, the loop never runs). Fold 0 hit this on every restart and was never evaluated.
- **Done.** Checkpoints record `finished`; resuming a finished or early-stopped fold trains nothing and goes
  to evaluation. Also: checkpoint overwrite used `rename`, which fails on Windows; now `replace`.

### B4. Interrupted epoch is skipped on resume (done 2026-10-27)
- The SIGTERM handler saves `latest.pt` with the current epoch, and resume starts at epoch + 1, so the rest
  of that epoch never trains (fold 0 epoch 11, fold 2 epoch 4).
- **Done.** The signal handler only sets a flag; the loop stops at the next optimizer-step boundary and saves
  `latest.pt` with `epoch_complete: false` and `batches_done`. Resume trains the remaining batches of that
  epoch, then validates. An interrupted fold is not evaluated; the run stops (second signal exits at once).

### B5. Leftover process on the GPU killed all remaining folds (done 2026-10-27)
- 2026-10-14 16:25: folds 2-7 all failed within one minute with CUDA OOM; "Process 19318 has 17.34 GiB"
  (an earlier training process still on the GPU). The runner moved on to the next fold after each failure.
- **Change.** Before (re)starting, check `nvidia-smi` for an old training process; stop the run instead of
  cycling through folds when a fold fails with OOM at step 0.
- **Done.** Before each fold, `check_gpu_free` refuses to start with less than `training.min_free_gpu_gb` (16)
  free and logs the GPU processes. CUDA OOM, a busy GPU or an interrupt stop the whole run (`StopRun`) instead
  of moving on to the next fold; other errors still skip only that fold.
- Gap 2026-10-09 09:21 to 2026-10-12 12:04: fold 1 started twice and completed no epoch; cause not in the log.

### B6. Training resolution (open, measure first)
- Training resizes to short side 800, long side at most 1333 (`odvg_dataset.py`, `RandomResize([800], 1333)`).
  About 90% of images are 640x640 (all of korean, sydney, closeup, AI-Dataset-4, Defect-Detection-6, AGDD,
  and 11,171 of RFDETR), so they are *upscaled* to 800x800: about 1.56x the pixels, no extra information.
  RFDETR also has 2,465 at 1024x1240 and 470 at 416x416; fuselage is 1120x960.
- Pre-resizing the files would not help (they are small already). Decision 2026-10-27: keep original sizes.
- **Option.** Training at 640 would be roughly 1.3-1.5x faster per epoch; small defects may suffer, and the
  pretrained model was trained near 800. Test on one fold before changing.

### B7. One input shape: pad to square, resize to 800x800 (done 2026-10-27, replaces aspect-ratio batching)
- Nearly all images become 800x800; RFDETR's 1024x1240 images become 800x969, so mixed batches are padded.
- **Done.** `data.fixed_square: true`: pad right/bottom to a square (ImageNet-mean fill), resize to 800x800, in
  train, val and test (`SquareResize`, `prepare_eval_image`). Boxes keep pixel coordinates through the padding;
  predictions are scaled by max(w, h) and clipped to the image. Tested on 73 real images of every shape in
  fold 1 (29 shapes): max box error 0.0001 px, perfect predictions give mAP 1.0.
  800 kept (pretraining size); training at 640 dropped.
- **Padding mask (done 2026-10-28, `data.pad_mask: true`).** The square padding is marked as padding in the
  model's mask (as in pretraining), so position encodings, encoder attention and query selection ignore it;
  boxes are normalised to the real image area and mapped back by (width, height). Tested: mask equals the
  padding exactly on 60 images / 28 shapes, box error 0.0001 px, perfect predictions mAP 1.0.
- **Batched test evaluation (done 2026-10-28, `evaluation.batch_size: 16`).** Batch 1 and 16 give identical
  detections and mAP (folds 0 and 2, 48 images each, stand-in model).

### B8. `cudnn.benchmark = True` with changing input sizes (done 2026-10-27)
- `lodo_runner.py:42`. cuDNN re-tunes for every new input shape. With B7, only a few shapes occur, so
  benchmark mode tunes once per shape; without B7, turn it off. Measure either way.
- **Done.** `cudnn.benchmark = data.fixed_square` (one shape: tune once).

### B9. Unreliable memory logging (done 2026-10-27)
- The log shows `mem=0.9GB` while the process held about 17 GB. Log `torch.cuda.max_memory_allocated()`
  instead, so batch size and headroom can be judged.
- **Done.** Logs `peak_alloc` (max_memory_allocated, reset each epoch) and `reserved`; epoch line and
  status.json carry the peak. Next: try batch 16 without accumulation if the peak allows.

### B10. Speed varied between runs (open)
- The same code ran at 1.18, 1.38, 2.0 and 3.44 s/iteration on different days, which points to other jobs
  sharing the server. Check `nvidia-smi` / `top` while training; profile one epoch (time waiting for data vs GPU step).

### B11. Smaller per-step overheads (open, low priority)
- CPU Hungarian matching for 7 decoder outputs x 8 images per step; 4 `.item()` GPU syncs per batch.

### B12. Further options (open)
- Text features: the caption is the same for every image and BERT is frozen; computing it once per run
  instead of per step saves a BERT forward per step (needs a change inside Grounding DINO's forward).
- bf16 instead of fp16 on the A5000 (no loss scaling, no skipped steps on overflow).
- Not planned: torch.compile / channels_last (Grounding DINO's custom deformable-attention ops).

## C. Data decisions 2026-10-28
- Defect-Detection-6 and closeup are one photo session (matched pairs are mostly consecutive frames); they are
  held out together (`lodo.hold_out_together`), and korean/sydney folds validate on AI-Dataset-4 instead of D6.
- Training uses only images with boxes (249 empty images; a sample showed unlabelled defects in many).
- Boxes with a side under 1 px (102, almost all RFDETR missing-head) are removed (`--min-box-side 1`).
- Class review (sheets: `mtp-data/lodo_v3/build/class_review`, regenerated for the final classes) -> 6 classes: crack, dent, corrosion, scratch,
  paint-off, missing-fastener.
  - `missing-head` was five source labels with different meanings. missing-fastener = RFDETR `missing-head` +
    sydney `Fastener Damage` (both mark empty fastener holes; without the merge fold 0 had no missing-head and
    fold 4 no fastener-damage training boxes). Dropped: AI-Dataset-4 `missing-rivet` (row-sized boxes along rivet
    seams), closeup `rivet_damage` (painted-over dimples, intact screws), D6 and fuselage `rivet_damage` (many
    intact screws; first tried as corrosion, rejected after the sample sheets).
  - rupture removed: sydney `Rupture` is mostly cracks and tears -> crack; closeup `Hole` (small pits) dropped.
  - closeup `Crack` dropped: the boxes are mostly along panel seams and edges.
  - closeup `Dent` dropped (2026-10-29): mostly tiny specks on flat panels, some boxes on seams.
  - A dropped label is also removed from that dataset's annotated classes, so unlabelled objects of that class
    are not counted as background (loss masking) or as false positives (evaluation).
- Val (2026-10-29): ~10% of every training dataset's images, in whole scene groups (a group is either train
  or val), selected like test (2 per sub-scene). Before, each fold's val was a whole other dataset
  (AI-Dataset-4 or korean), which annotates only 2 classes: in folds 1, 2, 4 and 5 val scored none of the
  classes the test set measures, so the best checkpoint was chosen on the wrong classes. Test is unchanged.
- Photo-matching filter over every dataset pair (2026-10-29): a sydney test image had a colour-inverted copy in
  korean training data (the filter compared sydney/korean only with RFDETR). Now every test/val image is
  matched against its 5 most similar training images in each dataset (its own included: val is a slice of
  the training datasets, so its likeliest copies are in its own dataset), on folds built without drops, and
  the drop list is computed once for all pair sets (`run_rebuild_valsplit.sh`).
- Val image with a copy or near-copy in training (photo match, or DINOv2 >= 0.80): its whole scene group
  moves from val to train (nothing is dropped, the scene stays on one side). Copies of test images are
  still dropped (from train or val). Moved groups can hold new copies, so matching repeats on the rebuilt
  folds (only new pairs) until nothing new is found.
- Near-copy clusters for val (2026-10-29): with random scene groups, 99-245 val groups per fold had a
  DINOv2 >= 0.80 near-copy in another group (neighbouring video frames, slightly different views: scene
  groups join only at >= 0.90) and moved to train, leaving val at 147-250 images (2-3%). Now groups linked by
  any pair >= 0.80 form a cluster that goes to val or train whole. Per dataset, isolated groups (no near-copy
  in any other group) fill val first; clusters of several groups are used only if the 10% target is not
  reached (AI-4: isolated groups hold 4-7% of its images, AGDD 8%, so AGDD stays near 8-9%).
- One copy per photo across training datasets (2026-10-29): 145-428 photos per fold were in two training
  datasets (RFDETR re-exports of sydney/korean, D6/closeup/fuselage from one session). Only one dataset's
  copies are kept, the best labelled (more annotated classes, then more boxes); test still keeps the
  held-out dataset's own copies.
- Val photo-match threshold 70 inliers (2026-10-29): about 30% of the new val images matched a training image
  at >= 25 inliers. A visual review of 10 random pairs per range (build/val_match_review/) found no copies at
  25-40 or 40-80 inliers (look-alike surfaces: rivet rows, seams) and copies at 80+ (rotated/cropped RFDETR
  re-exports, closeup shots of one panel). Val -> train moves (fold_geometric_filter.py --val-min-inliers) and
  the val leakage check use 70 (a margin below 80); copies of test images are still dropped at 25.
- Val own-dataset top 20 (2026-10-29): the first val leakage check failed on 1-4 of 200 sampled val images
  per fold, all same-dataset copies (RFDETR re-export families, one AI-4 frame; cosine 0.60-0.71, 74-640
  inliers) ranked 6th-10th most similar, past the filter's top 5 per dataset. Val images are now also
  matched against the 20 most similar training images of their own dataset (`--val-pairs "*:=:20"`, 22k
  more pairs); the build resumed at pass 8.
- Val copy rule extended (2026-10-29, visual review of the failed check and of 40-69 inlier pairs,
  build/val_match_review/40_70_score_*.jpg): 40-69 inliers with DINOv2 cosine >= 0.60 were mostly the same
  photo (rotated/cropped/recoloured re-exports, shots seconds apart), below 0.60 mostly different photos.
  A val image is a copy at >= 70 inliers, or 40-69 inliers with cosine >= 0.60 (filter and val check).
- Val leakage check (`scripts/check_val_leakage.py`, last step of the rebuild): reads only the fold files;
  per fold, 200 sampled val images are photo-matched against their 10 most similar training images of any
  dataset, and over all val images the highest cosine to training and near-duplicates inside val are
  reported. Fails the build on any same-photo match; sheets in `build/val_check/`.
- Val refill and class-aware val (2026-10-29; user: val about 10%, a little more or less; only a training
  group with no relative in any other training group may move to val; no leakage above all; balance of
  classes in val; option A). Before: val took ~10% once, and every val group moved to train for a
  photo-matched copy left a gap, so val ended at 4-9% per dataset (RFDETR 4.2-5.3%), also because val
  keeps at most 2 copies per photo against 5 in train; nothing looked at classes (fold 0: 18 dent boxes).
  Now `pick_val_groups`, by priority:
  1. only groups with no DINOv2 near-copy (>= 0.80) in any other group and not barred can go to val; a
     group with a val image photo-matched to training is barred from val for good; never relaxed;
  2. ~10% of every dataset, counted per image as val / (train + val), with real val sizes and without the
     copies of test dropped later; no group may push its dataset past 12%;
  3. first, while a class has < 50 boxes or < 15 groups in val, the next eligible group with that class
     is added (most short class first); then every dataset is filled to 10%. Shortfalls are logged,
     never forced.
  Groups are tried in a fixed hash order, so barring a group changes only its replacements. Option A:
  pick, photo-match only the picked val images, bar the matched ones, refill, repeat until a pass finds
  nothing (at most 20 passes). Independent review (subagent, 2026-10-29): ~10% per training dataset is
  sensible for checkpoint selection (cf. DomainBed's training-domain validation, Gulrajani & Lopez-Paz
  2020); what matters is boxes and independent groups per class (>= 50-100 boxes, >= 15-20 groups);
  select by smoothed val mAP and also log the last checkpoint. Preview: val 9.5-10.1% per fold; AGDD
  ~7% and AI-Dataset-4 3-7.5% (too few stand-alone groups, accepted); fold 0 dent 18 -> 92 boxes,
  missing-fastener 25 -> 56 (14 groups each: no eligible group left).
  Run script: python by absolute path, earlier copy moved to build/superseded_<time>/ before copying,
  and a copy = build check (SHA256 per split) after verify.
- Heavily filtered copies (2026-10-30). The val leakage check after the refill rebuild failed on 2 of 1,200
  sampled val images; visually both are the same photo as a training image, one copy embossed (grainy
  relief filter; one also rotated): fold 2 val RFDETR aircraft_003338 = train korean 40_24 (143 inliers,
  cosine 0.42); fold 4 val korean 118_24 = train korean 118_3 (74 inliers, cosine 0.44). The filter only
  matched pairs at cosine >= 0.45 (--min-cos), so such pairs were never matched. Removing the floor for
  all specs would add 181k pairs (~6-7 h); instead a new spec `*:+:10:0` matches every test and val image
  against its 10 most similar training images of any dataset at any cosine (as the val check does):
  19.2k extra pairs (~45 min). Spec syntax now `q:t:K[:MIN_COS[:MAX_COS]]`, `+` = all datasets in one pool.
  Visual review of its matches: real filtered (embossed, posterized) copies at every inlier level down to
  ~30, but at cosine >= 0.45 the deeper K=10 matched mostly look-alike fuselages (rows of windows, liveries)
  at >= 25 inliers (rounds 3-4: ~0 of 33 the same photo), and each drop pulled the next neighbour in:
  fold 0 lost 125 more training images and was not converging (22, 18, 15 per round). The spec is now
  limited to the gap it was meant for, `*:+:10:0:0.45` (below cosine 0.45 only); at >= 0.45 the earlier
  approved specs apply unchanged. Drop list recomputed from scratch (step 2), so the look-alike drops
  are undone.
- Final lodo_v3 build (2026-10-30 03:08, REBUILD DONE): converged at pass 7. train/val/test per fold:
  0 RFDETR 1983/223/4207; 1 korean 5977/652/864; 2 AI-4 7764/863/449; 3 D6 6583/695/615;
  4 sydney 3987/419/1523; 5 closeup 6464/690/1205. Val 10.0-10.6% per dataset except AGDD ~7%,
  AI-4 2.7-7.5% (too few stand-alone groups), fold 0 korean/sydney ~14%, fold 5 fuselage 15.9%.
  Checks: DINOv2 audit 0 at >= 0.80; verify passed; copy = build (SHA256) in all folds; annotations
  preserved; test photo audit 0%; val leakage check 0 same-photo in 1,200 sampled val images (all its
  pairs were already matched by the rounds, so it confirms them rather than adding new comparisons).
  Val boxes per class >= 44 in every fold (fold 0: crack 44, missing-fastener 48, dent 83).
- Korean photo numbers and fuselage hold-out (2026-10-30, user decision after visual review).
  korean-aerospace names its own variants `N_M` (photo N, variant M): all variants of a number are the
  same photo, recoloured, darkened, inverted or tinted (sheet relative_review/korean_same_number.jpg);
  3,362 files = 158 photos, and the grouping split each photo into 2-11 groups (strong filters fell
  outside the copy rules), so variants of one photo could sit in val and train (val 127_14 = train
  127_1). New rule `lodo.source_name_photo: {korean-aerospace: '^(\d+)_\d+$'}` (dedup.py): every variant
  of a number is one sub-scene, so the caps apply per photo (train <= 5, test/val <= 2). Preview: fold 1
  test 864 -> 262 (158 photos instead of recoloured duplicates), fold 0 test 4207 -> 4061 (RFDETR
  re-exports of korean photos).
  fuselage (Aircraft_Fuselage_DET2023) shows the same aircraft surfaces as closeup and D6 (202/389
  fuselage images have a closeup image at DINOv2 >= 0.80; same paint chips, rivets and holes in other
  shots): `hold_out_together: [D6, closeup, fuselage]`, so fuselage is not trained on in folds 3 and 5.
  The rebuild now starts with step 1 (folds without drops under the current grouping), and the
  comparison step checks every korean number is one sub-scene and that no held-out dataset is trained
  on, and reports test-set changes by file name against build/superseded_before_korean.
- korean copies kept with RFDETR (2026-10-30, user: "keep the korean copies as well ... wherever RFDETR
  is in train keep the korean as well"). With the photo-number rule, 122 of 131 korean photos also have
  RFDETR copies, and "one copy per photo across training datasets" kept only RFDETR's (better labelled),
  so korean nearly vanished from training (10-34 images). New `lodo.keep_copies_together:
  [[korean-aerospace, RFDETR_DATASET]]`: copies of one photo in these datasets are all kept in training,
  the caps count per dataset (<= 5 korean + <= 5 RFDETR per photo in train, <= 2 + 2 in val; thinning at
  DINOv2 >= 0.95 still applies). Leakage unchanged: a photo whose group touches test/val is not trained
  on in any dataset. Preview: korean train/val 239-365/52-100 in folds 2-5; fold 0 keeps 10 (its photos
  are RFDETR test scenes). verify_folds --per-dataset-caps and the comparison step allow these pairs.
  Rebuild resumed from the existing drop list, so the added korean images are photo-matched.
- Final lodo_v3 build v2 (2026-10-30 ~10:40, REBUILD DONE; korean photo numbers, fuselage held out with
  D6/closeup, korean copies kept with RFDETR). train/val/test: 0 RFDETR 1870/201/4061; 1 korean
  6401/698/262; 2 AI-4 7407/881/449; 3 D6 6158/684/615; 4 sydney 3564/389/1523; 5 closeup 6060/681/1205.
  Checks: DINOv2 audit 0 at >= 0.80; verify; copy = build; annotations preserved; test photo audit 0%;
  val leakage check 0 same-photo in 1,200 sampled val images (46 newly matched pairs). Known cost (not
  leakage): "match-anything" images (grainy/filtered variants matching >= 5 different photos at >= 25
  inliers) drop real training photos; option B (stricter rule for them, ~960 images back, most in folds 0
  and 4) was offered and declined by the user.
- Scene/augmentation analysis (2026-10-30): 61-82% of training photos have one copy; heavily augmented
  photos (4+ copies) are almost only RFDETR/korean. Under-represented classes by photos: fold 0 dent 14,
  missing-fastener 43, crack 86; fold 4 crack 162, missing-fastener 160; folds 3/5 paint-off ~290.
  korean variant numbers M (uploader's own augmentations; Roboflow export says none): 1 cool tint,
  2 greyscale, 3/4 rotate ~-20/+30 with black corners, 5 edge-enhance+brighter, 6 sharpen, 7/22 brighter,
  8 3x3 tile shuffle, 9/13/23 blur, 10 darker, 11 overexposed, 12 magenta shift, 14 negative, 15 small hue
  shift, 16 mirror, 17 shift, 18 upside down, 19 CLAHE contrast, 20 small rotation, 21 ~unchanged,
  24 emboss, 25 zoom on the defect. Sample sheets with boxes: build/relative_review/aug_sample_*.jpg
  (tile shuffle left out: cuts defects). Awaiting the user's choice of augmentations and amount.
- Dataset/class balancing in training (2026-10-30, user decision; no new files, dataset on the Hub
  unchanged). `data.balance` in base_config.yaml: per training image a repeat factor
  max(1, sqrt(t_ds/f_ds), sqrt(t_c/f_c) per class in the image), capped at 3 (t = equal share of the
  datasets / classes); a WeightedRandomSampler draws len(train) images per epoch with these weights
  (same epoch length). Every draw of an image with factor > 1 gets fresh light/colour changes
  (`PhotometricJitter`, RandAugment-style): every draw applies 3-6 changes chosen at random from 16:
  brightness 0.3-2.0, contrast 0.3-1.6, saturation 0-1.8, gamma 0.4-2.5, hue +-0.5 of the colour wheel,
  warm/cool tint +-0.4, blur radius / sharpen up to 3, motion blur up to 15 px, sensor noise up to std 12,
  JPEG quality down to 10, posterize down to 3 bits, shadow gradient up to 70%, haze up to 50%,
  histogram equalisation blended in up to 30%, and on/off greyscale and colour-channel swap. Light and
  colour changes run first, image-quality changes last (noise is never amplified afterwards). No
  geometric changes (user: rotation would spoil the boxes); the left-right flip stays. Strength (user,
  after the sample sheets: too mild; "not a bell curve around no change"; "more aggressive, more
  augmentations"; then "draws 3 and 6 destroy the image, the noise should not be this much"): direction
  by coin flip, strength u ~ Beta(3, 1) of the way to the limit (strong changes most likely).
  Safeguards, else redraw (up to 5 times, then the original): inside every box the fine detail (grey
  minus blurred grey) must correlate >= 0.4 with the original's; the whole image's detail >= 0.5; and
  each defect must still stand out from the background (box detail spread / image detail spread >= 60%
  of the original ratio). History: a plain contrast check let a shadow erase an AGDD scratch; noise 30 +
  full equalisation turned a flat fuselage panel into blotches; then contrast/saturation/equalisation
  amplified its faint texture until it drowned the scratches (pattern checks passed) - fixed by the
  stand-out check and the lower limits above. 300 training images: 1.8-2.0 draws per image, original
  kept for 3-5%, ~110-125 ms per image in a data worker incl. loading (8 workers: faster than the GPU).
  Sheets: build/final_review/aug_each_change.jpg (each change alone at its limits) and
  build/final_review/light_colour_samples.jpg (random draws as in training).
