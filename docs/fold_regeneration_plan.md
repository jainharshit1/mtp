# Plan: build the LODO folds once, freeze them, train directly

Goal: one versioned, self-contained set of folds (`lodo_v3`) built once with the rules below. Training
reads these files as they are: no regrouping, no re-splitting, no checks at training time. If a rule
changes, build a new version (`lodo_v4`) instead of editing v3.

Related: `improvement_steps.md` (A1-A6 data, B1-B11 training).

## The rules (defaults of `scripts/01_ingest_and_unify.py` + `configs/datasets.yaml` `lodo:`)

1. **Exact duplicates.** Byte-identical files (SHA256) are one image; a copy never counts twice.
   The local raw data already has the 864 byte-identical files removed (sydney 852, closeup 12).
2. **Scene groups** decide the split side. Two images are in one group if any of these links them,
   chained transitively: same Roboflow source name within a dataset; dihedral pHash distance <= 8
   (flips and 90-degree rotations) **and** DINOv2 cosine >= 0.7 (`--phash-min-cosine`); DINOv2-small
   cosine >= 0.90 (mild crops; consecutive video frames).
   Why 0.7: pHash <= 8 matches many unrelated low-texture images (plain panels). A first full build with
   a 0.5 guard chained 10,420 images (a third of the data) into one group, through a 5,189-image chain of
   pHash links alone; 23,802 pHash pairs scored below 0.5. Inside that group, pairs sharing a Roboflow
   name (certain copies) scored >= 0.8 except 26; sampled cross-dataset pairs were all copies at 0.85-0.90,
   nearly all at 0.80-0.85, mixed at 0.70-0.80 (RFDETR re-exports sydney/korean photos heavily recoloured,
   e.g. magenta or black-and-white) and mostly different photos below 0.70. With 0.7 the largest group is
   1,806 images. 0.7 keeps more recoloured copies together
   (leakage-safe); 0.8 would split the chain further but miss more copies.
3. **Sub-scenes** are what the copy caps count: images linked by SHA, Roboflow name or pHash only.
4. **6 folds.** Test datasets, one fold each: RFDETR_DATASET, korean-aerospace, Aircraft-AI-Dataset-4,
   Aircraft-Defect-Detection-6, sydney, closeup. AGDD (193 labelled images, 29 crack boxes) and fuselage
   (389 images, 630 boxes) are too small to test on and are always trained on.
5. **Each fold holds out two datasets.** Test = the fold's dataset; val = a second, whole dataset, handled
   exactly like test, so val measures the same thing as test (a source never trained on):

   | Fold | Test | Val |
   |---|---|---|
   | 0 | RFDETR_DATASET | Aircraft-AI-Dataset-4 |
   | 1 | korean-aerospace | Aircraft-Defect-Detection-6 |
   | 2 | Aircraft-AI-Dataset-4 | korean-aerospace |
   | 3 | Aircraft-Defect-Detection-6 | Aircraft-AI-Dataset-4 |
   | 4 | sydney | Aircraft-Defect-Detection-6 |
   | 5 | closeup | korean-aerospace |

   No two folds train on the same datasets; each val dataset is out of training in 3 of 6 folds.
   Any group containing an image of the test (or val) dataset goes to that side, using only that
   dataset's own images with boxes; the group's copies in other datasets are dropped from the fold.
6. **Which images of a group are kept:** best first (boxes, labelled classes, box count, resolution);
   drop near-repeats, i.e. DINOv2 cosine >= 0.95 to an image already kept from the group (about one
   frame in 3-4 of a video walk-around); at most 5 images per sub-scene in train, 2 in test and val.
6b. **Per-fold near-copy filter.** After selection, a training image is dropped if its DINOv2 cosine
   to any test or val image of that fold is >= 0.80 (`--proximity-filter`). Why: RFDETR re-exports
   photos of sydney, closeup and korean-aerospace rotated a few degrees (black corners) and recoloured,
   under new names (`aircraft_xxxxx`). These copies score DINOv2 0.80-0.90 and pHash 10-16, so no
   grouping rule links them; the audit of the build without this filter found 393 test images with a
   training image at >= 0.85 (168 in fold 4, sydney held out), and most inspected pairs were the same
   photo. Copies were seen down to about 0.82; at 0.80 the sampled pairs were different photos.
   Lowering the *linking* threshold instead does not work: 0.85 for cross-dataset links cost 8% of
   training images and still left copies at 0.80-0.85; 0.80 chained 13,646 images into one group and
   cost 34%. The filter cannot chain (it only removes training images near this fold's test/val) and
   costs about 2.4% of training images (most in fold 4, then fold 0). Sheets:
   (sample sheets deleted 2026-10-28).
7. **Guard at build time only.** `_check_fold` refuses to write folds if a group spans two splits, a
   test-dataset image is in train/val, or a val-dataset image is in train.
8. **Images keep their original size** (mostly 640x640). Batching by aspect ratio is a training-side task (B7/B8).

Not changed (recorded in `improvement_steps.md`): Roboflow name collisions in sydney still merge a
few different photos (A4; safe for leakage), and heavily cropped or rotated copies under different
names can still be missed (A5).

## Local build (2026-10-27), `D:\MTP\mtp-data\lodo_v3\`

`build/run_build.sh` runs every step after the embeddings and stops at the first failure:

| Step | Script | Output (in `lodo_v3/build/` unless noted) |
|---|---|---|
| 0 | `scripts/00_embed_images.py` (DINOv2 for all 31,715 images, cached by path, resumable) | `embed_cache.npz`, `00_embed.log` |
| 1 | `scripts/01_ingest_and_unify.py --embed-cache` (groups, sub-scenes, thinning, caps, 6 folds) | `processed/`, `odvg_abs/` (absolute local paths), `01_build.log` |
| 2 | `scripts/audit_leakage.py --embed-cache` (searches for near-duplicates across splits from scratch, without group ids) | `leakage_audit/` (CSVs, contact sheets, `summary.json`) |
| 3 | `scripts/build_portable_folds.py` (copies each used image once, original size; relative paths; annotations grouped and sorted per image) | `lodo_v3/images/`, `lodo_v3/fold_0..5/`, `lodo_v3/MANIFEST.json` |
| 4 | `scripts/verify_folds.py` (image SHA256, datasets per split, groups, caps, every box) | `lodo_v3/checks/verify_report.txt`, `overlay_<dataset>.jpg`, `fold_<k>_<split>.jpg` |
| 5 | `scripts/check_annotations.py` (each image's boxes in every fold file vs its raw annotation file, parsed independently of the pipeline) | `lodo_v3/checks/annotation_check.txt` |

Annotation pre-check (2026-10-27): the pipeline's ingest matches an independent parse of the raw files for
all 31,715 images and 85,071 boxes. Only raw boxes dropped on purpose: 215 AGDD `spot` (not in class_map).
Fixed on the way: YOLO and fuselage boxes crossing the left/top edge were shifted instead of cropped (no box
in this data does this, so the result is unchanged).

Config used: `build/datasets_local.yaml` (= `configs/datasets.yaml` with local `base_dir`s).

Notes from this machine:
- `transformers` also loaded TensorFlow next to torch; with both in one process the DINOv2 step crashed
  natively (segmentation fault / silent exit). Running with `USE_TF=0` fixed it. Set `USE_TF=0` on the
  server too if TensorFlow is installed there.
- Embedding on this CPU takes about 3.5 min per 1,000 images (about 1 h 50 min for all).

## After the build

1. **Visual check** of `lodo_v3/checks/fold_<k>_<split>.jpg` and `overlay_<dataset>.jpg`, and a look at the
   leakage audit's contact sheets (`build/leakage_audit/fold_<k>_*_top_pairs.jpg`).
2. **Freeze.** Copy `lodo_v3/` to the server (e.g. `mtp/data/lodo_v3`), make it read-only
   (`chmod -R a-w`), note the SHA256 of `MANIFEST.json` below, and set `configs/base_config.yaml`
   -> `data.odvg_dir: data/lodo_v3`. Training and evaluation resolve the relative image paths against
   that folder; fold ids are discovered from the `fold_*` folders (6).
3. **Clear old training state.** Move `outputs/fold_*` aside (e.g. `outputs_v2_archive/`): the trainer
   resumes from `outputs/fold_k/latest.pt` if it exists, which would continue a model trained on the old folds.
4. **Fix the training blockers, then train all 6 folds** (`improvement_steps.md`): B3 (crash when
   resuming a finished fold), B4 (interrupted epoch skipped), B5 (leftover GPU process), B1 (early
   stopping on val, now a held-out dataset), B7/B8 (aspect-ratio batching, cudnn).

## Frozen version record
- Version: lodo_v3
- Built: 2026-10-27 (local)
- Commit: (hash)
- MANIFEST.json SHA256: (hash)
