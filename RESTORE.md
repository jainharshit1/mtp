# Restoring the project on a fresh machine

Code lives in GitHub (`jainharshit1/mtp`). Everything heavy lives in a **private** Hugging Face
dataset repo (created by `scripts/hf_backup.py`), referred to below as `HF_REPO` (e.g. `USER/mtp-backup`).

## What is where

| Item | Location | Needed to resume training? |
|---|---|---|
| Code, configs, docs, scripts | GitHub | yes |
| `vendor/Open-GroundingDino` + local patch | upstream GitHub + `vendor_patches/` | yes |
| 8 raw image datasets (~2 GB) | HF `raw/*.tar` | yes (jsonl files point at the images) |
| `data/` (ODVG folds, metadata) | HF `data.tar` | yes (or regenerate with scripts 01/02) |
| `weights/groundingdino_swint_ogc.pth` | HF `weights/` | yes |
| `outputs/` (checkpoints, logs) | HF `outputs/` | yes, for resume |
| `bert-base-uncased` | auto-downloaded from the public HF hub | yes (needs internet once) |
| `.venv`, `wheels/` | not backed up | rebuild from `requirements_portable.txt` |

Resume works because each checkpoint (`outputs/fold_N/latest.pt`, `best.pt`) stores the LoRA weights,
optimizer, scheduler, grad scaler, EMA and RNG state. Without `outputs/`, training restarts from epoch 0.

## Steps

**Paths matter.** The ODVG `.jsonl` files and `configs/datasets.yaml` contain absolute paths
`/DATA/air_force_object_detection/...`. Restore into exactly that directory. If you must use a different
one, `sed -i 's#/DATA/air_force_object_detection#/new/path#g'` on `configs/datasets.yaml`,
`scripts/*.sh`, `data/odvg/*/*.jsonl`, `data/odvg/*/test_annotations.json`, and `data/processed/*.json`.

```bash
# 1. code
sudo mkdir -p /DATA && sudo chown $USER /DATA
git clone https://github.com/jainharshit1/mtp.git /DATA/air_force_object_detection
cd /DATA/air_force_object_detection

# 2. vendored Open-GroundingDino at the commit we used, plus our local patch
git clone https://github.com/longzw1997/Open-GroundingDino.git vendor/Open-GroundingDino
cat vendor_patches/README.txt          # shows the commit hash
cd vendor/Open-GroundingDino && git checkout <COMMIT_FROM_README> \
  && git apply ../../vendor_patches/open_groundingdino_local_changes.patch && cd ../..

# 3. python env (Python 3.10, CUDA 12.4 driver)
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements_portable.txt
#   optional: build deformable-attention CUDA ops (the pure-Python fallback is what ran so far)
#   cd vendor/Open-GroundingDino/models/GroundingDINO/ops && python setup.py build install

# 4. pull the backup from Hugging Face (private repo -> log in first)
hf auth login
hf download HF_REPO --repo-type dataset --local-dir /DATA/hf_download

# 5. put the files into the project
cd /DATA/air_force_object_detection
for t in /DATA/hf_download/raw/*.tar /DATA/hf_download/data.tar; do tar -xf "$t" -C .; done
mkdir -p weights && cp /DATA/hf_download/weights/groundingdino_swint_ogc.pth weights/
cp -r /DATA/hf_download/outputs ./outputs

# 6. sanity checks
python -c "import torch; print(torch.cuda.is_available())"
ls data/odvg/fold_0 outputs/fold_0
head -c 300 data/odvg/fold_0/train.jsonl     # the image paths in it must exist

# 7. resume (auto-resumes from outputs/fold_N/latest.pt)
./scripts/train_resilient.sh
./scripts/check_status.sh
```

The tar files unpack to the original relative layout (`AGDD/`, `DATASET/RFDETR_DATASET/`,
`Aircraft-Defect-Detection-*/`, `data/`), so no further moving is needed.

Optional: re-install the watchdog cron (see `docs/training_operations.md`):
```
@reboot  /DATA/air_force_object_detection/scripts/train_resilient.sh
*/5 * * * * /DATA/air_force_object_detection/scripts/train_resilient.sh >> /DATA/air_force_object_detection/outputs/training_launcher.log 2>&1
```

## Keeping the backup fresh

```bash
python scripts/hf_backup.py --repo HF_REPO --only checkpoints   # small (~100 MB), run after each fold
python scripts/hf_backup.py --repo HF_REPO                      # full backup (first time)
```

## Not backed up on purpose

`pre_processing/gpt.py` and `install_datasets.ipynb` (hardcoded Roboflow API key, kept out of git),
`aircraft-dataset/`, `DATASET/visualized_rf_detr/`, `ml_challnege_26.zip`, `wheels/`, `.venv/`.
