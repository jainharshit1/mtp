# Training Operations Guide

## Resilient Training System

Training is managed by `scripts/train_resilient.sh`, a wrapper that:

- Prevents duplicate instances via a lockfile
- Waits for GPU drivers after reboot
- Skips launch if training is already marked `complete`
- Auto-resumes from the latest checkpoint

### Cron Watchdog

Two cron entries keep training alive:

```
@reboot  /DATA/air_force_object_detection/scripts/train_resilient.sh
*/5 * * * * /DATA/air_force_object_detection/scripts/train_resilient.sh >> /DATA/air_force_object_detection/outputs/training_launcher.log 2>&1
```

- `@reboot` restarts training after a system reboot.
- `*/5 * * * *` checks every 5 minutes; if the process died (crash, OOM, manual kill), it relaunches. The lockfile prevents duplicates if training is already running.

### Crash Resilience Built Into the Trainer

| Scenario | Protection |
|---|---|
| System reboot / SIGTERM | Signal handler saves emergency checkpoint before exit |
| CUDA OOM on a batch | Catches error, clears cache, skips batch (aborts after 10 per epoch) |
| NaN/Inf loss | Detects non-finite loss, skips batch |
| Corrupt/truncated image | Dataset substitutes a random other sample |
| Checkpoint file corruption (killed mid-write) | Writes to `.tmp` then atomic rename; on load, falls back to `best.pt` |
| RNG state dtype mismatch on resume | Explicit CPU uint8 cast with fallback |

---

## Stopping Training

### Early Stop on Plateau, Still Compute mAP (recommended)

Use this when val_loss has plateaued before epoch 30 (the built-in early stopping
needs `patience=7` epochs without improvement) and you still want the fold's test mAP.

How it works: the trainer's SIGTERM handler saves `latest.pt`, finishes the current
epoch's validation, and exits the epoch loop. `LODORunner.run_fold` then loads
`best.pt` (lowest val_loss, EMA weights) and runs `evaluate_fold` on the fold's
`test_annotations.json`, writing `outputs/fold_N/eval_results.json` (mAP, mAP_50,
mAP_75, per-class AP). The `mAP=` line is also logged by `run_training.py`.

Do NOT use `./scripts/train_resilient.sh stop` for this. It sends SIGTERM and then
`kill -9` after 60 s, which can kill the process during evaluation, before mAP exists.
The cron watchdog would also relaunch training.

```bash
cd /DATA/air_force_object_detection

# 1. Disable the watchdog so nothing relaunches (restore with `crontab -e`, see below)
crontab -l > outputs/crontab.backup
crontab -l | grep -v train_resilient.sh | crontab -

# 2. Ask the trainer to stop gracefully (SIGTERM to the python process only)
kill -TERM "$(cat outputs/.training.pid)"

# 3. Wait for evaluation to finish (validation + test-set inference, can take several minutes)
tail -f outputs/training.log | grep --line-buffered -E "Early|interrupted|Loading best|mAP="
#    ...or: until [ -f outputs/fold_0/eval_results.json ]; do sleep 10; done

# 4. IMPORTANT: as soon as "mAP=" appears, stop the runner (see caveat below)
./scripts/train_resilient.sh stop

# 5. Read the result
python3 -c "import json; d=json.load(open('outputs/fold_0/eval_results.json')); print({k:v for k,v in d.items() if k.startswith('mAP')})"
```

Caveat: `run_training.py` iterates over all 8 folds, so after fold 0 is evaluated it
immediately starts fold 1. Step 4 must follow quickly. If `outputs/fold_1/` was
created, delete it before resuming (`rm -rf outputs/fold_1`).

Notes:
- The checkpoint evaluated is `best.pt` (lowest val_loss), not necessarily the last epoch.
- A signal mid-epoch trains on a partial epoch. Validation still runs on it, and
  `best.pt` is only replaced if that partial epoch has a lower val_loss.
- Replace `fold_0` with the fold that is running (`current_fold` in `outputs/status.json`).
- To re-enable the watchdog later: `crontab outputs/crontab.backup`.

### Permanent Stop (abandon training)

```bash
# Remove cron jobs, then stop the process
crontab -r
./scripts/train_resilient.sh stop
```

If you have other cron jobs you want to keep, use `crontab -e` and delete only the two training lines instead of `crontab -r`.

### Temporary Pause (free the GPU, resume later)

```bash
# 1. Mark as complete so the watchdog won't relaunch
python3 -c "
import json
d = json.load(open('outputs/status.json'))
d['state'] = 'complete'
json.dump(d, open('outputs/status.json', 'w'))
"

# 2. Stop the running process
./scripts/train_resilient.sh stop
```

To unpause:

```bash
# Set state back to training — the watchdog will pick it up within 5 minutes
python3 -c "
import json
d = json.load(open('outputs/status.json'))
d['state'] = 'training'
json.dump(d, open('outputs/status.json', 'w'))
"
```

Or launch immediately: `./scripts/train_resilient.sh`

---

## Monitoring

```bash
./scripts/check_status.sh          # Quick status
./scripts/check_status.sh -f       # Follow log live
./scripts/check_status.sh -a       # Full info (status + GPU + checkpoints + log)
./scripts/train_resilient.sh status # Check if process is running
```

Launcher activity log: `outputs/training_launcher.log`
