#!/bin/bash
# Training status monitor — run anytime to check progress
# Usage:
#   ./scripts/check_status.sh          # Quick status
#   ./scripts/check_status.sh -f       # Follow log (live tail)
#   ./scripts/check_status.sh -l 50    # Last 50 lines of log
#   ./scripts/check_status.sh -g       # GPU utilization
#   ./scripts/check_status.sh -a       # All info

cd /DATA/air_force_object_detection

STATUS_FILE="outputs/status.json"
LOG_FILE="outputs/training.log"

show_status() {
    echo "========================================"
    echo "  TRAINING STATUS"
    echo "========================================"
    if [ -f "$STATUS_FILE" ]; then
        python3 -c "
import json, sys
d = json.load(open('$STATUS_FILE'))
state = d.get('state', '?')
colors = {'training': '\033[33m', 'complete': '\033[32m', 'error': '\033[31m', 'fold_complete': '\033[36m'}
c = colors.get(state, '\033[0m')
print(f'  State:        {c}{state}\033[0m')
print(f'  Updated:      {d.get(\"updated_at\", \"?\")}')
if d.get('current_fold') is not None:
    print(f'  Current fold: {d[\"current_fold\"]} ({d.get(\"fold_progress\", \"?\")})')
if d.get('current_epoch') is not None:
    print(f'  Epoch:        {d[\"current_epoch\"]}/{d.get(\"max_epochs\", \"?\")}')
if d.get('last_train_loss') is not None:
    print(f'  Train loss:   {d[\"last_train_loss\"]:.4f}')
if d.get('last_val_loss') is not None:
    print(f'  Val loss:     {d[\"last_val_loss\"]:.4f}')
if d.get('last_mAP') is not None:
    print(f'  Last mAP:     {d[\"last_mAP\"]:.4f}')
if d.get('vram_gb') is not None:
    print(f'  VRAM:         {d[\"vram_gb\"]} GB')
if d.get('elapsed'):
    print(f'  Elapsed:      {d[\"elapsed\"]}')
if d.get('eta'):
    print(f'  ETA:          {d[\"eta\"]}')
if d.get('completed_folds'):
    print(f'  Done folds:   {d[\"completed_folds\"]}')
if d.get('error'):
    print(f'  \033[31mERROR: {d[\"error\"]}\033[0m')
if d.get('pid'):
    import os
    alive = os.path.exists(f'/proc/{d[\"pid\"]}')
    status = '\033[32malive\033[0m' if alive else '\033[31mdead\033[0m'
    print(f'  Process:      PID {d[\"pid\"]} ({status})')
"
    else
        echo "  No status file found. Training hasn't started yet."
    fi
    echo ""
}

show_gpu() {
    echo "========================================"
    echo "  GPU STATUS"
    echo "========================================"
    nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU:    %s\n  Temp:   %s°C\n  Util:   %s%%\n  Mem:    %s/%s MiB (%s%%)\n", $1, $2, $3, $5, $6, $4}'
    echo ""
}

show_log() {
    local lines=${1:-20}
    echo "========================================"
    echo "  LAST $lines LOG LINES"
    echo "========================================"
    if [ -f "$LOG_FILE" ]; then
        tail -n "$lines" "$LOG_FILE"
    else
        echo "  No log file found."
    fi
    echo ""
}

show_checkpoints() {
    echo "========================================"
    echo "  CHECKPOINTS"
    echo "========================================"
    for fold_dir in outputs/fold_*/; do
        if [ -d "$fold_dir" ]; then
            fold=$(basename "$fold_dir")
            files=$(ls -la "$fold_dir"*.pt 2>/dev/null | awk '{printf "%s (%s) ", $NF, $5}')
            if [ -n "$files" ]; then
                echo "  $fold: $files"
            fi
            if [ -f "${fold_dir}eval_results.json" ]; then
                mAP=$(python3 -c "import json; print(f'{json.load(open(\"${fold_dir}eval_results.json\")).get(\"mAP\", 0):.4f}')")
                echo "    → mAP: $mAP"
            fi
        fi
    done
    echo ""
}

# Parse args
case "${1:-}" in
    -f|--follow)
        echo "Following training log (Ctrl+C to stop)..."
        tail -f "$LOG_FILE"
        ;;
    -l|--log)
        show_log "${2:-50}"
        ;;
    -g|--gpu)
        show_gpu
        ;;
    -a|--all)
        show_status
        show_gpu
        show_checkpoints
        show_log 30
        ;;
    -h|--help)
        echo "Usage: $0 [option]"
        echo "  (none)    Quick status overview"
        echo "  -f        Follow log in real-time"
        echo "  -l [N]    Show last N log lines (default 50)"
        echo "  -g        GPU utilization"
        echo "  -a        All info (status + GPU + checkpoints + log)"
        echo "  -h        This help"
        ;;
    *)
        show_status
        show_log 10
        ;;
esac
