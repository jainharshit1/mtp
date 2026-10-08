#!/bin/bash
# Pause / resume LODO training safely.
#
#   ./scripts/pause_training.sh pause            # wait for the next epoch boundary, then stop (no work lost)
#   ./scripts/pause_training.sh pause --now      # stop immediately (loses the partial epoch, up to ~1h)
#   ./scripts/pause_training.sh pause --dry-run  # show what would happen, change nothing
#   ./scripts/pause_training.sh resume           # restore the cron watchdog and relaunch from latest.pt
#   ./scripts/pause_training.sh status
#
# Why not just `kill`:
#   - SIGTERM does not stop the run: the trainer saves a mid-epoch checkpoint, evaluates the
#     whole test set, and run_training.py then starts the next fold.
#   - `train_resilient.sh stop` sends SIGTERM, force-kills after 60 s, and the cron watchdog
#     relaunches training within 5 minutes.
# So we (1) remove the cron watchdog, (2) wait until a checkpoint has just been written and the
# next epoch has begun, (3) SIGKILL. Resume then continues exactly from that checkpoint.
# Run pause in the background if you do not want to wait: nohup ./scripts/pause_training.sh pause &

set -uo pipefail

PROJECT_DIR="/DATA/air_force_object_detection"
PIDFILE="$PROJECT_DIR/outputs/.training.pid"
LOGFILE="$PROJECT_DIR/outputs/training.log"
CRON_BACKUP="$PROJECT_DIR/outputs/crontab.backup"
CRON_MATCH="train_resilient.sh"
MAX_WAIT="${MAX_WAIT:-5400}"   # seconds to wait for an epoch boundary (an epoch is ~1h)

CMD="${1:-status}"
shift || true
NOW=0; DRY=0
for a in "$@"; do
    case "$a" in
        --now) NOW=1 ;;
        --dry-run) DRY=1 ;;
        *) echo "Unknown option: $a"; exit 1 ;;
    esac
done

say() { echo "$(date '+%H:%M:%S') | $*"; }
run() { if [ "$DRY" = 1 ]; then say "[dry-run] $*"; else eval "$@"; fi; }

train_pid() {
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        cat "$PIDFILE"
    fi
}

cron_has_watchdog() { crontab -l 2>/dev/null | grep -q "$CRON_MATCH"; }

disable_cron() {
    if cron_has_watchdog; then
        say "$([ "$DRY" = 1 ] && echo "[dry-run] ")Saving crontab to $CRON_BACKUP and removing the watchdog entries"
        if [ "$DRY" = 0 ]; then
            crontab -l > "$CRON_BACKUP"
            crontab -l | grep -v "$CRON_MATCH" | crontab -
        fi
    else
        say "Cron watchdog already removed"
    fi
}

enable_cron() {
    if cron_has_watchdog; then
        say "Cron watchdog already installed"
    elif [ -f "$CRON_BACKUP" ]; then
        say "Restoring watchdog entries from $CRON_BACKUP"
        { crontab -l 2>/dev/null; grep "$CRON_MATCH" "$CRON_BACKUP"; } | crontab -
    else
        say "No crontab backup found; add the entries from docs/training_operations.md by hand"
    fi
}

wait_for_epoch_boundary() {
    # A fresh "Saved checkpoint ... latest.pt" line followed by a fresh batch-0 line of the next
    # epoch means every checkpoint for the finished epoch is on disk.
    local start_line; start_line=$(wc -l < "$LOGFILE")
    local waited=0 saved=0
    say "Waiting for the next epoch boundary (up to ${MAX_WAIT}s)..."
    while [ "$waited" -lt "$MAX_WAIT" ]; do
        [ -n "$(train_pid)" ] || { say "Training exited by itself"; return 1; }
        local new; new=$(tail -n +"$((start_line + 1))" "$LOGFILE")
        if [ "$saved" = 0 ] && grep -q "Saved checkpoint: .*latest.pt" <<<"$new"; then
            saved=1; say "Checkpoint written; waiting for the next epoch to start"
        fi
        if [ "$saved" = 1 ] && grep -qE "\[[0-9]+\]\[0/[0-9]+\]" <<<"$(grep -A100000 'latest.pt' <<<"$new")"; then
            return 0
        fi
        sleep 3; waited=$((waited + 3))
    done
    say "Timed out waiting for an epoch boundary"
    return 2
}

case "$CMD" in
    status)
        pid=$(train_pid)
        [ -n "$pid" ] && say "Training RUNNING (PID $pid)" || say "Training NOT running"
        cron_has_watchdog && say "Cron watchdog: INSTALLED (will relaunch training if it dies)" \
                          || say "Cron watchdog: REMOVED (training stays stopped)"
        "$PROJECT_DIR/scripts/check_status.sh" 2>/dev/null | sed -n 4,12p
        ;;

    pause)
        pid=$(train_pid)
        disable_cron            # first, so nothing relaunches the run
        if [ -z "$pid" ]; then
            say "Training is not running. Nothing to stop."
            exit 0
        fi
        if [ "$NOW" = 0 ]; then
            if [ "$DRY" = 1 ]; then
                say "[dry-run] would wait for the next epoch boundary, then kill -9 $pid"
                exit 0
            fi
            wait_for_epoch_boundary
            rc=$?
            if [ "$rc" = 1 ]; then exit 0; fi
            if [ "$rc" = 2 ]; then
                say "Not killing. Re-run with --now to stop immediately, or 'resume' to restore the watchdog."
                exit 2
            fi
        fi
        say "Stopping training (PID $pid) with SIGKILL"
        run "kill -9 $pid"
        sleep 5
        run "pkill -9 -f scripts/run_training.py 2>/dev/null; true"   # stray dataloader workers
        run "rm -f '$PIDFILE'"
        if [ "$DRY" = 0 ]; then
            say "GPU memory in use: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)"
            say "Paused. Last checkpoint: $(ls -t "$PROJECT_DIR"/outputs/fold_*/latest.pt 2>/dev/null | head -1)"
            say "Run './scripts/pause_training.sh resume' to continue."
        fi
        ;;

    resume)
        if [ -n "$(train_pid)" ]; then
            say "Training is already running (PID $(train_pid))"
        else
            say "Launching training (resumes from the latest checkpoint)"
            nohup "$PROJECT_DIR/scripts/train_resilient.sh" >> "$PROJECT_DIR/outputs/training_launcher.log" 2>&1 &
            sleep 3
        fi
        enable_cron
        ;;

    *)
        echo "Usage: $0 {pause [--now] [--dry-run] | resume | status}"
        exit 1
        ;;
esac
