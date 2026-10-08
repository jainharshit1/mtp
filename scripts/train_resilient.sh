#!/bin/bash
# Resilient training launcher — survives reboots via @reboot cron.
# Automatically resumes from the latest checkpoint.
#
# Setup (one-time):
#   crontab -e   →   @reboot /DATA/air_force_object_detection/scripts/train_resilient.sh
#
# Manual run:
#   ./scripts/train_resilient.sh          # start/resume training
#   ./scripts/train_resilient.sh stop     # stop training gracefully
#   ./scripts/train_resilient.sh status   # check if running

set -euo pipefail

PROJECT_DIR="/DATA/air_force_object_detection"
LOCKFILE="$PROJECT_DIR/outputs/.training.lock"
PIDFILE="$PROJECT_DIR/outputs/.training.pid"
LOGFILE="$PROJECT_DIR/outputs/training_launcher.log"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') | $*" >> "$LOGFILE"; }

case "${1:-start}" in
    stop)
        if [ -f "$PIDFILE" ]; then
            pid=$(cat "$PIDFILE")
            if kill -0 "$pid" 2>/dev/null; then
                echo "Stopping training (PID $pid)..."
                kill -TERM "$pid"
                # Wait up to 60s for graceful shutdown
                for i in $(seq 1 60); do
                    kill -0 "$pid" 2>/dev/null || break
                    sleep 1
                done
                if kill -0 "$pid" 2>/dev/null; then
                    echo "Force-killing PID $pid..."
                    kill -9 "$pid"
                fi
                echo "Stopped."
            else
                echo "Process $pid is not running."
            fi
            rm -f "$PIDFILE" "$LOCKFILE"
        else
            echo "No PID file found. Training is not running."
        fi
        exit 0
        ;;
    status)
        if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
            echo "Training is running (PID $(cat "$PIDFILE"))"
        else
            echo "Training is NOT running"
        fi
        exit 0
        ;;
    start)
        ;;
    *)
        echo "Usage: $0 [start|stop|status]"
        exit 1
        ;;
esac

# Lock to prevent duplicate launches
exec 9>"$LOCKFILE"
if ! flock -n 9; then
    log "Another instance is already running. Exiting."
    exit 0
fi

# Check if training is already complete
if [ -f "$PROJECT_DIR/outputs/status.json" ]; then
    state=$(python3 -c "import json; print(json.load(open('$PROJECT_DIR/outputs/status.json')).get('state',''))" 2>/dev/null || true)
    if [ "$state" = "complete" ]; then
        log "Training already complete. Exiting."
        exit 0
    fi
fi

# Wait for GPU to be available (after reboot, drivers may take a moment)
for i in $(seq 1 30); do
    if nvidia-smi > /dev/null 2>&1; then
        break
    fi
    log "Waiting for GPU driver... ($i/30)"
    sleep 10
done

if ! nvidia-smi > /dev/null 2>&1; then
    log "ERROR: GPU not available after 5 minutes. Aborting."
    exit 1
fi

cd "$PROJECT_DIR"
source "$PROJECT_DIR/.venv/bin/activate"

log "Starting resilient training launcher"
log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

# Launch training (resume is on by default)
python3 scripts/run_training.py &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$PIDFILE"
log "Launched training with PID $TRAIN_PID"

# Wait for it to finish
wait $TRAIN_PID
EXIT_CODE=$?

log "Training exited with code $EXIT_CODE"
rm -f "$PIDFILE"

# Release the lock
flock -u 9
