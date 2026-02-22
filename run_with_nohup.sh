#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PID_FILE=".protea.pid"
LOG_FILE="protea.log"

# Check if already running
if [ -f "$PID_FILE" ]; then
    old_pid=$(cat "$PID_FILE")
    if kill -0 "$old_pid" 2>/dev/null; then
        echo "Protea is already running (pid=$old_pid)"
        echo "  Stop it first:  kill $old_pid"
        exit 1
    fi
    rm -f "$PID_FILE"
fi

# Watchdog loop with exponential backoff (5s → 60s).
# Stops restarting on clean exit (exit 0).
_run_watchdog() {
    local delay=5
    local max_delay=60
    while true; do
        echo "$(date '+%Y-%m-%d %H:%M:%S') [watchdog] Starting Protea..." >> "$LOG_FILE"
        .venv/bin/python run.py >> "$LOG_FILE" 2>&1
        rc=$?
        if [ $rc -eq 0 ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') [watchdog] Protea exited cleanly." >> "$LOG_FILE"
            break
        fi
        echo "$(date '+%Y-%m-%d %H:%M:%S') [watchdog] Protea exited with code $rc, restarting in ${delay}s..." >> "$LOG_FILE"
        sleep "$delay"
        delay=$(( delay * 2 ))
        if [ $delay -gt $max_delay ]; then
            delay=$max_delay
        fi
    done
    rm -f "$PID_FILE"
}

nohup bash -c "$(declare -f _run_watchdog); PID_FILE='$PID_FILE' LOG_FILE='$LOG_FILE' _run_watchdog" >> "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "Protea started in background with watchdog (pid=$!)"
echo "  Logs:  tail -f $LOG_FILE"
echo "  Stop:  kill $!"
