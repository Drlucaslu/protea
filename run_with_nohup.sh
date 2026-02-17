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

nohup .venv/bin/python run.py >> "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "Protea started in background (pid=$!)"
echo "  Logs:  tail -f $LOG_FILE"
echo "  Stop:  kill $!"
