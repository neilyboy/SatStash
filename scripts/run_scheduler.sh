#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$ROOT_DIR/venv"
PYTHON_BIN="$VENV_DIR/bin/python"
SCHEDULER="$ROOT_DIR/scheduler_daemon.py"
LOG_DIR="$HOME/.satstash/logs"
LOG_FILE="$LOG_DIR/scheduler.log"

mkdir -p "$LOG_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Virtualenv not found. Run ./install.sh first." >&2
    exit 1
fi

exec "$PYTHON_BIN" "$SCHEDULER" >>"$LOG_FILE" 2>&1
