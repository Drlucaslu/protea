#!/usr/bin/env bash
# Protea Restore — restores memory, config, and data from a backup archive.
#
# Usage:
#   ./protea-restore.sh backups/protea-20260228-143000.tar.gz
#
# The script will:
#   1. Verify the archive
#   2. Back up current data (pre-restore safety copy)
#   3. Extract and restore files
#   4. Show what was restored
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_green()  { printf '\033[32m%s\033[0m' "$*"; }
_yellow() { printf '\033[33m%s\033[0m' "$*"; }
_red()    { printf '\033[31m%s\033[0m' "$*"; }
_bold()   { printf '\033[1m%s\033[0m' "$*"; }
_dim()    { printf '\033[2m%s\033[0m' "$*"; }

ok()   { echo "  $(_green '[ok]') $*"; }
info() { echo "  $(_yellow '[..]') $*"; }
warn() { echo "  $(_yellow '[!!]') $*"; }
fail() { echo "  $(_red '[!!]') $*"; exit 1; }

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

if [ $# -lt 1 ]; then
    echo "Usage: $0 <backup_archive.tar.gz>"
    echo
    echo "Examples:"
    echo "  $0 backups/protea-20260228-143000.tar.gz"
    echo
    if [ -d backups ]; then
        echo "Available backups:"
        ls -1t backups/*.tar.gz 2>/dev/null | head -5 | while read f; do
            size=$(du -h "$f" | cut -f1)
            echo "  $f  ($size)"
        done
    fi
    exit 1
fi

ARCHIVE="$1"

if [ ! -f "$ARCHIVE" ]; then
    fail "Archive not found: $ARCHIVE"
fi

echo
echo "$(_bold '=== Protea Restore ===')"
echo
echo "  Archive: $ARCHIVE"
echo

# ---------------------------------------------------------------------------
# Step 1: Verify archive contents
# ---------------------------------------------------------------------------

info "Verifying archive..."
TMPDIR=$(mktemp -d)
trap "rm -rf '$TMPDIR'" EXIT

tar -xzf "$ARCHIVE" -C "$TMPDIR"

# Find the extracted directory (should be protea-YYYYMMDD-HHMMSS).
EXTRACTED=$(ls -1 "$TMPDIR" | head -1)
if [ -z "$EXTRACTED" ]; then
    fail "Archive appears empty"
fi
SRC="${TMPDIR}/${EXTRACTED}"

# List what's in the backup.
echo "  Contents:"
HAS_DB=false; HAS_ENV=false; HAS_CONFIG=false; HAS_RING2=false; HAS_GENES=false; HAS_MEMDB=false
[ -f "$SRC/protea.db" ]              && HAS_DB=true     && echo "    - protea.db (main database)"
[ -f "$SRC/memory.db" ]              && HAS_MEMDB=true   && echo "    - memory.db"
[ -f "$SRC/.env" ]                   && HAS_ENV=true     && echo "    - .env (configuration)"
[ -f "$SRC/config/config.toml" ]     && HAS_CONFIG=true  && echo "    - config/config.toml"
[ -f "$SRC/genes/core_genes.json" ]  && HAS_GENES=true   && echo "    - genes/core_genes.json"
[ -d "$SRC/ring2" ]                  && HAS_RING2=true   && echo "    - ring2/ (evolved code)"
echo

# ---------------------------------------------------------------------------
# Step 2: Confirm
# ---------------------------------------------------------------------------

# Check if Protea is running.
if [ -f data/sentinel.pid ]; then
    PID=$(cat data/sentinel.pid 2>/dev/null || true)
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        warn "Protea appears to be running (PID $PID)."
        warn "Stop it first: ./stop_run.sh"
        echo
        read -p "  Continue anyway? [y/N] " answer < /dev/tty 2>/dev/null || answer="N"
        [ "${answer:-N}" = "y" ] || [ "${answer:-N}" = "Y" ] || { echo "  Aborted."; exit 0; }
    fi
fi

read -p "  Restore from this backup? [Y/n] " answer < /dev/tty 2>/dev/null || answer="Y"
if [ "${answer:-Y}" = "n" ] || [ "${answer:-Y}" = "N" ]; then
    echo "  Aborted."
    exit 0
fi
echo

# ---------------------------------------------------------------------------
# Step 3: Safety backup of current data
# ---------------------------------------------------------------------------

SAFETY_TS=$(date +%Y%m%d-%H%M%S)
if [ -f data/protea.db ]; then
    cp data/protea.db "data/protea.db.pre-restore-${SAFETY_TS}"
    ok "Current protea.db saved as protea.db.pre-restore-${SAFETY_TS}"
fi

# ---------------------------------------------------------------------------
# Step 4: Restore
# ---------------------------------------------------------------------------

mkdir -p data

if $HAS_DB; then
    cp "$SRC/protea.db" data/protea.db
    ok "Restored data/protea.db"
fi

if $HAS_MEMDB; then
    cp "$SRC/memory.db" data/memory.db
    ok "Restored data/memory.db"
fi

if $HAS_ENV; then
    if [ -f .env ]; then
        cp .env ".env.pre-restore-${SAFETY_TS}"
        ok "Current .env saved as .env.pre-restore-${SAFETY_TS}"
    fi
    cp "$SRC/.env" .env
    ok "Restored .env"
fi

if $HAS_CONFIG; then
    mkdir -p config
    if [ -f config/config.toml ]; then
        cp config/config.toml "config/config.toml.pre-restore-${SAFETY_TS}"
    fi
    cp "$SRC/config/config.toml" config/config.toml
    ok "Restored config/config.toml"
fi

if $HAS_GENES; then
    mkdir -p genes
    cp "$SRC/genes/core_genes.json" genes/core_genes.json
    ok "Restored genes/core_genes.json"
fi

if $HAS_RING2; then
    if [ -d "$SRC/ring2/.git" ]; then
        # Full ring2 with git history.
        rm -rf ring2
        cp -r "$SRC/ring2" ring2
        ok "Restored ring2/ (with git history)"
    elif [ -f "$SRC/ring2/main.py" ]; then
        mkdir -p ring2
        cp "$SRC/ring2/main.py" ring2/main.py
        ok "Restored ring2/main.py"
    fi
fi

# ---------------------------------------------------------------------------
# Step 5: Summary
# ---------------------------------------------------------------------------

echo
echo "$(_bold 'Restore summary:')"

if $HAS_DB && command -v sqlite3 &>/dev/null; then
    MEM_COUNT=$(sqlite3 data/protea.db "SELECT COUNT(*) FROM memory;" 2>/dev/null || echo "?")
    FIT_COUNT=$(sqlite3 data/protea.db "SELECT COUNT(*) FROM fitness_log;" 2>/dev/null || echo "?")
    MAX_GEN=$(sqlite3 data/protea.db "SELECT MAX(generation) FROM fitness_log;" 2>/dev/null || echo "?")
    SKILL_COUNT=$(sqlite3 data/protea.db "SELECT COUNT(*) FROM skills;" 2>/dev/null || echo "n/a")
    echo "  Memory entries: $MEM_COUNT"
    echo "  Fitness records: $FIT_COUNT (up to gen $MAX_GEN)"
    echo "  Skills: $SKILL_COUNT"
fi

echo
echo "  $(_green "$(_bold 'Restore complete.')")"
echo "  Start Protea: .venv/bin/python run.py"
echo
