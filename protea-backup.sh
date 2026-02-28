#!/usr/bin/env bash
# Protea Backup — archives memory, config, and optionally Ring 2 history.
#
# Usage:
#   ./protea-backup.sh              # standard backup
#   ./protea-backup.sh --full       # include Ring 2 git history
#   ./protea-backup.sh -o /path/    # custom output directory
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
fail() { echo "  $(_red '[!!]') $*"; exit 1; }

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

FULL=false
OUT_DIR="backups"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full)  FULL=true; shift ;;
        -o|--output) OUT_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--full] [-o output_dir]"
            echo "  --full    Include Ring 2 git history"
            echo "  -o DIR    Output directory (default: backups/)"
            exit 0
            ;;
        *) fail "Unknown option: $1" ;;
    esac
done

# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BACKUP_NAME="protea-${TIMESTAMP}"
BACKUP_DIR="${OUT_DIR}/${BACKUP_NAME}"

echo
echo "$(_bold '=== Protea Backup ===')"
echo

mkdir -p "$BACKUP_DIR"

# 1. Main database
if [ -f data/protea.db ]; then
    # Use SQLite backup for consistency (avoid copying mid-write).
    if command -v sqlite3 &>/dev/null; then
        sqlite3 data/protea.db ".backup '${BACKUP_DIR}/protea.db'"
        ok "data/protea.db (SQLite safe copy)"
    else
        cp data/protea.db "${BACKUP_DIR}/protea.db"
        ok "data/protea.db (file copy)"
    fi
else
    info "data/protea.db not found — skipping"
fi

# 2. Secondary memory DB
if [ -f data/memory.db ]; then
    cp data/memory.db "${BACKUP_DIR}/memory.db"
    ok "data/memory.db"
fi

# 3. Environment config
if [ -f .env ]; then
    cp .env "${BACKUP_DIR}/.env"
    ok ".env"
else
    info ".env not found — skipping"
fi

# 4. Config file
if [ -f config/config.toml ]; then
    mkdir -p "${BACKUP_DIR}/config"
    cp config/config.toml "${BACKUP_DIR}/config/config.toml"
    ok "config/config.toml"
fi

# 5. Core genes
if [ -f genes/core_genes.json ]; then
    mkdir -p "${BACKUP_DIR}/genes"
    cp genes/core_genes.json "${BACKUP_DIR}/genes/core_genes.json"
    ok "genes/core_genes.json"
fi

# 6. Ring 2 (optional --full)
if $FULL && [ -d ring2 ]; then
    cp -r ring2 "${BACKUP_DIR}/ring2"
    ok "ring2/ (full git history)"
elif [ -f ring2/main.py ]; then
    mkdir -p "${BACKUP_DIR}/ring2"
    cp ring2/main.py "${BACKUP_DIR}/ring2/main.py"
    ok "ring2/main.py (current code only)"
fi

# 7. Create tar.gz
ARCHIVE="${OUT_DIR}/${BACKUP_NAME}.tar.gz"
tar -czf "$ARCHIVE" -C "$OUT_DIR" "$BACKUP_NAME"
rm -rf "$BACKUP_DIR"
ok "Archive created: $ARCHIVE"

# 8. Stats
SIZE=$(du -h "$ARCHIVE" | cut -f1)
echo
echo "$(_bold 'Backup summary:')"
echo "  Archive: $ARCHIVE"
echo "  Size:    $SIZE"
echo "  Mode:    $(if $FULL; then echo 'full (with Ring 2 history)'; else echo 'standard'; fi)"

# Show memory/skill counts if sqlite3 available.
if [ -f data/protea.db ] && command -v sqlite3 &>/dev/null; then
    MEM_COUNT=$(sqlite3 data/protea.db "SELECT COUNT(*) FROM memory;" 2>/dev/null || echo "?")
    FIT_COUNT=$(sqlite3 data/protea.db "SELECT COUNT(*) FROM fitness_log;" 2>/dev/null || echo "?")
    echo "  Memory entries: $MEM_COUNT"
    echo "  Fitness records: $FIT_COUNT"
    # Try to get skill count (table may not exist in all versions).
    SKILL_COUNT=$(sqlite3 data/protea.db "SELECT COUNT(*) FROM skills;" 2>/dev/null || echo "n/a")
    echo "  Skills: $SKILL_COUNT"
fi

echo
echo "  $(_dim 'Restore with:') ./protea-restore.sh $ARCHIVE"
echo
