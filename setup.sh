#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "=== Protea Setup ==="
echo

# 1. Check Python >= 3.11
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.11+ first."
    exit 1
fi

py_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
py_major=$(echo "$py_version" | cut -d. -f1)
py_minor=$(echo "$py_version" | cut -d. -f2)

if [ "$py_major" -lt 3 ] || { [ "$py_major" -eq 3 ] && [ "$py_minor" -lt 11 ]; }; then
    echo "ERROR: Python 3.11+ required (found $py_version)"
    exit 1
fi
echo "[ok] Python $py_version"

# 2. Check Git
if ! command -v git &>/dev/null; then
    echo "ERROR: git not found. Install Git first."
    exit 1
fi
echo "[ok] Git $(git --version | awk '{print $3}')"

# 3. Create venv if needed
if [ ! -d .venv ]; then
    echo "[..] Creating virtual environment..."
    python3 -m venv .venv
    echo "[ok] Created .venv"
else
    echo "[ok] .venv already exists"
fi

# Activate venv
source .venv/bin/activate

# 4. Copy .env.example -> .env if needed
if [ ! -f .env ]; then
    cp .env.example .env
    echo "[ok] Created .env from .env.example"
    echo "     >>> Edit .env and set CLAUDE_API_KEY (required) <<<"
else
    echo "[ok] .env already exists"
fi

# 5. Initialize Ring 2 git repo if needed
if [ ! -d ring2/.git ]; then
    echo "[..] Initializing Ring 2 git repo..."
    git init ring2
    git -C ring2 add -A
    git -C ring2 commit -m "init"
    echo "[ok] Ring 2 git repo initialized"
else
    echo "[ok] Ring 2 git repo already exists"
fi

# 6. Create data/ and output/ directories
mkdir -p data output
echo "[ok] data/ and output/ directories ready"

# 7. Run tests
echo
echo "=== Running Tests ==="
python -m pytest tests/ -v

# 8. Done
echo
echo "=== Setup Complete ==="
echo
echo "Next steps:"
echo "  1. Edit .env and set CLAUDE_API_KEY (required)"
echo "  2. Optionally set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID"
echo "  3. Run: source .venv/bin/activate && python run.py"
