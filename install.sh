#!/usr/bin/env bash
# Protea Installer — beginner-friendly setup with auto-dependency handling.
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/Drlucaslu/protea/main/install.sh | bash
#   or:  bash install.sh
#
# After dependencies are resolved, launches setup_wizard.py for configuration.
set -euo pipefail

REPO_URL="https://github.com/Drlucaslu/protea.git"
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=11

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_green()  { printf '\033[32m%s\033[0m' "$*"; }
_yellow() { printf '\033[33m%s\033[0m' "$*"; }
_red()    { printf '\033[31m%s\033[0m' "$*"; }
_bold()   { printf '\033[1m%s\033[0m' "$*"; }

ok()   { echo "  $(_green '[ok]') $*"; }
info() { echo "  $(_yellow '[..]') $*"; }
fail() { echo "  $(_red '[!!]') $*"; exit 1; }

# ---------------------------------------------------------------------------
# Step 0: Ensure we are in the project root
# ---------------------------------------------------------------------------

echo
echo "$(_bold '=== Protea Installer ===')"
echo

if [ -f run.py ] && [ -d ring0 ]; then
    : # already in protea root
elif [ -f "$(dirname "$0")/run.py" ] 2>/dev/null; then
    cd "$(dirname "$0")"
else
    info "Cloning protea..."
    git clone --depth 1 "$REPO_URL"
    cd protea
    ok "Cloned into $(pwd)"
fi

# ---------------------------------------------------------------------------
# Step 1: Find or install Python >= 3.11
# ---------------------------------------------------------------------------

find_python() {
    for cmd in python3.13 python3.12 python3.11 python3; do
        if command -v "$cmd" &>/dev/null; then
            local ver
            ver=$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
            local major minor
            major=$(echo "$ver" | cut -d. -f1)
            minor=$(echo "$ver" | cut -d. -f2)
            if [ "$major" -ge $MIN_PYTHON_MAJOR ] && [ "$minor" -ge $MIN_PYTHON_MINOR ]; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

install_python() {
    info "Installing Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+..."
    case "$(uname -s)" in
        Darwin)
            if command -v brew &>/dev/null; then
                info "Installing via Homebrew..."
                brew install python@3.11
            else
                info "Homebrew not found — installing Homebrew first..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" < /dev/tty
                eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || /usr/local/bin/brew shellenv 2>/dev/null)"
                brew install python@3.11
            fi
            ;;
        Linux)
            if [ -f /etc/debian_version ]; then
                info "Installing via apt (deadsnakes PPA)..."
                sudo apt-get update -qq
                sudo apt-get install -y -qq software-properties-common
                sudo add-apt-repository -y ppa:deadsnakes/ppa
                sudo apt-get update -qq
                sudo apt-get install -y -qq python3.11 python3.11-venv
            elif [ -f /etc/redhat-release ]; then
                info "Installing via dnf..."
                sudo dnf install -y python3.11
            else
                fail "Unsupported Linux distribution. Please install Python 3.11+ manually."
            fi
            ;;
        *)
            fail "Unsupported OS ($(uname -s)). Please install Python 3.11+ manually."
            ;;
    esac
}

PYTHON_CMD=$(find_python || true)

if [ -z "$PYTHON_CMD" ]; then
    if command -v python3 &>/dev/null; then
        cur_ver=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        info "Python $cur_ver found, but ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+ required."
    else
        info "Python not found."
    fi
    read -p "  Install Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}? [Y/n] " answer < /dev/tty
    if [ "${answer:-Y}" != "n" ] && [ "${answer:-Y}" != "N" ]; then
        install_python
        PYTHON_CMD=$(find_python || true)
        [ -z "$PYTHON_CMD" ] && fail "Python installed but not found in PATH."
    else
        fail "Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+ is required. Aborted."
    fi
fi

py_version=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
ok "Python $py_version ($PYTHON_CMD)"

# ---------------------------------------------------------------------------
# Step 2: Check Git
# ---------------------------------------------------------------------------

if ! command -v git &>/dev/null; then
    fail "Git not found. Please install Git first."
fi
ok "Git $(git --version | awk '{print $3}')"

# ---------------------------------------------------------------------------
# Step 3: Create venv
# ---------------------------------------------------------------------------

if [ ! -d .venv ]; then
    info "Creating virtual environment..."
    $PYTHON_CMD -m venv .venv
fi
if [ ! -f .venv/bin/activate ]; then
    fail "venv creation failed. On Debian/Ubuntu: sudo apt install python3-venv"
fi
ok "Virtual environment ready"

PY=.venv/bin/python

# ---------------------------------------------------------------------------
# Step 4: Install dependencies
# ---------------------------------------------------------------------------

info "Installing dependencies (this may take a minute)..."
$PY -m pip install --quiet --upgrade pip 2>&1 | tail -1 || true
$PY -m pip install --quiet -e ".[dev]" 2>&1 | tail -1 || true
ok "Dependencies installed"

# ---------------------------------------------------------------------------
# Step 5: Launch Setup Wizard
# ---------------------------------------------------------------------------

echo
echo "$(_bold '=== Configuration ===')"
echo
echo "  Choose setup mode:"
echo "    1. Web UI  — opens setup page at http://localhost:8899"
echo "    2. CLI     — step-by-step terminal prompts"
echo

read -p "  Mode [1]: " mode < /dev/tty
mode="${mode:-1}"

case "$mode" in
    1|web|Web)
        $PY setup_wizard.py --web
        ;;
    2|cli|CLI)
        $PY setup_wizard.py --cli
        ;;
    *)
        info "Invalid choice, defaulting to CLI."
        $PY setup_wizard.py --cli
        ;;
esac
