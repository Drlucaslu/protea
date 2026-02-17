#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/Drlucaslu/protea.git"

echo "=== Protea Setup ==="
echo

# 0. Clone repo if not already inside it
if [ -f run.py ] && [ -d ring0 ]; then
    : # already in protea root
elif [ -f "$(dirname "$0")/run.py" ] 2>/dev/null; then
    cd "$(dirname "$0")"
else
    echo "[..] Cloning protea..."
    git clone "$REPO_URL"
    cd protea
    echo "[ok] Cloned into $(pwd)"
fi

# 1. Find or install Python >= 3.11
find_python() {
    # Try common names in order of preference
    for cmd in python3.13 python3.12 python3.11 python3; do
        if command -v "$cmd" &>/dev/null; then
            local ver
            ver=$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
            local major minor
            major=$(echo "$ver" | cut -d. -f1)
            minor=$(echo "$ver" | cut -d. -f2)
            if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

install_python() {
    echo "[..] Attempting to install Python 3.11+..."
    case "$(uname -s)" in
        Darwin)
            if command -v brew &>/dev/null; then
                echo "[..] Installing via Homebrew..."
                brew install python@3.11
            else
                echo "[..] Homebrew not found, installing Homebrew first..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" < /dev/tty
                eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || /usr/local/bin/brew shellenv 2>/dev/null)"
                brew install python@3.11
            fi
            ;;
        Linux)
            if [ -f /etc/debian_version ]; then
                echo "[..] Installing via apt (deadsnakes PPA)..."
                sudo apt-get update
                sudo apt-get install -y software-properties-common
                sudo add-apt-repository -y ppa:deadsnakes/ppa
                sudo apt-get update
                sudo apt-get install -y python3.11 python3.11-venv
            elif [ -f /etc/redhat-release ]; then
                echo "[..] Installing via dnf..."
                sudo dnf install -y python3.11
            else
                echo "ERROR: Unsupported Linux distribution. Please install Python 3.11+ manually."
                exit 1
            fi
            ;;
        *)
            echo "ERROR: Unsupported OS. Please install Python 3.11+ manually."
            exit 1
            ;;
    esac
}

PYTHON_CMD=$(find_python || true)

if [ -z "$PYTHON_CMD" ]; then
    if command -v python3 &>/dev/null; then
        cur_ver=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        echo "Python $cur_ver found, but 3.11+ is required."
    else
        echo "Python not found."
    fi
    read -p "Install Python 3.11? [Y/n] " answer < /dev/tty
    if [ "${answer:-Y}" != "n" ] && [ "${answer:-Y}" != "N" ]; then
        install_python
        PYTHON_CMD=$(find_python || true)
        if [ -z "$PYTHON_CMD" ]; then
            echo "ERROR: Installation succeeded but Python 3.11+ still not found in PATH."
            exit 1
        fi
    else
        echo "Aborted."
        exit 1
    fi
fi

py_version=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "[ok] Python $py_version ($PYTHON_CMD)"

# 2. Check Git
if ! command -v git &>/dev/null; then
    echo "ERROR: git not found. Install Git first."
    exit 1
fi
echo "[ok] Git $(git --version | awk '{print $3}')"

# 3. Create venv if needed
if [ ! -d .venv ]; then
    echo "[..] Creating virtual environment..."
    $PYTHON_CMD -m venv .venv
fi

# Verify venv was created properly
if [ ! -f .venv/bin/activate ]; then
    echo "ERROR: venv creation failed — .venv/bin/activate not found."
    echo "  On Debian/Ubuntu, install: sudo apt install python3-venv"
    exit 1
fi
echo "[ok] Virtual environment ready"

# Use venv python directly (no need to source activate)
PY=.venv/bin/python

# 4. Install dependencies
echo "[..] Installing dependencies..."
$PY -m pip install --quiet --upgrade pip
$PY -m pip install --quiet -e ".[dev]"
echo "[ok] Dependencies installed"

# 5. Configure .env interactively
if [ ! -f .env ]; then
    echo
    echo "=== Configuration ==="
    echo

    # Read from /dev/tty so this works even when piped via curl
    read -p "CLAUDE_API_KEY (required): " api_key < /dev/tty
    if [ -z "$api_key" ]; then
        echo "ERROR: CLAUDE_API_KEY is required."
        exit 1
    fi

    read -p "TELEGRAM_BOT_TOKEN (optional, press Enter to skip): " tg_token < /dev/tty

    cat > .env <<EOF
# Claude API Key (used by Ring 1 evolution engine)
CLAUDE_API_KEY=$api_key

# Telegram Bot Token
TELEGRAM_BOT_TOKEN=$tg_token

# Telegram Chat ID (auto-detected from first message if empty)
TELEGRAM_CHAT_ID=
EOF
    echo "[ok] Created .env"
else
    echo "[ok] .env already exists"
fi

# 6. Initialize Ring 2 git repo if needed
if [ ! -d ring2/.git ]; then
    echo "[..] Initializing Ring 2 git repo..."
    git init ring2
    git -C ring2 add -A
    git -C ring2 commit -m "init"
    echo "[ok] Ring 2 git repo initialized"
else
    echo "[ok] Ring 2 git repo already exists"
fi

# 7. Create data/ and output/ directories
mkdir -p data output
echo "[ok] data/ and output/ directories ready"

# 8. Run tests
echo
echo "=== Running Tests ==="
$PY -m pytest tests/ -v

# 9. Ensure scripts are executable
chmod +x run_with_nohup.sh 2>/dev/null || true

# 10. Done

echo
echo "=== Setup Complete ==="
echo
echo "To start Protea:"
echo "  .venv/bin/python run.py            # foreground"
echo "  ./run_with_nohup.sh                # background (nohup)"
