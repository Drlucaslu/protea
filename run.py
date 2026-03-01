#!/usr/bin/env python3
"""Protea entry point — launches Ring 0 Sentinel."""

import os
import sys
import pathlib

if sys.version_info < (3, 11):
    root = pathlib.Path(__file__).resolve().parent
    venv_py = root / ".venv" / "bin" / "python"
    print(f"ERROR: Python 3.11+ required (found {sys.version_info.major}.{sys.version_info.minor})")
    if venv_py.exists():
        print(f"  Run: {venv_py} run.py")
    else:
        print("  Run: bash setup.sh")
    sys.exit(1)

# Load .env file if present (does not override existing env vars).
_env_file = pathlib.Path(__file__).resolve().parent / ".env"
if _env_file.is_file():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        key, _, value = _line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value

# Ensure project root is on sys.path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from ring0.sentinel import main


if __name__ == "__main__":
    main()
