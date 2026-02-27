#!/usr/bin/env python3
"""Protea entry point — launches Ring 0 Sentinel."""

import fcntl
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


def _acquire_lock(project_root):
    """Acquire exclusive PID lock. Exit if another sentinel is running."""
    lock_path = project_root / "data" / "sentinel.pid"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"ERROR: Another sentinel is already running (lock: {lock_path})")
        sys.exit(1)
    fh.write(str(os.getpid()))
    fh.flush()
    return fh  # must keep reference to prevent GC closing the fd


if __name__ == "__main__":
    _project_root = pathlib.Path(__file__).resolve().parent
    _lock_fh = _acquire_lock(_project_root)
    main()
