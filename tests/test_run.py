"""Tests for run.py PID lock."""

from __future__ import annotations

import fcntl
import os
import pathlib
import subprocess
import sys

import pytest


def _project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent


class TestPidLock:
    """Verify PID lock prevents duplicate sentinels."""

    def test_acquire_lock_succeeds(self, tmp_path):
        """First process should acquire the lock successfully."""
        sys.path.insert(0, str(_project_root()))
        from run import _acquire_lock

        fh = _acquire_lock(tmp_path)
        assert fh is not None
        pid_file = tmp_path / "data" / "sentinel.pid"
        assert pid_file.exists()
        assert pid_file.read_text().strip() == str(os.getpid())
        # Clean up
        fh.close()

    def test_second_acquire_fails(self, tmp_path):
        """Second process should fail to acquire the lock."""
        sys.path.insert(0, str(_project_root()))
        from run import _acquire_lock

        fh = _acquire_lock(tmp_path)
        try:
            with pytest.raises(SystemExit):
                _acquire_lock(tmp_path)
        finally:
            fh.close()

    def test_lock_released_after_close(self, tmp_path):
        """Lock should be re-acquirable after the holder closes the fd."""
        sys.path.insert(0, str(_project_root()))
        from run import _acquire_lock

        fh = _acquire_lock(tmp_path)
        fh.close()  # releases the flock

        # Should succeed now
        fh2 = _acquire_lock(tmp_path)
        assert fh2 is not None
        fh2.close()
