"""Tests for sentinel PID lock."""

from __future__ import annotations

import os
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from ring0.sentinel import _acquire_lock


class TestPidLock:
    """Verify PID lock prevents duplicate sentinels."""

    def test_acquire_lock_succeeds(self, tmp_path):
        """First process should acquire the lock successfully."""
        fh = _acquire_lock(tmp_path)
        assert fh is not None
        pid_file = tmp_path / "data" / "sentinel.pid"
        assert pid_file.exists()
        assert pid_file.read_text().strip() == str(os.getpid())
        fh.close()

    def test_second_acquire_fails(self, tmp_path):
        """Second process should fail to acquire the lock."""
        fh = _acquire_lock(tmp_path)
        try:
            fh2 = _acquire_lock(tmp_path)
            assert fh2 is None
        finally:
            fh.close()

    def test_lock_released_after_close(self, tmp_path):
        """Lock should be re-acquirable after the holder closes the fd."""
        fh = _acquire_lock(tmp_path)
        fh.close()  # releases the flock

        # Should succeed now
        fh2 = _acquire_lock(tmp_path)
        assert fh2 is not None
        fh2.close()
