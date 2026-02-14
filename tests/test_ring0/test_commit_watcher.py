"""Tests for ring0.commit_watcher.CommitWatcher."""

from __future__ import annotations

import subprocess
import threading
import time
from unittest import mock

import pytest

from ring0.commit_watcher import CommitWatcher


class TestGetHead:
    """Unit tests for CommitWatcher._get_head()."""

    def test_returns_hash_in_git_repo(self, tmp_path):
        # Create a real git repo with one commit.
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
        subprocess.run(["git", "commit", "--allow-empty", "-m", "init"],
                       cwd=tmp_path, capture_output=True, check=True)

        watcher = CommitWatcher(tmp_path, threading.Event())
        head = watcher._get_head()
        assert head is not None
        assert len(head) == 40  # full SHA-1

    def test_returns_none_outside_git_repo(self, tmp_path):
        watcher = CommitWatcher(tmp_path, threading.Event())
        assert watcher._get_head() is None


class TestPolling:
    """Integration tests for the polling loop."""

    def test_no_trigger_when_head_unchanged(self, tmp_path):
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
        subprocess.run(["git", "commit", "--allow-empty", "-m", "init"],
                       cwd=tmp_path, capture_output=True, check=True)

        restart_event = threading.Event()
        watcher = CommitWatcher(tmp_path, restart_event, interval=1)

        thread = threading.Thread(target=watcher.run, daemon=True)
        thread.start()
        time.sleep(2.5)  # let it poll a couple of times
        watcher.stop()
        thread.join(timeout=3)

        assert not restart_event.is_set()

    def test_triggers_on_new_commit(self, tmp_path):
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
        subprocess.run(["git", "commit", "--allow-empty", "-m", "init"],
                       cwd=tmp_path, capture_output=True, check=True)

        restart_event = threading.Event()
        watcher = CommitWatcher(tmp_path, restart_event, interval=1)

        thread = threading.Thread(target=watcher.run, daemon=True)
        thread.start()

        # Make a new commit while watcher is running.
        time.sleep(0.5)
        subprocess.run(["git", "commit", "--allow-empty", "-m", "change"],
                       cwd=tmp_path, capture_output=True, check=True)

        thread.join(timeout=5)
        assert restart_event.is_set()

    def test_stop_terminates_loop(self, tmp_path):
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
        subprocess.run(["git", "commit", "--allow-empty", "-m", "init"],
                       cwd=tmp_path, capture_output=True, check=True)

        restart_event = threading.Event()
        watcher = CommitWatcher(tmp_path, restart_event, interval=60)

        thread = threading.Thread(target=watcher.run, daemon=True)
        thread.start()
        time.sleep(0.3)
        watcher.stop()
        thread.join(timeout=3)

        assert not thread.is_alive()
        assert not restart_event.is_set()
