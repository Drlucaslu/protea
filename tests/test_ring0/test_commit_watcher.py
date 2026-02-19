"""Tests for ring0.commit_watcher.CommitWatcher."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from unittest import mock

import pytest

from ring0.commit_watcher import CommitWatcher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git(cwd, *args):
    """Run a git command in the given directory."""
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True,
    )


def _init_repo(path):
    """Create a git repo with one commit and return its HEAD hash."""
    _git(path, "init")
    _git(path, "config", "user.email", "test@test.com")
    _git(path, "config", "user.name", "Test")
    _git(path, "commit", "--allow-empty", "-m", "init")
    result = _git(path, "rev-parse", "HEAD")
    return result.stdout.strip()


def _make_remote(tmp_path):
    """Create a bare remote repo + a local clone. Returns (local_path, remote_path)."""
    remote = tmp_path / "remote.git"
    remote.mkdir()
    _git(remote, "init", "--bare")

    local = tmp_path / "local"
    subprocess.run(
        ["git", "clone", str(remote), str(local)],
        capture_output=True, text=True, check=True,
    )
    _git(local, "config", "user.email", "test@test.com")
    _git(local, "config", "user.name", "Test")
    # Initial commit — ensure branch is named "main" regardless of git defaults.
    (local / "README.md").write_text("hello")
    _git(local, "add", "README.md")
    _git(local, "commit", "-m", "init")
    _git(local, "branch", "-M", "main")
    _git(local, "push", "-u", "origin", "main")
    return local, remote


# ---------------------------------------------------------------------------
# TestGetHead — basic HEAD detection (unchanged from original)
# ---------------------------------------------------------------------------

class TestGetHead:
    def test_returns_hash_in_git_repo(self, tmp_path):
        _init_repo(tmp_path)
        watcher = CommitWatcher(tmp_path, threading.Event())
        head = watcher._get_head()
        assert head is not None
        assert len(head) == 40

    def test_returns_none_outside_git_repo(self, tmp_path):
        watcher = CommitWatcher(tmp_path, threading.Event())
        assert watcher._get_head() is None


# ---------------------------------------------------------------------------
# TestLocalHeadPolling — detect manual pull / local commits
# ---------------------------------------------------------------------------

class TestLocalHeadPolling:
    def test_no_trigger_when_head_unchanged(self, tmp_path):
        _init_repo(tmp_path)
        restart_event = threading.Event()
        watcher = CommitWatcher(tmp_path, restart_event, interval=1)

        thread = threading.Thread(target=watcher.run, daemon=True)
        thread.start()
        time.sleep(2.5)
        watcher.stop()
        thread.join(timeout=3)

        assert not restart_event.is_set()

    def test_triggers_on_new_commit(self, tmp_path):
        _init_repo(tmp_path)
        restart_event = threading.Event()
        watcher = CommitWatcher(tmp_path, restart_event, interval=1)

        thread = threading.Thread(target=watcher.run, daemon=True)
        thread.start()

        time.sleep(0.5)
        _git(tmp_path, "commit", "--allow-empty", "-m", "change")

        thread.join(timeout=5)
        assert restart_event.is_set()

    def test_stop_terminates_loop(self, tmp_path):
        _init_repo(tmp_path)
        restart_event = threading.Event()
        watcher = CommitWatcher(tmp_path, restart_event, interval=60)

        thread = threading.Thread(target=watcher.run, daemon=True)
        thread.start()
        time.sleep(0.3)
        watcher.stop()
        thread.join(timeout=3)

        assert not thread.is_alive()
        assert not restart_event.is_set()


# ---------------------------------------------------------------------------
# TestRing2Filter — ring2/ files excluded from sync
# ---------------------------------------------------------------------------

class TestRing2Filter:
    def test_filters_ring2_files(self, tmp_path):
        local, remote = _make_remote(tmp_path)
        watcher = CommitWatcher(local, threading.Event())

        # Create two commits on local — one with ring2/ and one with ring0/.
        (local / "ring2").mkdir(exist_ok=True)
        (local / "ring2" / "main.py").write_text("print('hi')")
        (local / "ring0").mkdir(exist_ok=True)
        (local / "ring0" / "sentinel.py").write_text("# sentinel")
        _git(local, "add", ".")
        _git(local, "commit", "-m", "mixed changes")
        head = _git(local, "rev-parse", "HEAD").stdout.strip()
        parent = _git(local, "rev-parse", "HEAD~1").stdout.strip()

        changes = watcher._get_changed_files(parent, head)
        paths = [p for _, p in changes]
        assert "ring0/sentinel.py" in paths
        assert all(not p.startswith("ring2/") for _, p in changes)

    def test_only_ring2_returns_empty(self, tmp_path):
        local, remote = _make_remote(tmp_path)
        watcher = CommitWatcher(local, threading.Event())

        (local / "ring2").mkdir(exist_ok=True)
        (local / "ring2" / "agent.py").write_text("# agent")
        _git(local, "add", ".")
        _git(local, "commit", "-m", "ring2 only")
        head = _git(local, "rev-parse", "HEAD").stdout.strip()
        parent = _git(local, "rev-parse", "HEAD~1").stdout.strip()

        changes = watcher._get_changed_files(parent, head)
        assert changes == []


# ---------------------------------------------------------------------------
# TestSyncAndRollback — file sync + rollback
# ---------------------------------------------------------------------------

class TestSyncAndRollback:
    def test_sync_writes_files(self, tmp_path):
        local, remote = _make_remote(tmp_path)
        watcher = CommitWatcher(local, threading.Event())

        # Push a new file from a "contributor" clone.
        contrib = tmp_path / "contrib"
        subprocess.run(
            ["git", "clone", str(remote), str(contrib)],
            capture_output=True, text=True, check=True,
        )
        _git(contrib, "config", "user.email", "test@test.com")
        _git(contrib, "config", "user.name", "Test")
        (contrib / "new_file.txt").write_text("new content")
        _git(contrib, "add", "new_file.txt")
        _git(contrib, "commit", "-m", "add new_file")
        _git(contrib, "push", "origin", "main")

        # Fetch in local so we have the remote ref.
        _git(local, "fetch", "origin")
        remote_head = _git(local, "rev-parse", "origin/main").stdout.strip()
        local_head = _git(local, "rev-parse", "HEAD").stdout.strip()

        changes = watcher._get_changed_files(local_head, remote_head)
        assert len(changes) == 1
        assert changes[0] == ("A", "new_file.txt")

        originals = watcher._sync_files(changes, "origin/main")
        assert (local / "new_file.txt").read_text() == "new content"
        assert originals["new_file.txt"] is None  # didn't exist before

    def test_rollback_restores_files(self, tmp_path):
        local, remote = _make_remote(tmp_path)
        watcher = CommitWatcher(local, threading.Event())

        # Write an original file.
        (local / "existing.txt").write_text("original")

        originals = {"existing.txt": b"original", "new.txt": None}

        # Simulate sync had written these.
        (local / "existing.txt").write_text("modified")
        (local / "new.txt").write_text("added")

        watcher._rollback(originals)

        assert (local / "existing.txt").read_text() == "original"
        assert not (local / "new.txt").exists()

    def test_sync_handles_deleted_files(self, tmp_path):
        local, remote = _make_remote(tmp_path)
        watcher = CommitWatcher(local, threading.Event())

        # Create a file locally.
        (local / "to_delete.txt").write_text("bye")

        changes = [("D", "to_delete.txt")]
        originals = watcher._sync_files(changes, "origin/main")

        assert not (local / "to_delete.txt").exists()
        assert originals["to_delete.txt"] == b"bye"

        # Rollback should restore it.
        watcher._rollback(originals)
        assert (local / "to_delete.txt").read_text() == "bye"


# ---------------------------------------------------------------------------
# TestStatePersistence — state file save/load
# ---------------------------------------------------------------------------

class TestStatePersistence:
    def test_save_and_load(self, tmp_path):
        _init_repo(tmp_path)
        watcher = CommitWatcher(tmp_path, threading.Event())
        test_hash = "abc123def456"

        watcher._save_state(test_hash)
        loaded = watcher._load_state()
        assert loaded == test_hash

    def test_load_returns_none_when_no_file(self, tmp_path):
        _init_repo(tmp_path)
        watcher = CommitWatcher(tmp_path, threading.Event())
        assert watcher._load_state() is None

    def test_state_survives_new_instance(self, tmp_path):
        _init_repo(tmp_path)
        test_hash = "deadbeef1234"

        w1 = CommitWatcher(tmp_path, threading.Event())
        w1._save_state(test_hash)

        w2 = CommitWatcher(tmp_path, threading.Event())
        assert w2._last_synced_hash == test_hash

    def test_state_file_location(self, tmp_path):
        _init_repo(tmp_path)
        watcher = CommitWatcher(tmp_path, threading.Event())
        watcher._save_state("abc")

        state_file = tmp_path / "data" / ".commit_watcher_state"
        assert state_file.exists()
        data = json.loads(state_file.read_text())
        assert data["last_synced_hash"] == "abc"


# ---------------------------------------------------------------------------
# TestTestGate — pytest gate blocks restart on failure
# ---------------------------------------------------------------------------

class TestTestGate:
    def test_run_tests_returns_true_on_success(self, tmp_path):
        _init_repo(tmp_path)
        watcher = CommitWatcher(tmp_path, threading.Event())

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout="all passed")
            assert watcher._run_tests() is True

    def test_run_tests_returns_false_on_failure(self, tmp_path):
        _init_repo(tmp_path)
        watcher = CommitWatcher(tmp_path, threading.Event())

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=1, stdout="FAILED")
            assert watcher._run_tests() is False

    def test_run_tests_returns_false_on_timeout(self, tmp_path):
        _init_repo(tmp_path)
        watcher = CommitWatcher(tmp_path, threading.Event())

        with mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired("pytest", 300)):
            assert watcher._run_tests() is False


# ---------------------------------------------------------------------------
# TestRemoteSync — full remote sync flow
# ---------------------------------------------------------------------------

class TestRemoteSync:
    def test_skips_when_no_new_remote_commits(self, tmp_path):
        local, remote = _make_remote(tmp_path)
        restart_event = threading.Event()
        watcher = CommitWatcher(local, restart_event, fetch_interval=0)

        # Set last_synced_hash to current remote head.
        _git(local, "fetch", "origin")
        remote_head = _git(local, "rev-parse", "origin/main").stdout.strip()
        watcher._last_synced_hash = remote_head

        assert watcher._try_remote_sync() is False
        assert not restart_event.is_set()

    def test_skips_failed_hash(self, tmp_path):
        local, remote = _make_remote(tmp_path)
        watcher = CommitWatcher(local, threading.Event(), fetch_interval=0)

        _git(local, "fetch", "origin")
        remote_head = _git(local, "rev-parse", "origin/main").stdout.strip()
        watcher._failed_hash = remote_head
        watcher._last_synced_hash = "old_hash"

        assert watcher._try_remote_sync() is False

    def test_sync_success_triggers_restart(self, tmp_path):
        local, remote = _make_remote(tmp_path)
        restart_event = threading.Event()
        watcher = CommitWatcher(local, restart_event, fetch_interval=0)

        # Push a change from contributor.
        contrib = tmp_path / "contrib"
        subprocess.run(
            ["git", "clone", str(remote), str(contrib)],
            capture_output=True, text=True, check=True,
        )
        _git(contrib, "config", "user.email", "test@test.com")
        _git(contrib, "config", "user.name", "Test")
        _create_file(contrib, "ring0/__init__.py", "# new")
        _git(contrib, "add", ".")
        _git(contrib, "commit", "-m", "update ring0")
        _git(contrib, "push", "origin", "main")

        # Mock _run_tests to pass.
        with mock.patch.object(watcher, "_run_tests", return_value=True):
            result = watcher._try_remote_sync()

        assert result is True
        assert (local / "ring0" / "__init__.py").read_text() == "# new"

    def test_sync_failure_rolls_back(self, tmp_path):
        local, remote = _make_remote(tmp_path)
        watcher = CommitWatcher(local, threading.Event(), fetch_interval=0)

        # Write an existing file.
        (local / "config.txt").write_text("original config")
        _git(local, "add", "config.txt")
        _git(local, "commit", "-m", "add config")
        _git(local, "push", "origin", "main")

        # Push a change from contributor.
        contrib = tmp_path / "contrib"
        subprocess.run(
            ["git", "clone", str(remote), str(contrib)],
            capture_output=True, text=True, check=True,
        )
        _git(contrib, "config", "user.email", "test@test.com")
        _git(contrib, "config", "user.name", "Test")
        (contrib / "config.txt").write_text("modified config")
        _git(contrib, "add", "config.txt")
        _git(contrib, "commit", "-m", "modify config")
        _git(contrib, "push", "origin", "main")

        # Mock _run_tests to fail.
        with mock.patch.object(watcher, "_run_tests", return_value=False):
            result = watcher._try_remote_sync()

        assert result is False
        # File should be rolled back.
        assert (local / "config.txt").read_text() == "original config"
        # Failed hash should be recorded.
        remote_head = _git(local, "rev-parse", "origin/main").stdout.strip()
        assert watcher._failed_hash == remote_head

    def test_only_ring2_changes_skip_sync(self, tmp_path):
        local, remote = _make_remote(tmp_path)
        watcher = CommitWatcher(local, threading.Event(), fetch_interval=0)

        # Push a ring2-only change from contributor.
        contrib = tmp_path / "contrib"
        subprocess.run(
            ["git", "clone", str(remote), str(contrib)],
            capture_output=True, text=True, check=True,
        )
        _git(contrib, "config", "user.email", "test@test.com")
        _git(contrib, "config", "user.name", "Test")
        (contrib / "ring2").mkdir(exist_ok=True)
        (contrib / "ring2" / "main.py").write_text("# ring2 code")
        _git(contrib, "add", ".")
        _git(contrib, "commit", "-m", "ring2 only")
        _git(contrib, "push", "origin", "main")

        result = watcher._try_remote_sync()
        assert result is False
        # Hash should still be updated (ring2-only changes are acknowledged).
        remote_head = _git(local, "rev-parse", "origin/main").stdout.strip()
        assert watcher._last_synced_hash == remote_head

    def test_fetch_interval_throttling(self, tmp_path):
        local, remote = _make_remote(tmp_path)
        watcher = CommitWatcher(local, threading.Event(), fetch_interval=9999)

        # First call should fetch.
        with mock.patch.object(watcher, "_fetch", return_value=True) as mock_fetch:
            with mock.patch.object(watcher, "_get_remote_head", return_value=watcher._last_synced_hash):
                watcher._try_remote_sync()
                assert mock_fetch.called

        # Second call should be throttled.
        with mock.patch.object(watcher, "_fetch") as mock_fetch:
            watcher._try_remote_sync()
            assert not mock_fetch.called


def _create_file(repo_path, rel_path, content):
    """Helper to create a file with parent dirs."""
    full = repo_path / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content)
