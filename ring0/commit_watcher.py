"""CommitWatcher — polls Git HEAD and auto-syncs from remote.

Flow:
  1. Every poll_interval: check local HEAD change (manual pull) → restart
  2. Every fetch_interval: git fetch → diff → selective sync (skip ring2/) →
     pytest gate → restart on success / rollback on failure
"""

from __future__ import annotations

import json
import logging
import pathlib
import subprocess
import sys
import threading
import time

log = logging.getLogger("protea.commit_watcher")

STATE_FILE = "data/.commit_watcher_state"


class CommitWatcher:
    """Poll local HEAD and remote origin for changes, sync selectively."""

    def __init__(
        self,
        project_root,
        restart_event: threading.Event,
        interval: int = 10,
        fetch_interval: int = 120,
    ):
        self._root = str(project_root)
        self._root_path = pathlib.Path(project_root)
        self._restart_event = restart_event
        self._interval = interval
        self._fetch_interval = fetch_interval
        self._stop_event = threading.Event()
        self._last_fetch_time: float = 0.0
        self._failed_hash: str | None = None
        self._state_path = self._root_path / STATE_FILE
        self._last_synced_hash = self._load_state()

    # -- state persistence --

    def _load_state(self) -> str | None:
        """Load last synced remote hash from state file."""
        try:
            data = self._state_path.read_text().strip()
            if data:
                state = json.loads(data)
                return state.get("last_synced_hash")
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        return None

    def _save_state(self, remote_hash: str) -> None:
        """Persist last synced remote hash."""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(json.dumps({"last_synced_hash": remote_hash}))

    # -- git helpers --

    def _git(self, *args: str, timeout: int = 30) -> subprocess.CompletedProcess:
        """Run a git command in the project root."""
        return subprocess.run(
            ["git", *args],
            cwd=self._root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def _get_head(self) -> str | None:
        """Return the current local HEAD hash, or None on error."""
        try:
            result = self._git("rev-parse", "HEAD", timeout=5)
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    def _fetch(self) -> bool:
        """Run git fetch origin. Returns True on success."""
        try:
            result = self._git("fetch", "origin", timeout=60)
            return result.returncode == 0
        except Exception:
            log.warning("git fetch failed", exc_info=True)
            return False

    def _get_remote_head(self) -> str | None:
        """Return the hash of origin/main (or origin/master)."""
        for branch in ("origin/main", "origin/master"):
            try:
                result = self._git("rev-parse", branch, timeout=5)
                if result.returncode == 0:
                    return result.stdout.strip()
            except Exception:
                continue
        return None

    def _get_changed_files(self, from_hash: str, to_hash: str) -> list[tuple[str, str]]:
        """Return list of (status, path) between two commits, excluding ring2/."""
        try:
            result = self._git("diff", "--name-status", "--no-renames", from_hash, to_hash)
            if result.returncode != 0:
                return []
        except Exception:
            return []

        changes = []
        for line in result.stdout.strip().splitlines():
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            status, path = parts
            # Skip ring2/ files — ring2 is managed by GitManager snapshots.
            if path.startswith("ring2/"):
                continue
            changes.append((status, path))
        return changes

    def _sync_files(
        self, changes: list[tuple[str, str]], remote_ref: str
    ) -> dict[str, bytes | None]:
        """Write changed files from remote_ref to disk. Returns originals for rollback.

        originals maps path → original bytes (or None if file didn't exist).
        """
        originals: dict[str, bytes | None] = {}

        for status, rel_path in changes:
            full_path = self._root_path / rel_path

            # Save original for rollback.
            if full_path.exists():
                originals[rel_path] = full_path.read_bytes()
            else:
                originals[rel_path] = None

            if status == "D":
                # Deleted in remote.
                if full_path.exists():
                    full_path.unlink()
            else:
                # Added or modified — read content from remote ref.
                try:
                    result = self._git("show", f"{remote_ref}:{rel_path}", timeout=10)
                    if result.returncode != 0:
                        log.warning("git show failed for %s: %s", rel_path, result.stderr.strip())
                        continue
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    full_path.write_text(result.stdout)
                except Exception:
                    log.warning("Failed to sync %s", rel_path, exc_info=True)

        return originals

    def _rollback(self, originals: dict[str, bytes | None]) -> None:
        """Restore files from saved originals after a failed sync."""
        for rel_path, content in originals.items():
            full_path = self._root_path / rel_path
            if content is None:
                # File didn't exist before — remove it.
                if full_path.exists():
                    full_path.unlink()
            else:
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_bytes(content)

    def _run_tests(self) -> bool:
        """Run pytest tests/ -x -q --tb=line. Returns True if tests pass."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "-x", "-q", "--tb=line"],
                cwd=self._root,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                log.info("Tests passed after remote sync")
                return True
            else:
                log.warning("Tests failed after remote sync:\n%s", result.stdout[-500:] if result.stdout else "")
                return False
        except subprocess.TimeoutExpired:
            log.warning("Test suite timed out (300s)")
            return False
        except Exception:
            log.warning("Failed to run tests", exc_info=True)
            return False

    # -- auto-push logic --

    def _try_push_local(self) -> None:
        """Push local commits to origin if local is ahead of remote."""
        try:
            result = self._git("rev-list", "--count", "origin/main..HEAD", timeout=5)
            if result.returncode != 0:
                return
            ahead = int(result.stdout.strip())
            if ahead == 0:
                return

            log.info("Local is %d commit(s) ahead, pushing to origin", ahead)
            push_result = self._git("push", "origin", "main", timeout=60)
            if push_result.returncode == 0:
                log.info("Push successful")
                # Update synced hash so we don't re-pull our own commits.
                head = self._get_head()
                if head:
                    self._last_synced_hash = head
                    self._save_state(head)
            else:
                log.warning("Push failed: %s", push_result.stderr.strip())
        except Exception:
            log.warning("Push failed", exc_info=True)

    # -- remote sync logic --

    def _try_remote_sync(self) -> bool:
        """Attempt to fetch and sync from remote. Returns True if restart needed."""
        now = time.monotonic()
        if now - self._last_fetch_time < self._fetch_interval:
            return False
        self._last_fetch_time = now

        if not self._fetch():
            log.warning("git fetch origin failed")
            return False

        self._try_push_local()

        remote_head = self._get_remote_head()
        if remote_head is None:
            log.warning("Could not resolve origin/main or origin/master")
            return False

        # Already synced this hash (or it's the initial state).
        if remote_head == self._last_synced_hash:
            log.info("Remote %s: up to date", remote_head[:12])
            return False

        # Skip hash that previously failed tests.
        if remote_head == self._failed_hash:
            return False

        # Need a base hash to diff against.
        base_hash = self._last_synced_hash or self._get_head()
        if base_hash is None:
            return False

        changes = self._get_changed_files(base_hash, remote_head)
        if not changes:
            # Only ring2/ changes or no changes — just update hash.
            self._last_synced_hash = remote_head
            self._save_state(remote_head)
            log.info("Remote %s: only ring2 or no relevant changes, skipping", remote_head[:12])
            return False

        log.info("Syncing %d files from remote %s", len(changes), remote_head[:12])
        originals = self._sync_files(changes, remote_head)

        if self._run_tests():
            self._last_synced_hash = remote_head
            self._save_state(remote_head)
            self._failed_hash = None
            log.info("Remote sync successful, triggering restart")
            return True
        else:
            log.warning("Rolling back %d files after failed tests", len(originals))
            self._rollback(originals)
            self._failed_hash = remote_head
            return False

    # -- main loop --

    def run(self) -> None:
        """Polling loop — intended to run in a daemon thread."""
        initial = self._get_head()
        if initial is None:
            log.debug("Not a git repo or git unavailable — CommitWatcher disabled")
            return
        log.info("CommitWatcher started  HEAD=%s", initial[:12])
        last_head = initial

        # Initialize last_synced_hash if not loaded from state file.
        if self._last_synced_hash is None:
            self._last_synced_hash = initial

        while not self._stop_event.is_set():
            self._stop_event.wait(self._interval)
            if self._stop_event.is_set():
                break

            # 1. Check local HEAD change (manual pull).
            head = self._get_head()
            if head is not None and head != last_head:
                log.info("Local HEAD changed: %s → %s", last_head[:12], head[:12])
                self._restart_event.set()
                return

            # 2. Periodically fetch and sync from remote.
            if self._try_remote_sync():
                self._restart_event.set()
                return

    def stop(self) -> None:
        """Signal the polling loop to stop."""
        self._stop_event.set()
