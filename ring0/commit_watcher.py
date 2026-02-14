"""CommitWatcher — polls Git HEAD and triggers restart on new commits."""

from __future__ import annotations

import logging
import subprocess
import threading

log = logging.getLogger("protea.commit_watcher")


class CommitWatcher:
    """Poll ``git rev-parse HEAD`` and set *restart_event* when it changes."""

    def __init__(self, project_root, restart_event: threading.Event, interval: int = 5):
        self._root = str(project_root)
        self._restart_event = restart_event
        self._interval = interval
        self._stop_event = threading.Event()

    # -- helpers --

    def _get_head(self) -> str | None:
        """Return the current HEAD hash, or *None* on any git error."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self._root,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return None
        except Exception:
            return None

    # -- main loop --

    def run(self) -> None:
        """Polling loop — intended to run in a daemon thread."""
        initial = self._get_head()
        if initial is None:
            log.debug("Not a git repo or git unavailable — CommitWatcher disabled")
            return
        log.info("CommitWatcher started  HEAD=%s", initial[:12])
        last_head = initial

        while not self._stop_event.is_set():
            self._stop_event.wait(self._interval)
            if self._stop_event.is_set():
                break
            head = self._get_head()
            if head is None:
                continue
            if head != last_head:
                log.info("New commit detected: %s → %s", last_head[:12], head[:12])
                self._restart_event.set()
                return

    def stop(self) -> None:
        """Signal the polling loop to stop."""
        self._stop_event.set()
