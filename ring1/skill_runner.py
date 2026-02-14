"""SkillRunner — run a crystallized skill as an independent subprocess.

Only one skill can run at a time.  Subprocess output is captured to a
temporary log file for inspection via get_output().
"""

from __future__ import annotations

import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import time

log = logging.getLogger("protea.skill_runner")


class SkillRunner:
    """Manage a single independently-running skill process."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._skill_name: str = ""
        self._start_time: float = 0.0
        self._log_path: str = ""
        self._log_fh = None
        self._script_path: str = ""
        self._port: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, skill_name: str, source_code: str) -> tuple[int, str]:
        """Start a skill process.  Returns (pid, message).

        If a skill is already running it is stopped first.
        """
        if self.is_running():
            self.stop()

        # Write source to a temp file.
        fd, script_path = tempfile.mkstemp(suffix=".py", prefix=f"protea_skill_{skill_name}_")
        with os.fdopen(fd, "w") as f:
            f.write(source_code)
        self._script_path = script_path

        # Prepare log file.
        log_fd, log_path = tempfile.mkstemp(suffix=".log", prefix=f"protea_skill_{skill_name}_")
        log_fh = os.fdopen(log_fd, "w")
        self._log_path = log_path
        self._log_fh = log_fh

        # Prepare env — set PROTEA_HEARTBEAT to a temp path so heartbeat
        # code inside the skill doesn't fail.
        hb_fd, hb_path = tempfile.mkstemp(prefix="protea_hb_")
        os.close(hb_fd)
        env = {**os.environ, "PROTEA_HEARTBEAT": hb_path, "PYTHONUNBUFFERED": "1"}

        proc = subprocess.Popen(
            [sys.executable, script_path],
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        self._proc = proc
        self._skill_name = skill_name
        self._start_time = time.time()
        self._port = None

        log.info("Skill '%s' started  pid=%d", skill_name, proc.pid)

        # Wait briefly and try to detect a port from output.
        time.sleep(1)
        self._detect_port()

        parts = [f"Skill *{skill_name}* started (PID {proc.pid})."]
        if self._port:
            parts.append(f"Port: {self._port}")
        parts.append("Use /stop to stop, /running to check status.")
        return proc.pid, "\n".join(parts)

    def stop(self) -> bool:
        """Stop the current skill process.  Returns True if stopped."""
        proc = self._proc
        if proc is None or proc.poll() is not None:
            self._cleanup()
            return False

        # Kill the entire process group so child processes (e.g. HTTP
        # servers spawned by the skill) are also terminated.
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            proc.wait()
        log.info("Skill '%s' stopped  pid=%d", self._skill_name, proc.pid)
        self._cleanup()
        return True

    def is_running(self) -> bool:
        """Return True if the skill process is alive."""
        if self._proc is None:
            return False
        if self._proc.poll() is not None:
            return False
        return True

    def get_output(self, max_lines: int = 30) -> str:
        """Read the tail of the skill's log output."""
        if not self._log_path:
            return ""
        try:
            with open(self._log_path, errors="replace") as f:
                lines = f.read().splitlines()
            return "\n".join(lines[-max_lines:])
        except FileNotFoundError:
            return ""

    def get_info(self) -> dict | None:
        """Return info dict or None if no skill is loaded."""
        if self._proc is None:
            return None
        running = self.is_running()
        uptime = time.time() - self._start_time if running else 0.0
        # Re-detect port if we haven't found one yet.
        if self._port is None and running:
            self._detect_port()
        return {
            "skill_name": self._skill_name,
            "pid": self._proc.pid,
            "running": running,
            "uptime": uptime,
            "port": self._port,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    _PORT_RE = re.compile(r"(?:port\s+|localhost:)(\d{2,5})", re.IGNORECASE)

    def _detect_port(self) -> None:
        """Try to detect a port number from the log output."""
        output = self.get_output(max_lines=50)
        m = self._PORT_RE.search(output)
        if m:
            self._port = int(m.group(1))

    def _cleanup(self) -> None:
        """Close file handles and reset state."""
        if self._log_fh:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None
        self._proc = None
        self._port = None
