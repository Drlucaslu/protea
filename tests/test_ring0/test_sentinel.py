"""Integration tests for ring0.sentinel."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time
import tomllib

import pytest

from ring0.git_manager import GitManager
from ring0.heartbeat import HeartbeatMonitor
from ring0.sentinel import (
    _classify_failure,
    _read_ring2_output,
    _start_ring2,
    _stop_ring2,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent.parent


def _write_ring2_script(ring2_path: pathlib.Path, script: str) -> None:
    """Write a custom Ring 2 main.py for testing."""
    (ring2_path / "main.py").write_text(script)


# Minimal Ring 2 that heartbeats correctly
_GOOD_RING2 = """\
import os, pathlib, time
hb = pathlib.Path(os.environ["PROTEA_HEARTBEAT"])
pid = os.getpid()
while True:
    hb.write_text(f"{pid}\\n{time.time()}\\n")
    time.sleep(1)
"""

# Ring 2 that crashes immediately
_BAD_RING2 = """\
import sys
sys.exit(1)
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRing2Lifecycle:
    """Start / stop Ring 2 and verify heartbeat protocol."""

    def test_start_and_heartbeat(self, tmp_path):
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        _write_ring2_script(ring2, _GOOD_RING2)
        hb_path = ring2 / ".heartbeat"

        proc = _start_ring2(ring2, hb_path)
        try:
            monitor = HeartbeatMonitor(hb_path, timeout_sec=6.0)
            assert monitor.wait_for_heartbeat(startup_timeout=5.0)
            assert monitor.is_alive()
        finally:
            _stop_ring2(proc)

        assert proc.poll() is not None  # process terminated

    def test_stop_dead_process_is_safe(self):
        _stop_ring2(None)  # should not raise

    def test_crashing_ring2_detected(self, tmp_path):
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        _write_ring2_script(ring2, _BAD_RING2)
        hb_path = ring2 / ".heartbeat"

        proc = _start_ring2(ring2, hb_path)
        proc.wait(timeout=5)
        monitor = HeartbeatMonitor(hb_path, timeout_sec=2.0)
        assert not monitor.is_alive()


class TestRollbackIntegration:
    """Verify Git snapshot + rollback works with Ring 2 code."""

    def test_rollback_restores_ring2(self, tmp_path):
        ring2 = tmp_path / "ring2"
        ring2.mkdir()

        gm = GitManager(ring2)
        gm.init_repo()

        # Version 1: good script
        _write_ring2_script(ring2, _GOOD_RING2)
        good_hash = gm.snapshot("good version")

        # Version 2: bad script
        _write_ring2_script(ring2, _BAD_RING2)
        gm.snapshot("bad version")

        # Rollback to good
        gm.rollback(good_hash)
        content = (ring2 / "main.py").read_text()
        assert "sys.exit(1)" not in content
        assert "time.sleep" in content


class TestOutputCapture:
    """Verify Ring 2 output is captured to .output.log."""

    def test_start_creates_log_file(self, tmp_path):
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        _write_ring2_script(ring2, _BAD_RING2)
        hb_path = ring2 / ".heartbeat"

        proc = _start_ring2(ring2, hb_path)
        proc.wait(timeout=5)
        _stop_ring2(proc)

        log_file = ring2 / ".output.log"
        assert log_file.exists()

    def test_output_written_to_log(self, tmp_path):
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        script = 'import sys\nprint("hello_from_ring2")\nsys.exit(0)\n'
        _write_ring2_script(ring2, script)
        hb_path = ring2 / ".heartbeat"

        proc = _start_ring2(ring2, hb_path)
        proc.wait(timeout=5)
        _stop_ring2(proc)

        log_file = ring2 / ".output.log"
        content = log_file.read_text()
        assert "hello_from_ring2" in content

    def test_stderr_captured(self, tmp_path):
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        script = 'import sys\nprint("err_msg", file=sys.stderr)\nsys.exit(1)\n'
        _write_ring2_script(ring2, script)
        hb_path = ring2 / ".heartbeat"

        proc = _start_ring2(ring2, hb_path)
        proc.wait(timeout=5)
        _stop_ring2(proc)

        log_file = ring2 / ".output.log"
        content = log_file.read_text()
        assert "err_msg" in content


class TestReadRing2Output:
    """Verify _read_ring2_output reads the tail of the log."""

    def test_reads_last_n_lines(self, tmp_path):
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        # Create 200 lines of output.
        lines = [f"line {i}" for i in range(200)]
        script = "import sys\n" + "\n".join(f'print("line {i}")' for i in range(200)) + "\nsys.exit(0)\n"
        _write_ring2_script(ring2, script)
        hb_path = ring2 / ".heartbeat"

        proc = _start_ring2(ring2, hb_path)
        proc.wait(timeout=10)
        _stop_ring2(proc)

        output = _read_ring2_output(proc, max_lines=10)
        result_lines = output.splitlines()
        assert len(result_lines) == 10
        assert "line 199" in result_lines[-1]

    def test_no_log_path_returns_empty(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        output = _read_ring2_output(proc)
        assert output == ""

    def test_missing_log_file_returns_empty(self, tmp_path):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        proc._log_path = tmp_path / "nonexistent.log"
        output = _read_ring2_output(proc)
        assert output == ""


class TestClassifyFailure:
    """Verify _classify_failure correctly categorises exit reasons."""

    def test_heartbeat_timeout_process_running(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            result = _classify_failure(proc, "")
            assert "heartbeat timeout" in result
            assert "still running" in result
        finally:
            proc.kill()
            proc.wait()

    def test_exit_code_nonzero(self):
        proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(42)"])
        proc.wait()
        result = _classify_failure(proc, "some output")
        assert "exit code 42" in result

    def test_traceback_extracted(self):
        proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(1)"])
        proc.wait()
        output = (
            "Starting up...\n"
            "Traceback (most recent call last):\n"
            '  File "main.py", line 10\n'
            "ZeroDivisionError: division by zero"
        )
        result = _classify_failure(proc, output)
        assert "Traceback" in result
        assert "ZeroDivisionError" in result

    def test_clean_exit_but_heartbeat_lost(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        assert proc.returncode == 0
        result = _classify_failure(proc, "")
        assert "clean exit" in result

    def test_signal_kill(self):
        import signal
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        time.sleep(0.1)
        proc.send_signal(signal.SIGKILL)
        proc.wait()
        result = _classify_failure(proc, "")
        assert "killed by signal" in result
        assert "SIGKILL" in result


class TestLogRotation:
    """Verify Ring 2 .output.log is truncated when too large."""

    def test_large_log_truncated(self, tmp_path):
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        _write_ring2_script(ring2, _BAD_RING2)
        hb_path = ring2 / ".heartbeat"

        # Pre-populate log with 600 lines.
        log_file = ring2 / ".output.log"
        log_file.write_text("\n".join(f"old line {i}" for i in range(600)) + "\n")

        proc = _start_ring2(ring2, hb_path)
        proc.wait(timeout=5)
        _stop_ring2(proc)

        content = log_file.read_text()
        lines = content.strip().splitlines()
        # Should have been truncated to ~200 old lines + any new output.
        # The old lines should only contain the tail (line 400+).
        assert "old line 0" not in content
        assert "old line 599" in content

    def test_small_log_not_truncated(self, tmp_path):
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        _write_ring2_script(ring2, _BAD_RING2)
        hb_path = ring2 / ".heartbeat"

        # Pre-populate log with only 100 lines (under 500 threshold).
        log_file = ring2 / ".output.log"
        log_file.write_text("\n".join(f"old line {i}" for i in range(100)) + "\n")

        proc = _start_ring2(ring2, hb_path)
        proc.wait(timeout=5)
        _stop_ring2(proc)

        content = log_file.read_text()
        # All old lines should still be present.
        assert "old line 0" in content
        assert "old line 99" in content


class TestConfigLoading:
    """Verify the default config.toml is parseable."""

    def test_config_loads(self):
        root = _project_root()
        cfg_path = root / "config" / "config.toml"
        with open(cfg_path, "rb") as f:
            cfg = tomllib.load(f)
        assert "ring0" in cfg
        assert cfg["ring0"]["heartbeat_interval_sec"] > 0
        assert cfg["ring0"]["heartbeat_timeout_sec"] > 0
