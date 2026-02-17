"""Resource monitor — CPU, memory, disk usage (pure stdlib)."""

import os
import platform
import shutil
import subprocess


def get_cpu_percent() -> float:
    """Return estimated CPU usage as a percentage.

    Uses os.getloadavg() (1-min load) normalised by CPU count.
    """
    load1, _, _ = os.getloadavg()
    cpus = os.cpu_count() or 1
    return round(load1 / cpus * 100, 2)


def get_memory_percent() -> float:
    """Return physical memory usage percentage (supports macOS and Linux)."""
    if platform.system() == "Darwin":
        return _get_memory_percent_darwin()
    return _get_memory_percent_linux()


def _get_memory_percent_linux() -> float:
    """Return memory usage percentage via /proc/meminfo."""
    info: dict[str, int] = {}
    with open("/proc/meminfo") as f:
        for line in f:
            parts = line.split(":")
            if len(parts) == 2:
                key = parts[0].strip()
                val = parts[1].strip().split()[0]  # value in kB
                if val.isdigit():
                    info[key] = int(val)
    total = info.get("MemTotal", 0)
    available = info.get("MemAvailable", 0)
    if total == 0:
        return 0.0
    used = total - available
    return round(used / total * 100, 2)


def _get_memory_percent_darwin() -> float:
    """Return memory usage percentage via vm_stat (macOS)."""
    out = subprocess.check_output(["vm_stat"], text=True)
    info: dict[str, int] = {}
    for line in out.splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        val = val.strip().rstrip(".")
        if val.isdigit():
            info[key.strip()] = int(val)
    page_size = 16384  # default on Apple Silicon; 4096 on Intel
    try:
        ps_out = subprocess.check_output(["sysctl", "-n", "hw.pagesize"], text=True)
        page_size = int(ps_out.strip())
    except Exception:
        pass
    total_out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
    total = int(total_out.strip())
    free = info.get("Pages free", 0) * page_size
    inactive = info.get("Pages inactive", 0) * page_size
    speculative = info.get("Pages speculative", 0) * page_size
    purgeable = info.get("Pages purgeable", 0) * page_size
    used = total - free - inactive - speculative - purgeable
    return round(used / total * 100, 2) if total else 0.0


def get_disk_percent(path: str = "/") -> float:
    """Return disk usage percentage for *path*."""
    usage = shutil.disk_usage(path)
    return round(usage.used / usage.total * 100, 2) if usage.total else 0.0


def check_resources(
    max_cpu: float = 90.0,
    max_mem: float = 90.0,
    max_disk: float = 90.0,
) -> tuple[bool, str]:
    """Return (ok, message). ok is True when all metrics are within limits."""
    cpu, mem, disk = get_cpu_percent(), get_memory_percent(), get_disk_percent()
    violations: list[str] = []
    if cpu > max_cpu:
        violations.append(f"CPU {cpu:.1f}% > {max_cpu:.1f}%")
    if mem > max_mem:
        violations.append(f"Memory {mem:.1f}% > {max_mem:.1f}%")
    if disk > max_disk:
        violations.append(f"Disk {disk:.1f}% > {max_disk:.1f}%")
    if violations:
        return False, "Resource limit exceeded: " + "; ".join(violations)
    return True, "All resources within limits"
