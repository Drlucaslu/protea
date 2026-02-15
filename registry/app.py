"""Standalone Skill Registry process entry point.

Loads [registry] config from config.toml, creates a RegistryStore + Server,
and runs until interrupted.
"""

from __future__ import annotations

import logging
import pathlib
import signal
import sys
import tomllib

from registry.server import RegistryServer
from registry.store import RegistryStore

log = logging.getLogger("protea.registry_app")


def run(config_path: str | None = None) -> None:
    """Start the Skill Registry service (blocking)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    project_root = pathlib.Path(__file__).resolve().parent.parent
    cfg_path = pathlib.Path(config_path) if config_path else project_root / "config" / "config.toml"

    with open(cfg_path, "rb") as f:
        cfg = tomllib.load(f)

    reg_cfg = cfg.get("registry", {})
    host = reg_cfg.get("host", "127.0.0.1")
    port = reg_cfg.get("port", 8761)
    db_path = project_root / reg_cfg.get("db_path", "data/registry.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    store = RegistryStore(db_path)
    server = RegistryServer(store, host=host, port=port)

    def _shutdown(signum, frame):
        log.info("Received signal %s — shutting down", signum)
        server.stop()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    log.info("Skill Registry starting (db=%s)", db_path)
    server.run()
