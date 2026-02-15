"""Import OpenClaw SKILL.md files into Protea.

Usage:
    python run_import.py --dir /path/to/openclaw/skills/skills/
    python run_import.py --file /path/to/SKILL.md
    python run_import.py --url https://raw.githubusercontent.com/.../SKILL.md
    python run_import.py --dir /path/to/skills/ --dry-run
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("protea.run_import")

    parser = argparse.ArgumentParser(description="Import SKILL.md files into Protea")
    parser.add_argument("--dir", type=str, help="Recursively import from directory")
    parser.add_argument("--file", type=str, help="Import a single SKILL.md file")
    parser.add_argument("--url", type=str, help="Import from a URL")
    parser.add_argument("--db", type=str, default="data/protea.db",
                        help="SkillStore database path (default: data/protea.db)")
    parser.add_argument("--registry", type=str, default=None,
                        help="Registry URL to also publish to (optional)")
    parser.add_argument("--node-id", type=str, default="clawhub",
                        help="Registry node ID (default: clawhub)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse only, do not import")

    args = parser.parse_args()

    if not args.dir and not args.file and not args.url:
        parser.error("At least one of --dir, --file, or --url is required")

    # Create stores (unless dry-run).
    skill_store = None
    registry_client = None

    if not args.dry_run:
        try:
            from ring0.skill_store import SkillStore
            db_path = pathlib.Path(args.db)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            skill_store = SkillStore(db_path)
            log.info("SkillStore: %s", db_path)
        except Exception as exc:
            log.warning("Could not create SkillStore: %s", exc)

        if args.registry:
            try:
                from ring1.registry_client import RegistryClient
                registry_client = RegistryClient(args.registry, args.node_id)
                log.info("Registry: %s (node_id=%s)", args.registry, args.node_id)
            except Exception as exc:
                log.warning("Could not create RegistryClient: %s", exc)

    from registry.importer import SkillImporter
    importer = SkillImporter(
        skill_store=skill_store,
        registry_client=registry_client,
    )

    if args.file:
        path = pathlib.Path(args.file)
        if args.dry_run:
            text = path.read_text(encoding="utf-8", errors="replace")
            skill = importer.parse_skill_md(text)
            if skill:
                log.info("[DRY-RUN] Parsed: %s — %s", skill["name"], skill["description"])
                log.info("  tags: %s", skill.get("tags", []))
                log.info("  template: %s...", skill["prompt_template"][:100])
            else:
                log.warning("[DRY-RUN] Could not parse: %s", path)
        else:
            result = importer.import_file(path)
            if result:
                log.info("Imported: %s", result["name"])
            else:
                log.warning("Skipped: %s", path)

    if args.dir:
        dir_path = pathlib.Path(args.dir)
        if args.dry_run:
            count = 0
            for md_path in sorted(dir_path.rglob("SKILL.md")):
                text = md_path.read_text(encoding="utf-8", errors="replace")
                skill = importer.parse_skill_md(text)
                if skill:
                    count += 1
                    log.info("[DRY-RUN] %3d. %s — %s", count, skill["name"], skill["description"][:60])
                else:
                    log.warning("[DRY-RUN] Could not parse: %s", md_path)
            log.info("[DRY-RUN] Total parseable: %d", count)
        else:
            result = importer.import_directory(dir_path)
            log.info("Import complete: %d imported, %d skipped, %d errors",
                     result.imported, result.skipped, len(result.errors))
            for err in result.errors:
                log.error("  %s", err)

    if args.url:
        if args.dry_run:
            import urllib.request
            req = urllib.request.Request(args.url, method="GET")
            with urllib.request.urlopen(req, timeout=15) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            skill = importer.parse_skill_md(text)
            if skill:
                log.info("[DRY-RUN] Parsed: %s — %s", skill["name"], skill["description"])
            else:
                log.warning("[DRY-RUN] Could not parse URL: %s", args.url)
        else:
            result = importer.import_from_url(args.url)
            if result:
                log.info("Imported: %s", result["name"])
            else:
                log.warning("Skipped URL: %s", args.url)


if __name__ == "__main__":
    main()
