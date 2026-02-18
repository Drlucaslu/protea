"""Schedule tool — lets the LLM create/list/remove/enable/disable scheduled tasks.

Pure stdlib.
"""

from __future__ import annotations

import logging

from ring1.tool_registry import Tool

log = logging.getLogger("protea.tools.schedule")


def make_schedule_tool(scheduled_store) -> Tool:
    """Create a Tool that manages scheduled tasks via *scheduled_store*.

    Args:
        scheduled_store: ScheduledTaskStore instance.
    """

    def _exec_schedule(inp: dict) -> str:
        action = inp.get("action", "")

        if action == "create":
            return _create(inp)
        elif action == "list":
            return _list()
        elif action == "remove":
            return _remove(inp)
        elif action == "enable":
            return _enable(inp)
        elif action == "disable":
            return _disable(inp)
        else:
            return f"Unknown action: {action!r}. Use one of: create, list, remove, enable, disable."

    def _create(inp: dict) -> str:
        name = inp.get("name", "").strip()
        cron_expr = inp.get("cron_expr", "").strip()
        task_text = inp.get("task_text", "").strip()
        schedule_type = inp.get("schedule_type", "cron")

        if not name:
            return "Error: 'name' is required for create."
        if not cron_expr:
            return "Error: 'cron_expr' is required for create."
        if not task_text:
            return "Error: 'task_text' is required for create."
        if schedule_type not in ("cron", "once"):
            return f"Error: schedule_type must be 'cron' or 'once', got {schedule_type!r}."

        # Validate cron expression
        if schedule_type == "cron":
            try:
                from ring0.cron import next_run, describe
                from datetime import datetime
                next_run(cron_expr, datetime.now())
            except Exception as exc:
                return f"Error: invalid cron expression {cron_expr!r} — {exc}"
        else:
            # Validate ISO datetime for once
            try:
                from datetime import datetime
                datetime.fromisoformat(cron_expr)
            except (ValueError, TypeError) as exc:
                return f"Error: invalid ISO datetime {cron_expr!r} — {exc}"

        # Check for duplicate name
        existing = scheduled_store.get_by_name(name)
        if existing:
            return f"Error: a schedule named {name!r} already exists (id={existing['schedule_id']})."

        try:
            schedule_id = scheduled_store.add(
                name=name,
                task_text=task_text,
                cron_expr=cron_expr,
                schedule_type=schedule_type,
            )
        except Exception as exc:
            log.warning("Schedule create failed: %s", exc)
            return f"Error creating schedule: {exc}"

        # Build confirmation with human-readable description
        desc = cron_expr
        if schedule_type == "cron":
            try:
                from ring0.cron import describe as _describe
                desc = _describe(cron_expr)
            except Exception:
                pass

        return (
            f"Schedule created:\n"
            f"  ID: {schedule_id}\n"
            f"  Name: {name}\n"
            f"  Schedule: {desc}\n"
            f"  Type: {schedule_type}\n"
            f"  Task: {task_text}"
        )

    def _list() -> str:
        try:
            tasks = scheduled_store.get_all()
        except Exception as exc:
            return f"Error listing schedules: {exc}"

        if not tasks:
            return "No scheduled tasks."

        lines = [f"Scheduled tasks ({len(tasks)}):"]
        for t in tasks:
            status = "enabled" if t.get("enabled") else "disabled"
            desc = t.get("cron_expr", "")
            if t.get("schedule_type") == "cron":
                try:
                    from ring0.cron import describe as _describe
                    desc = _describe(t["cron_expr"])
                except Exception:
                    pass
            lines.append(
                f"  [{status}] {t['name']} ({t['schedule_id']})\n"
                f"    Schedule: {desc} | Runs: {t.get('run_count', 0)}\n"
                f"    Task: {t['task_text'][:80]}"
            )
        return "\n".join(lines)

    def _remove(inp: dict) -> str:
        name = inp.get("name", "").strip()
        if not name:
            return "Error: 'name' is required for remove."

        task = scheduled_store.get_by_name(name)
        if not task:
            return f"Error: no schedule named {name!r} found."

        try:
            scheduled_store.remove(task["schedule_id"])
        except Exception as exc:
            return f"Error removing schedule: {exc}"
        return f"Schedule {name!r} removed."

    def _enable(inp: dict) -> str:
        name = inp.get("name", "").strip()
        if not name:
            return "Error: 'name' is required for enable."

        task = scheduled_store.get_by_name(name)
        if not task:
            return f"Error: no schedule named {name!r} found."

        try:
            scheduled_store.enable(task["schedule_id"])
        except Exception as exc:
            return f"Error enabling schedule: {exc}"
        return f"Schedule {name!r} enabled."

    def _disable(inp: dict) -> str:
        name = inp.get("name", "").strip()
        if not name:
            return "Error: 'name' is required for disable."

        task = scheduled_store.get_by_name(name)
        if not task:
            return f"Error: no schedule named {name!r} found."

        try:
            scheduled_store.disable(task["schedule_id"])
        except Exception as exc:
            return f"Error disabling schedule: {exc}"
        return f"Schedule {name!r} disabled."

    return Tool(
        name="manage_schedule",
        description=(
            "Create, list, remove, enable, or disable scheduled/recurring tasks. "
            "Use this when the user wants to set up a timer, reminder, cron job, "
            "or any repeating/one-shot task. For cron tasks, provide a standard "
            "5-field cron expression (minute hour day month weekday). "
            "For one-shot tasks, provide an ISO datetime string."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "list", "remove", "enable", "disable"],
                    "description": "The action to perform.",
                },
                "name": {
                    "type": "string",
                    "description": "Name for the schedule (required for create/remove/enable/disable).",
                },
                "cron_expr": {
                    "type": "string",
                    "description": (
                        "Cron expression (5-field: minute hour day month weekday) "
                        "or ISO datetime for one-shot. Required for create."
                    ),
                },
                "schedule_type": {
                    "type": "string",
                    "enum": ["cron", "once"],
                    "description": "Type of schedule. Default: 'cron'.",
                },
                "task_text": {
                    "type": "string",
                    "description": "The task text to execute when the schedule fires. Required for create.",
                },
            },
            "required": ["action"],
        },
        execute=_exec_schedule,
    )
