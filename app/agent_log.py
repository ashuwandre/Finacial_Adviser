"""JSONL run log for supervisor, specialists, tools, and HITL."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG_PATH = Path(__file__).resolve().parent / "agent_runs.jsonl"


def log_event(
    event: str,
    *,
    thread_id: str = "",
    agent: str = "",
    detail: str = "",
    ok: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "thread_id": thread_id,
        "agent": agent,
        "detail": detail,
        "ok": ok,
    }
    if extra:
        record.update(extra)
    try:
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
    except OSError:
        pass
    return record
