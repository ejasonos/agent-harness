from __future__ import annotations

import json
import tempfile
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


_ACTIVE_JOURNAL: ContextVar[TaskJournal | None] = ContextVar(
    "active_task_journal",
    default=None,
)


class TaskJournal:
    """Append-only, run-scoped JSONL evidence stored in the temp directory."""

    def __init__(self, task: str) -> None:
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix="local-agent-",
            suffix=".jsonl",
            delete=False,
        )
        self.path = Path(handle.name)
        handle.close()
        self._lock = threading.Lock()
        self.append("task", content=task)

    def append(self, event: str, **fields: Any) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "event": event,
            **fields,
        }
        serialized = json.dumps(
            entry,
            default=str,
            ensure_ascii=True,
        )
        with self._lock, self.path.open("a", encoding="utf-8") as output:
            output.write(serialized + "\n")
            output.flush()

    def read_text(self) -> str:
        with self._lock:
            return self.path.read_text(encoding="utf-8")

    def close(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def __enter__(self) -> TaskJournal:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


@contextmanager
def use_task_journal(journal: TaskJournal) -> Iterator[None]:
    token = _ACTIVE_JOURNAL.set(journal)
    try:
        yield
    finally:
        _ACTIVE_JOURNAL.reset(token)


def record_task_event(event: str, **fields: Any) -> None:
    journal = _ACTIVE_JOURNAL.get()
    if journal is not None:
        journal.append(event, **fields)


def read_task_journal() -> str:
    journal = _ACTIVE_JOURNAL.get()
    if journal is None:
        return ""
    return journal.read_text()