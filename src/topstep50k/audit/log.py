"""Structured audit trail.

Every state mutation in a backtest goes through the AuditLog. Each event
records:
  * the clock's `now()` at the moment of the event (NOT wall-clock)
  * the event type (bar, signal, order, fill, breach, day_close, ...)
  * the structured payload used to reach the decision

The log is append-only. It is the source of truth for "what did the
backtester actually do" — the equity curve and trade list are
derivable from it. This is what makes results reproducible.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator, Protocol


@dataclass(frozen=True)
class AuditEvent:
    ts: datetime  # clock.now() at the time of the event
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.astimezone(timezone.utc).isoformat()
    if hasattr(obj, "_asdict"):
        return obj._asdict()
    if hasattr(obj, "value"):  # Enum
        return obj.value
    raise TypeError(f"Not JSON serialisable: {type(obj).__name__}")


class AuditLog(Protocol):
    def record(self, event: AuditEvent) -> None: ...


class InMemoryAuditLog:
    """Append-only in-memory log. Good for unit tests and small runs."""

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        if self._events and event.ts < self._events[-1].ts:
            raise ValueError(
                f"Audit out of order: prev={self._events[-1].ts} new={event.ts}"
            )
        self._events.append(event)

    def __iter__(self) -> Iterator[AuditEvent]:
        return iter(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def of_kind(self, kind: str) -> list[AuditEvent]:
        return [e for e in self._events if e.kind == kind]


class JsonlAuditLog:
    """Streams events to a JSONL file. Survives crashes, can be tail-followed.

    Note: opens the file in append mode; if you want a clean log for a
    new run, delete the file first.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self._path.open("a", encoding="utf-8")
        self._last_ts: datetime | None = None

    def record(self, event: AuditEvent) -> None:
        if self._last_ts is not None and event.ts < self._last_ts:
            raise ValueError(f"Audit out of order: prev={self._last_ts} new={event.ts}")
        self._last_ts = event.ts
        self._fp.write(
            json.dumps(
                {
                    "ts": event.ts.astimezone(timezone.utc).isoformat(),
                    "kind": event.kind,
                    "payload": event.payload,
                },
                default=_json_default,
            )
        )
        self._fp.write("\n")
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()

    def __enter__(self) -> "JsonlAuditLog":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
