"""Request tracing with no external service: one JSON line per request.

    with trace("support_request") as t:
        with span("retrieve", mode="hybrid") as s:
            ...
            s.set(top_score=0.71)
        t.set(escalated=True)

Each line in traces/traces.jsonl holds the request's total latency, its attributes
and a flat list of timed spans (with parent ids). ``trace_report.py`` turns a file
of them into latency percentiles, token usage and escalation/refusal rates.

Tracing never breaks the app: write failures are logged and ignored. Set
TRACING=0 to turn it off, TRACE_FILE to move the file, and TRACE_CONTENT=1 to
also record raw query text (off by default, since questions can hold sensitive details).
"""
import contextvars
import hashlib
import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

log = logging.getLogger("itagent.observability")

_current = contextvars.ContextVar("current_trace", default=None)


def _enabled():
    return os.getenv("TRACING", "1") != "0"


def _path():
    return os.getenv("TRACE_FILE", os.path.join("traces", "traces.jsonl"))


def describe_text(text):
    """What to record about user text: length and a short hash, or the text itself if opted in."""
    if os.getenv("TRACE_CONTENT") == "1":
        return {"text": text}
    return {"chars": len(text), "sha": hashlib.sha256(text.encode()).hexdigest()[:10]}


class _Null:
    """Returned when there is nothing to record into, so call sites never need an if."""

    def set(self, **attrs):
        pass


class _SpanHandle:
    def __init__(self, record):
        self._record = record

    def set(self, **attrs):
        self._record["attrs"].update(attrs)


class _Trace:
    def __init__(self, name, attrs):
        self.id = uuid.uuid4().hex[:12]
        self.name = name
        self.attrs = dict(attrs)
        self.spans = []
        self._stack = []
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._t0 = time.perf_counter()

    def set(self, **attrs):
        self.attrs.update(attrs)


@contextmanager
def span(name, **attrs):
    current = _current.get()
    if current is None:
        yield _Null()
        return

    record = {
        "id": len(current.spans),
        "parent": current._stack[-1] if current._stack else None,
        "name": name,
        "attrs": dict(attrs),
        "duration_ms": None,
        "error": None,
    }
    current.spans.append(record)
    current._stack.append(record["id"])
    t0 = time.perf_counter()
    try:
        yield _SpanHandle(record)
    except BaseException as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"[:300]
        raise
    finally:
        record["duration_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        current._stack.pop()


def annotate(**attrs):
    """Set attributes on the active trace from anywhere in the call stack."""
    current = _current.get()
    if current is not None:
        current.set(**attrs)


@contextmanager
def trace(name, **attrs):
    if not _enabled():
        yield _Null()
        return

    current = _Trace(name, attrs)
    token = _current.set(current)
    try:
        yield current
    except BaseException as exc:
        current.attrs["error"] = f"{type(exc).__name__}: {exc}"[:300]
        raise
    finally:
        _current.reset(token)
        _write({
            "trace_id": current.id,
            "name": current.name,
            "started_at": current.started_at,
            "duration_ms": round((time.perf_counter() - current._t0) * 1000, 1),
            "attrs": current.attrs,
            "spans": current.spans,
        })


def _write(record):
    try:
        path = _path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError as exc:  # tracing must never take the app down
        log.warning("could not write trace: %s", exc)
