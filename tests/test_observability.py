import json

import pytest

import observability
from observability import annotate, describe_text, span, trace
from trace_report import percentile, render, summarize


@pytest.fixture
def trace_file(tmp_path, monkeypatch):
    path = tmp_path / "traces.jsonl"
    monkeypatch.setenv("TRACING", "1")
    monkeypatch.setenv("TRACE_FILE", str(path))
    return path


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_trace_writes_one_json_line_with_nested_spans(trace_file):
    with trace("request", turn=1):
        with span("outer"):
            with span("inner", model="m") as s:
                s.set(prompt_tokens=10)
        annotate(escalated=True)

    (record,) = read(trace_file)
    assert record["name"] == "request" and record["attrs"] == {"turn": 1, "escalated": True}
    outer, inner = record["spans"]
    assert inner["parent"] == outer["id"] and outer["parent"] is None
    assert inner["attrs"] == {"model": "m", "prompt_tokens": 10}
    assert record["duration_ms"] >= outer["duration_ms"] >= inner["duration_ms"] >= 0


def test_span_records_errors_and_still_raises(trace_file):
    with pytest.raises(RuntimeError):
        with trace("request"):
            with span("llm.generate"):
                raise RuntimeError("boom")
    (record,) = read(trace_file)
    assert "RuntimeError: boom" in record["spans"][0]["error"]
    assert "boom" in record["attrs"]["error"]


def test_tracing_can_be_switched_off(trace_file, monkeypatch):
    monkeypatch.setenv("TRACING", "0")
    with trace("request"):
        with span("x") as s:
            s.set(a=1)  # must not fail
    assert not trace_file.exists()


def test_spans_outside_a_trace_are_harmless(trace_file):
    with span("orphan") as s:
        s.set(a=1)
    annotate(b=2)
    assert not trace_file.exists()


def test_unwritable_trace_path_does_not_break_the_app(monkeypatch, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setenv("TRACING", "1")
    monkeypatch.setenv("TRACE_FILE", str(blocker / "sub" / "t.jsonl"))
    with trace("request"):
        pass


def test_query_text_is_not_recorded_unless_opted_in(monkeypatch):
    monkeypatch.delenv("TRACE_CONTENT", raising=False)
    assert "text" not in describe_text("my password is hunter2")
    monkeypatch.setenv("TRACE_CONTENT", "1")
    assert describe_text("hello") == {"text": "hello"}


def test_percentile_nearest_rank():
    assert percentile([], 50) is None
    assert percentile([1, 2, 3, 4], 50) == 2
    assert percentile([1, 2, 3, 4], 95) == 4


def test_summary_reports_latency_tokens_and_rates(trace_file):
    for refused, escalated in [(False, True), (True, False)]:
        with trace("support_request") as t:
            with span("llm.generate", model="gemini-x") as s:
                s.set(prompt_tokens=100, output_tokens=20)
            with span("tool.create_ticket"):
                pass
            t.set(answered_from_kb=not refused, escalated=escalated)

    from trace_report import load

    summary = summarize(load(trace_file))
    assert summary["requests"] == 2
    assert summary["refusal_rate"] == 0.5 and summary["escalation_rate"] == 0.5
    assert summary["tokens_by_model"]["gemini-x"] == {"calls": 2, "prompt": 200, "output": 40}
    assert summary["tool_calls"] == {"create_ticket": 2}
    assert summary["llm_calls_per_request"] == 1.0
    assert "p95" in render(summary)
