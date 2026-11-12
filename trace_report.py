"""Summarise traces/traces.jsonl: latency, token usage, escalation and refusal rates.

    python trace_report.py [path/to/traces.jsonl]
"""
import json
import math
import os
import sys
from collections import defaultdict


def load(path):
    traces = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    return traces


def percentile(values, p):
    """Nearest-rank percentile; None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def _rate(count, total):
    return round(count / total, 3) if total else None


def summarize(traces):
    n = len(traces)
    latencies = [t["duration_ms"] for t in traces]

    by_span = defaultdict(list)
    tokens = defaultdict(lambda: {"calls": 0, "prompt": 0, "output": 0})
    errors = 0
    llm_calls = 0
    tools = defaultdict(int)

    for t in traces:
        if t["attrs"].get("error"):
            errors += 1
        for s in t["spans"]:
            by_span[s["name"]].append(s["duration_ms"])
            if s["name"] == "llm.generate":
                llm_calls += 1
                bucket = tokens[s["attrs"].get("model", "unknown")]
                bucket["calls"] += 1
                bucket["prompt"] += s["attrs"].get("prompt_tokens", 0) or 0
                bucket["output"] += s["attrs"].get("output_tokens", 0) or 0
            if s["name"].startswith("tool."):
                tools[s["name"].removeprefix("tool.")] += 1

    refused = sum(1 for t in traces if t["attrs"].get("answered_from_kb") is False)
    escalated = sum(1 for t in traces if t["attrs"].get("escalated"))

    return {
        "requests": n,
        "error_rate": _rate(errors, n),
        "latency_ms": {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95), "max": max(latencies, default=None)},
        "spans": {
            name: {"count": len(v), "p50_ms": percentile(v, 50), "p95_ms": percentile(v, 95)}
            for name, v in sorted(by_span.items())
        },
        "llm_calls_per_request": round(llm_calls / n, 2) if n else None,
        "tokens_by_model": dict(tokens),
        "refusal_rate": _rate(refused, n),
        "escalation_rate": _rate(escalated, n),
        "tool_calls": dict(tools),
    }


def render(summary):
    if not summary["requests"]:
        return "No traces found."
    lat = summary["latency_ms"]
    lines = [
        f"Requests: {summary['requests']}    error rate: {summary['error_rate']}",
        f"Latency ms: p50={lat['p50']}  p95={lat['p95']}  max={lat['max']}",
        f"LLM calls per request: {summary['llm_calls_per_request']}",
        f"Refused (no relevant KB content): {summary['refusal_rate']}    escalated to a ticket: {summary['escalation_rate']}",
        f"Tool calls: {summary['tool_calls'] or 'none'}",
        "",
        f"{'span':<18}{'count':>7}{'p50 ms':>10}{'p95 ms':>10}",
    ]
    for name, s in summary["spans"].items():
        lines.append(f"{name:<18}{s['count']:>7}{s['p50_ms']:>10}{s['p95_ms']:>10}")
    lines += ["", f"{'model':<28}{'calls':>7}{'prompt tok':>12}{'output tok':>12}"]
    for model, t in summary["tokens_by_model"].items():
        lines.append(f"{model:<28}{t['calls']:>7}{t['prompt']:>12}{t['output']:>12}")
    return "\n".join(lines)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.getenv("TRACE_FILE", os.path.join("traces", "traces.jsonl"))
    if not os.path.exists(path):
        raise SystemExit(f"No trace file at {path}. Run the agent first (tracing is on by default).")
    print(render(summarize(load(path))))
