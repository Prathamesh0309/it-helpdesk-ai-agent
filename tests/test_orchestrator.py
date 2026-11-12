import asyncio
from types import SimpleNamespace

import orchestrator


class FakeSession:
    """Stands in for the MCP ClientSession and records which tools were called."""

    def __init__(self):
        self.calls = []

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        return SimpleNamespace(content=[SimpleNamespace(text=f"{name} ok")])


def run(monkeypatch, *, chunks, decision, query="my laptop is broken"):
    monkeypatch.setattr(orchestrator, "retrieve", lambda q, h: chunks)
    monkeypatch.setattr(orchestrator, "generate_answer", lambda q, c, h: "kb answer")
    monkeypatch.setattr(orchestrator, "decide_actions", lambda q, a, h: dict(decision))
    session = FakeSession()
    answer = asyncio.run(orchestrator.orchestrate(session, query, "Asha", []))
    return session, answer


GOOD = [{"text": "t", "source": "s.txt", "score": 0.9}]
WEAK = [{"text": "t", "source": "s.txt", "score": 0.2}]


def test_no_tools_when_answer_is_sufficient(monkeypatch):
    session, answer = run(monkeypatch, chunks=GOOD, decision={"needs_ticket": False, "needs_slack": False, "reasoning": "ok"})
    assert session.calls == []
    assert answer == "kb answer"


def test_empty_knowledge_base_hit_always_opens_a_ticket(monkeypatch):
    # Even if the LLM says no ticket is needed, a human must pick up a question
    # the knowledge base could not answer.
    session, answer = run(monkeypatch, chunks=WEAK, decision={"needs_ticket": False, "needs_slack": False, "reasoning": "ok"})
    assert [name for name, _ in session.calls] == ["create_ticket"]
    assert session.calls[0][1]["priority"] == "medium"
    assert "Actions Taken" in answer


def test_status_check_does_not_open_a_new_ticket(monkeypatch):
    decision = {"needs_ticket": False, "needs_slack": False, "ticket_id_to_check": "TKT-ABC123", "reasoning": "status"}
    session, _ = run(monkeypatch, chunks=WEAK, decision=decision)
    assert [name for name, _ in session.calls] == ["check_ticket_status"]


def test_urgent_issue_notifies_slack(monkeypatch):
    decision = {"needs_ticket": True, "ticket_priority": "high", "needs_slack": True, "slack_urgency": "urgent", "reasoning": "outage"}
    session, _ = run(monkeypatch, chunks=GOOD, decision=decision)
    assert [name for name, _ in session.calls] == ["create_ticket", "notify_slack"]
    assert session.calls[1][1]["urgency"] == "urgent"


def test_decision_failure_falls_back_to_no_action(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(orchestrator, "generate", broken)
    decision = orchestrator.decide_actions("q", "answer", [])
    assert decision["needs_ticket"] is False and decision["needs_slack"] is False
