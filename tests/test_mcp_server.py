import json

import pytest

import mcp_server


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_server, "TICKETS_FILE", str(tmp_path / "tickets.json"))


def test_create_then_check_ticket():
    created = json.loads(mcp_server.create_ticket("Asha", "VPN keeps dropping", "high"))
    assert created["success"] and created["ticket_id"].startswith("TKT-")

    status = json.loads(mcp_server.check_ticket_status(created["ticket_id"]))
    assert status["success"]
    assert status["status"] == "open"
    assert status["issue"] == "VPN keeps dropping"


def test_invalid_priority_falls_back_to_medium():
    created = json.loads(mcp_server.create_ticket("Asha", "printer", "urgent!!"))
    assert created["details"]["priority"] == "medium"


def test_ticket_lookup_ignores_case_and_whitespace():
    created = json.loads(mcp_server.create_ticket("Asha", "monitor", "low"))
    lookup = json.loads(mcp_server.check_ticket_status(f"  {created['ticket_id'].lower()} "))
    assert lookup["success"]


def test_unknown_ticket_reports_not_found():
    result = json.loads(mcp_server.check_ticket_status("TKT-NOPE00"))
    assert result["success"] is False


def test_notify_slack_marks_urgency():
    result = json.loads(mcp_server.notify_slack("it-help", "server down", "urgent"))
    assert result["success"]
    assert "URGENT" in result["notification"]["message"]
