import json
from types import SimpleNamespace

import pytest
from google.genai import errors

import config
from observability import trace


class FakeModels:
    def __init__(self, failures):
        self.failures = list(failures)
        self.calls = 0

    def generate_content(self, model, contents, config=None):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        usage = SimpleNamespace(prompt_token_count=11, candidates_token_count=7)
        return SimpleNamespace(text="ok", usage_metadata=usage)


def install(monkeypatch, failures):
    models = FakeModels(failures)
    monkeypatch.setattr(config, "gemini_client", SimpleNamespace(models=models))
    monkeypatch.setattr(config.time, "sleep", lambda s: None)
    return models


def api_error(cls, code):
    return cls(code, {"error": {"code": code, "message": "x", "status": "X"}})


def test_retries_server_errors_then_succeeds(monkeypatch):
    models = install(monkeypatch, [api_error(errors.ServerError, 503)])
    assert config.generate("m", "p").text == "ok"
    assert models.calls == 2


def test_retries_rate_limits(monkeypatch):
    models = install(monkeypatch, [api_error(errors.ClientError, 429), api_error(errors.ClientError, 429)])
    assert config.generate("m", "p").text == "ok"
    assert models.calls == 3


def test_does_not_retry_other_client_errors(monkeypatch):
    models = install(monkeypatch, [api_error(errors.ClientError, 400)])
    with pytest.raises(errors.ClientError):
        config.generate("m", "p")
    assert models.calls == 1


def test_gives_up_after_the_retry_budget(monkeypatch):
    models = install(monkeypatch, [api_error(errors.ServerError, 503)] * 5)
    with pytest.raises(errors.ServerError):
        config.generate("m", "p", retries=2)
    assert models.calls == 3


def test_generate_span_records_model_tokens_and_retries(monkeypatch, tmp_path):
    install(monkeypatch, [api_error(errors.ServerError, 503)])
    path = tmp_path / "t.jsonl"
    monkeypatch.setenv("TRACING", "1")
    monkeypatch.setenv("TRACE_FILE", str(path))
    with trace("request"):
        config.generate("gemini-x", "p")
    (record,) = [json.loads(line) for line in path.read_text().splitlines()]
    (span,) = record["spans"]
    assert span["name"] == "llm.generate"
    assert span["attrs"] == {"model": "gemini-x", "retries": 1, "prompt_tokens": 11, "output_tokens": 7}
