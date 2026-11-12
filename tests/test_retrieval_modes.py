import pytest

import phase1_rag
from config import MIN_RELEVANCE


@pytest.fixture
def fake_index(monkeypatch):
    dense = [
        {"id": "a", "text": "A", "source": "s.txt", "title": "A", "score": 0.80},
        {"id": "b", "text": "B", "source": "s.txt", "title": "B", "score": 0.40},
    ]
    lexical = [
        {"id": "c", "text": "C", "source": "s.txt", "title": "C", "score": None, "bm25": 3.0},
        {"id": "a", "text": "A", "source": "s.txt", "title": "A", "score": None, "bm25": 1.0},
    ]
    monkeypatch.setattr(phase1_rag, "rewrite_query", lambda q, h: q)
    monkeypatch.setattr(phase1_rag, "dense_search", lambda q, n: [dict(c) for c in dense])
    monkeypatch.setattr(phase1_rag, "lexical_search", lambda q, n: [dict(c) for c in lexical])


def test_dense_mode_uses_only_embeddings(fake_index):
    chunks = phase1_rag.retrieve("q", mode="dense")
    assert [c["id"] for c in chunks] == ["a", "b"]


def test_bm25_mode_uses_only_lexical_results(fake_index):
    chunks = phase1_rag.retrieve("q", mode="bm25")
    assert [c["id"] for c in chunks] == ["c", "a"]


def test_hybrid_mode_surfaces_a_chunk_only_bm25_found(fake_index):
    ids = [c["id"] for c in phase1_rag.retrieve("q", mode="hybrid")]
    assert ids[0] == "a"          # both retrievers rank it highly
    assert "c" in ids             # dense alone would have missed it


def test_unknown_mode_is_rejected(fake_index):
    with pytest.raises(ValueError):
        phase1_rag.retrieve("q", mode="semantic")


def test_guardrail_ignores_lexical_only_chunks():
    lexical_only = [{"id": "c", "text": "C", "score": None, "bm25": 9.0}]
    # BM25-only mode has no cosine to check, so a match counts...
    assert phase1_rag.is_relevant(lexical_only)
    # ...but in hybrid mode a weak dense score must still trigger the refusal,
    # even when a lexical-only chunk is also in the list.
    mixed = [{"id": "a", "score": MIN_RELEVANCE - 0.1}, {"id": "c", "score": None}]
    assert not phase1_rag.is_relevant(mixed)
    assert phase1_rag.is_relevant([{"id": "a", "score": MIN_RELEVANCE}])
    assert not phase1_rag.is_relevant([])
