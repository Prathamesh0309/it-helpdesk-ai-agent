import pytest

from rag.bm25 import BM25, tokenize
from rag.chunking import chunk_documents
from rag.fusion import fuse_chunks, rrf

DOC = (
    "TITLE\n=====\n\nIssue: VPN keeps disconnecting\n------\nSteps to fix: set MTU size to 1300 on the adapter.\n\n---\n\n"
    "Issue: Printer is offline\n------\nRestart the print spooler service and add the printer again by name.\n"
)


def test_tokenize_stems_and_drops_stopwords():
    assert tokenize("The VPN is dropping") == ["vpn", "drop"]
    assert tokenize("drops") == tokenize("dropped") == tokenize("dropping")
    assert tokenize("minute") == tokenize("minutes")


def test_bm25_ranks_the_matching_document_first():
    bm25 = BM25(["vpn drops often set mtu 1300", "printer offline restart spooler", "password reset self service"])
    hits = bm25.search("printer is offline")
    assert hits[0][0] == 1
    assert all(score > 0 for _, score in hits)


def test_bm25_rare_terms_outweigh_common_ones():
    docs = ["the vpn setting", "the vpn setting", "the vpn setting", "mtu 1300 fix"]
    assert BM25(docs).search("vpn mtu")[0][0] == 3


def test_bm25_returns_nothing_for_unmatched_or_empty_queries():
    bm25 = BM25(["printer offline"])
    assert bm25.search("quantum entanglement") == []
    assert bm25.search("the a an") == []


def test_rrf_rewards_agreement_between_rankings():
    fused = [item for item, _ in rrf([["a", "b", "c"], ["c", "a"]])]
    assert fused[:2] == ["a", "c"]  # in both lists
    assert fused[2] == "b"          # in one list only


def test_rrf_is_deterministic_on_ties():
    assert [i for i, _ in rrf([["x"], ["y"]])] == ["x", "y"]


def test_fuse_chunks_keeps_both_scores_and_marks_lexical_only_chunks():
    dense = [{"id": "a", "text": "A", "score": 0.7}, {"id": "b", "text": "B", "score": 0.6}]
    lexical = [{"id": "b", "text": "B", "score": None, "bm25": 2.0}, {"id": "c", "text": "C", "score": None, "bm25": 1.0}]
    chunks = fuse_chunks(dense, lexical, k=60, top_k=3)
    by_id = {c["id"]: c for c in chunks}
    assert chunks[0]["id"] == "b"                      # ranked by both retrievers
    assert by_id["b"]["score"] == 0.6 and by_id["b"]["bm25"] == 2.0
    assert by_id["c"]["score"] is None                 # lexical-only: no cosine
    assert all("rrf" in c for c in chunks)


def test_section_chunking_keeps_problem_and_fix_together():
    chunks = chunk_documents([{"text": DOC, "source": "kb.txt"}], "section")
    vpn = [c for c in chunks if "VPN" in c["text"]]
    assert len(vpn) == 1 and "MTU size to 1300" in vpn[0]["text"]
    assert vpn[0]["title"] == "VPN keeps disconnecting"
    assert all(c["strategy"] == "section" for c in chunks)


def test_fixed_chunking_cuts_across_sections():
    long_doc = {"text": ("word " * 400), "source": "kb.txt"}
    chunks = chunk_documents([long_doc], "fixed")
    assert len(chunks) > 1 and all(c["strategy"] == "fixed" for c in chunks)
    assert max(len(c["text"]) for c in chunks) <= 500


def test_unknown_chunking_strategy_is_rejected():
    with pytest.raises(ValueError):
        chunk_documents([], "semantic")
