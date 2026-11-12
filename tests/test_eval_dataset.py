import collections

from evals.retrieval_eval import (
    DOCS_DIR,
    contains_span,
    evaluate_bm25,
    first_hit_rank,
    load_dataset,
    summarize,
)
from rag.chunking import chunk_documents, load_documents

DATASET = load_dataset()
DOCUMENTS = load_documents(DOCS_DIR)
KB_TEXT = " ".join(d["text"] for d in DOCUMENTS)


def test_dataset_is_well_formed():
    ids = [c["id"] for c in DATASET]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    for case in DATASET:
        assert case["question"].strip() and case["ground_truth"].strip()
        assert case["type"] in {"direct", "paraphrase", "lexical_gap", "out_of_scope"}
        assert (case["type"] == "out_of_scope") == (case["answer_span"] is None)


def test_every_answer_span_exists_in_the_knowledge_base():
    missing = [c["id"] for c in DATASET if c["answer_span"] and not contains_span(KB_TEXT, c["answer_span"])]
    assert not missing, f"answer_span not found in data/: {missing}"


def test_dataset_covers_every_question_type():
    counts = collections.Counter(c["type"] for c in DATASET)
    assert all(counts[t] >= 3 for t in ("direct", "paraphrase", "lexical_gap", "out_of_scope"))


def test_span_matching_ignores_case_and_whitespace():
    assert contains_span("Set  MTU\nsize to 1300", "set mtu size to 1300")
    assert first_hit_rank(["nope", "has the FACT here"], "the fact") == 2
    assert first_hit_rank(["nope"], "the fact") is None


def test_summarize_computes_hit_rate_and_mrr():
    result = summarize({"direct": [1, 2, None, 5]}, k=3)
    assert result["overall"]["hit@3"] == 0.5
    assert result["overall"]["mrr"] == round((1 + 0.5) / 4, 3)


def test_bm25_baseline_does_not_regress():
    """The CI gate: section chunking + BM25 must keep clearing these floors (measured 0.867 / 0.817)."""
    result = evaluate_bm25(DATASET, chunk_documents(DOCUMENTS, "section"), k=3)
    assert result["overall"]["hit@3"] >= 0.80
    assert result["overall"]["mrr"] >= 0.75


def test_section_chunking_beats_fixed_window_on_bm25():
    section = evaluate_bm25(DATASET, chunk_documents(DOCUMENTS, "section"), k=3)["overall"]
    fixed = evaluate_bm25(DATASET, chunk_documents(DOCUMENTS, "fixed"), k=3)["overall"]
    assert section["hit@3"] > fixed["hit@3"] and section["mrr"] > fixed["mrr"]
