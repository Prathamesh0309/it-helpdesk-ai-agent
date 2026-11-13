import collections

from evals.retrieval_eval import (
    DOCS_DIR,
    OUT_OF_SCOPE_TYPES,
    calibrate_threshold,
    case_rank,
    case_spans,
    contains_span,
    evaluate_bm25,
    first_hit_rank,
    load_dataset,
    select_cases,
    summarize,
)
from rag.chunking import chunk_documents
from rag.loaders import load_documents

DATASET = load_dataset()
DOCUMENTS = load_documents(DOCS_DIR)
KB_TEXT = " ".join(d["text"] for d in DOCUMENTS)


def test_dataset_is_well_formed():
    ids = [c["id"] for c in DATASET]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    for case in DATASET:
        assert case["question"].strip() and case["ground_truth"].strip()
        assert case["type"] in {"direct", "paraphrase", "lexical_gap", "typo", "multi_fact",
                                "followup", "out_of_scope", "near_miss"}
        assert case["split"] in {"dev", "test"}
        assert (case["type"] in OUT_OF_SCOPE_TYPES) == (case["answer_span"] is None)
        if case["type"] == "multi_fact":
            assert len(case["answer_spans"]) >= 2
        if case["type"] == "followup":
            assert case["history"] and all(t["role"] in {"user", "assistant"} for t in case["history"])


def test_dataset_is_large_and_split_has_both_halves():
    assert len(DATASET) >= 100
    splits = collections.Counter(c["split"] for c in DATASET)
    assert splits["test"] >= 30 and splits["dev"] >= 60


def test_knowledge_base_includes_pdf_markdown_and_text_sources():
    formats = {d["format"] for d in DOCUMENTS}
    assert formats == {"text", "markdown", "pdf"}


def test_every_answer_span_exists_in_the_knowledge_base():
    missing = [c["id"] for c in DATASET for span in case_spans(c) if not contains_span(KB_TEXT, span)]
    assert not missing, f"answer_span not found in data/: {missing}"


def test_dataset_covers_every_question_type():
    counts = collections.Counter(c["type"] for c in DATASET)
    assert all(counts[t] >= 3 for t in ("direct", "paraphrase", "lexical_gap", "typo", "multi_fact",
                                         "followup", "out_of_scope", "near_miss"))


def test_span_matching_ignores_case_and_whitespace():
    assert contains_span("Set  MTU\nsize to 1300", "set mtu size to 1300")
    assert first_hit_rank(["nope", "has the FACT here"], "the fact") == 2
    assert first_hit_rank(["nope"], "the fact") is None


def test_summarize_computes_hit_rate_and_mrr():
    result = summarize({"direct": [1, 2, None, 5]}, k=3)
    assert result["overall"]["hit@3"] == 0.5
    assert result["overall"]["mrr"] == round((1 + 0.5) / 4, 3)


def test_multi_fact_rank_needs_every_span():
    case = {"answer_span": "alpha", "answer_spans": ["alpha", "beta"]}
    assert case_rank(["alpha here", "beta here"], case) == 2
    assert case_rank(["alpha here", "nothing"], case) is None


def test_followups_are_excluded_from_offline_eval():
    assert all(not c.get("history") for c in select_cases(DATASET, offline=True))
    assert any(c.get("history") for c in DATASET)


def test_calibrate_threshold_separates_the_two_groups():
    result = calibrate_threshold([0.70, 0.66, 0.64], [0.55, 0.60, 0.58])
    assert 0.60 < result["threshold"] <= 0.64
    assert result["best"]["answered"] == 1.0 and result["best"]["refused"] == 1.0


def test_calibrate_threshold_needs_both_groups():
    import pytest
    with pytest.raises(ValueError):
        calibrate_threshold([0.7], [])


def test_bm25_baseline_does_not_regress():
    """The CI gate: structure chunking + BM25 must keep clearing these floors (measured 0.846 / 0.785)."""
    result = evaluate_bm25(DATASET, chunk_documents(DOCUMENTS, "structure"), k=3)
    assert result["overall"]["hit@3"] >= 0.80
    assert result["overall"]["mrr"] >= 0.72


def test_heading_aware_chunking_beats_fixed_window_on_bm25():
    structure = evaluate_bm25(DATASET, chunk_documents(DOCUMENTS, "structure"), k=3)["overall"]
    fixed = evaluate_bm25(DATASET, chunk_documents(DOCUMENTS, "fixed"), k=3)["overall"]
    assert structure["hit@3"] > fixed["hit@3"] and structure["mrr"] > fixed["mrr"]


def test_held_out_split_is_not_far_worse_than_dev():
    """A large gap would mean the knowledge base or settings were fitted to the dev split."""
    chunks = chunk_documents(DOCUMENTS, "structure")
    dev = evaluate_bm25(select_cases(DATASET, "dev"), chunks, k=3)["overall"]["hit@3"]
    test = evaluate_bm25(select_cases(DATASET, "test"), chunks, k=3)["overall"]["hit@3"]
    assert abs(dev - test) < 0.15
