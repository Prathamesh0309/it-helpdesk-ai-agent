import phase1_rag
from config import MIN_RELEVANCE
from evaluate_rag import _parse_score


def chunk(score):
    return {"text": "t", "source": "s.txt", "score": score}


def test_is_relevant_threshold():
    assert phase1_rag.is_relevant([chunk(MIN_RELEVANCE)])
    assert not phase1_rag.is_relevant([chunk(MIN_RELEVANCE - 0.01)])
    assert not phase1_rag.is_relevant([])


def test_irrelevant_retrieval_never_reaches_the_llm(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("the LLM must not be called when retrieval found nothing relevant")

    monkeypatch.setattr(phase1_rag, "generate", boom)
    answer = phase1_rag.generate_answer("parental leave policy?", [chunk(0.3)])
    assert answer == phase1_rag.FALLBACK_ANSWER


def test_format_history():
    assert phase1_rag.format_history([]) == "No previous conversation."
    text = phase1_rag.format_history(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    )
    assert text == "Employee: hi\nCopilot: hello"


def test_parse_judge_scores():
    text = "FAITHFULNESS: 0.9\nANSWER_RELEVANCY: 1\nREASONING: fine"
    assert _parse_score(text, "FAITHFULNESS") == 0.9
    assert _parse_score(text, "ANSWER_RELEVANCY") == 1.0
    assert _parse_score("no scores here", "FAITHFULNESS") is None
