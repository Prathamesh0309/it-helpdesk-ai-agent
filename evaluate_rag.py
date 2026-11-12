"""Live evaluation of the whole RAG pipeline (calls Gemini, so it needs GEMINI_API_KEY).

Retrieval and generation are scored separately, so a drop in score points at the
half that broke:

1. RETRIEVAL   hit@k and MRR on whether the retrieved text contains the fact needed
               to answer (no LLM involved). Compare modes with --compare.
2. GUARDRAIL   out-of-scope questions must be refused; in-scope ones must not be.
3. GENERATION  faithfulness and answer relevancy from an LLM judge that is a
               different model from the generator.

    python evaluate_rag.py                      # everything, current RETRIEVAL_MODE
    python evaluate_rag.py --compare            # retrieval only: dense vs bm25 vs hybrid
    python evaluate_rag.py --min-faithfulness 0.9 --min-hit-rate 0.85   # gate: exit 1 if below

The offline, API-free half of this (BM25 + chunking ablation) is evals/retrieval_eval.py.
Rebuild the index first (python phase1_ingest.py) after changing the data or CHUNKING.
"""
import argparse
import json
import re
import sys
from collections import defaultdict

from google.genai import types

from config import JUDGE_MODEL, RETRIEVAL_MODE, TOP_K, generate
from evals.retrieval_eval import first_hit_rank, load_dataset, summarize
from observability import trace
from phase1_rag import MODES, generate_answer, is_relevant, retrieve

JUDGE_PROMPT = """You are a strict evaluator for a RAG system. Score two metrics from 0.0 to 1.0.

QUESTION: {question}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

REFERENCE ANSWER (for judging relevancy/correctness only):
{ground_truth}

1. FAITHFULNESS: judge ONLY against the RETRIEVED CONTEXT. Every claim in the answer must be
   supported by the context. 1.0 = fully supported, 0.0 = made up.
   (Politely declining because the context lacks the answer counts as faithful.)
2. ANSWER_RELEVANCY: does the answer correctly address the question, consistent with the reference?

Respond in this EXACT format:
FAITHFULNESS: <score>
ANSWER_RELEVANCY: <score>
REASONING: <one sentence>"""


def _parse_score(text, label):
    m = re.search(rf"{label}\s*:\s*([01](?:\.\d+)?)", text)
    return float(m.group(1)) if m else None


def judge(question, answer, context, ground_truth):
    response = generate(
        JUDGE_MODEL,
        JUDGE_PROMPT.format(
            question=question, context=context, answer=answer, ground_truth=ground_truth
        ),
        config=types.GenerateContentConfig(temperature=0),
    )
    text = response.text or ""
    reasoning = re.search(r"REASONING\s*:\s*(.+)", text)
    return {
        "faithfulness": _parse_score(text, "FAITHFULNESS"),
        "answer_relevancy": _parse_score(text, "ANSWER_RELEVANCY"),
        "reasoning": reasoning.group(1).strip() if reasoning else "N/A",
    }


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else float("nan")


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else float("nan")


def evaluate_retrieval(dataset, mode, k=TOP_K):
    """Retrieval hit@k / MRR and guardrail behaviour for one retrieval mode."""
    ranks_by_type = defaultdict(list)
    refused_ok, false_refusals, misses = [], [], []
    for case in dataset:
        chunks = retrieve(case["question"], mode=mode)
        if case["answer_span"] is None:
            refused_ok.append(0.0 if is_relevant(chunks) else 1.0)
            continue
        rank = first_hit_rank([c["text"] for c in chunks], case["answer_span"])
        ranks_by_type[case["type"]].append(rank)
        false_refusals.append(0.0 if is_relevant(chunks) else 1.0)
        if rank is None:
            misses.append(case["id"])
    result = summarize(ranks_by_type, k)
    result.update(
        out_of_scope_refused=round(mean(refused_ok), 3),
        in_scope_wrongly_refused=round(mean(false_refusals), 3),
        misses=misses,
    )
    return result


def compare_modes(dataset, k=TOP_K):
    results = {mode: evaluate_retrieval(dataset, mode, k) for mode in MODES}
    print(f"\n{'mode':<9}{'hit@' + str(k):>8}{'MRR':>8}{'OOS refused':>13}{'wrongly refused':>17}   by type (hit@{k})")
    for mode, r in results.items():
        by_type = "  ".join(f"{t}={s[f'hit@{k}']:.2f}" for t, s in r["by_type"].items())
        print(f"{mode:<9}{r['overall'][f'hit@{k}']:>8.3f}{r['overall']['mrr']:>8.3f}"
              f"{r['out_of_scope_refused']:>13.3f}{r['in_scope_wrongly_refused']:>17.3f}   {by_type}")
    return results


def run_evaluation(dataset, mode, with_generation=True):
    print(f"=== RAG evaluation: retrieval mode = {mode}, {len(dataset)} cases ===\n")
    retrieval = evaluate_retrieval(dataset, mode)
    k = TOP_K
    o = retrieval["overall"]
    print(f"  Retrieval hit@{k}:        {o[f'hit@{k}']:.3f}   (MRR {o['mrr']:.3f})")
    print(f"  Out-of-scope refused:    {retrieval['out_of_scope_refused']:.3f}")
    print(f"  In-scope wrongly refused:{retrieval['in_scope_wrongly_refused']:>6.3f}")
    if retrieval["misses"]:
        print(f"  Retrieval misses: {', '.join(retrieval['misses'])}")
    summary = {"mode": mode, "retrieval": retrieval}
    if not with_generation:
        return summary

    faith, rel, by_type = [], [], defaultdict(list)
    for i, case in enumerate(dataset, 1):
        with trace("eval_case", case=case["id"], mode=mode):
            chunks = retrieve(case["question"], mode=mode)
            answer = generate_answer(case["question"], chunks)
        context = "\n---\n".join(c["text"] for c in chunks)
        scores = judge(case["question"], answer, context, case["ground_truth"])
        faith.append(scores["faithfulness"])
        rel.append(scores["answer_relevancy"])
        by_type[case["type"]].append(scores["faithfulness"])
        print(f"[{i}/{len(dataset)}] {case['id']:<16} faithfulness={scores['faithfulness']}  "
              f"relevancy={scores['answer_relevancy']}  {scores['reasoning']}")

    summary["generation"] = {
        "faithfulness": round(mean(faith), 3),
        "answer_relevancy": round(mean(rel), 3),
        "faithfulness_by_type": {t: round(mean(v), 3) for t, v in sorted(by_type.items())},
    }
    print(f"\n  Avg faithfulness:        {summary['generation']['faithfulness']:.3f}")
    print(f"  Avg answer relevancy:    {summary['generation']['answer_relevancy']:.3f}")
    print(f"  (judge model: {JUDGE_MODEL}; a small hand-written set and an LLM judge give a rough signal, not a benchmark)")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=MODES, default=RETRIEVAL_MODE)
    parser.add_argument("--compare", action="store_true", help="retrieval only, all three modes side by side")
    parser.add_argument("--retrieval-only", action="store_true", help="skip generation and the judge")
    parser.add_argument("--json", help="write the results to this path")
    parser.add_argument("--min-hit-rate", type=float)
    parser.add_argument("--min-faithfulness", type=float)
    parser.add_argument("--min-relevancy", type=float)
    args = parser.parse_args(argv)

    dataset = load_dataset()
    if args.compare:
        results = compare_modes(dataset)
        if args.json:
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        return 0

    summary = run_evaluation(dataset, args.mode, with_generation=not args.retrieval_only)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    failed = []
    hit = summary["retrieval"]["overall"][f"hit@{TOP_K}"]
    if args.min_hit_rate is not None and hit < args.min_hit_rate:
        failed.append(f"hit@{TOP_K} {hit} < {args.min_hit_rate}")
    gen = summary.get("generation")
    if gen and args.min_faithfulness is not None and gen["faithfulness"] < args.min_faithfulness:
        failed.append(f"faithfulness {gen['faithfulness']} < {args.min_faithfulness}")
    if gen and args.min_relevancy is not None and gen["answer_relevancy"] < args.min_relevancy:
        failed.append(f"answer relevancy {gen['answer_relevancy']} < {args.min_relevancy}")
    if failed:
        print("\nRegression gate FAILED: " + "; ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
