"""Offline retrieval evaluation. Makes no API calls, so it runs in CI.

Each case has an ``answer_span``: a fact that must appear in the retrieved text
for the question to be answerable. A retrieval is a hit if any of the top-k
chunks contains that span. Because the check is on content rather than on which
chunk id came back, it is fair across chunking strategies.

    python -m evals.retrieval_eval --chunking all
    python -m evals.retrieval_eval --chunking all --split test
    python -m evals.retrieval_eval --chunking structure --min-hit-rate 0.8 --min-mrr 0.6   # CI gate

Case schema (evals/dataset.jsonl): ``type``, ``split`` (dev/test), ``question``,
``ground_truth``, ``answer_span`` (None for questions the knowledge base cannot
answer), optional ``answer_spans`` (multi-fact: every span must be retrieved) and
optional ``history`` (follow-ups; they need query rewriting, so they are live-only).
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

from rag.bm25 import BM25
from rag.chunking import STRATEGIES, chunk_documents, load_documents

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "dataset.jsonl")
DOCS_DIR = os.path.join(os.path.dirname(HERE), "data")


def load_dataset(path=DATASET):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


OUT_OF_SCOPE_TYPES = frozenset({"out_of_scope", "near_miss"})
SPLITS = ("dev", "test")


def case_spans(case):
    """Every fact that must be retrieved for the case to be answerable."""
    if case.get("answer_spans"):
        return list(case["answer_spans"])
    return [case["answer_span"]] if case.get("answer_span") else []


def select_cases(dataset, split=None, offline=False):
    """Filter by split; offline drops follow-ups, which need an LLM to rewrite the query."""
    return [
        c for c in dataset
        if (split is None or c.get("split") == split) and not (offline and c.get("history"))
    ]


def _norm(text):
    return re.sub(r"\s+", " ", text).lower()


def contains_span(text, span):
    return _norm(span) in _norm(text)


def first_hit_rank(chunk_texts, span):
    """1-based rank of the first chunk containing the span, or None."""
    for rank, text in enumerate(chunk_texts, start=1):
        if contains_span(text, span):
            return rank
    return None


def case_rank(chunk_texts, case):
    """Rank at which the case is answerable: the worst first-hit rank over its spans.

    A multi-fact question is only answerable once every span has been retrieved,
    so its rank is that of the last span to appear (None if any never does).
    """
    ranks = [first_hit_rank(chunk_texts, span) for span in case_spans(case)]
    return None if not ranks or None in ranks else max(ranks)


def calibrate_threshold(in_scope_scores, out_of_scope_scores, candidates=None):
    """Pick the relevance threshold that best separates answerable from unanswerable questions.

    Inputs are the best dense similarity per question. A question is answered when
    its score is >= the threshold. Returns the threshold maximizing balanced accuracy
    (mean of the answer rate on in-scope and the refusal rate on out-of-scope),
    breaking ties toward the higher threshold, plus the full sweep.
    """
    if not in_scope_scores or not out_of_scope_scores:
        raise ValueError("need scores from both in-scope and out-of-scope questions")
    if candidates is None:
        pool = sorted(set(in_scope_scores) | set(out_of_scope_scores))
        candidates = [round(x, 4) for x in pool]
    sweep = []
    for t in candidates:
        answered = sum(1 for x in in_scope_scores if x >= t) / len(in_scope_scores)
        refused = sum(1 for x in out_of_scope_scores if x < t) / len(out_of_scope_scores)
        sweep.append({"threshold": t, "answered": round(answered, 3), "refused": round(refused, 3),
                      "balanced": round((answered + refused) / 2, 3)})
    best = max(sweep, key=lambda r: (r["balanced"], r["threshold"]))
    return {"threshold": best["threshold"], "best": best, "sweep": sweep}


def summarize(ranks_by_type, k):
    """ranks_by_type: {type: [rank-or-None, ...]} for in-scope cases only."""
    def stats(ranks):
        n = len(ranks)
        hits = [r for r in ranks if r is not None and r <= k]
        return {
            "n": n,
            f"hit@{k}": round(len(hits) / n, 3) if n else None,
            "mrr": round(sum(1 / r for r in ranks if r is not None and r <= k) / n, 3) if n else None,
        }

    everything = [r for ranks in ranks_by_type.values() for r in ranks]
    return {"overall": stats(everything), "by_type": {t: stats(r) for t, r in sorted(ranks_by_type.items())}}


def evaluate_bm25(dataset, chunks, k=3):
    bm25 = BM25([c["text"] for c in chunks])
    ranks_by_type = defaultdict(list)
    misses = []
    for case in select_cases(dataset, offline=True):
        if not case_spans(case):
            continue  # out-of-scope cases are about refusal, not retrieval
        top = [chunks[i]["text"] for i, _ in bm25.search(case["question"], n=k)]
        rank = case_rank(top, case)
        ranks_by_type[case["type"]].append(rank)
        if rank is None:
            misses.append(case["id"])
    result = summarize(ranks_by_type, k)
    result["misses"] = misses
    result["chunks"] = len(chunks)
    return result


def run(chunking_choices, k=3, split=None):
    dataset = select_cases(load_dataset(), split=split)
    documents = load_documents(DOCS_DIR)
    return {
        strategy: evaluate_bm25(dataset, chunk_documents(documents, strategy), k)
        for strategy in chunking_choices
    }


def render(results, k):
    lines = [f"BM25 retrieval, top-{k}, answerable questions only (follow-ups excluded: they need an LLM)", ""]
    header = f"{'chunking':<10}{'chunks':>8}{'hit@' + str(k):>9}{'MRR':>8}   by question type (hit@{k})"
    lines.append(header)
    for strategy, r in results.items():
        by_type = "  ".join(f"{t}={s[f'hit@{k}']:.2f}" for t, s in r["by_type"].items())
        lines.append(f"{strategy:<10}{r['chunks']:>8}{r['overall'][f'hit@{k}']:>9.3f}{r['overall']['mrr']:>8.3f}   {by_type}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunking", choices=[*STRATEGIES, "all", "both"], default="structure",
                        help="'both' is kept as an alias for 'all'")
    parser.add_argument("--split", choices=SPLITS, help="score only the dev or the test split")
    parser.add_argument("-k", type=int, default=3)
    parser.add_argument("--min-hit-rate", type=float, help="exit 1 if the gated strategy's hit rate is below this")
    parser.add_argument("--min-mrr", type=float, help="exit 1 if the gated strategy's MRR is below this")
    parser.add_argument("--json", help="write full results to this path")
    args = parser.parse_args(argv)

    choices = list(STRATEGIES) if args.chunking in ("all", "both") else [args.chunking]
    results = run(choices, args.k, args.split)
    print(render(results, args.k))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    gate = results.get("structure") or next(iter(results.values()))
    failed = []
    if args.min_hit_rate is not None and gate["overall"][f"hit@{args.k}"] < args.min_hit_rate:
        failed.append(f"hit@{args.k} {gate['overall'][f'hit@{args.k}']} < {args.min_hit_rate}")
    if args.min_mrr is not None and gate["overall"]["mrr"] < args.min_mrr:
        failed.append(f"MRR {gate['overall']['mrr']} < {args.min_mrr}")
    if failed:
        print("\nRegression gate FAILED: " + "; ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
