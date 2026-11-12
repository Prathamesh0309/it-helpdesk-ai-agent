"""Offline retrieval evaluation. Makes no API calls, so it runs in CI.

Each case has an ``answer_span``: a fact that must appear in the retrieved text
for the question to be answerable. A retrieval is a hit if any of the top-k
chunks contains that span. Because the check is on content rather than on which
chunk id came back, it is fair across chunking strategies.

    python -m evals.retrieval_eval --chunking both
    python -m evals.retrieval_eval --min-hit-rate 0.8 --min-mrr 0.6   # CI gate
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
    for case in dataset:
        if not case["answer_span"]:
            continue  # out-of-scope cases are about refusal, not retrieval
        top = [chunks[i]["text"] for i, _ in bm25.search(case["question"], n=k)]
        rank = first_hit_rank(top, case["answer_span"])
        ranks_by_type[case["type"]].append(rank)
        if rank is None:
            misses.append(case["id"])
    result = summarize(ranks_by_type, k)
    result["misses"] = misses
    result["chunks"] = len(chunks)
    return result


def run(chunking_choices, k=3):
    dataset = load_dataset()
    documents = load_documents(DOCS_DIR)
    return {
        strategy: evaluate_bm25(dataset, chunk_documents(documents, strategy), k)
        for strategy in chunking_choices
    }


def render(results, k):
    lines = [f"BM25 retrieval, top-{k}, in-scope questions only", ""]
    header = f"{'chunking':<10}{'chunks':>8}{'hit@' + str(k):>9}{'MRR':>8}   by question type (hit@{k})"
    lines.append(header)
    for strategy, r in results.items():
        by_type = "  ".join(f"{t}={s[f'hit@{k}']:.2f}" for t, s in r["by_type"].items())
        lines.append(f"{strategy:<10}{r['chunks']:>8}{r['overall'][f'hit@{k}']:>9.3f}{r['overall']['mrr']:>8.3f}   {by_type}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunking", choices=[*STRATEGIES, "both"], default="section")
    parser.add_argument("-k", type=int, default=3)
    parser.add_argument("--min-hit-rate", type=float, help="exit 1 if section-chunking hit rate is below this")
    parser.add_argument("--min-mrr", type=float, help="exit 1 if section-chunking MRR is below this")
    parser.add_argument("--json", help="write full results to this path")
    args = parser.parse_args(argv)

    choices = list(STRATEGIES) if args.chunking == "both" else [args.chunking]
    results = run(choices, args.k)
    print(render(results, args.k))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    gate = results.get("section") or next(iter(results.values()))
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
