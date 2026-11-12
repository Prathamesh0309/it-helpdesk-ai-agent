"""Reciprocal Rank Fusion: merge ranked lists without comparing their scores.

BM25 scores and cosine similarities live on different scales, so fusing by rank
avoids having to normalise or weight them.
"""


def rrf(rankings, k=60):
    """Fuse several ranked id lists. Returns [(id, fused_score)] best first.

    score(id) = sum over lists of 1 / (k + rank), rank starting at 1.
    Ties break on the order of first appearance, so results are deterministic.
    """
    scores, first_seen = {}, {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(item, len(first_seen))
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))


def fuse_chunks(dense, lexical, k=60, top_k=3):
    """Fuse two ranked lists of chunk dicts (each with an "id") into the top_k chunks.

    A chunk found by both retrievers keeps both scores: "score" (cosine similarity
    from the dense side) and "bm25". A chunk only the lexical side found has
    score=None, and the relevance guardrail ignores it. "rrf" holds the fused score.
    """
    merged = {c["id"]: dict(c) for c in lexical}
    for c in dense:
        merged[c["id"]] = {**merged.get(c["id"], {}), **c}

    fused = rrf([[c["id"] for c in dense], [c["id"] for c in lexical]], k)
    chunks = []
    for chunk_id, score in fused[:top_k]:
        chunk = merged[chunk_id]
        chunk["rrf"] = round(score, 4)
        chunks.append(chunk)
    return chunks
