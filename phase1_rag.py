from functools import lru_cache

import chromadb

from config import (
    CANDIDATE_POOL,
    CHROMA_DIR,
    COLLECTION_NAME,
    GENERATION_MODEL,
    MIN_RELEVANCE,
    RETRIEVAL_MODE,
    RRF_K,
    TOP_K,
    generate,
)
from observability import span
from phase1_ingest import GeminiEmbeddingFunction
from rag.bm25 import BM25
from rag.fusion import fuse_chunks

MODES = ("dense", "bm25", "hybrid")

FALLBACK_ANSWER = (
    "I don't have information about that. Please contact it-support@company.com"
)

_embedder = GeminiEmbeddingFunction()


@lru_cache(maxsize=1)
def load_collection():
    """Open the Chroma collection once per process, not once per question."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    return chroma_client.get_collection(
        name=COLLECTION_NAME, embedding_function=_embedder
    )


def format_history(history):
    if not history:
        return "No previous conversation."
    lines = []
    for turn in history:
        role = "Employee" if turn["role"] == "user" else "Copilot"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


def rewrite_query(query, history):
    """Make a follow-up self-contained so retrieval can find it.

    "It still doesn't work" embeds to nothing useful on its own; combined with
    the history it becomes "VPN still disconnects after changing MTU".
    """
    if not history:
        return query
    prompt = f"""Rewrite the employee's latest message as one standalone search query
for an IT knowledge base, using the conversation for missing context.
Return ONLY the rewritten query, no quotes, no explanation.

CONVERSATION:
{format_history(history[-6:])}

LATEST MESSAGE: {query}

STANDALONE QUERY:"""
    response = generate(GENERATION_MODEL, prompt)
    return (response.text or "").strip() or query


@lru_cache(maxsize=1)
def load_lexical_index():
    """BM25 index built from the chunks already in Chroma, so the two can never disagree."""
    data = load_collection().get(include=["documents", "metadatas"])
    chunks = [
        {"id": i, "text": doc, "source": meta["source"], "title": meta.get("title", ""), "score": None}
        for i, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])
    ]
    return BM25([c["text"] for c in chunks]), chunks


def dense_search(search_query, n):
    collection = load_collection()
    results = collection.query(
        # Embed explicitly as a *query* so the RETRIEVAL_QUERY task type is used.
        query_embeddings=_embedder.embed_query([search_query]),
        n_results=min(n, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "id": chunk_id,
            "text": doc,
            "source": meta["source"],
            "title": meta.get("title", ""),
            # Chroma returns cosine *distance*; similarity = 1 - distance.
            "score": round(1 - dist, 3),
        }
        for chunk_id, doc, meta, dist in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


def lexical_search(search_query, n):
    bm25, chunks = load_lexical_index()
    return [{**chunks[i], "bm25": round(score, 3)} for i, score in bm25.search(search_query, n)]


def retrieve(query, history=None, mode=None):
    """Return the top-K chunks for a query.

    mode: "dense" (embeddings), "bm25" (lexical) or "hybrid" (both, fused with
    Reciprocal Rank Fusion). Defaults to RETRIEVAL_MODE.
    """
    mode = mode or RETRIEVAL_MODE
    if mode not in MODES:
        raise ValueError(f"unknown retrieval mode {mode!r}; choose from {MODES}")

    search_query = rewrite_query(query, history or [])

    with span("retrieve", mode=mode) as s:
        dense = dense_search(search_query, CANDIDATE_POOL) if mode != "bm25" else []
        lexical = lexical_search(search_query, CANDIDATE_POOL) if mode != "dense" else []

        if mode == "dense":
            chunks = dense[:TOP_K]
        elif mode == "bm25":
            chunks = lexical[:TOP_K]
        else:
            chunks = fuse_chunks(dense, lexical, k=RRF_K, top_k=TOP_K)

        s.set(
            sources=[c["source"] for c in chunks],
            top_dense_score=max((c["score"] for c in chunks if c.get("score") is not None), default=None),
            relevant=is_relevant(chunks),
            query_rewritten=search_query != query,
        )
    return chunks


def is_relevant(chunks):
    """The guardrail: is anything we retrieved close enough to answer from?

    Uses dense cosine similarity, which is calibrated (see MIN_RELEVANCE). BM25
    scores have no absolute scale, so chunks only the lexical side found carry
    score=None and are skipped; in BM25-only mode there is no cosine to check, so
    any match counts.
    """
    if not chunks:
        return False
    scores = [c["score"] for c in chunks if c.get("score") is not None]
    return max(scores) >= MIN_RELEVANCE if scores else True


def generate_answer(query, chunks, history=None):
    # Guardrail: if retrieval found nothing close, don't ask the LLM to guess.
    if not is_relevant(chunks):
        return FALLBACK_ANSWER

    context = ""
    for i, chunk in enumerate(chunks, 1):
        context += f"\nSource {i} ({chunk['source']}, relevance: {chunk['score']}):\n"
        context += chunk["text"] + "\n"

    prompt = f"""You are a helpful IT Support Copilot for a tech company.
Answer the employee's question using ONLY the context provided below.
Be conversational and empathetic. Explain WHY each step helps, don't just list them.

CONVERSATION HISTORY:
{format_history(history)}

KNOWLEDGE BASE CONTEXT:
{context}

INSTRUCTIONS:
- Take conversation history into account when answering
- Don't repeat suggestions already made in the conversation history
- If the employee said they already tried something, acknowledge it and move on
- If the answer is not in the context say: "{FALLBACK_ANSWER}"

EMPLOYEE LATEST QUESTION: {query}

ANSWER:"""

    response = generate(GENERATION_MODEL, prompt)
    return response.text


def rag_query(query, history=None):
    print(f"\n{'─'*50}\nQuery: {query}\n{'─'*50}")

    chunks = retrieve(query, history)

    print(f"\nTop {TOP_K} retrieved chunks ({RETRIEVAL_MODE}):")
    for i, c in enumerate(chunks, 1):
        print(f"  {i}. [{c['source']}] score={c.get('score')}")
        print(f"     \"{c['text'][:80]}...\"")

    return generate_answer(query, chunks, history)


if __name__ == "__main__":
    print("=== IT Support Copilot — RAG Query ===")
    print("Type 'quit' to exit\n")

    history = []
    while True:
        user_input = input("Your question: ").strip()

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if not user_input:
            continue

        answer = rag_query(user_input, history)
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer})
        print(f"\nIT Copilot:\n{answer}")
