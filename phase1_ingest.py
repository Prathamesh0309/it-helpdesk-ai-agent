import chromadb
from google.genai import types

from config import (
    CHROMA_DIR,
    CHUNKING,
    COLLECTION_NAME,
    DOCS_DIR,
    EMBEDDING_MODEL,
    gemini_client,
)
from observability import span
from rag.chunking import chunk_documents as chunk_all
from rag.chunking import load_documents as load_docs

EMBED_BATCH_SIZE = 50


def load_documents():
    documents = load_docs(DOCS_DIR)
    for doc in documents:
        print(f"Loaded: {doc['source']}")
    return documents


def chunk_documents(documents, strategy=CHUNKING):
    """Chunk with the configured strategy (see rag/chunking.py for the options)."""
    chunks = chunk_all(documents, strategy)
    for source in dict.fromkeys(c["source"] for c in chunks):
        print(f"Created {sum(c['source'] == source for c in chunks)} chunks from {source} ({strategy})")
    return chunks


class GeminiEmbeddingFunction:
    """Chroma-compatible embedding function.

    Gemini embeddings are asymmetric: documents and queries should be embedded
    with different task types (RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY) so a short
    question lands near the passage that answers it.
    """

    def name(self):
        return "GeminiEmbeddingFunction"

    def _embed(self, texts, task_type):
        embeddings = []
        # One API call per batch instead of one per chunk.
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = list(texts[i:i + EMBED_BATCH_SIZE])
            with span("embed", model=EMBEDDING_MODEL, task_type=task_type, texts=len(batch)):
                result = gemini_client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
            embeddings.extend(e.values for e in result.embeddings)
        return embeddings

    # Chroma calls this when it has to embed on its own; treat as documents.
    def __call__(self, input):
        return self._embed(input, "RETRIEVAL_DOCUMENT")

    def embed_documents(self, input):
        return self._embed(input, "RETRIEVAL_DOCUMENT")

    def embed_query(self, input):
        return self._embed(input, "RETRIEVAL_QUERY")


def store_in_chromadb(chunks):
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Full rebuild: simplest way to guarantee the index matches data/ exactly.
    existing = [c.name for c in chroma_client.list_collections()]
    if COLLECTION_NAME in existing:
        chroma_client.delete_collection(COLLECTION_NAME)

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=GeminiEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        documents=[c["text"] for c in chunks],
        ids=[c["id"] for c in chunks],
        metadatas=[{"source": c["source"], "title": c["title"], "strategy": c["strategy"]} for c in chunks],
    )
    print(f"Stored {collection.count()} chunks in ChromaDB")


if __name__ == "__main__":
    print("=== Phase 1: Ingestion ===")

    print("\n[1/3] Loading documents...")
    documents = load_documents()
    if not documents:
        raise SystemExit(f"No .txt files found in {DOCS_DIR}")

    print("\n[2/3] Chunking...")
    chunks = chunk_documents(documents)

    print("\n[3/3] Storing in ChromaDB...")
    store_in_chromadb(chunks)

    print("\nDone! Run phase1_rag.py to query.")
