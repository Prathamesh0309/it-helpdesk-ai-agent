"""Shared settings and the single Gemini client used by every module."""
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors

from observability import span

load_dotenv()

_api_key = os.getenv("GEMINI_API_KEY")
if not _api_key:
    raise RuntimeError("GEMINI_API_KEY is not set. Add it to a .env file (see README).")

gemini_client = genai.Client(api_key=_api_key)

# Models
EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-3.1-flash-lite"
# The judge is deliberately a different model from the generator, so the system
# is not grading its own homework.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-3.8-flash")

# Vector store
DOCS_DIR = "./data"
CHROMA_DIR = "./db"
COLLECTION_NAME = "it_knowledge_base"

# Retrieval
# dense  = embeddings only (Gemini + ChromaDB)
# bm25   = lexical only (rag/bm25.py)
# hybrid = both, merged with Reciprocal Rank Fusion
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")
# section = split on the knowledge base's own '---' boundaries; fixed = sliding window
CHUNKING = os.getenv("CHUNKING", "section")
RRF_K = 60
# How many candidates each retriever contributes before fusion.
CANDIDATE_POOL = 50
TOP_K = 3
# Cosine similarity below this means "nothing relevant found"; we skip the LLM
# and escalate instead of letting it improvise.
# Calibrated empirically on the original two-document KB: in-scope questions
# scored 0.66-0.74, out-of-scope ones 0.51-0.60, so 0.62 sat in the gap. The KB has
# grown since: re-check with `python evaluate_rag.py --compare` and adjust.
MIN_RELEVANCE = 0.62


def generate(model, contents, config=None, retries=6):
    """generate_content with exponential backoff on transient errors.

    Retries 5xx (server overload) and 429 (rate limit / quota burst). Other 4xx
    errors (bad model name, blocked request) are not fixed by waiting, so they
    propagate immediately. Each call is a traced span with latency and token use.
    """
    with span("llm.generate", model=model) as s:
        for attempt in range(retries + 1):
            try:
                response = gemini_client.models.generate_content(
                    model=model, contents=contents, config=config
                )
            except (errors.ServerError, errors.ClientError) as e:
                transient = isinstance(e, errors.ServerError) or getattr(e, "code", None) == 429
                if not transient or attempt == retries:
                    raise
                s.set(retries=attempt + 1)
                time.sleep(min(2 ** (attempt + 1), 30))
                continue

            usage = getattr(response, "usage_metadata", None)
            if usage is not None:
                s.set(
                    prompt_tokens=getattr(usage, "prompt_token_count", None),
                    output_tokens=getattr(usage, "candidates_token_count", None),
                )
            return response
