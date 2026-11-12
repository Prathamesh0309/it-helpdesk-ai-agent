"""Chunking strategies for the knowledge base.

Two strategies so the choice can be measured instead of assumed:

* ``section``: split on the ``---`` separators the knowledge base is written
  with, so a problem and its fix stay in one chunk. Oversized sections fall
  back to a sliding window.
* ``fixed``: a plain sliding character window over the whole document, the
  usual default. Kept as the baseline for the ablation in ``evals/``.
"""
import glob
import os
import re

MAX_CHUNK_CHARS = 800
CHUNK_OVERLAP = 100
MIN_CHUNK_CHARS = 50

FIXED_WINDOW = 500
FIXED_OVERLAP = 50

STRATEGIES = ("section", "fixed")


def load_documents(docs_dir):
    documents = []
    for filepath in sorted(glob.glob(os.path.join(docs_dir, "*.txt"))):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        documents.append({"text": text, "source": os.path.basename(filepath)})
    return documents


def _window(text, size, overlap):
    if len(text) <= size:
        return [text]
    pieces, start = [], 0
    while start < len(text):
        pieces.append(text[start:start + size])
        start += size - overlap
    return pieces


def _title(text):
    """Human-readable label for a chunk: its first "Issue:" line, else its first real line."""
    lines = [line.strip() for line in text.splitlines()]
    for line in lines:
        if line.startswith("Issue:"):
            return line.removeprefix("Issue:").strip()
    for line in lines:
        if line and not set(line) <= set("=-"):
            return line
    return ""


def _section_chunks(doc):
    # Match only a line that is exactly '---'. Longer '-----' lines under each
    # "Issue:" title are underlines, not separators.
    sections = re.split(r"(?m)^---[ \t]*$", doc["text"])
    for section in sections:
        yield from _window(section.strip(), MAX_CHUNK_CHARS, CHUNK_OVERLAP)


def _fixed_chunks(doc):
    yield from _window(doc["text"].strip(), FIXED_WINDOW, FIXED_OVERLAP)


def chunk_documents(documents, strategy="section"):
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown chunking strategy {strategy!r}; choose from {STRATEGIES}")
    make = _section_chunks if strategy == "section" else _fixed_chunks

    chunks = []
    for doc in documents:
        index = 0
        for piece in make(doc):
            piece = piece.strip()
            if len(piece) < MIN_CHUNK_CHARS:
                continue
            chunks.append({
                "id": f"{doc['source']}_chunk_{index}",
                "text": piece,
                "source": doc["source"],
                "title": _title(piece),
                "strategy": strategy,
            })
            index += 1
    return chunks
