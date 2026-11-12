"""Chunking strategies for the knowledge base.

Three strategies so the choice can be measured instead of assumed:

* ``section``: split on the ``---`` separators the knowledge base is written
  with, so a problem and its fix stay in one chunk. Oversized sections fall
  back to a sliding window.
* ``fixed``: a plain sliding character window over the whole document, the
  usual default. Kept as the baseline for the ablation in ``evals/``.
* ``structure``: split on the document's own headings (markdown ``##``, numbered
  PDF headings, ``---`` blocks in text files), pack sub-sections up to the size
  limit, and prefix every chunk with a contextual header such as
  ``Leave and Time Off Policy > Sick Leave`` so a chunk still says where it
  came from once it is cut out of the document.
"""
import re

from rag.loaders import load_documents  # noqa: F401  (re-exported for callers)

MAX_CHUNK_CHARS = 800
CHUNK_OVERLAP = 100
MIN_CHUNK_CHARS = 50

FIXED_WINDOW = 500
FIXED_OVERLAP = 50

STRATEGIES = ("section", "fixed", "structure")


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
        for piece in _window(section.strip(), MAX_CHUNK_CHARS, CHUNK_OVERLAP):
            yield piece, None


def _fixed_chunks(doc):
    for piece in _window(doc["text"].strip(), FIXED_WINDOW, FIXED_OVERLAP):
        yield piece, None


# --- structure-aware chunking ------------------------------------------------

_MD_H1 = re.compile(r"^# +(.+)$")
_MD_H2 = re.compile(r"^## +(.+)$")
_NUM_H1 = re.compile(r"^(\d{1,2})\. +([A-Z].{1,80})$")
_NUM_H2 = re.compile(r"^\d{1,2}\.\d{1,2} +([A-Z].{1,80})$")
_BANNER_UNDERLINE = re.compile(r"^={3,}\s*$")


def _split_units(lines, is_sub):
    """Group lines into units, starting a new unit at every sub-heading."""
    units, current = [], []
    for line in lines:
        if is_sub(line) and current:
            units.append("\n".join(current).strip())
            current = []
        current.append(line)
    if current:
        units.append("\n".join(current).strip())
    return [u for u in units if u]


def _pack(units, limit):
    """Greedily merge consecutive units up to ``limit``; windows any single oversized unit."""
    packed, buf = [], ""
    for unit in units:
        if buf and len(buf) + 1 + len(unit) > limit:
            packed.append(buf)
            buf = ""
        if len(unit) > limit:
            if buf:
                packed.append(buf)
                buf = ""
            packed.extend(_window(unit, limit, CHUNK_OVERLAP))
        else:
            buf = f"{buf}\n{unit}" if buf else unit
    if buf:
        packed.append(buf)
    return packed


def _heading_chunks(doc, h1, h2_match, doc_title):
    """Split a heading-structured document into (text, title) pairs with context headers."""
    sections, title, body = [], None, []
    for line in doc["text"].splitlines():
        match = h1(line.strip())
        if match:
            if title is not None:
                sections.append((title, body))
            title, body = match, []
        elif title is not None:
            body.append(line.rstrip())
    if title is not None:
        sections.append((title, body))

    for title, body in sections:
        units = _split_units(body, lambda ln: h2_match(ln.strip()))
        for piece in _pack(units, MAX_CHUNK_CHARS):
            header = f"{doc_title} > {title}" if doc_title else title
            yield f"{header}\n{piece}", title


def _banner_chunks(doc):
    """``---``-separated text files: carry the nearest "TITLE\n=====" banner into each chunk."""
    banner = ""
    for section in re.split(r"(?m)^---[ \t]*$", doc["text"]):
        section = section.strip()
        lines = section.splitlines()
        for i, line in enumerate(lines[:-1]):
            if _BANNER_UNDERLINE.match(lines[i + 1]) and line.strip():
                banner = line.strip()
        for piece in _window(section, MAX_CHUNK_CHARS, CHUNK_OVERLAP):
            has_banner = banner and banner.lower() in piece.lower()
            yield (piece if has_banner or not banner else f"{banner}\n{piece}"), None


def _structure_chunks(doc):
    lines = [ln.strip() for ln in doc["text"].splitlines() if ln.strip()]
    if doc.get("format") == "markdown":
        title = next((m.group(1) for m in map(_MD_H1.match, lines) if m), "")
        yield from _heading_chunks(
            doc, lambda ln: (m.group(1) if (m := _MD_H2.match(ln)) else None),
            lambda ln: False, title)
    elif doc.get("format") == "pdf":
        title = lines[0] if lines else ""
        yield from _heading_chunks(
            doc, lambda ln: (f"{m.group(2)}" if (m := _NUM_H1.match(ln)) else None),
            lambda ln: bool(_NUM_H2.match(ln)), title)
    else:
        yield from _banner_chunks(doc)


_STRATEGY_FUNCS = {
    "section": _section_chunks,
    "fixed": _fixed_chunks,
    "structure": _structure_chunks,
}


def chunk_documents(documents, strategy="section"):
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown chunking strategy {strategy!r}; choose from {STRATEGIES}")
    make = _STRATEGY_FUNCS[strategy]

    chunks = []
    for doc in documents:
        index = 0
        for piece, title in make(doc):
            piece = piece.strip()
            if len(piece) < MIN_CHUNK_CHARS:
                continue
            chunks.append({
                "id": f"{doc['source']}_chunk_{index}",
                "text": piece,
                "source": doc["source"],
                "title": title or _title(piece),
                "strategy": strategy,
            })
            index += 1
    return chunks
