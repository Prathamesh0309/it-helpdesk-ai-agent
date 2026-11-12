"""Document loaders: turn .txt, .md and .pdf files into clean plain text.

Text and markdown are read as-is. PDFs are the interesting case: extraction
returns the running page header and "Page X of N" footer on every page, can
glue the header onto the first words of the next page, hyphenates words across
line breaks, and breaks sections mid-sentence at page boundaries. ``clean_pages``
removes that furniture and re-joins the pages so that a section that spans two
pages comes out as one continuous piece of text.
"""
import glob
import os
import re
from collections import Counter

LOADERS = {".txt": "text", ".md": "markdown", ".pdf": "pdf"}


def _common_prefix(pages):
    """First-line header repeated at the start of at least half the pages (and at least two)."""
    firsts = {p.split("\n", 1)[0].strip() for p in pages if p.strip()}
    best = ""
    for candidate in firsts:
        if len(candidate) < 8:
            continue
        hits = sum(1 for p in pages if p.lstrip().startswith(candidate))
        if hits >= max(2, len(pages) / 2) and len(candidate) > len(best):
            best = candidate
    return best


def _footer_pattern(pages):
    """Regex for a last line that repeats once its digits are masked, e.g. 'Page 2 of 5'."""
    masked = Counter()
    for p in pages:
        lines = [ln.strip() for ln in p.strip().splitlines() if ln.strip()]
        if lines:
            masked[re.sub(r"\d+", "#", lines[-1])] += 1
    for shape, count in masked.most_common(1):
        if count >= max(2, len(pages) / 2) and "#" in shape:
            return re.compile("^" + re.escape(shape).replace("\\#", r"\d+") + "$")
    return None


def clean_pages(pages):
    """Join extracted page texts into one string, dropping repeated headers/footers."""
    header = _common_prefix(pages) if len(pages) > 1 else ""
    footer = _footer_pattern(pages) if len(pages) > 1 else None
    cleaned = []
    for page in pages:
        text = page.lstrip()
        if header and text.startswith(header):
            text = text[len(header):].lstrip("\n")
        lines = text.splitlines()
        while lines and not lines[-1].strip():
            lines.pop()
        if footer and lines and footer.match(lines[-1].strip()):
            lines.pop()
        cleaned.append("\n".join(lines))
    text = "\n".join(cleaned)
    # Words hyphenated at a line break ("config-\nuration"), but keep "Wi-Fi" style compounds.
    text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def read_pdf(path):
    from pypdf import PdfReader  # imported lazily: only needed when PDFs are present

    reader = PdfReader(path)
    return clean_pages([page.extract_text() or "" for page in reader.pages])


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_documents(docs_dir):
    """Load every supported file in ``docs_dir`` as {text, source, format}."""
    documents = []
    paths = sorted(
        p for p in glob.glob(os.path.join(docs_dir, "*"))
        if os.path.splitext(p)[1].lower() in LOADERS
    )
    for path in paths:
        ext = os.path.splitext(path)[1].lower()
        text = read_pdf(path) if ext == ".pdf" else read_text(path)
        documents.append({"text": text, "source": os.path.basename(path), "format": LOADERS[ext]})
    return documents
