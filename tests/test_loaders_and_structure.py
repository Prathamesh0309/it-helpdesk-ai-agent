import os

import pytest

from rag.chunking import STRATEGIES, chunk_documents
from rag.loaders import clean_pages, load_documents

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
PAGES = [
    "Acme | Handbook | Confidential\nTitle\n1. First\nBody one continues within\nPage 1 of 2",
    "Acme | Handbook | Confidential12 months of the text.\n2. Second\nMore text\nPage 2 of 2",
]


def test_clean_pages_strips_header_footer_and_rejoins_across_pages():
    text = clean_pages(PAGES)
    assert "Confidential" not in text and "Page " not in text
    assert "continues within\n12 months of the text." in text  # sentence split by the page break is whole again


def test_clean_pages_dehyphenates_but_keeps_compounds():
    text = clean_pages(["config-\nuration of Wi-Fi"])
    assert "configuration" in text and "Wi-Fi" in text


def test_clean_pages_leaves_a_single_page_alone():
    assert clean_pages(["Only page\nPage 1 of 1"]) == "Only page\nPage 1 of 1"


def test_load_documents_reads_all_three_formats():
    docs = {d["source"]: d for d in load_documents(DATA)}
    assert docs["vpn_and_accounts.txt"]["format"] == "text"
    assert docs["dev_tools_and_access.md"]["format"] == "markdown"
    pdf = docs["hr_leave_policy.pdf"]
    assert pdf["format"] == "pdf" and "Sick Leave" in pdf["text"]
    assert "Page 1 of" not in pdf["text"] and "Confidential" not in pdf["text"]


def test_pdf_section_spanning_a_page_break_is_one_chunk():
    docs = [d for d in load_documents(DATA) if d["source"] == "hr_leave_policy.pdf"]
    chunks = chunk_documents(docs, "structure")
    parental = next(c for c in chunks if c["title"] == "Parental Leave")
    assert "16 weeks" in parental["text"] and "12 weeks" in parental["text"] and "phased return" in parental["text"]


def test_structure_chunks_carry_a_contextual_header():
    docs = [d for d in load_documents(DATA) if d["source"] == "hr_leave_policy.pdf"]
    chunks = chunk_documents(docs, "structure")
    assert all(c["text"].startswith("Leave and Time Off Policy > ") for c in chunks)
    assert {"Sick Leave", "Parental Leave", "Bereavement Leave"} <= {c["title"] for c in chunks}


def test_markdown_splits_on_h2_headings():
    docs = [d for d in load_documents(DATA) if d["source"] == "dev_tools_and_access.md"]
    titles = [c["title"] for c in chunk_documents(docs, "structure")]
    assert "SSH keys and commit signing" in titles and len(titles) >= 5


def test_text_files_get_their_banner_as_context():
    docs = [d for d in load_documents(DATA) if d["source"] == "vpn_and_accounts.txt"]
    chunks = chunk_documents(docs, "structure")
    assert any(c["text"].startswith("VPN TROUBLESHOOTING") for c in chunks)
    assert any(c["text"].startswith("PASSWORD & ACCOUNT ISSUES") for c in chunks)


def test_section_chunking_has_no_separators_in_pdfs_so_it_falls_back_to_windows():
    docs = [d for d in load_documents(DATA) if d["source"] == "hr_leave_policy.pdf"]
    section = chunk_documents(docs, "section")
    assert all(len(c["text"]) <= 800 for c in section)
    assert not any(c["title"] == "Sick Leave" for c in section)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_strategy_yields_unique_ids_and_no_tiny_chunks(strategy):
    chunks = chunk_documents(load_documents(DATA), strategy)
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(len(c["text"]) >= 50 for c in chunks)
