"""Deterministic HTML extraction and section-bounded chunking (no model calls)."""

import hashlib
import re
from datetime import datetime

from bs4 import BeautifulSoup

from app.rag_models import FilingChunk
from app.sec_models import Filing

# Version the chunk identity when extraction/chunking behavior changes.
PIPELINE_VERSION = "annual-html-v1"
ITEM = re.compile(r"^item\s+(\d{1,2}[a-c]?)\s*[.\-–—:]?\s*(.*)$", re.I)


def parse_sections(html: str) -> tuple[list[tuple[str, str]], list[str]]:
    soup = BeautifulSoup(html, "html.parser")
    if soup.find(["html", "body"]) is None:
        raise ValueError(
            "Expected an HTML annual filing; plain text and PDF are unsupported"
        )
    for tag in list(
        soup.find_all(["script", "style", "head", "noscript", "ix:header", "ix:hidden"])
    ):
        if tag.parent is not None:
            tag.decompose()
    for tag in list(soup.find_all(attrs={"hidden": True})):
        if tag.parent is not None:
            tag.decompose()
    for tag in list(soup.find_all(style=True)):
        if tag.parent is not None and re.search(
            r"display\s*:\s*none|visibility\s*:\s*hidden", str(tag.get("style")), re.I
        ):
            tag.decompose()
    # SEC tables of contents commonly contain several internal navigation links.
    for table in list(soup.find_all("table")):
        if table.parent is not None and len(table.select('a[href^="#"]')) >= 3:
            table.decompose()
    for tag in soup.find_all(
        ["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "li", "br"]
    ):
        tag.insert_before("\n")
        tag.insert_after("\n")
    for tag in soup.find_all(["td", "th"]):
        tag.insert_after(" ")
    lines = [re.sub(r"\s+", " ", line).strip() for line in soup.get_text().splitlines()]
    sections: list[tuple[str, str]] = []
    title = "Unidentified section"
    body: list[str] = []
    identified = False
    for line in lines:
        if not line:
            continue
        match = ITEM.fullmatch(line) if len(line) <= 200 else None
        if match and 1 <= int(re.match(r"\d+", match[1])[0]) <= 16:
            if body:
                sections.append((title, "\n\n".join(body)))
            title = f"Item {match[1].upper()}"
            if match[2]:
                title += f". {match[2]}"
            body = []
            identified = True
        else:
            body.append(line)
    if body:
        sections.append((title, "\n\n".join(body)))
    if not sections:
        raise ValueError("Annual filing has no extractable text")
    warnings = (
        []
        if identified
        else [
            "No Item headings identified; evidence uses an unidentified section label"
        ]
    )
    return sections, warnings


def chunk_filing(
    filing: Filing,
    retrieved_at: datetime,
    sections: list[tuple[str, str]],
    *,
    chunk_size: int = 1800,
    overlap: int = 200,
) -> list[FilingChunk]:
    if chunk_size < 100 or not 0 <= overlap < chunk_size:
        raise ValueError(
            "Chunk size must be at least 100 and overlap smaller than size"
        )
    chunks = []
    for section, text in sections:
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start + chunk_size // 2, end)
                if boundary > start:
                    end = boundary
            while start < end and text[start].isspace():
                start += 1
            content = text[start:end]
            identity = (
                f"{PIPELINE_VERSION}:{filing.cik}:{filing.accession_number}:"
                f"{len(chunks)}:{section}:{start}:{content}"
            )
            chunks.append(
                FilingChunk(
                    chunk_id=hashlib.sha256(identity.encode()).hexdigest(),
                    filing=filing,
                    retrieved_at=retrieved_at,
                    section=section,
                    chunk_index=len(chunks),
                    char_start=start,
                    char_end=end,
                    text=content,
                )
            )
            if end == len(text):
                break
            start = max(start + 1, end - overlap)
    return chunks
