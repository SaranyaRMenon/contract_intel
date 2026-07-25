"""Document ingestion: read PDFs/DOCX/text and chunk for embedding."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from . import config


@dataclass
class Chunk:
    contract_id: str
    chunk_index: int
    text: str
    page: int | None = None


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> list[tuple[int, str]]:
    """Return list of (page_number, text) tuples (1-indexed pages)."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    out: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        out.append((i, text))
    return out


def read_docx(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def load_document(path: Path) -> list[tuple[int | None, str]]:
    """Return list of (page_or_none, text) sections."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return [(p, t) for p, t in read_pdf(path)]
    if suffix in (".docx",):
        return [(None, read_docx(path))]
    if suffix in (".txt", ".md"):
        return [(None, read_text_file(path))]
    raise ValueError(f"Unsupported file type: {suffix}")


def _normalize(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    text = _normalize(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    step = max(chunk_size - overlap, 200)
    for start in range(0, len(text), step):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
    return chunks


def build_chunks(contract_id: str, sections: list[tuple[int | None, str]]) -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 0
    for page, section_text in sections:
        for piece in chunk_text(section_text):
            chunks.append(Chunk(contract_id=contract_id, chunk_index=idx, text=piece, page=page))
            idx += 1
    return chunks


def full_text(sections: list[tuple[int | None, str]]) -> str:
    return _normalize("\n\n".join(s for _, s in sections))
