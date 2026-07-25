"""End-to-end ingestion pipeline tying file → chunks → vector store → metadata."""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from . import config, ingestion, vectorstore


@dataclass
class IngestResult:
    contract_id: str
    chunk_count: int
    pages: int
    bytes: int
    stored_path: str


def _slugify(name: str) -> str:
    base = "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_")
    return base or "contract"


def _contract_meta_path(contract_id: str) -> Path:
    return config.CONTRACTS_DIR / f"{contract_id}.meta.json"


def _contract_path(contract_id: str) -> Path | None:
    for ext in (".pdf", ".docx", ".txt", ".md"):
        p = config.CONTRACTS_DIR / f"{contract_id}{ext}"
        if p.exists():
            return p
    return None


def ingest_file(source_path: Path, contract_id: str | None = None) -> IngestResult:
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if contract_id is None:
        contract_id = _slugify(source_path.stem)

    dest = config.CONTRACTS_DIR / f"{contract_id}{source_path.suffix.lower()}"
    if source_path.resolve() != dest.resolve():
        shutil.copy2(source_path, dest)

    sections = ingestion.load_document(dest)
    chunks = ingestion.build_chunks(contract_id, sections)
    vectorstore.delete_contract(contract_id)
    count = vectorstore.upsert_chunks(chunks, metadata_extra={"filename": dest.name})

    meta = {
        "contract_id": contract_id,
        "filename": dest.name,
        "ingested_at": datetime.utcnow().isoformat() + "Z",
        "pages": sum(1 for _ in sections),
        "chunks": count,
        "bytes": dest.stat().st_size,
    }
    _contract_meta_path(contract_id).write_text(json.dumps(meta, indent=2))
    return IngestResult(
        contract_id=contract_id,
        chunk_count=count,
        pages=meta["pages"],
        bytes=meta["bytes"],
        stored_path=str(dest),
    )


def ingest_text(contract_id: str, text: str) -> IngestResult:
    contract_id = _slugify(contract_id)
    dest = config.CONTRACTS_DIR / f"{contract_id}.txt"
    dest.write_text(text)
    return ingest_file(dest, contract_id=contract_id)


def list_contracts() -> list[dict]:
    out = []
    for p in sorted(config.CONTRACTS_DIR.glob("*.meta.json")):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            continue
    return out


def delete_contract(contract_id: str) -> bool:
    found = False
    for p in config.CONTRACTS_DIR.glob(f"{contract_id}.*"):
        p.unlink()
        found = True
    vectorstore.delete_contract(contract_id)
    return found
