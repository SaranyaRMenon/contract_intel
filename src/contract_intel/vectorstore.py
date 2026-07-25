"""Chroma vector store wrapper using local embedding model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from . import config
from .ingestion import Chunk


@dataclass
class RetrievedChunk:
    contract_id: str
    chunk_index: int
    text: str
    page: int | None
    distance: float


_collection = None


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        embed_fn = embedding_functions.DefaultEmbeddingFunction()
        _collection = client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            embedding_function=embed_fn,
        )
    return _collection


def upsert_chunks(chunks: list[Chunk], metadata_extra: dict[str, Any] | None = None) -> int:
    if not chunks:
        return 0
    coll = get_collection()
    ids = [f"{c.contract_id}::{c.chunk_index}" for c in chunks]
    docs = [c.text for c in chunks]
    metadatas = []
    for c in chunks:
        meta: dict[str, Any] = {
            "contract_id": c.contract_id,
            "chunk_index": c.chunk_index,
        }
        if c.page is not None:
            meta["page"] = c.page
        if metadata_extra:
            meta.update(metadata_extra)
        metadatas.append(meta)
    coll.upsert(ids=ids, documents=docs, metadatas=metadatas)
    return len(ids)


def delete_contract(contract_id: str) -> None:
    coll = get_collection()
    coll.delete(where={"contract_id": contract_id})


def list_contracts() -> list[str]:
    coll = get_collection()
    res = coll.get(include=["metadatas"])
    seen: set[str] = set()
    for m in res.get("metadatas") or []:
        if m and "contract_id" in m:
            seen.add(m["contract_id"])
    return sorted(seen)


def search(
    query: str,
    k: int = 5,
    contract_id: str | None = None,
) -> list[RetrievedChunk]:
    coll = get_collection()
    where = {"contract_id": contract_id} if contract_id else None
    res = coll.query(query_texts=[query], n_results=k, where=where)
    out: list[RetrievedChunk] = []
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        out.append(
            RetrievedChunk(
                contract_id=meta.get("contract_id", ""),
                chunk_index=int(meta.get("chunk_index", 0)),
                text=doc,
                page=meta.get("page"),
                distance=float(dist),
            )
        )
    return out
