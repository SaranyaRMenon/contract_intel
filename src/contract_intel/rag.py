"""RAG Q&A over indexed contracts."""
from __future__ import annotations

from dataclasses import dataclass

from . import llm, vectorstore


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[dict]


SYSTEM_PROMPT = (
    "You are a contract Q&A assistant. Answer ONLY using the provided context. "
    "If the answer is not present, say so explicitly — do NOT speculate. "
    "When you state a fact, reference the source as [#] using the indices in the context."
)


def _format_context(chunks) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        page = f" p.{c.page}" if c.page else ""
        parts.append(f"[{i}] (contract={c.contract_id}{page})\n{c.text}")
    return "\n\n---\n\n".join(parts)


def ask(
    question: str,
    contract_id: str | None = None,
    k: int = 6,
    model: str | None = None,
) -> Answer:
    chunks = vectorstore.search(question, k=k, contract_id=contract_id)
    if not chunks:
        return Answer(question=question, answer="No indexed content found.", citations=[])
    context = _format_context(chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Question: {question}\n\nContext:\n{context}",
        },
    ]
    text = llm.chat_text(messages, model=model)
    citations = [
        {
            "n": i + 1,
            "contract_id": c.contract_id,
            "chunk_index": c.chunk_index,
            "page": c.page,
            "distance": round(c.distance, 4),
        }
        for i, c in enumerate(chunks)
    ]
    return Answer(question=question, answer=text, citations=citations)
