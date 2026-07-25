"""FastAPI service exposing the same operations as the CLI."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import logging
import traceback

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import agent, chat as chat_module, extraction, pipeline, rag, risk, summarize, vectorstore
from .ingestion import full_text, load_document
from .pipeline import _contract_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("contract_intel.api")

app = FastAPI(title="Contract Intelligence API", version="0.1.0")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": exc.__class__.__name__,
            "detail": str(exc),
            "traceback": traceback.format_exc().splitlines()[-8:],
        },
    )


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/contracts")
def contracts_list():
    return {"contracts": pipeline.list_contracts()}


@app.post("/contracts/upload")
async def contracts_upload(
    file: UploadFile = File(...),
    contract_id: str | None = Query(default=None),
):
    suffix = Path(file.filename or "upload").suffix.lower() or ".txt"
    if suffix not in {".pdf", ".docx", ".txt", ".md"}:
        raise HTTPException(400, f"Unsupported file type: {suffix}")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        result = pipeline.ingest_file(tmp_path, contract_id=contract_id)
        return result.__dict__
    finally:
        tmp_path.unlink(missing_ok=True)


@app.delete("/contracts/{contract_id}")
def contracts_delete(contract_id: str):
    ok = pipeline.delete_contract(contract_id)
    if not ok:
        raise HTTPException(404, "not found")
    return {"deleted": contract_id}


@app.get("/contracts/{contract_id}/extract")
def contracts_extract(contract_id: str):
    path = _contract_path(contract_id)
    if path is None:
        raise HTTPException(404, "not found")
    text = full_text(load_document(path))
    return extraction.extract(text).model_dump()


@app.get("/contracts/{contract_id}/risks")
def contracts_risks(contract_id: str):
    path = _contract_path(contract_id)
    if path is None:
        raise HTTPException(404, "not found")
    text = full_text(load_document(path))
    return {"risks": [r.model_dump() for r in risk.detect_risks(text)]}


@app.get("/contracts/{contract_id}/summary")
def contracts_summary(contract_id: str):
    path = _contract_path(contract_id)
    if path is None:
        raise HTTPException(404, "not found")
    text = full_text(load_document(path))
    return {"summary": summarize.summarize(text)}


class AskBody(BaseModel):
    question: str
    contract_id: str | None = None
    k: int = 6


@app.post("/ask")
def ask_endpoint(body: AskBody):
    ans = rag.ask(body.question, contract_id=body.contract_id, k=body.k)
    return {"question": ans.question, "answer": ans.answer, "citations": ans.citations}


class AgentBody(BaseModel):
    prompt: str


@app.post("/agent")
def agent_endpoint(body: AgentBody):
    text, trace = agent.run_agent(body.prompt)
    return {"answer": text, "trace": trace.steps}


class ChatBody(BaseModel):
    message: str
    history: list[dict] = []
    contract_id: str | None = None


@app.post("/chat")
def chat_endpoint(body: ChatBody):
    history = [
        chat_module.ChatMessage(role=m["role"], content=m["content"])
        for m in body.history
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    response = chat_module.chat(
        user_message=body.message,
        history=history,
        contract_id=body.contract_id,
    )
    return {
        "reply": response.reply,
        "history": [{"role": m.role, "content": m.content} for m in response.history],
        "tools_used": response.tool_calls_made,
    }


class SearchBody(BaseModel):
    query: str
    k: int = 6
    contract_id: str | None = None


@app.post("/search")
def search_endpoint(body: SearchBody):
    chunks = vectorstore.search(body.query, k=body.k, contract_id=body.contract_id)
    return {
        "results": [
            {
                "contract_id": c.contract_id,
                "chunk_index": c.chunk_index,
                "page": c.page,
                "text": c.text,
                "distance": c.distance,
            }
            for c in chunks
        ]
    }
