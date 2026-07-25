"""Agentic Contract Reviewer using OpenAI tool/function calling.

The agent decides which tools to call (search, extract, risks, summarize) to
satisfy a user request.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from . import config, extraction, llm, rag, risk, vectorstore
from .ingestion import full_text, load_document
from .pipeline import _contract_path


@dataclass
class AgentTrace:
    steps: list[dict[str, Any]] = field(default_factory=list)


SYSTEM_PROMPT = (
    "You are a Contract Reviewer agent. You help users analyze indexed contracts. "
    "Use tools when you need facts from the contracts. Prefer `search_contract` for "
    "specific factual questions. Use `extract_clauses` and `detect_risks` for full "
    "structured reviews. Stop and answer the user once you have enough information. "
    "Do not call tools you don't need. When you cite information, mention the source "
    "contract id and page if available."
)


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_contracts",
            "description": "List all indexed contract IDs available to query.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_contract",
            "description": "Semantic search across indexed contracts. Returns top-k passages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "contract_id": {
                        "type": ["string", "null"],
                        "description": "Limit search to this contract; null = all.",
                    },
                    "k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_clauses",
            "description": "Run structured clause extraction over an indexed contract.",
            "parameters": {
                "type": "object",
                "properties": {"contract_id": {"type": "string"}},
                "required": ["contract_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_risks",
            "description": "Detect risks in an indexed contract using rules + LLM.",
            "parameters": {
                "type": "object",
                "properties": {"contract_id": {"type": "string"}},
                "required": ["contract_id"],
            },
        },
    },
]


def _load_contract_text(contract_id: str) -> str:
    path = _contract_path(contract_id)
    if path is None:
        raise FileNotFoundError(f"Contract '{contract_id}' not found on disk.")
    return full_text(load_document(path))


def _dispatch(name: str, args: dict[str, Any]) -> Any:
    if name == "list_contracts":
        return {"contracts": vectorstore.list_contracts()}
    if name == "search_contract":
        chunks = vectorstore.search(
            args["query"], k=int(args.get("k", 5)), contract_id=args.get("contract_id")
        )
        return {
            "results": [
                {
                    "contract_id": c.contract_id,
                    "page": c.page,
                    "text": c.text[:600],
                    "distance": round(c.distance, 4),
                }
                for c in chunks
            ]
        }
    if name == "extract_clauses":
        text = _load_contract_text(args["contract_id"])
        result = extraction.extract(text)
        return result.model_dump()
    if name == "detect_risks":
        text = _load_contract_text(args["contract_id"])
        risks = risk.detect_risks(text)
        return {"risks": [r.model_dump() for r in risks]}
    return {"error": f"unknown tool {name}"}


def run_agent(user_input: str, model: str | None = None, max_iters: int = 6) -> tuple[str, AgentTrace]:
    trace = AgentTrace()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]
    for step in range(max_iters):
        msg = llm.chat(
            messages,
            model=model or config.SMART_MODEL,
            tools=TOOLS,
            tool_choice="auto",
            max_completion_tokens=4096,
        )
        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            text = (msg.content or "").strip()
            trace.steps.append({"step": step, "kind": "final", "content": text})
            return text, trace
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )
        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            try:
                result = _dispatch(tc.function.name, args)
            except Exception as e:
                result = {"error": str(e)}
            trace.steps.append(
                {"step": step, "kind": "tool", "name": tc.function.name, "args": args, "result_preview": str(result)[:300]}
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result)[:8000],
                }
            )
    trace.steps.append({"step": max_iters, "kind": "halted", "reason": "max_iters"})
    return "Agent exceeded max iterations without producing a final answer.", trace
