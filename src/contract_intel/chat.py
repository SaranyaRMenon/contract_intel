"""Unified chatbot: RAG + agent tools + conversation memory."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from . import config, extraction, llm, risk, vectorstore
from .ingestion import full_text, load_document
from .pipeline import _contract_path


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ChatResponse:
    reply: str
    history: list[ChatMessage]
    tool_calls_made: list[str] = field(default_factory=list)


TOOL_GATHER_PROMPT = """You are a contract data retrieval agent. Your ONLY job is to call the right tools to collect contract information needed to answer the user's question.

Tool selection rules — follow strictly:
- search_contract: use for most questions. One or two targeted queries is usually enough.
- detect_risks: use ONLY when the user explicitly asks about risks, concerns, or problems in the contract.
- extract_clauses: use ONLY when the user explicitly asks for a full extraction, all clauses, or all parties/dates. Do NOT call this for simple factual questions like payment terms, governing law, or specific clauses.
- list_contracts: use only when the user asks what contracts are available.

Examples:
- "What are the payment terms?" → search_contract("payment terms")
- "What is the governing law?" → search_contract("governing law")
- "Could there be operational disruption?" → search_contract("termination notice"), search_contract("liability cessation")
- "What are the risks?" → detect_risks only
- "Extract all clauses" → extract_clauses only

Do NOT call multiple heavy tools for a simple question. Do NOT write any answer — only call tools.
"""

GROUNDED_SYNTHESIS_PROMPT = """You are a grounded contract analyst. Answer the user's question by reasoning from the contract excerpts below.

Grounding rules:
1. Every claim you make must trace back to a specific clause or passage in the excerpts.
2. When you cite a fact, reference its source (e.g. "Section 6 states..." or "The termination clause specifies...").
3. You MAY draw analytical conclusions from the contract text — for example, if the contract says the Client must stop using software immediately on termination, you can conclude that quick termination creates operational risk for the Client. This is reasoning from evidence, not speculation.
4. You MUST NOT add general legal advice, negotiation tips, or knowledge that is not derivable from the contract text.
5. If a specific detail is genuinely absent from the excerpts, say: "The contract does not specify this."
6. If the user asks "what steps" or "what can be done" — only list steps or remedies that are explicitly stated in the contract. If none are stated, say so clearly.

Contract excerpts:
{context}
"""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_contract",
            "description": "Semantic search across indexed contracts. Use for specific questions about contract content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "contract_id": {
                        "type": ["string", "null"],
                        "description": "Limit to this contract ID. Use null to search all.",
                    },
                    "k": {"type": "integer", "default": 6, "description": "Number of passages to retrieve"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_clauses",
            "description": "Run full structured clause extraction on a contract. Returns parties, dates, payment terms, termination, IP, liability, governing law, etc.",
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
            "description": "Detect legal and business risks in a contract using rules + LLM analysis.",
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
            "name": "list_contracts",
            "description": "List all available contract IDs that have been indexed.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


_CONTRACT_KEYWORDS = {
    "contract", "clause", "payment", "termination", "liability", "risk",
    "party", "parties", "agreement", "section", "law", "governing", "ip",
    "intellectual", "confidential", "warranty", "indemnif", "damages",
    "notice", "breach", "default", "license", "fee", "term", "renewal",
    "scope", "service", "sla", "uptime", "force", "majeure", "penalty",
    "interest", "invoice", "effective", "expir", "automat", "obligation",
    "right", "permitted", "prohibit", "restrict", "comply", "jurisdiction",
    "arbitrat", "dispute", "compensat", "refund", "credit", "deliver",
    "milestone", "delay", "support", "maintenance", "data", "privacy",
    "security", "ownership", "assign", "transfer", "sublicens", "non-compete",
    "solicit", "exclusiv", "perpetual", "irrevocabl", "waiver", "sever",
}


def _is_conversational(message: str) -> bool:
    """Return True if the message is a greeting or chitchat with no contract intent."""
    msg = message.lower().strip()
    if any(kw in msg for kw in _CONTRACT_KEYWORDS):
        return False
    return len(msg.split()) <= 10


def _dispatch(name: str, args: dict[str, Any]) -> Any:
    if name == "list_contracts":
        return {"contracts": vectorstore.list_contracts()}
    if name == "search_contract":
        chunks = vectorstore.search(
            args["query"],
            k=int(args.get("k", 6)),
            contract_id=args.get("contract_id"),
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
        path = _contract_path(args["contract_id"])
        if path is None:
            return {"error": f"Contract '{args['contract_id']}' not found."}
        text = full_text(load_document(path))
        return extraction.extract(text).model_dump()
    if name == "detect_risks":
        path = _contract_path(args["contract_id"])
        if path is None:
            return {"error": f"Contract '{args['contract_id']}' not found."}
        text = full_text(load_document(path))
        risks = risk.detect_risks(text)
        return {"risks": [r.model_dump() for r in risks]}
    return {"error": f"Unknown tool: {name}"}


def _gather_context(
    user_message: str,
    history: list[ChatMessage],
    contract_id: str | None,
    model: str | None,
    max_iters: int,
) -> tuple[str, list[str]]:
    """Stage 1: Run tool-calling loop to collect raw contract data.
    Returns (collected_context_text, list_of_tool_names_called).
    """
    contract_note = (
        f"Active contract ID: '{contract_id}'. Use this as default for all tool calls unless told otherwise."
        if contract_id
        else "No contract is currently selected — tell the user to upload one."
    )

    gather_messages: list[dict[str, Any]] = [
        {"role": "system", "content": TOOL_GATHER_PROMPT + "\n\n" + contract_note},
    ]
    for msg in history[-6:]:
        gather_messages.append({"role": msg.role, "content": msg.content})
    gather_messages.append({"role": "user", "content": user_message})

    tool_calls_made: list[str] = []
    collected_chunks: list[str] = []

    for _ in range(max_iters):
        msg = llm.chat(
            gather_messages,
            model=config.FAST_MODEL,
            tools=TOOLS,
            tool_choice="auto",
            max_completion_tokens=2048,
        )
        tool_calls = getattr(msg, "tool_calls", None) or []

        if not tool_calls:
            break

        gather_messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            }
        )

        for tc in tool_calls:
            tool_calls_made.append(tc.function.name)
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            try:
                result = _dispatch(tc.function.name, args)
            except Exception as e:
                result = {"error": str(e)}

            result_text = json.dumps(result)
            collected_chunks.append(f"[Tool: {tc.function.name}]\n{result_text[:3000]}")
            gather_messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result_text[:8000]}
            )

    return "\n\n---\n\n".join(collected_chunks) if collected_chunks else "", tool_calls_made


def chat(
    user_message: str,
    history: list[ChatMessage],
    contract_id: str | None = None,
    max_iters: int = 6,
    model: str | None = None,
) -> ChatResponse:
    """Two-stage grounded chat.

    Stage 1 — Tool gathering: collect raw contract data via tools.
    Stage 2 — Grounded synthesis: answer ONLY using the collected data.
    """
    # ── Conversational shortcut — skip tools for greetings / chitchat ──────
    if _is_conversational(user_message):
        convo_messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a friendly AI Contract Assistant. "
                    "Respond briefly and naturally to the user's message. "
                    "If they seem ready to ask a contract question, let them know you're ready to help."
                ),
            }
        ]
        for msg in history[-4:]:
            convo_messages.append({"role": msg.role, "content": msg.content})
        convo_messages.append({"role": "user", "content": user_message})
        reply = llm.chat_text(convo_messages, model=config.FAST_MODEL, max_completion_tokens=256)
        updated_history = list(history) + [
            ChatMessage(role="user", content=user_message),
            ChatMessage(role="assistant", content=reply),
        ]
        return ChatResponse(reply=reply, history=updated_history, tool_calls_made=[])

    # ── Stage 1: gather contract data ──────────────────────────────────────
    context, tool_calls_made = _gather_context(
        user_message, history, contract_id, model, max_iters
    )

    if not context:
        context = "No relevant contract information was retrieved."

    # ── Stage 2: grounded synthesis — no tools, only the retrieved text ────
    synthesis_messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": GROUNDED_SYNTHESIS_PROMPT.format(context=context),
        },
    ]
    for msg in history[-6:]:
        synthesis_messages.append({"role": msg.role, "content": msg.content})
    synthesis_messages.append({"role": "user", "content": user_message})

    reply = llm.chat_text(
        synthesis_messages,
        model=model or config.SMART_MODEL,
        max_completion_tokens=2048,
    )

    updated_history = list(history) + [
        ChatMessage(role="user", content=user_message),
        ChatMessage(role="assistant", content=reply),
    ]
    return ChatResponse(reply=reply, history=updated_history, tool_calls_made=tool_calls_made)
