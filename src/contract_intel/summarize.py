"""Executive summary generation with prompt engineering (role + few-shot tone)."""
from __future__ import annotations

from . import llm


SYSTEM_PROMPT = (
    "You are an executive briefing writer for in-house legal teams. "
    "Produce concise, scannable summaries optimized for a busy GC. "
    "Tone: precise, business-aware, never vague. Avoid legalese. "
    "Format with these sections (markdown):\n"
    "## Overview\n"
    "## Parties & Term\n"
    "## Key Commercial Terms\n"
    "## Obligations\n"
    "## Risks to Flag\n"
    "## Recommended Next Steps"
)


def summarize(
    text: str,
    extras: str = "",
    model: str | None = None,
    max_completion_tokens: int = 2000,
) -> str:
    truncated = text[:60_000]
    user = "Summarize the following contract.\n\n"
    if extras:
        user += f"Additional context:\n{extras}\n\n"
    user += f"=== CONTRACT TEXT ===\n{truncated}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    return llm.chat_text(messages, model=model, max_completion_tokens=max_completion_tokens)
