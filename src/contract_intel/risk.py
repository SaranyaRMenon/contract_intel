"""Risk detection: hybrid rule-based + LLM with structured output guardrails."""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ValidationError

from . import llm

Severity = Literal["low", "medium", "high"]


class Risk(BaseModel):
    title: str
    severity: Severity
    category: str
    explanation: str
    suggestion: str | None = None
    quote: str | None = None
    source: Literal["rule", "llm"] = "llm"


RISK_SCHEMA = {
    "type": "object",
    "properties": {
        "risks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "category": {"type": "string"},
                    "explanation": {"type": "string"},
                    "suggestion": {"type": ["string", "null"]},
                    "quote": {"type": ["string", "null"]},
                },
                "required": ["title", "severity", "category", "explanation"],
            },
        }
    },
    "required": ["risks"],
}


RULES: list[tuple[str, str, Severity, str]] = [
    (
        "Auto-renewal without notice window",
        r"automatic(?:ally)?\s+renew",
        "medium",
        "Contract renews automatically. Verify the cancellation notice window.",
    ),
    (
        "Unlimited liability",
        r"(unlimited|no\s+cap\s+on)\s+liability",
        "high",
        "Counterparty's liability is uncapped — major financial exposure.",
    ),
    (
        "Indemnification by us",
        r"\bwe\s+(shall|will|agree to)\s+indemnif",
        "medium",
        "We are accepting an indemnification obligation; check scope.",
    ),
    (
        "Foreign governing law",
        r"governed by the laws of\s+(?!the state of (california|delaware|new york))[^.\n]+",
        "low",
        "Governing law is a non-US/non-standard jurisdiction; legal cost may rise.",
    ),
    (
        "Perpetual license grant",
        r"perpetual\s+(?:and\s+irrevocable\s+)?license",
        "medium",
        "Perpetual license can outlast termination. Confirm this is intentional.",
    ),
    (
        "Class-action waiver",
        r"class[-\s]action\s+waiver",
        "low",
        "Disputes restricted from class actions; usually enforceable in US.",
    ),
    (
        "Broad assignment of IP",
        r"assigns?\s+(all|any)\s+(right|title|interest).*(intellectual property|inventions)",
        "medium",
        "Broad IP assignment clause. Verify scope vs. background IP.",
    ),
]


def rule_scan(text: str) -> list[Risk]:
    risks: list[Risk] = []
    for title, pattern, severity, explanation in RULES:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            start = max(0, m.start() - 80)
            end = min(len(text), m.end() + 80)
            risks.append(
                Risk(
                    title=title,
                    severity=severity,
                    category="rule",
                    explanation=explanation,
                    quote=text[start:end].strip(),
                    source="rule",
                )
            )
    return risks


SYSTEM_PROMPT = (
    "You are a senior contracts attorney. Identify risks in the provided contract text. "
    "Be concrete: cite the issue, explain the impact in business terms, and suggest a "
    "redline if reasonable. Severity levels: high (material financial/legal exposure), "
    "medium (notable concern, negotiate), low (note for awareness). Return JSON only."
)


def llm_scan(text: str, model: str | None = None) -> list[Risk]:
    truncated = text[:60_000]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Analyze the following contract for risks. Identify 5-12 distinct risks "
                "spanning liability, indemnity, IP, data, term/termination, payment, "
                "and compliance.\n\n"
                f"=== CONTRACT TEXT ===\n{truncated}"
            ),
        },
    ]
    raw = llm.chat_json(messages, RISK_SCHEMA, schema_name="RiskReport", model=model)
    risks: list[Risk] = []
    for r in raw.get("risks", []):
        try:
            risks.append(Risk.model_validate({**r, "source": "llm"}))
        except ValidationError:
            continue
    return risks


def detect_risks(text: str, model: str | None = None) -> list[Risk]:
    rule_risks = rule_scan(text)
    llm_risks = llm_scan(text, model=model)
    seen = {(r.title.lower()) for r in rule_risks}
    merged = list(rule_risks)
    for r in llm_risks:
        if r.title.lower() not in seen:
            merged.append(r)
            seen.add(r.title.lower())
    sev_order = {"high": 0, "medium": 1, "low": 2}
    merged.sort(key=lambda r: sev_order.get(r.severity, 3))
    return merged
