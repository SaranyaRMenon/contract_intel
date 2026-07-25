"""Structured clause extraction with Pydantic schemas + JSON-mode LLM call.

Pydantic models double as guardrails: any LLM output that doesn't conform is
rejected. We retry once with the validation error fed back to the model.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from . import llm


class Party(BaseModel):
    name: str
    role: Optional[str] = Field(
        default=None, description="e.g. 'Customer', 'Vendor', 'Licensor', 'Licensee'"
    )


class Clause(BaseModel):
    name: str = Field(description="Short clause name, e.g. 'Limitation of Liability'")
    category: str = Field(
        description="One of: payment, term, termination, liability, indemnity, "
        "ip, confidentiality, data_privacy, governing_law, dispute_resolution, "
        "warranty, sla, renewal, other"
    )
    summary: str = Field(description="Two-to-four sentence plain-English summary")
    quote: Optional[str] = Field(
        default=None, description="Short verbatim excerpt from the contract"
    )


class ContractExtraction(BaseModel):
    title: Optional[str] = None
    contract_type: Optional[str] = Field(
        default=None,
        description="e.g. NDA, MSA, SOW, Employment Agreement, Lease, License",
    )
    effective_date: Optional[str] = None
    term_length: Optional[str] = None
    governing_law: Optional[str] = None
    parties: list[Party] = Field(default_factory=list)
    clauses: list[Clause] = Field(default_factory=list)


EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": ["string", "null"]},
        "contract_type": {"type": ["string", "null"]},
        "effective_date": {"type": ["string", "null"]},
        "term_length": {"type": ["string", "null"]},
        "governing_law": {"type": ["string", "null"]},
        "parties": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": ["string", "null"]},
                },
                "required": ["name"],
            },
        },
        "clauses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "category": {"type": "string"},
                    "summary": {"type": "string"},
                    "quote": {"type": ["string", "null"]},
                },
                "required": ["name", "category", "summary"],
            },
        },
    },
    "required": ["clauses"],
}


SYSTEM_PROMPT = (
    "You are a precise contract analyst. Extract structured information from the "
    "provided contract text. Use only what is in the text — do not invent terms. "
    "If a field is missing, set it to null. Return strict JSON conforming to the schema."
)


def extract(contract_text: str, model: str | None = None) -> ContractExtraction:
    """Run structured extraction. Validates with Pydantic; retries once on failure."""
    truncated = contract_text[:60_000]
    user = (
        "Extract the contract metadata, parties, and a comprehensive list of clauses "
        "from the contract below. Aim for 8-20 clauses covering all distinct sections.\n\n"
        f"=== CONTRACT TEXT ===\n{truncated}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

    raw = llm.chat_json(messages, EXTRACTION_SCHEMA, schema_name="ContractExtraction", model=model)
    try:
        return ContractExtraction.model_validate(raw)
    except ValidationError as e:
        retry_messages = messages + [
            {"role": "assistant", "content": str(raw)},
            {
                "role": "user",
                "content": (
                    "Your previous JSON failed validation with these errors:\n"
                    f"{e.errors()}\n\n"
                    "Return ONLY corrected JSON conforming to the schema."
                ),
            },
        ]
        raw = llm.chat_json(
            retry_messages, EXTRACTION_SCHEMA, schema_name="ContractExtraction", model=model
        )
        return ContractExtraction.model_validate(raw)
