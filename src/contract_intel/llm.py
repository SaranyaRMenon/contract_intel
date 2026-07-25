"""Thin OpenAI client wrapper that talks to Replit AI Integrations proxy."""
from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from . import config


_client: OpenAI | None = None


def _is_openai_native() -> bool:
    """OpenAI's own API uses max_completion_tokens; everyone else uses max_tokens."""
    base = (config.OPENAI_BASE_URL or "").lower()
    return "openai.com" in base


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not config.OPENAI_BASE_URL or not config.OPENAI_API_KEY:
            raise RuntimeError(
                "OpenAI integration env vars are missing. "
                "Expected AI_INTEGRATIONS_OPENAI_BASE_URL and AI_INTEGRATIONS_OPENAI_API_KEY."
            )
        _client = OpenAI(
            base_url=config.OPENAI_BASE_URL,
            api_key=config.OPENAI_API_KEY,
        )
    return _client


def chat(
    messages: list[dict[str, Any]],
    model: str | None = None,
    response_format: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    max_completion_tokens: int = 4096,
) -> Any:
    """Single chat completion call. Returns the raw `choices[0].message`."""
    client = get_client()
    kwargs: dict[str, Any] = {
        "model": model or config.DEFAULT_MODEL,
        "messages": messages,
    }
    if _is_openai_native():
        kwargs["max_completion_tokens"] = max_completion_tokens
    else:
        kwargs["max_tokens"] = max_completion_tokens
    if response_format is not None:
        kwargs["response_format"] = response_format
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        msg = str(e).lower()
        if "max_completion_tokens" in msg or "max_tokens" in msg:
            kwargs.pop("max_completion_tokens", None)
            kwargs.pop("max_tokens", None)
            other = "max_tokens" if _is_openai_native() else "max_completion_tokens"
            kwargs[other] = max_completion_tokens
            resp = client.chat.completions.create(**kwargs)
        elif "response_format" in msg and response_format is not None:
            kwargs.pop("response_format", None)
            kwargs["messages"] = list(messages) + [
                {
                    "role": "system",
                    "content": "Respond with valid JSON only. No prose, no markdown fences.",
                }
            ]
            resp = client.chat.completions.create(**kwargs)
        else:
            raise
    return resp.choices[0].message


def chat_text(messages: list[dict[str, Any]], **kwargs: Any) -> str:
    msg = chat(messages, **kwargs)
    return (msg.content or "").strip()


def chat_json(
    messages: list[dict[str, Any]],
    schema: dict[str, Any],
    schema_name: str = "Response",
    model: str | None = None,
    max_completion_tokens: int = 4096,
) -> dict[str, Any]:
    """Structured-output chat call using JSON schema. Returns parsed dict."""
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "schema": schema,
            "strict": False,
        },
    }
    msg = chat(
        messages,
        model=model,
        response_format=response_format,
        max_completion_tokens=max_completion_tokens,
    )
    raw = (msg.content or "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            return json.loads(raw[start : end + 1])
        raise
