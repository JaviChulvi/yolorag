from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any


async def text_event_stream(text: str, chunk_size: int = 80) -> AsyncIterator[str]:
    for chunk in _chunks(text, chunk_size=chunk_size):
        yield data_event({"content": chunk})
    yield "data: [DONE]\n\n"


async def content_event_stream(
    chunks: AsyncIterator[str | dict[str, Any]],
    error_prefix: str = "Stream failed",
) -> AsyncIterator[str]:
    try:
        async for chunk in chunks:
            if isinstance(chunk, dict):
                yield data_event(chunk)
            else:
                yield data_event({"content": chunk})
    except Exception as exc:
        async for event in error_event_stream(f"{error_prefix}: {exc}"):
            yield event
        return
    yield "data: [DONE]\n\n"


async def typed_event_stream(
    events: AsyncIterator[dict[str, Any]],
    error_prefix: str = "Stream failed",
) -> AsyncIterator[str]:
    try:
        async for event in events:
            yield data_event(event)
    except Exception as exc:
        async for event in typed_error_event_stream(f"{error_prefix}: {exc}"):
            yield event
        return
    yield "data: [DONE]\n\n"


async def error_event_stream(message: str) -> AsyncIterator[str]:
    yield data_event({"error": message})
    yield "data: [DONE]\n\n"


async def typed_error_event_stream(message: str) -> AsyncIterator[str]:
    yield data_event({"type": "error", "error": message})
    yield "data: [DONE]\n\n"


def data_event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _chunks(text: str, chunk_size: int) -> list[str]:
    if not text:
        return [""]
    return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]
