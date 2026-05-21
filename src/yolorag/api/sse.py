from __future__ import annotations

import json
from collections.abc import AsyncIterator


async def text_event_stream(text: str, chunk_size: int = 80) -> AsyncIterator[str]:
    for chunk in _chunks(text, chunk_size=chunk_size):
        yield _data_event({"content": chunk})
    yield "data: [DONE]\n\n"


async def error_event_stream(message: str) -> AsyncIterator[str]:
    yield _data_event({"error": message})
    yield "data: [DONE]\n\n"


def _data_event(payload: dict[str, str]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _chunks(text: str, chunk_size: int) -> list[str]:
    if not text:
        return [""]
    return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]
