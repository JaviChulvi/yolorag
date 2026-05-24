from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from starlette.responses import StreamingResponse

from yolorag.api.schemas import ChatRequest
from yolorag.api.sse import (
    content_event_stream,
    error_event_stream,
    typed_error_event_stream,
    typed_event_stream,
)
from yolorag.runtime import (
    YoloRAGAgentRuntime,
    YoloRAGRuntime,
    build_deep_runtime,
    build_runtime,
)


router = APIRouter()


@router.post("/chat")
async def chat(payload: ChatRequest, request: Request) -> StreamingResponse:
    return await chat_fast(payload, request)


@router.post("/chat/fast")
async def chat_fast(payload: ChatRequest, request: Request) -> StreamingResponse:
    user_message = payload.latest_user_message()
    if user_message is None:
        raise HTTPException(status_code=400, detail="At least one user message is required.")

    session_id = payload.session_id or str(uuid4())
    request_id = str(uuid4())
    total_messages, active_messages = _message_counts(payload)

    try:
        runtime = _fast_runtime(request)
        composed_user_message = _compose_user_message(payload, user_message.content)
        stream = content_event_stream(
            runtime.stream_answer(
                user_message=composed_user_message,
                conversation_id=session_id,
                conversation_messages=_model_messages(payload, composed_user_message),
                raw_user_message=user_message.content,
                request_id=request_id,
                user_message_index=_latest_user_message_index(payload),
            ),
            error_prefix="Chat generation failed",
        )
    except Exception as exc:
        stream = error_event_stream(f"Chat generation failed: {exc}")

    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Session-ID": session_id,
            "X-Total-User-Messages": str(total_messages),
            "X-Active-User-Messages": str(active_messages),
            "X-Chat-Mode": "fast",
        },
    )


@router.post("/chat/deep")
async def chat_deep(payload: ChatRequest, request: Request) -> StreamingResponse:
    user_message = payload.latest_user_message()
    if user_message is None:
        raise HTTPException(status_code=400, detail="At least one user message is required.")

    session_id = payload.session_id or str(uuid4())
    request_id = str(uuid4())
    total_messages, active_messages = _message_counts(payload)

    try:
        runtime = _deep_runtime(request)
        composed_user_message = _compose_user_message(payload, user_message.content)
        stream = _text_from_deep_events(
            runtime.stream_answer(
                user_message=composed_user_message,
                conversation_id=session_id,
                conversation_messages=_model_messages(payload, composed_user_message),
                raw_user_message=user_message.content,
                request_id=request_id,
                user_message_index=_latest_user_message_index(payload),
            ),
            error_prefix="Deep chat generation failed",
        )
    except Exception as exc:
        stream = _text_error_stream(f"Deep chat generation failed: {exc}")

    return StreamingResponse(
        stream,
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Session-ID": session_id,
            "X-Total-User-Messages": str(total_messages),
            "X-Active-User-Messages": str(active_messages),
            "X-Chat-Mode": "deep",
        },
    )


@router.post("/chat/deep/events")
async def chat_deep_events(payload: ChatRequest, request: Request) -> StreamingResponse:
    user_message = payload.latest_user_message()
    if user_message is None:
        raise HTTPException(status_code=400, detail="At least one user message is required.")

    session_id = payload.session_id or str(uuid4())
    request_id = str(uuid4())
    total_messages, active_messages = _message_counts(payload)

    try:
        runtime = _deep_runtime(request)
        composed_user_message = _compose_user_message(payload, user_message.content)
        stream = typed_event_stream(
            runtime.stream_answer(
                user_message=composed_user_message,
                conversation_id=session_id,
                conversation_messages=_model_messages(payload, composed_user_message),
                raw_user_message=user_message.content,
                request_id=request_id,
                user_message_index=_latest_user_message_index(payload),
            ),
            error_prefix="Deep chat generation failed",
        )
    except Exception as exc:
        stream = typed_error_event_stream(f"Deep chat generation failed: {exc}")

    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Session-ID": session_id,
            "X-Total-User-Messages": str(total_messages),
            "X-Active-User-Messages": str(active_messages),
            "X-Chat-Mode": "deep",
            "X-Stream-Format": "agent-events",
        },
    )


def _fast_runtime(request: Request) -> YoloRAGRuntime:
    runtime = getattr(request.app.state, "fast_runtime", None)
    if runtime is None:
        runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        runtime = build_runtime(mode="fast")
        request.app.state.fast_runtime = runtime
        request.app.state.runtime = runtime
    return runtime


def _deep_runtime(request: Request) -> YoloRAGAgentRuntime:
    runtime = getattr(request.app.state, "deep_runtime", None)
    if runtime is None:
        runtime = build_deep_runtime()
        request.app.state.deep_runtime = runtime
    return runtime


def _message_counts(payload: ChatRequest) -> tuple[int, int]:
    user_message_count = sum(1 for message in payload.messages if message.role == "user")
    return user_message_count, user_message_count


def _model_messages(payload: ChatRequest, composed_user_message: str) -> list[dict[str, str]]:
    latest_user_index = _latest_user_message_index(payload)
    messages = []
    for index, message in enumerate(payload.messages):
        content = (
            composed_user_message
            if latest_user_index is not None and index == latest_user_index
            else message.content
        )
        messages.append({"role": message.role, "content": content})
    return messages


def _latest_user_message_index(payload: ChatRequest) -> int | None:
    for index in range(len(payload.messages) - 1, -1, -1):
        if payload.messages[index].role == "user":
            return index
    return None


def _compose_user_message(payload: ChatRequest, content: str) -> str:
    parts = []
    if payload.instructions:
        parts.append(f"Instructions:\n{payload.instructions}")
    if payload.context:
        context = payload.context.compact_text()
        if context:
            parts.append(f"Page context:\n{context}")
    if not parts:
        return content
    parts.append(f"User message:\n{content}")
    return "\n\n".join(parts)


async def _text_from_deep_events(
    events: AsyncIterator[dict[str, Any]],
    error_prefix: str,
) -> AsyncIterator[str]:
    try:
        async for event in events:
            if event.get("type") != "content":
                continue
            content = event.get("content")
            if content:
                yield str(content)
    except Exception as exc:
        yield f"{error_prefix}: {exc}"


async def _text_error_stream(message: str) -> AsyncIterator[str]:
    yield message
