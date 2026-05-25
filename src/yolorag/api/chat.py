from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
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
from yolorag.config.settings import getenv
from yolorag.runtime import (
    YoloRAGAgentRuntime,
    YoloRAGRuntime,
    build_deep_runtime,
    build_runtime,
)


router = APIRouter()

DEFAULT_LLM_PROVIDER = "deepseek"
DEFAULT_KNOWLEDGE_PROVIDER = "mongodb"
VALID_LLM_PROVIDERS = {"openai", "deepseek"}
VALID_KNOWLEDGE_PROVIDERS = {"mongodb", "postgresql"}


@dataclass(frozen=True)
class RuntimeOptions:
    provider: str
    knowledge_provider: str
    conversation_provider: str
    explicitly_selected: bool

    @property
    def cache_key(self) -> tuple[str, str, str]:
        return (self.provider, self.knowledge_provider, self.conversation_provider)


@router.post("/chat")
async def chat(payload: ChatRequest, request: Request) -> StreamingResponse:
    return await chat_fast(payload, request)


@router.post("/chat/fast")
async def chat_fast(payload: ChatRequest, request: Request) -> StreamingResponse:
    runtime_options = _runtime_options(request)
    user_message_index = _latest_user_message_index(payload)
    if user_message_index is None:
        raise HTTPException(status_code=400, detail="At least one user message is required.")
    user_message_content = payload.messages[user_message_index].content

    session_id = payload.session_id or str(uuid4())
    request_id = str(uuid4())
    total_messages, active_messages = _message_counts(payload)

    try:
        runtime = _fast_runtime(request, runtime_options)
        stream = content_event_stream(
            runtime.stream_answer(
                user_message=user_message_content,
                conversation_id=session_id,
                conversation_messages=_model_messages(payload),
                request_id=request_id,
                user_message_index=user_message_index,
                include_metrics=payload.include_metrics,
                persist=payload.analytics,
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
            **_runtime_headers(runtime_options),
        },
    )


@router.post("/chat/deep")
async def chat_deep(payload: ChatRequest, request: Request) -> StreamingResponse:
    runtime_options = _runtime_options(request)
    user_message_index = _latest_user_message_index(payload)
    if user_message_index is None:
        raise HTTPException(status_code=400, detail="At least one user message is required.")
    user_message_content = payload.messages[user_message_index].content

    session_id = payload.session_id or str(uuid4())
    request_id = str(uuid4())
    total_messages, active_messages = _message_counts(payload)

    try:
        runtime = _deep_runtime(request, runtime_options)
        stream = _text_from_deep_events(
            runtime.stream_answer(
                user_message=user_message_content,
                conversation_id=session_id,
                conversation_messages=_model_messages(payload),
                request_id=request_id,
                user_message_index=user_message_index,
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
            **_runtime_headers(runtime_options),
        },
    )


@router.post("/chat/deep/events")
async def chat_deep_events(payload: ChatRequest, request: Request) -> StreamingResponse:
    runtime_options = _runtime_options(request)
    user_message_index = _latest_user_message_index(payload)
    if user_message_index is None:
        raise HTTPException(status_code=400, detail="At least one user message is required.")
    user_message_content = payload.messages[user_message_index].content

    session_id = payload.session_id or str(uuid4())
    request_id = str(uuid4())
    total_messages, active_messages = _message_counts(payload)

    try:
        runtime = _deep_runtime(request, runtime_options)
        stream = typed_event_stream(
            runtime.stream_answer(
                user_message=user_message_content,
                conversation_id=session_id,
                conversation_messages=_model_messages(payload),
                request_id=request_id,
                user_message_index=user_message_index,
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
            **_runtime_headers(runtime_options),
        },
    )


def _fast_runtime(request: Request, options: RuntimeOptions) -> YoloRAGRuntime:
    runtime = getattr(request.app.state, "fast_runtime", None)
    if runtime is not None and not options.explicitly_selected:
        return runtime
    cache = _runtime_cache(request, mode="fast")
    if options.cache_key in cache:
        return cache[options.cache_key]
    runtime = build_runtime(
        provider_name=options.provider,
        mode="fast",
        knowledge_provider=options.knowledge_provider,
        conversation_provider=options.conversation_provider,
    )
    cache[options.cache_key] = runtime
    return runtime


def _deep_runtime(request: Request, options: RuntimeOptions) -> YoloRAGAgentRuntime:
    runtime = getattr(request.app.state, "deep_runtime", None)
    if runtime is not None and not options.explicitly_selected:
        return runtime
    cache = _runtime_cache(request, mode="deep")
    if options.cache_key in cache:
        return cache[options.cache_key]
    runtime = build_deep_runtime(
        provider_name=options.provider,
        knowledge_provider=options.knowledge_provider,
        conversation_provider=options.conversation_provider,
    )
    cache[options.cache_key] = runtime
    return runtime


def _runtime_cache(
    request: Request,
    *,
    mode: str,
) -> dict[tuple[str, str, str], YoloRAGRuntime | YoloRAGAgentRuntime]:
    caches = getattr(request.app.state, "runtime_caches", None)
    if caches is None:
        caches = {"fast": {}, "deep": {}}
        request.app.state.runtime_caches = caches
    return caches[mode]


def _runtime_options(request: Request) -> RuntimeOptions:
    provider_param = _first_query_value(request, "provider", "llm_provider")
    knowledge_param = _first_query_value(request, "knowledge_provider", "db", "database")
    conversation_param = _first_query_value(request, "conversation_provider")

    provider = _normalize_choice(
        provider_param or getenv("YOLORAG_API_PROVIDER", DEFAULT_LLM_PROVIDER),
        allowed=VALID_LLM_PROVIDERS,
        field="provider",
    )
    knowledge_provider = _normalize_choice(
        knowledge_param or getenv("YOLORAG_KNOWLEDGE_PROVIDER", DEFAULT_KNOWLEDGE_PROVIDER),
        allowed=VALID_KNOWLEDGE_PROVIDERS,
        field="knowledge_provider",
    )
    conversation_provider = _normalize_choice(
        conversation_param
        or getenv("YOLORAG_CONVERSATION_PROVIDER")
        or knowledge_provider,
        allowed=VALID_KNOWLEDGE_PROVIDERS,
        field="conversation_provider",
    )

    return RuntimeOptions(
        provider=provider,
        knowledge_provider=knowledge_provider,
        conversation_provider=conversation_provider,
        explicitly_selected=any(
            value is not None for value in (provider_param, knowledge_param, conversation_param)
        ),
    )


def _first_query_value(request: Request, *names: str) -> str | None:
    for name in names:
        value = request.query_params.get(name)
        if value is not None:
            return value
    return None


def _normalize_choice(
    value: str | None,
    *,
    allowed: set[str],
    field: str,
) -> str:
    normalized = (value or "").strip().lower()
    if normalized in allowed:
        return normalized
    choices = ", ".join(sorted(allowed))
    raise HTTPException(
        status_code=400,
        detail=f"Unsupported {field} {value!r}. Expected one of: {choices}.",
    )


def _runtime_headers(options: RuntimeOptions) -> dict[str, str]:
    return {
        "X-LLM-Provider": options.provider,
        "X-Knowledge-Provider": options.knowledge_provider,
        "X-Conversation-Provider": options.conversation_provider,
    }


def _message_counts(payload: ChatRequest) -> tuple[int, int]:
    user_message_count = sum(1 for message in payload.messages if message.role == "user")
    return user_message_count, user_message_count


def _model_messages(payload: ChatRequest) -> list[dict[str, str]]:
    return [
        {"role": message.role, "content": message.content}
        for message in payload.messages
    ]


def _latest_user_message_index(payload: ChatRequest) -> int | None:
    for index in range(len(payload.messages) - 1, -1, -1):
        if payload.messages[index].role == "user":
            return index
    return None


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
