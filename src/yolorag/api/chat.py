from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from starlette.responses import StreamingResponse

from yolorag.api.schemas import ChatRequest
from yolorag.api.sse import content_event_stream, error_event_stream
from yolorag.runtime import YoloRAGRuntime, build_runtime


router = APIRouter()


@router.post("/chat")
async def chat(payload: ChatRequest, request: Request) -> StreamingResponse:
    user_message = payload.latest_user_message()
    if user_message is None:
        raise HTTPException(status_code=400, detail="At least one user message is required.")

    session_id = payload.session_id or str(uuid4())
    total_messages = 0
    active_messages = 0

    try:
        runtime = _runtime(request)
        total_messages, active_messages = _pending_message_counts(runtime, session_id)
        stream = content_event_stream(
            runtime.stream_answer(
                user_message=_compose_user_message(payload, user_message.content),
                conversation_id=session_id,
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
        },
    )


def _runtime(request: Request) -> YoloRAGRuntime:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        runtime = build_runtime()
        request.app.state.runtime = runtime
    return runtime


def _pending_message_counts(runtime: YoloRAGRuntime, session_id: str) -> tuple[int, int]:
    total_messages, active_messages = runtime.message_counts(session_id)
    return total_messages + 1, active_messages + 1


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
