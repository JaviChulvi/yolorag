from __future__ import annotations

import asyncio
import logging

from yolorag.core.conversation import ConversationLogger, ConversationMessageLog


logger = logging.getLogger(__name__)


def schedule_user_message_write(
    conversation_logger: ConversationLogger | None,
    *,
    conversation_id: str,
    request_id: str | None,
    raw_user_message: str,
    user_message_index: int | None,
) -> None:
    schedule_transcript_write(
        conversation_logger,
        [
            ConversationMessageLog(
                conversation_id=conversation_id,
                role="user",
                content=raw_user_message,
                request_id=request_id,
                message_index=user_message_index,
            )
        ],
    )


def schedule_assistant_message_write(
    conversation_logger: ConversationLogger | None,
    *,
    conversation_id: str,
    request_id: str | None,
    assistant_message: str,
    user_message_index: int | None,
    retrieved_document_ids: list[str],
    provider: str,
    model: str,
) -> None:
    schedule_transcript_write(
        conversation_logger,
        [
            ConversationMessageLog(
                conversation_id=conversation_id,
                role="assistant",
                content=assistant_message,
                request_id=request_id,
                message_index=None if user_message_index is None else user_message_index + 1,
                provider=provider,
                model=model,
                retrieved_document_ids=list(retrieved_document_ids),
            )
        ],
    )


def schedule_transcript_write(
    conversation_logger: ConversationLogger | None,
    messages: list[ConversationMessageLog],
) -> None:
    if conversation_logger is None or not messages:
        return
    asyncio.create_task(_append_transcript_messages(conversation_logger, messages))


async def _append_transcript_messages(
    conversation_logger: ConversationLogger,
    messages: list[ConversationMessageLog],
) -> None:
    try:
        await asyncio.to_thread(conversation_logger.append_messages, messages)
    except Exception:
        logger.warning("Failed to persist chat transcript messages.", exc_info=True)
