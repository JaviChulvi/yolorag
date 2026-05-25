from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from yolorag.core.conversation import ConversationLogger
from yolorag.core.orchestrator import MAIN_SYSTEM_PROMPT
from yolorag.core.transcripts import (
    schedule_assistant_message_write,
    schedule_user_message_write,
)
from yolorag.providers.base import LLMProvider, LLMRequest, Message
from yolorag.tools.base import ToolCallRequest, ToolCallResult
from yolorag.tools.router import ToolRouter


logger = logging.getLogger(__name__)

MAINTAINER_OPERATING_PROTOCOL = """\
Maintainer operating protocol for GitHub/support threads:

1. Before drafting, classify the thread type:
   - bug report
   - usage question
   - docs/UX gap
   - feature request
   - environment/install issue
   - model/export/deployment issue
   - platform/backend issue
   - resolved/follow-up/thanks

2. Identify the thread stage before choosing the reply shape:
   initial report, clarification needed, new evidence provided, root cause likely,
   workaround confirmed, PR/fix merged, or closure/thanks.

3. Keep strict evidence discipline. Exact versions, dates, timestamps, paths,
   metrics, commands, and error text must come from the user, retrieved docs,
   repository/tool evidence, or prior conversation. If evidence is missing, say
   what is missing instead of inventing details.

4. Gather bounded evidence. Use docs_search for Ultralytics docs/product/API,
   install, training, export, deployment, or troubleshooting claims unless the
   needed evidence is already present. Use repository/GitHub tools when available
   and allowed. Ask only for the next artifacts required to decide, not a generic
   laundry list.

5. Distinguish root-cause fixes from downstream workarounds. Prefer the layer
   that introduced a broken invariant when suggesting a PR, and label defensive
   workarounds as workarounds.

6. If docs and code disagree, state the current code behavior first, then mention
   the docs wording as a possible documentation cleanup. Prefer current code for
   current behavior.

7. Compile the final public reply in maintainer style: brief acknowledgement,
   likely mechanism or uncertainty, one to three concrete next actions, only the
   missing artifacts needed for the next decision, then stop. Default to 80-140
   words unless an MRE or complex technical explanation truly needs more.

8. Keep the final answer external-facing. Do not narrate internal drafting,
   evaluation, search, or tool-use process. Cite docs or code naturally when
   useful, but write the final message as the public support reply itself.
"""

DEEP_AGENT_SYSTEM_PROMPT = f"""\
{MAIN_SYSTEM_PROMPT}

You are running in deep agent mode. Use tools when they materially improve the answer,
especially docs_search for Ultralytics documentation questions. Do not expose hidden
reasoning. Stream concise progress summaries through the application, then provide a
direct final answer. If a tool fails, recover gracefully and continue with the best
available information. For Ultralytics documentation, product, API, install, training,
export, deployment, or troubleshooting questions, call docs_search before the final
answer unless the needed documentation context is already present in the conversation.
When GitHub tools are available, use them only for the repositories allowed by their
tool descriptions and never try to inspect unrelated repositories.

{MAINTAINER_OPERATING_PROTOCOL}
"""

class DeepAgentOrchestrator:
    def __init__(
        self,
        *,
        provider: LLMProvider,
        model: str,
        tool_router: ToolRouter,
        conversation_logger: ConversationLogger | None = None,
        max_steps: int = 6,
        tool_timeout_seconds: float = 20.0,
        max_tokens: int = 1800,
    ) -> None:
        self.provider = provider
        self.model = model
        self.tool_router = tool_router
        self.conversation_logger = conversation_logger
        self.max_steps = max_steps
        self.tool_timeout_seconds = tool_timeout_seconds
        self.max_tokens = max_tokens

    async def stream_answer(
        self,
        *,
        user_message: str,
        conversation_id: str,
        conversation_messages: list[Message] | None = None,
        raw_user_message: str | None = None,
        request_id: str | None = None,
        user_message_index: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        schedule_user_message_write(
            self.conversation_logger,
            conversation_id=conversation_id,
            request_id=request_id,
            raw_user_message=raw_user_message or user_message,
            user_message_index=user_message_index,
        )

        messages = self._initial_messages(
            user_message=user_message,
            conversation_messages=conversation_messages,
        )
        answer_chunks: list[str] = []
        tool_call_count = 0
        started = time.perf_counter()

        yield {
            "type": "status",
            "message": "Starting deep agent",
            "step": 0,
        }

        for step_index in range(1, self.max_steps + 1):
            yield {
                "type": "status",
                "message": "Thinking",
                "step": step_index,
            }
            tool_schemas = await self.tool_router.openai_schemas()
            response = await self.provider.complete(
                LLMRequest(
                    messages=messages,
                    model=self.model,
                    mode="deep",
                    max_tokens=self.max_tokens,
                    tools=tool_schemas or None,
                    metadata={"agent_step": step_index},
                )
            )

            if response.tool_calls:
                messages.append(_assistant_tool_message(response.content, response.tool_calls))
                for tool_call in response.tool_calls:
                    tool_name, arguments, call_id = _parse_tool_call(tool_call)
                    if not tool_name:
                        continue

                    tool_call_count += 1
                    yield {
                        "type": "tool_call",
                        "tool": tool_name,
                        "arguments": _safe_arguments(arguments),
                        "step": step_index,
                    }
                    tool_result = await self._call_tool(
                        ToolCallRequest(
                            name=tool_name,
                            arguments=arguments,
                            call_id=call_id,
                        )
                    )
                    yield {
                        "type": "tool_result",
                        "tool": tool_name,
                        "summary": _tool_result_summary(tool_result),
                        "error": tool_result.error,
                        "step": step_index,
                    }
                    messages.append(_tool_result_message(call_id, tool_result))
                continue

            answer = response.content.strip()
            if not answer:
                answer = "I could not produce a final answer from the available context."

            for chunk in _chunks(answer):
                answer_chunks.append(chunk)
                yield {"type": "content", "content": chunk}

            final_answer = "".join(answer_chunks)
            schedule_assistant_message_write(
                self.conversation_logger,
                conversation_id=conversation_id,
                request_id=request_id,
                assistant_message=final_answer,
                user_message_index=user_message_index,
                retrieved_document_ids=[],
                provider=response.provider,
                model=response.model,
            )
            yield {
                "type": "done",
                "step_count": step_index,
                "tool_call_count": tool_call_count,
                "latency_ms": _elapsed_ms(started),
            }
            return

        fallback = (
            "I reached the deep agent step limit before completing the task. "
            "Here is the best partial answer I can give from the work so far."
        )
        for chunk in _chunks(fallback):
            answer_chunks.append(chunk)
            yield {"type": "content", "content": chunk}
        final_answer = "".join(answer_chunks)
        schedule_assistant_message_write(
            self.conversation_logger,
            conversation_id=conversation_id,
            request_id=request_id,
            assistant_message=final_answer,
            user_message_index=user_message_index,
            retrieved_document_ids=[],
            provider=self.provider.provider_name,
            model=self.model,
        )
        yield {
            "type": "done",
            "step_count": self.max_steps,
            "tool_call_count": tool_call_count,
            "latency_ms": _elapsed_ms(started),
            "stopped_reason": "max_steps",
        }

    async def _call_tool(self, request: ToolCallRequest) -> ToolCallResult:
        try:
            return await self._call_tool_with_timeout(request)
        except Exception as exc:
            logger.warning("Tool call %s failed.", request.name, exc_info=True)
            return ToolCallResult(
                name=request.name,
                output={"error": f"{type(exc).__name__}: {exc}"},
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _call_tool_with_timeout(self, request: ToolCallRequest) -> ToolCallResult:
        import asyncio

        return await asyncio.wait_for(
            self.tool_router.call(request),
            timeout=self.tool_timeout_seconds,
        )

    def _initial_messages(
        self,
        *,
        user_message: str,
        conversation_messages: list[Message] | None,
    ) -> list[Message]:
        body = (
            [dict(message) for message in conversation_messages]
            if conversation_messages
            else [{"role": "user", "content": user_message}]
        )
        return [{"role": "system", "content": DEEP_AGENT_SYSTEM_PROMPT}, *body]


def _assistant_tool_message(content: str, tool_calls: list[dict[str, Any]]) -> Message:
    return {
        "role": "assistant",
        "content": content or "",
        "tool_calls": tool_calls,
    }


def _tool_result_message(call_id: str | None, result: ToolCallResult) -> Message:
    message: Message = {
        "role": "tool",
        "content": _json_dumps(result.output),
    }
    if call_id:
        message["tool_call_id"] = call_id
    return message


def _parse_tool_call(tool_call: dict[str, Any]) -> tuple[str | None, dict[str, Any], str | None]:
    function = tool_call.get("function") or {}
    name = function.get("name")
    raw_arguments = function.get("arguments") or "{}"
    if isinstance(raw_arguments, dict):
        arguments = raw_arguments
    else:
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            arguments = {}
    return name, arguments, tool_call.get("id")


def _safe_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    safe = dict(arguments)
    for key in list(safe):
        if any(secret in key.lower() for secret in ("key", "token", "secret", "password")):
            safe[key] = "[redacted]"
    return safe


def _tool_result_summary(result: ToolCallResult) -> str:
    if result.error:
        return result.error
    output = result.output
    if isinstance(output, dict):
        results = output.get("results")
        if isinstance(results, list):
            return f"{len(results)} result(s)"
        content = output.get("content")
        if isinstance(content, list):
            return f"{len(content)} content block(s)"
    text = _json_dumps(output)
    return text[:240] + ("..." if len(text) > 240 else "")


def _json_dumps(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) > 16000:
        return text[:15997] + "..."
    return text


def _chunks(text: str, chunk_size: int = 120) -> list[str]:
    if not text:
        return [""]
    return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
