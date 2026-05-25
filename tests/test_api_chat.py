from __future__ import annotations

import unittest
from asyncio import run
from decimal import Decimal

from fastapi.testclient import TestClient

from yolorag.api.app import create_app
from yolorag.core.agent import DeepAgentOrchestrator
from yolorag.core.orchestrator import RAGOrchestrator
from yolorag.providers.base import LLMRequest, LLMResponse, LLMStreamEvent
from yolorag.retrieval.base import Document, RetrievalResult
from yolorag.runtime import YoloRAGAgentRuntime, YoloRAGRuntime
from yolorag.tools.base import ToolCallRequest, ToolCallResult
from yolorag.tools.router import ToolRouter
from yolorag.usage.models import CostBreakdown, TokenUsage


class RecordingProvider:
    provider_name = "test"

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []
        self.complete_calls = 0
        self.stream_calls = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.complete_calls += 1
        self.requests.append(request)
        user_message = request.messages[-1]["content"]
        return LLMResponse(
            content=f"Echo: {user_message}",
            provider=self.provider_name,
            model=request.model,
            usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            cost=CostBreakdown(total_usd=Decimal("0.000001"), pricing_source="test"),
            latency_ms=1,
            raw_response={"test": True},
        )

    async def stream_complete(self, request: LLMRequest):
        self.stream_calls += 1
        self.requests.append(request)
        user_message = request.messages[-1]["content"]
        yield LLMStreamEvent(content="Echo: ")
        yield LLMStreamEvent(content=user_message)
        yield LLMStreamEvent(
            usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            cost=CostBreakdown(total_usd=Decimal("0.000001"), pricing_source="test"),
        )


class ChatApiTests(unittest.TestCase):
    def test_chat_streams_sse_and_returns_session_headers(self) -> None:
        provider = RecordingProvider()
        runtime = YoloRAGRuntime(
            orchestrator=RAGOrchestrator(provider=provider, model="test-model")
        )
        client = TestClient(create_app(runtime=runtime))

        response = client.post(
            "/api/chat/fast",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertTrue(response.headers["X-Session-ID"])
        self.assertEqual(response.headers["X-Total-User-Messages"], "1")
        self.assertEqual(response.headers["X-Chat-Mode"], "fast")
        self.assertEqual(provider.stream_calls, 1)
        self.assertEqual(provider.complete_calls, 0)
        self.assertIn('data: {"content": "Echo: "}', response.text)
        self.assertIn('data: {"content": "hello"}', response.text)
        self.assertNotIn('"type": "metrics"', response.text)
        self.assertIn("data: [DONE]", response.text)

    def test_legacy_chat_route_aliases_fast_chat(self) -> None:
        provider = RecordingProvider()
        runtime = YoloRAGRuntime(
            orchestrator=RAGOrchestrator(provider=provider, model="test-model")
        )
        client = TestClient(create_app(runtime=runtime))

        response = client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Chat-Mode"], "fast")
        self.assertEqual(provider.stream_calls, 1)

    def test_fast_chat_can_stream_metrics_when_requested(self) -> None:
        provider = RecordingProvider()
        runtime = YoloRAGRuntime(
            orchestrator=RAGOrchestrator(provider=provider, model="test-model")
        )
        client = TestClient(create_app(runtime=runtime))

        response = client.post(
            "/api/chat/fast",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "include_metrics": True,
                "analytics": False,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn('data: {"content": "Echo: "}', response.text)
        self.assertIn('"type": "metrics"', response.text)
        self.assertIn('"provider": "test"', response.text)
        self.assertIn('"input_tokens": 10', response.text)
        self.assertIn('"estimated_cost_usd": 1e-06', response.text)
        self.assertIn("data: [DONE]", response.text)
        self.assertEqual(provider.stream_calls, 1)
        self.assertEqual(provider.complete_calls, 0)

    def test_chat_streams_llm_answer_when_retrieval_fails(self) -> None:
        provider = RecordingProvider()
        runtime = YoloRAGRuntime(
            orchestrator=RAGOrchestrator(
                provider=provider,
                model="test-model",
                retriever=FailingRetriever(),
            )
        )
        client = TestClient(create_app(runtime=runtime))

        with self.assertLogs("yolorag.core.orchestrator", level="WARNING"):
            response = client.post(
                "/api/chat/fast",
                json={"messages": [{"role": "user", "content": "How do I export a YOLO model?"}]},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn('data: {"content": "Echo: "}', response.text)
        self.assertIn('data: {"content": "How do I export a YOLO model?"}', response.text)
        self.assertIn("data: [DONE]", response.text)
        self.assertNotIn("Chat generation failed", response.text)

    def test_chat_reuses_session_history_for_follow_up_turns(self) -> None:
        provider = RecordingProvider()
        runtime = YoloRAGRuntime(
            orchestrator=RAGOrchestrator(provider=provider, model="test-model")
        )
        client = TestClient(create_app(runtime=runtime))

        first = client.post(
            "/api/chat/fast",
            json={"messages": [{"role": "user", "content": "first"}]},
        )
        session_id = first.headers["X-Session-ID"]
        second = client.post(
            "/api/chat/fast",
            json={
                "session_id": session_id,
                "messages": [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "Echo: first"},
                    {"role": "user", "content": "second"},
                ],
            },
        )

        self.assertEqual(second.headers["X-Session-ID"], session_id)
        self.assertEqual(second.headers["X-Total-User-Messages"], "2")
        second_request_messages = provider.requests[1].messages
        self.assertIn({"role": "user", "content": "first"}, second_request_messages)
        self.assertIn({"role": "assistant", "content": "Echo: first"}, second_request_messages)
        self.assertEqual(second_request_messages[-1], {"role": "user", "content": "second"})

    def test_chat_includes_page_context_when_present(self) -> None:
        provider = RecordingProvider()
        runtime = YoloRAGRuntime(
            orchestrator=RAGOrchestrator(provider=provider, model="test-model")
        )
        client = TestClient(create_app(runtime=runtime))

        client.post(
            "/api/chat/fast",
            json={
                "messages": [{"role": "user", "content": "what page is this?"}],
                "context": {
                    "title": "YOLO Export",
                    "url": "https://docs.ultralytics.com/modes/export/",
                    "path": "/modes/export/",
                    "description": "Export docs",
                },
            },
        )

        sent_message = provider.requests[0].messages[-1]["content"]
        self.assertIn("Page context:", sent_message)
        self.assertIn("Title: YOLO Export", sent_message)
        self.assertIn("User message:\nwhat page is this?", sent_message)

    def test_fast_chat_routes_retrieval_on_raw_user_message_not_instructions(self) -> None:
        provider = RecordingProvider()
        retriever = StaticRetriever(score=0.2)
        runtime = YoloRAGRuntime(
            orchestrator=RAGOrchestrator(
                provider=provider,
                model="test-model",
                retriever=retriever,
            )
        )
        client = TestClient(create_app(runtime=runtime))

        response = client.post(
            "/api/chat/fast",
            json={
                "messages": [{"role": "user", "content": "hola me llamo javi"}],
                "instructions": "You are the YoloRAG assistant for Ultralytics docs.",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(retriever.calls, [("hola me llamo javi", 2)])
        self.assertNotIn("Sources:", response.text)
        self.assertFalse(
            any(
                "Relevant retrieved context" in message["content"]
                for message in provider.requests[0].messages
            )
        )

    def test_deep_chat_runs_tool_loop_and_returns_final_text_only(self) -> None:
        provider = ToolCallingProvider()
        deep_runtime = YoloRAGAgentRuntime(
            orchestrator=DeepAgentOrchestrator(
                provider=provider,
                model="deep-model",
                tool_router=ToolRouter(tools=[FakeDocsTool()]),
                max_steps=3,
            )
        )
        client = TestClient(create_app(deep_runtime=deep_runtime))

        response = client.post(
            "/api/chat/deep",
            json={"messages": [{"role": "user", "content": "how do I export?"}]},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/plain", response.headers["content-type"])
        self.assertEqual(response.headers["X-Chat-Mode"], "deep")
        self.assertEqual(response.text, "Use yolo export.")
        self.assertNotIn("tool_call", response.text)
        self.assertNotIn("tool_result", response.text)
        self.assertNotIn("data:", response.text)
        self.assertEqual(provider.complete_calls, 2)
        self.assertTrue(provider.requests[0].tools)
        self.assertEqual(provider.requests[1].messages[-1]["role"], "tool")

    def test_deep_chat_events_stream_typed_agent_events(self) -> None:
        provider = ToolCallingProvider()
        deep_runtime = YoloRAGAgentRuntime(
            orchestrator=DeepAgentOrchestrator(
                provider=provider,
                model="deep-model",
                tool_router=ToolRouter(tools=[FakeDocsTool()]),
                max_steps=3,
            )
        )
        client = TestClient(create_app(deep_runtime=deep_runtime))

        response = client.post(
            "/api/chat/deep/events",
            json={"messages": [{"role": "user", "content": "how do I export?"}]},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertEqual(response.headers["X-Chat-Mode"], "deep")
        self.assertEqual(response.headers["X-Stream-Format"], "agent-events")
        self.assertIn('"type": "status"', response.text)
        self.assertIn('"message": "Starting deep agent"', response.text)
        self.assertIn('"type": "tool_call"', response.text)
        self.assertIn('"tool": "docs_search"', response.text)
        self.assertIn('"type": "tool_result"', response.text)
        self.assertIn('"summary": "1 result(s)"', response.text)
        self.assertIn('"type": "content"', response.text)
        self.assertIn('"content": "Use yolo export."', response.text)
        self.assertIn('"type": "done"', response.text)
        self.assertIn('"tool_call_count": 1', response.text)
        self.assertIn('"latency_ms":', response.text)
        self.assertIn("data: [DONE]", response.text)
        self.assertEqual(provider.complete_calls, 2)


class StaticRetriever:
    def __init__(self, score: float = 0.9) -> None:
        self.score = score
        self.calls: list[tuple[str, int]] = []

    async def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        self.calls.append((query, top_k))
        return [
            RetrievalResult(
                document=Document(
                    id="doc-1",
                    title="Training Docs",
                    content="Use model.train(data='coco8.yaml', epochs=100).",
                ),
                score=self.score,
                reason=f"Test retriever top_k={top_k}",
            )
        ]


class FailingRetriever:
    async def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        raise RuntimeError("vector store unavailable")


class FakeDocsTool:
    name = "docs_search"
    description = "Search test docs."
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    async def call(self, request: ToolCallRequest) -> ToolCallResult:
        return ToolCallResult(
            name=self.name,
            output={
                "results": [
                    {
                        "title": "Export",
                        "url": "https://docs.ultralytics.com/modes/export/",
                        "content": "Use yolo export.",
                    }
                ]
            },
        )


class ToolCallingProvider(RecordingProvider):
    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.complete_calls += 1
        self.requests.append(request)
        if self.complete_calls == 1:
            content = ""
            tool_calls = [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "docs_search",
                        "arguments": '{"query": "export"}',
                    },
                }
            ]
        else:
            content = "Use yolo export."
            tool_calls = []
        return LLMResponse(
            content=content,
            provider=self.provider_name,
            model=request.model,
            usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            cost=CostBreakdown(total_usd=Decimal("0.000001"), pricing_source="test"),
            latency_ms=1,
            raw_response={"test": True},
            tool_calls=tool_calls,
        )


class OrchestratorRetrievalTests(unittest.TestCase):
    def test_main_prompt_does_not_expose_internal_mode_or_budget(self) -> None:
        provider = RecordingProvider()
        orchestrator = RAGOrchestrator(provider=provider, model="test-model")

        run(orchestrator.answer("hello", mode="deep"))

        main_prompt = provider.requests[0].messages[0]["content"]
        self.assertIn("YoloRAG", main_prompt)
        self.assertIn("Ultralytics YOLO documentation", main_prompt)
        self.assertNotIn("Mode=", main_prompt)
        self.assertNotIn("reasoning_budget", main_prompt)
        self.assertEqual(provider.requests[0].mode, "deep")

    def test_fast_retrieves_high_confidence_context_without_text_gate(self) -> None:
        provider = RecordingProvider()
        orchestrator = RAGOrchestrator(
            provider=provider,
            model="test-model",
            retriever=StaticRetriever(),
            retrieval_top_k=5,
        )

        result = run(orchestrator.answer("Can this work in my pipeline?", mode="fast"))

        sent_messages = provider.requests[0].messages
        context_messages = [
            message["content"]
            for message in sent_messages
            if message["role"] == "system" and "Relevant retrieved context" in message["content"]
        ]
        self.assertEqual(len(context_messages), 1)
        self.assertTrue(result.trace.retrieval_used)
        self.assertIn("Fast mode uses lightweight retrieval", result.trace.route_reason)

    def test_fast_skips_low_confidence_context(self) -> None:
        provider = RecordingProvider()
        retriever = StaticRetriever(score=0.2)
        orchestrator = RAGOrchestrator(
            provider=provider,
            model="test-model",
            retriever=retriever,
            retrieval_top_k=5,
        )

        result = run(orchestrator.answer("hello", mode="fast"))

        self.assertEqual(retriever.calls, [("hello", 5)])
        self.assertFalse(result.trace.retrieval_used)
        self.assertFalse(
            any(
                "Relevant retrieved context" in message["content"]
                for message in provider.requests[0].messages
            )
        )
        self.assertIn("filters low-confidence context", result.trace.route_reason)

    def test_fast_routes_retrieval_with_raw_user_message_when_prompt_is_composed(self) -> None:
        provider = RecordingProvider()
        retriever = StaticRetriever(score=0.2)
        orchestrator = RAGOrchestrator(
            provider=provider,
            model="test-model",
            retriever=retriever,
            retrieval_top_k=5,
        )
        composed_message = (
            "Instructions:\nYou are the YoloRAG assistant for Ultralytics docs.\n\n"
            "User message:\nhola me llamo javi"
        )

        result = run(
            orchestrator.answer(
                composed_message,
                mode="fast",
                raw_user_message="hola me llamo javi",
            )
        )

        self.assertEqual(retriever.calls, [("hola me llamo javi", 5)])
        self.assertFalse(result.trace.retrieval_used)
        self.assertFalse(
            any(
                "Relevant retrieved context" in message["content"]
                for message in provider.requests[0].messages
            )
        )

    def test_fast_retrieves_context_on_domain_follow_up_turns(self) -> None:
        provider = RecordingProvider()
        orchestrator = RAGOrchestrator(
            provider=provider,
            model="test-model",
            retriever=StaticRetriever(),
            retrieval_top_k=5,
        )

        run(orchestrator.answer("How do I train a YOLO model?", conversation_id="thread-1"))
        run(orchestrator.answer("How do I export a YOLO model?", conversation_id="thread-1"))

        second_request_messages = provider.requests[1].messages
        self.assertTrue(
            any(
                message["role"] == "system"
                and "Relevant retrieved context" in message["content"]
                for message in second_request_messages
            )
        )

    def test_retrieval_failure_falls_back_to_llm_only_answer(self) -> None:
        provider = RecordingProvider()
        orchestrator = RAGOrchestrator(
            provider=provider,
            model="test-model",
            retriever=FailingRetriever(),
        )

        with self.assertLogs("yolorag.core.orchestrator", level="WARNING"):
            result = run(orchestrator.answer("How do I export a YOLO model?"))

        self.assertEqual(result.answer, "Echo: How do I export a YOLO model?")
        self.assertFalse(result.trace.retrieval_used)
        self.assertEqual(
            result.trace.retrieval_error,
            "RuntimeError: vector store unavailable",
        )
        self.assertFalse(
            any(
                "Relevant retrieved context" in message["content"]
                for message in provider.requests[0].messages
            )
        )


if __name__ == "__main__":
    unittest.main()
