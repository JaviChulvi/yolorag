from __future__ import annotations

import json
import unittest
from asyncio import run
from decimal import Decimal
from unittest.mock import patch

from fastapi.testclient import TestClient

from yolorag.api.app import create_app
from yolorag.core.agent import DeepAgentOrchestrator
from yolorag.core.orchestrator import RAGOrchestrator
from yolorag.providers.base import LLMRequest, LLMResponse, LLMStreamEvent
from yolorag.retrieval.base import Document, RetrievalResult, RetrievalTrace
from yolorag.runtime import YoloRAGAgentRuntime, YoloRAGRuntime
from yolorag.tools.base import ToolCallRequest, ToolCallResult
from yolorag.tools.docs_search import DocsSearchTool
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

    def test_fast_chat_selects_runtime_from_query_params(self) -> None:
        provider = RecordingProvider()
        runtime = YoloRAGRuntime(
            orchestrator=RAGOrchestrator(provider=provider, model="test-model")
        )
        client = TestClient(create_app())

        with patch("yolorag.api.chat.build_runtime", return_value=runtime) as build:
            response = client.post(
                "/api/chat/fast?provider=deepseek&knowledge_provider=postgresql",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )

        self.assertEqual(response.status_code, 200)
        build.assert_called_once_with(
            provider_name="deepseek",
            mode="fast",
            knowledge_provider="postgresql",
            conversation_provider="postgresql",
        )
        self.assertEqual(response.headers["X-LLM-Provider"], "deepseek")
        self.assertEqual(response.headers["X-Knowledge-Provider"], "postgresql")
        self.assertEqual(response.headers["X-Conversation-Provider"], "postgresql")
        self.assertEqual(provider.stream_calls, 1)

    def test_fast_chat_defaults_to_deepseek_and_mongodb(self) -> None:
        provider = RecordingProvider()
        runtime = YoloRAGRuntime(
            orchestrator=RAGOrchestrator(provider=provider, model="test-model")
        )
        client = TestClient(create_app())

        with patch.dict("os.environ", {}, clear=True):
            with patch("yolorag.api.chat.build_runtime", return_value=runtime) as build:
                response = client.post(
                    "/api/chat/fast",
                    json={"messages": [{"role": "user", "content": "hello"}]},
                )

        self.assertEqual(response.status_code, 200)
        build.assert_called_once_with(
            provider_name="deepseek",
            mode="fast",
            knowledge_provider="mongodb",
            conversation_provider="mongodb",
        )
        self.assertEqual(response.headers["X-LLM-Provider"], "deepseek")
        self.assertEqual(response.headers["X-Knowledge-Provider"], "mongodb")

    def test_fast_chat_reuses_runtime_cache_per_provider_and_database(self) -> None:
        client = TestClient(create_app())

        with patch("yolorag.api.chat.build_runtime") as build:
            build.side_effect = lambda **_: YoloRAGRuntime(
                orchestrator=RAGOrchestrator(provider=RecordingProvider(), model="test-model")
            )
            for provider_name, knowledge_provider in [
                ("openai", "mongodb"),
                ("openai", "mongodb"),
                ("openai", "postgresql"),
            ]:
                response = client.post(
                    f"/api/chat/fast?provider={provider_name}&knowledge_provider={knowledge_provider}",
                    json={"messages": [{"role": "user", "content": "hello"}]},
                )
                self.assertEqual(response.status_code, 200)

        self.assertEqual(build.call_count, 2)

    def test_chat_rejects_unknown_runtime_selector(self) -> None:
        client = TestClient(create_app())

        response = client.post(
            "/api/chat/fast?provider=bogus",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported provider", response.text)

    def test_fast_chat_runs_bounded_tool_pass_then_streams_answer(self) -> None:
        provider = FastToolStreamingProvider()
        docs_tool = CountingDocsTool()
        runtime = YoloRAGRuntime(
            orchestrator=RAGOrchestrator(
                provider=provider,
                model="test-model",
                tool_router=ToolRouter(tools=[docs_tool]),
            )
        )
        client = TestClient(create_app(runtime=runtime))

        response = client.post(
            "/api/chat/fast",
            json={"messages": [{"role": "user", "content": "how do I export?"}]},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn('data: {"content": "Use "}', response.text)
        self.assertIn('data: {"content": "yolo export."}', response.text)
        self.assertIn("data: [DONE]", response.text)
        self.assertNotIn("tool_call", response.text)
        self.assertEqual(provider.complete_calls, 1)
        self.assertEqual(provider.stream_calls, 1)
        self.assertTrue(provider.requests[0].tools)
        self.assertIsNone(provider.requests[1].tools)
        self.assertEqual(docs_tool.calls, ["docs_search"])
        self.assertEqual(provider.requests[1].messages[-1]["role"], "tool")

    def test_fast_chat_skips_tools_when_planner_returns_no_tool(self) -> None:
        cases = [
            [{"role": "user", "content": "hello"}],
            [{"role": "user", "content": "what is the capital of France?"}],
            [
                {"role": "user", "content": "how do I export YOLO?"},
                {"role": "assistant", "content": "Use yolo export."},
                {"role": "user", "content": "what did I ask?"},
            ],
        ]

        for messages in cases:
            with self.subTest(latest_user_message=messages[-1]["content"]):
                provider = NoToolPlanningProvider()
                docs_tool = CountingDocsTool()
                runtime = YoloRAGRuntime(
                    orchestrator=RAGOrchestrator(
                        provider=provider,
                        model="test-model",
                        tool_router=ToolRouter(tools=[docs_tool]),
                    )
                )
                client = TestClient(create_app(runtime=runtime))

                response = client.post(
                    "/api/chat/fast",
                    json={"messages": messages},
                )

                self.assertEqual(response.status_code, 200)
                self.assertIn('data: {"content": "Hi there."}', response.text)
                self.assertEqual(provider.complete_calls, 1)
                self.assertEqual(provider.stream_calls, 1)
                self.assertEqual(docs_tool.calls, [])
                self.assertFalse(
                    any(
                        message["role"] == "tool"
                        for message in provider.requests[1].messages
                    )
                )

    def test_fast_tool_selection_uses_raw_user_message_when_prompt_is_composed(self) -> None:
        provider = NoToolPlanningProvider()
        docs_tool = CountingDocsTool()
        orchestrator = RAGOrchestrator(
            provider=provider,
            model="test-model",
            tool_router=ToolRouter(tools=[docs_tool]),
        )
        composed_message = (
            "Instructions:\nYou are the YoloRAG assistant for Ultralytics docs.\n\n"
            "User message:\nhola me llamo javi"
        )

        run(
            _collect_events(
                orchestrator.stream_answer(
                    composed_message,
                    mode="fast",
                    raw_user_message="hola me llamo javi",
                )
            )
        )

        self.assertEqual(provider.requests[0].messages[-1]["content"], "hola me llamo javi")
        self.assertEqual(provider.requests[1].messages[-1]["content"], composed_message)
        self.assertEqual(docs_tool.calls, [])

    def test_fast_chat_metrics_include_docs_search_retrieval_trace(self) -> None:
        provider = FastToolStreamingProvider()
        docs_tool = DocsSearchTool(StaticRetrieverWithTrace(score=0.9))
        runtime = YoloRAGRuntime(
            orchestrator=RAGOrchestrator(
                provider=provider,
                model="test-model",
                tool_router=ToolRouter(tools=[docs_tool]),
            )
        )
        client = TestClient(create_app(runtime=runtime))

        response = client.post(
            "/api/chat/fast",
            json={
                "messages": [{"role": "user", "content": "how do I export?"}],
                "include_metrics": True,
                "analytics": False,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn('"type": "metrics"', response.text)
        self.assertIn('"retrieval": 41', response.text)
        self.assertIn('"query_embedding": 5', response.text)
        self.assertIn('"vector_search": 17', response.text)
        self.assertIn('"rerank": 19', response.text)
        self.assertIn('"used": true', response.text)
        self.assertIn('"reranked": true', response.text)
        self.assertIn('"candidate_count": 3', response.text)
        self.assertIn('"returned_count": 1', response.text)
        self.assertIn('"document_ids": ["doc-1"]', response.text)
        metrics = _metrics_from_sse(response.text)
        timings = metrics["timings_ms"]
        self.assertEqual(timings["retrieval"], 41)
        self.assertEqual(timings["query_embedding"], 5)
        self.assertEqual(timings["vector_search"], 17)
        self.assertEqual(timings["rerank"], 19)
        self.assertGreaterEqual(timings["llm"], 1)
        self.assertEqual(
            timings["total"],
            timings["retrieval"] + timings["llm"] + timings["orchestration_overhead"],
        )

    def test_fast_chat_streams_llm_answer_without_direct_retrieval(self) -> None:
        provider = RecordingProvider()
        runtime = YoloRAGRuntime(
            orchestrator=RAGOrchestrator(
                provider=provider,
                model="test-model",
                retriever=FailingRetriever(),
            )
        )
        client = TestClient(create_app(runtime=runtime))

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

    def test_fast_chat_fallback_skips_direct_retrieval_for_casual_turn(self) -> None:
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
        self.assertEqual(retriever.calls, [])
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


class StaticRetrieverWithTrace(StaticRetriever):
    async def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        self.calls.append((query, top_k))
        return [
            RetrievalResult(
                document=Document(
                    id="doc-1",
                    title="Export Docs",
                    content="Use yolo export.",
                ),
                score=self.score,
                reason=f"Test retriever top_k={top_k}",
                trace=RetrievalTrace(
                    provider="test",
                    total_ms=41,
                    query_embedding_ms=5,
                    vector_search_ms=17,
                    rerank_ms=19,
                    candidate_count=3,
                    returned_count=1,
                    reranked=True,
                ),
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


class CountingDocsTool(FakeDocsTool):
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def call(self, request: ToolCallRequest) -> ToolCallResult:
        self.calls.append(request.name)
        return await super().call(request)


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


class FastToolStreamingProvider(RecordingProvider):
    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.complete_calls += 1
        self.requests.append(request)
        return LLMResponse(
            content="",
            provider=self.provider_name,
            model=request.model,
            usage=TokenUsage(input_tokens=8, output_tokens=2, total_tokens=10),
            cost=CostBreakdown(total_usd=Decimal("0.000001"), pricing_source="test"),
            latency_ms=1,
            raw_response={"test": True},
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "docs_search",
                        "arguments": '{"query": "export"}',
                    },
                }
            ],
        )

    async def stream_complete(self, request: LLMRequest):
        self.stream_calls += 1
        self.requests.append(request)
        yield LLMStreamEvent(content="Use ")
        yield LLMStreamEvent(content="yolo export.")
        yield LLMStreamEvent(
            usage=TokenUsage(input_tokens=12, output_tokens=4, total_tokens=16),
            cost=CostBreakdown(total_usd=Decimal("0.000002"), pricing_source="test"),
        )


class NoToolPlanningProvider(RecordingProvider):
    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.complete_calls += 1
        self.requests.append(request)
        return LLMResponse(
            content="NO_TOOL",
            provider=self.provider_name,
            model=request.model,
            usage=TokenUsage(input_tokens=8, output_tokens=1, total_tokens=9),
            cost=CostBreakdown(total_usd=Decimal("0.000001"), pricing_source="test"),
            latency_ms=1,
            raw_response={"test": True},
        )

    async def stream_complete(self, request: LLMRequest):
        self.stream_calls += 1
        self.requests.append(request)
        yield LLMStreamEvent(content="Hi there.")
        yield LLMStreamEvent(
            usage=TokenUsage(input_tokens=10, output_tokens=3, total_tokens=13),
            cost=CostBreakdown(total_usd=Decimal("0.000002"), pricing_source="test"),
        )


def _metrics_from_sse(response_text: str) -> dict:
    for frame in response_text.split("\n\n"):
        data = "".join(
            line.removeprefix("data:").strip()
            for line in frame.splitlines()
            if line.startswith("data:")
        )
        if not data or data == "[DONE]":
            continue
        payload = json.loads(data)
        if payload.get("type") == "metrics":
            return payload["metrics"]
    raise AssertionError("metrics event not found")


async def _collect_events(stream):
    return [event async for event in stream]


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

    def test_deep_retrieves_high_confidence_context_without_text_gate(self) -> None:
        provider = RecordingProvider()
        orchestrator = RAGOrchestrator(
            provider=provider,
            model="test-model",
            retriever=StaticRetriever(),
            retrieval_top_k=5,
        )

        result = run(orchestrator.answer("Can this work in my pipeline?", mode="deep"))

        sent_messages = provider.requests[0].messages
        context_messages = [
            message["content"]
            for message in sent_messages
            if message["role"] == "system" and "Relevant retrieved context" in message["content"]
        ]
        self.assertEqual(len(context_messages), 1)
        self.assertTrue(result.trace.retrieval_used)
        self.assertIn("Deep mode uses retrieval", result.trace.route_reason)

    def test_fast_fallback_skips_retrieval_for_non_docs_turns(self) -> None:
        cases = [
            "hello",
            "what is the capital of France?",
            "what did I ask?",
        ]

        for user_message in cases:
            with self.subTest(user_message=user_message):
                provider = RecordingProvider()
                retriever = StaticRetriever(score=0.2)
                orchestrator = RAGOrchestrator(
                    provider=provider,
                    model="test-model",
                    retriever=retriever,
                    retrieval_top_k=5,
                )

                result = run(orchestrator.answer(user_message, mode="fast"))

                self.assertEqual(retriever.calls, [])
                self.assertFalse(result.trace.retrieval_used)
                self.assertFalse(
                    any(
                        "Relevant retrieved context" in message["content"]
                        for message in provider.requests[0].messages
                    )
                )
                self.assertIn("skips direct fallback retrieval", result.trace.route_reason)

    def test_deep_skips_low_confidence_context(self) -> None:
        provider = RecordingProvider()
        retriever = StaticRetriever(score=0.2)
        orchestrator = RAGOrchestrator(
            provider=provider,
            model="test-model",
            retriever=retriever,
            retrieval_top_k=5,
        )

        result = run(orchestrator.answer("How do I export a YOLO model?", mode="deep"))

        self.assertEqual(retriever.calls, [("How do I export a YOLO model?", 5)])
        self.assertFalse(result.trace.retrieval_used)
        self.assertFalse(
            any(
                "Relevant retrieved context" in message["content"]
                for message in provider.requests[0].messages
            )
        )
        self.assertIn("filters low-confidence context", result.trace.route_reason)

    def test_fast_fallback_skips_retrieval_when_prompt_is_composed(self) -> None:
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

        self.assertEqual(retriever.calls, [])
        self.assertFalse(result.trace.retrieval_used)
        self.assertFalse(
            any(
                "Relevant retrieved context" in message["content"]
                for message in provider.requests[0].messages
            )
        )

    def test_deep_retrieves_context_on_domain_follow_up_turns(self) -> None:
        provider = RecordingProvider()
        orchestrator = RAGOrchestrator(
            provider=provider,
            model="test-model",
            retriever=StaticRetriever(),
            retrieval_top_k=5,
        )

        run(orchestrator.answer("How do I train a YOLO model?", conversation_id="thread-1", mode="deep"))
        run(orchestrator.answer("How do I export a YOLO model?", conversation_id="thread-1", mode="deep"))

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
            result = run(orchestrator.answer("How do I export a YOLO model?", mode="deep"))

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
