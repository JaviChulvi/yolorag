from __future__ import annotations

import time

from openai import AsyncOpenAI

from yolorag.providers.base import LLMRequest, LLMResponse, LLMStreamEvent
from yolorag.usage.cost_calculator import CostCalculator
from yolorag.usage.extractors import OpenAIUsageExtractor


class OpenAIProvider:
    provider_name = "openai"
    reasoning_model_prefixes = ("gpt-5", "o1", "o3", "o4")
    verbosity_model_prefixes = ("gpt-5",)

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        provider_name: str = "openai",
        cost_calculator: CostCalculator | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.client = AsyncOpenAI(api_key=api_key, base_url=api_base)
        self.usage_extractor = OpenAIUsageExtractor()
        self.cost_calculator = cost_calculator or CostCalculator()

    async def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        kwargs = self._completion_kwargs(request)

        raw = await self.client.chat.completions.create(**kwargs)
        raw_dict = raw.model_dump()
        usage = self.usage_extractor.extract(raw_dict)
        cost = self.cost_calculator.calculate(
            provider=self.provider_name,
            model=request.model,
            usage=usage,
        )

        first_choice = raw.choices[0] if raw.choices else None
        message = first_choice.message if first_choice else None
        content = message.content if message and message.content else ""
        tool_calls = [
            call.model_dump()
            for call in getattr(message, "tool_calls", []) or []
        ]

        return LLMResponse(
            content=content,
            provider=self.provider_name,
            model=request.model,
            usage=usage,
            cost=cost,
            latency_ms=int((time.perf_counter() - started) * 1000),
            raw_response=raw_dict,
            tool_calls=tool_calls,
        )

    async def stream_complete(self, request: LLMRequest):
        kwargs = self._stream_completion_kwargs(request)
        stream = await self.client.chat.completions.create(**kwargs)

        async for chunk in stream:
            raw_chunk = chunk.model_dump()
            usage = (
                self.usage_extractor.extract(raw_chunk)
                if raw_chunk.get("usage")
                else None
            )
            for choice in raw_chunk.get("choices", []):
                delta = choice.get("delta") or {}
                content = delta.get("content") or ""
                if content:
                    yield LLMStreamEvent(
                        content=content,
                        usage=usage,
                        raw_chunk=raw_chunk,
                    )
            if usage is not None:
                yield LLMStreamEvent(usage=usage, raw_chunk=raw_chunk)

    def _completion_kwargs(self, request: LLMRequest) -> dict:
        kwargs = {
            "model": request.model,
            "messages": request.messages,
        }
        if self._uses_reasoning_controls(request.model):
            kwargs["reasoning_effort"] = "high" if request.mode == "deep" else "low"
            if self._uses_verbosity_control(request.model):
                kwargs["verbosity"] = "medium" if request.mode == "deep" else "low"
            if request.max_tokens is not None:
                kwargs["max_completion_tokens"] = request.max_tokens
        else:
            kwargs["temperature"] = request.temperature
            if request.max_tokens is not None:
                kwargs["max_tokens"] = request.max_tokens
        if request.tools:
            kwargs["tools"] = request.tools
            kwargs.pop("reasoning_effort", None)
            kwargs.pop("verbosity", None)
        return kwargs

    def _stream_completion_kwargs(self, request: LLMRequest) -> dict:
        kwargs = self._completion_kwargs(request)
        kwargs["stream"] = True
        if self.provider_name == "openai":
            kwargs["stream_options"] = {"include_usage": True}
        return kwargs

    def _uses_reasoning_controls(self, model: str) -> bool:
        return _normalized_model_name(model).startswith(self.reasoning_model_prefixes)

    def _uses_verbosity_control(self, model: str) -> bool:
        return _normalized_model_name(model).startswith(self.verbosity_model_prefixes)


def _normalized_model_name(model: str) -> str:
    lowered = model.lower()
    if lowered.startswith("ft:"):
        return lowered.split(":", 1)[1]
    return lowered
