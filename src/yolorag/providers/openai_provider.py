from __future__ import annotations

import time

from openai import AsyncOpenAI

from yolorag.providers.base import LLMRequest, LLMResponse
from yolorag.usage.cost_calculator import CostCalculator
from yolorag.usage.extractors import OpenAIUsageExtractor


class OpenAIProvider:
    provider_name = "openai"

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

    def _completion_kwargs(self, request: LLMRequest) -> dict:
        kwargs = {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "reasoning_effort": "high" if request.mode == "deep" else "low",
            "verbosity": "medium" if request.mode == "deep" else "low",
        }
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.tools:
            kwargs["tools"] = request.tools
        return kwargs
