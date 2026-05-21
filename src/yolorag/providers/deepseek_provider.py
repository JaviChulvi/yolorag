from __future__ import annotations

from yolorag.providers.base import LLMRequest
from yolorag.providers.openai_provider import OpenAIProvider
from yolorag.usage.cost_calculator import CostCalculator


DEEPSEEK_API_BASE = "https://api.deepseek.com"


class DeepSeekProvider(OpenAIProvider):
    provider_name = "deepseek"

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        cost_calculator: CostCalculator | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            api_base=api_base or DEEPSEEK_API_BASE,
            provider_name=self.provider_name,
            cost_calculator=cost_calculator,
        )

    def _completion_kwargs(self, request: LLMRequest) -> dict:
        kwargs = {
            "model": request.model,
            "messages": request.messages,
        }
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.tools:
            kwargs["tools"] = request.tools

        if request.mode == "deep":
            kwargs["reasoning_effort"] = "high"
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        else:
            kwargs["temperature"] = request.temperature
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        return kwargs
