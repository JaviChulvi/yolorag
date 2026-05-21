from __future__ import annotations

from typing import Any, Protocol

from yolorag.usage.models import TokenUsage


class UsageExtractor(Protocol):
    def extract(self, raw: dict[str, Any]) -> TokenUsage:
        ...


class OpenAIUsageExtractor:
    def extract(self, raw: dict[str, Any]) -> TokenUsage:
        usage = raw.get("usage") or {}
        input_details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
        output_details = (
            usage.get("output_tokens_details")
            or usage.get("completion_tokens_details")
            or {}
        )

        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))

        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=usage.get("total_tokens", input_tokens + output_tokens),
            cached_input_tokens=input_details.get("cached_tokens", 0),
            reasoning_tokens=output_details.get("reasoning_tokens", 0),
        )


class GeminiUsageExtractor:
    def extract(self, raw: dict[str, Any]) -> TokenUsage:
        usage = raw.get("usage_metadata") or raw.get("usageMetadata") or {}
        input_tokens = usage.get("prompt_token_count", usage.get("promptTokenCount", 0))
        output_tokens = usage.get(
            "candidates_token_count",
            usage.get("candidatesTokenCount", 0),
        )

        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=usage.get("total_token_count", usage.get("totalTokenCount", 0)),
            reasoning_tokens=usage.get("thoughts_token_count", usage.get("thoughtsTokenCount", 0)),
        )


def extractor_for_provider(provider: str) -> UsageExtractor:
    provider_key = provider.lower()
    if provider_key in {"openai", "openai-compatible", "deepseek", "kimi"}:
        return OpenAIUsageExtractor()
    if provider_key == "gemini":
        return GeminiUsageExtractor()
    raise ValueError(f"No usage extractor registered for provider {provider!r}")
