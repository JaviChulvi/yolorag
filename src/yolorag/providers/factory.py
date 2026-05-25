from __future__ import annotations

from collections.abc import Callable

from yolorag.config.settings import getenv
from yolorag.providers.base import LLMProvider
from yolorag.providers.deepseek_provider import DeepSeekProvider
from yolorag.providers.openai_provider import OpenAIProvider


ProviderBuilder = Callable[[str | None], LLMProvider]


def get_llm_provider(provider_name: str, api_base: str | None = None) -> LLMProvider:
    normalized_name = _normalize_provider_name(provider_name)
    try:
        builder = PROVIDERS[normalized_name]
    except KeyError:
        supported = ", ".join(sorted(PROVIDERS))
        raise ValueError(
            f"Unsupported provider {provider_name!r}. Supported providers: {supported}."
        ) from None
    return builder(api_base)


def _build_openai(api_base: str | None = None) -> LLMProvider:
    return OpenAIProvider(
        api_key=_require_env("OPENAI_API_KEY"),
        api_base=api_base or getenv("OPENAI_BASE_URL"),
    )


def _build_deepseek(api_base: str | None = None) -> LLMProvider:
    return DeepSeekProvider(
        api_key=_require_env("DEEPSEEK_API_KEY"),
        api_base=api_base or getenv("DEEPSEEK_BASE_URL"),
    )


PROVIDERS: dict[str, ProviderBuilder] = {
    "openai": _build_openai,
    "deepseek": _build_deepseek,
}


def _normalize_provider_name(provider_name: str) -> str:
    return provider_name.lower().strip()


def _require_env(name: str) -> str:
    value = getenv(name)
    if value:
        return value
    raise RuntimeError(f"Missing required environment variable {name}.")
