from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from yolorag.config.model_defaults import model_matrix
from yolorag.config.settings import getenv
from yolorag.providers.base import LLMProvider
from yolorag.providers.deepseek_provider import DeepSeekProvider
from yolorag.providers.gemini_provider import GeminiProvider
from yolorag.providers.openai_provider import OpenAIProvider


ProviderBuilder = Callable[[str | None], LLMProvider]


@dataclass(frozen=True)
class ProviderInfo:
    """Identity + pricing metadata for a provider shown in a UI picker.

    ``vision`` gates whether a provider appears in the describe picker. The
    model list is deliberately NOT stored here — it is derived from
    ``config/model_defaults.py`` (the single source of truth), so a provider's
    models are configured in one place for both chat resolution and the picker.
    ``pricing_provider`` is the genai-prices provider id when it differs from
    ``name`` (Gemini -> ``google``).
    """

    name: str
    label: str
    # Env vars checked in order for the API key (first non-empty wins).
    env_keys: list[str]
    pricing_provider: str = ""
    vision: bool = False
    extras: dict = field(default_factory=dict)

    def api_key(self) -> str | None:
        for key in self.env_keys:
            value = getenv(key)
            if value:
                return value
        return None


# Providers exposed in a UI picker. Registering a vision provider here makes it
# show up in the describe UI; its model list comes from ``model_defaults`` and it
# is built through ``get_llm_provider`` below — one source of truth per concern.
PROVIDER_INFO: dict[str, ProviderInfo] = {
    "gemini": ProviderInfo(
        name="gemini",
        label="Gemini",
        env_keys=["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        pricing_provider="google",
        vision=True,
    ),
}


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


def _configured_models(provider_name: str) -> list[str]:
    """Distinct models configured for a provider in ``model_defaults`` (fast first)."""
    modes = model_matrix().get(_normalize_provider_name(provider_name), {})
    return list(dict.fromkeys(modes.values()))


def default_describe_model(provider_name: str) -> str | None:
    """Model pre-selected for a describe request when the caller sends none.

    The provider's ``fast`` default (falling back to its first configured
    model), so the dropdown default and the endpoint default stay in sync.
    """
    modes = model_matrix().get(_normalize_provider_name(provider_name), {})
    if not modes:
        return None
    return modes.get("fast") or next(iter(modes.values()), None)


def list_vision_providers() -> list[dict]:
    """Vision-provider metadata for the UI — models + whether a key is set.

    Models come from ``config/model_defaults.py`` so the picker only ever shows
    configured models. Never includes secrets; ``available`` just reflects
    env-key presence.
    """
    providers: list[dict] = []
    for info in PROVIDER_INFO.values():
        if not info.vision:
            continue
        models = _configured_models(info.name)
        providers.append(
            {
                "name": info.name,
                "label": info.label,
                "models": models,
                "default_model": default_describe_model(info.name) or (models[0] if models else ""),
                "available": info.api_key() is not None,
                "env_keys": list(info.env_keys),
            }
        )
    return providers


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


def _build_gemini(api_base: str | None = None) -> LLMProvider:
    return GeminiProvider(
        api_key=_require_env_any(["GEMINI_API_KEY", "GOOGLE_API_KEY"]),
        api_base=api_base or getenv("GEMINI_API_BASE"),
    )


PROVIDERS: dict[str, ProviderBuilder] = {
    "openai": _build_openai,
    "deepseek": _build_deepseek,
    "gemini": _build_gemini,
}


def _normalize_provider_name(provider_name: str) -> str:
    return provider_name.lower().strip()


def _require_env(name: str) -> str:
    value = getenv(name)
    if value:
        return value
    raise RuntimeError(f"Missing required environment variable {name}.")


def _require_env_any(names: list[str]) -> str:
    for name in names:
        value = getenv(name)
        if value:
            return value
    raise RuntimeError(f"Missing required environment variable {' or '.join(names)}.")
