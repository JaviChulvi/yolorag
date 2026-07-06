from __future__ import annotations

from yolorag.providers.base import ResponseMode


BUILT_IN_MODEL_DEFAULTS: dict[str, dict[ResponseMode, str]] = {
    "openai": {
        "fast": "gpt-5.4-mini",
        "deep": "gpt-5.5",
    },
    "deepseek": {
        "fast": "deepseek-v4-flash",
        "deep": "deepseek-v4-pro",
    },
    "gemini": {
        "fast": "gemini-3.1-flash-lite",
        "deep": "gemma-4-31b-it",
    },
}


def default_model_for(provider_name: str, mode: ResponseMode) -> str:
    provider_defaults = model_matrix().get(provider_name)
    if provider_defaults is None:
        raise ValueError(f"No default models configured for provider {provider_name!r}.")
    return provider_defaults[mode]


def model_matrix() -> dict[str, dict[ResponseMode, str]]:
    return BUILT_IN_MODEL_DEFAULTS
