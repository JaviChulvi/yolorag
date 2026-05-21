from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from yolorag.config.settings import getenv
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
}


DEFAULT_MODEL_CONFIG_PATH = Path("models.json")


def default_model_for(
    provider_name: str,
    mode: ResponseMode,
    config_path: str | Path | None = None,
) -> str:
    provider_defaults = model_matrix(config_path=config_path).get(provider_name)
    if provider_defaults is None:
        raise ValueError(f"No default models configured for provider {provider_name!r}.")
    return provider_defaults[mode]


def model_matrix(config_path: str | Path | None = None) -> dict[str, dict[ResponseMode, str]]:
    external_defaults = _load_external_model_defaults(config_path=config_path)
    return external_defaults or BUILT_IN_MODEL_DEFAULTS


def _load_external_model_defaults(
    config_path: str | Path | None = None,
) -> dict[str, dict[ResponseMode, str]] | None:
    path = _resolve_config_path(config_path=config_path)
    if path is None or not path.exists():
        return None

    raw = json.loads(path.read_text())
    return _normalize_model_defaults(raw=raw, path=path)


def _resolve_config_path(config_path: str | Path | None) -> Path | None:
    if config_path:
        return Path(config_path)

    configured_path = getenv("YOLORAG_MODELS_CONFIG")
    if configured_path:
        return Path(configured_path)

    if DEFAULT_MODEL_CONFIG_PATH.exists():
        return DEFAULT_MODEL_CONFIG_PATH

    return None


def _normalize_model_defaults(
    raw: dict[str, Any],
    path: Path,
) -> dict[str, dict[ResponseMode, str]]:
    normalized: dict[str, dict[ResponseMode, str]] = {}

    for provider_name, provider_config in raw.items():
        if not isinstance(provider_config, dict):
            raise ValueError(f"Invalid provider config for {provider_name!r} in {path}.")

        normalized[provider_name] = {
            "fast": _extract_model_name(provider_name, "fast", provider_config, path),
            "deep": _extract_model_name(provider_name, "deep", provider_config, path),
        }

    return normalized


def _extract_model_name(
    provider_name: str,
    mode: ResponseMode,
    provider_config: dict[str, Any],
    path: Path,
) -> str:
    mode_config = provider_config.get(mode)
    if not isinstance(mode_config, dict):
        raise ValueError(
            f"Missing {provider_name}.{mode}.model in model config {path}."
        )

    model = mode_config.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError(
            f"Missing {provider_name}.{mode}.model in model config {path}."
        )

    return model
