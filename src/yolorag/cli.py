from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from decimal import Decimal

from yolorag.config.model_defaults import default_model_for
from yolorag.config.settings import getenv
from yolorag.core.orchestrator import RAGOrchestrator
from yolorag.providers.base import ResponseMode
from yolorag.providers.deepseek_provider import DeepSeekProvider
from yolorag.providers.openai_provider import OpenAIProvider
from yolorag.retrieval.base import Document
from yolorag.retrieval.in_memory import InMemoryRetriever
from yolorag.review.simple_reviewer import SimpleReviewer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the yolorag orchestrator prototype.")
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--mode", choices=["fast", "deep"], default="fast")
    parser.add_argument("--conversation-id", default="default")
    parser.add_argument("--provider", choices=["openai", "deepseek"], default="openai")
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name to call. If omitted, uses provider/mode env vars "
            "such as YOLORAG_OPENAI_FAST_MODEL or built-in defaults."
        ),
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help=(
            "Optional OpenAI-compatible base URL. Also reads OPENAI_BASE_URL "
            "or DEEPSEEK_BASE_URL based on provider."
        ),
    )
    parser.add_argument(
        "--models-config",
        default=None,
        help=(
            "Optional JSON model matrix. Also reads YOLORAG_MODELS_CONFIG. "
            "Defaults to ./models.json when present."
        ),
    )
    args = parser.parse_args()

    asyncio.run(
        _run(
            question=args.question,
            mode=args.mode,
            conversation_id=args.conversation_id,
            provider_name=args.provider,
            model=args.model,
            api_base=args.api_base,
            models_config=args.models_config,
        )
    )


async def _run(
    question: str,
    mode: ResponseMode,
    conversation_id: str,
    provider_name: str,
    model: str | None,
    api_base: str | None,
    models_config: str | None,
) -> None:
    selected_model = _resolve_model(
        provider_name=provider_name,
        mode=mode,
        model=model,
        models_config=models_config,
    )
    provider = _build_provider(provider_name=provider_name, api_base=api_base)
    orchestrator = RAGOrchestrator(
        provider=provider,
        model=selected_model,
        retriever=InMemoryRetriever(_sample_documents()),
        reviewer=SimpleReviewer(),
    )

    result = await orchestrator.answer(
        user_message=question,
        conversation_id=conversation_id,
        mode=mode,
    )

    print(result.answer)
    print("\nTrace:")
    print(json.dumps(asdict(result.trace), indent=2, default=_json_default))


def _build_provider(provider_name: str, api_base: str | None) -> OpenAIProvider | DeepSeekProvider:
    if provider_name == "openai":
        return OpenAIProvider(
            api_key=_require_env("OPENAI_API_KEY"),
            api_base=api_base or getenv("OPENAI_BASE_URL"),
        )
    if provider_name == "deepseek":
        return DeepSeekProvider(
            api_key=_require_env("DEEPSEEK_API_KEY"),
            api_base=api_base or getenv("DEEPSEEK_BASE_URL"),
        )
    raise ValueError(f"Unsupported provider {provider_name!r}")


def _resolve_model(
    provider_name: str,
    mode: ResponseMode,
    model: str | None,
    models_config: str | None = None,
) -> str:
    if model:
        return model

    provider_key = provider_name.upper()
    mode_key = "THINKING" if mode == "deep" else "FAST"
    mode_env_name = f"YOLORAG_{provider_key}_{mode_key}_MODEL"
    legacy_env_name = f"YOLORAG_{provider_key}_MODEL"

    configured_model = getenv(mode_env_name) or getenv(legacy_env_name)
    if configured_model:
        return configured_model

    return default_model_for(
        provider_name=provider_name,
        mode=mode,
        config_path=models_config,
    )


def _require_env(name: str) -> str:
    value = getenv(name)
    if value:
        return value
    raise SystemExit(f"Missing required environment variable {name}.")


def _sample_documents() -> list[Document]:
    return [
        Document(
            id="ultralytics-yolo-export",
            title="Ultralytics YOLO Export",
            content=(
                "Ultralytics YOLO models can be exported to deployment formats. "
                "Export and inference troubleshooting often benefits from focused "
                "retrieval because exact framework versions and target formats matter."
            ),
        ),
        Document(
            id="rag-selective-retrieval",
            title="Selective Retrieval Guidance",
            content=(
                "Retrieval should be skipped for generic chat and used when external "
                "knowledge materially improves answer quality. Repeated turns should "
                "avoid reinjecting the same documents."
            ),
        ),
    ]


def _json_default(value: object) -> str:
    if isinstance(value, Decimal):
        return str(value)
    return str(value)


if __name__ == "__main__":
    main()
