from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from yolorag.api.chat import router as chat_router
from yolorag.config.settings import getenv
from yolorag.runtime import YoloRAGAgentRuntime, YoloRAGRuntime


EXPOSED_WIDGET_HEADERS = [
    "X-Session-ID",
    "X-Total-User-Messages",
    "X-Active-User-Messages",
    "X-Chat-Mode",
]


def create_app(
    runtime: YoloRAGRuntime | None = None,
    deep_runtime: YoloRAGAgentRuntime | None = None,
) -> FastAPI:
    app = FastAPI(title="YoloRAG API", version="0.1.0")
    if runtime is not None:
        app.state.runtime = runtime
        app.state.fast_runtime = runtime
    if deep_runtime is not None:
        app.state.deep_runtime = deep_runtime

    origins = _cors_origins()
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["POST", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=EXPOSED_WIDGET_HEADERS,
        )

    app.include_router(chat_router, prefix="/api")
    return app


def _cors_origins() -> list[str]:
    configured = getenv("YOLORAG_CORS_ORIGINS")
    if not configured:
        return []
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


app = create_app()
