# YoloRAG FastAPI Integration Plan for the `llm` Widget

## Goal

Make `src/yolorag` work as the backend for the sibling `../llm` chat widget by exposing the HTTP contract that the widget already expects:

- `POST /api/chat` for Server-Sent Events chat streaming.
- `POST /api/search` for search-mode JSON results.
- `POST /api/feedback` for optional analytics feedback.

The `llm` folder should be treated as the client/widget contract. The core LLM, RAG, routing, provider, usage, and cost logic should remain owned by `src/yolorag`.

## Stakeholder Reference Notes

The website chat widget is the latency-critical path. It should feel super-fast for the yellow-pill chat entry points in `ultralytics/llm`: <https://github.com/ultralytics/llm>.

GitHub issue responses can use slower orchestration. Reference examples:

- Minimal assistant response: <https://github.com/ultralytics/yolo-ios-app/issues/234#issuecomment-4279029465>.
- More contextualized Paula-style response with a human face for engagement: <https://github.com/ultralytics/yolo-ios-app/issues/234#issuecomment-4283908126>.

Timing expectations differ by surface: the assistant GitHub response can take around 60 seconds, while a Paula-style contextualized response may arrive several hours later. The fast `/api/chat` path should not inherit those slower latency budgets.

Current RAG coverage scrapes docs and website content, but GitHub coverage is incomplete. GitHub Issues are not ingested today and may need a search tool or MCP-backed path. Source-code coverage is also uneven: `ultralytics/ultralytics` gets inline source-code reference pages in the docs, but other repositories such as `ultralytics/yolo-ios-app` do not have equivalent source-code coverage, which makes repo-specific answers weaker without GitHub source/search/MCP support.

The expected deployment stack is Python, Google Cloud Run, FastAPI, MongoDB, and Voyage embeddings/rerankers. The LLM layer should remain vendor-agnostic, including the harder cases of search, MCP, and tool calling. Dedicated vector databases remain open for consideration if they provide a clear operational or retrieval-quality benefit.

## Current State

`src/yolorag` already has useful backend primitives:

- `providers/`: OpenAI and DeepSeek provider adapters.
- `core/orchestrator.py`: RAG orchestration entrypoint.
- `core/conversation.py`: in-memory conversation tracking.
- `retrieval/`: retriever protocol and in-memory retriever.
- `review/`: simple answer review.
- `usage/`: token usage extraction and cost calculation.
- `runtime.py`: API runtime wiring for provider/model/orchestrator setup.

The `../llm` folder currently provides:

- `js/chat.js`: production chat widget.
- `js/chat.min.js`: minified widget.
- `docs/API.md`: documented widget/backend contract.
- `examples/web/demo.html`: browser demo.
- `ultralytics_llm/client.py`: placeholder Python client.

It does not yet provide a real FastAPI backend.

## Required Backend Contract

### `POST /api/chat`

Request body expected by the widget:

```json
{
  "messages": [{ "role": "user", "content": "What is YOLO11?" }],
  "session_id": "optional-session-id",
  "context": {
    "url": "https://docs.ultralytics.com/models/yolo11/",
    "title": "YOLO11",
    "description": "Page text or meta description",
    "path": "/models/yolo11/"
  },
  "analytics": true,
  "edit_index": 3,
  "instructions": "optional host instructions",
  "tools": ["search", "github"]
}
```

Response:

```text
Content-Type: text/event-stream
X-Session-ID: generated-or-existing-session-id
X-Total-User-Messages: 3
X-Active-User-Messages: 3

data: {"content": "Hello "}
data: {"content": "from YoloRAG."}
data: [DONE]
```

Error events should use:

```text
data: {"error": "Friendly error message"}
data: [DONE]
```

### `POST /api/search`

Request:

```json
{
  "query": "YOLO training parameters"
}
```

Response:

```json
{
  "results": [
    {
      "title": "Training Configuration",
      "url": "https://docs.ultralytics.com/usage/training/",
      "text": "Step-by-step instructions for configuring training jobs...",
      "score": 0.95
    }
  ]
}
```

### `POST /api/feedback`

Request sent by the widget when analytics are enabled:

```json
{
  "session_id": "session-id",
  "query_index": 0,
  "vote": true
}
```

Initial response can be:

```json
{
  "ok": true
}
```

This can start as a no-op or in-memory log, then later persist to a real analytics store.

## Proposed File Structure

Add:

```text
src/yolorag/api/
  __init__.py
  app.py
  routes.py
  schemas.py
  sse.py

src/yolorag/runtime.py
```

Optional later:

```text
src/yolorag/api/feedback.py
src/yolorag/api/sessions.py
src/yolorag/retrieval/docs_index.py
```

## Runtime Wiring

Create `src/yolorag/runtime.py` to centralize API runtime setup:

- Resolve provider: `openai` or `deepseek`.
- Resolve mode: `fast` or `deep`.
- Resolve model from env vars or built-in defaults.
- Build `RAGOrchestrator`.
- Build retriever and reviewer.
- Share one `InMemoryConversationStore` across requests.

This keeps the FastAPI app as the only runtime entrypoint for now.

Suggested environment variables:

```env
YOLORAG_API_PROVIDER=openai
YOLORAG_API_MODE=fast
YOLORAG_API_HOST=127.0.0.1
YOLORAG_API_PORT=8000
YOLORAG_CORS_ORIGINS=http://localhost:8000,http://127.0.0.1:8000
```

Existing model/provider env vars should keep working:

```env
OPENAI_API_KEY=
DEEPSEEK_API_KEY=
OPENAI_BASE_URL=
DEEPSEEK_BASE_URL=https://api.deepseek.com
YOLORAG_OPENAI_FAST_MODEL=
YOLORAG_OPENAI_THINKING_MODEL=
YOLORAG_DEEPSEEK_FAST_MODEL=
YOLORAG_DEEPSEEK_THINKING_MODEL=
```

## Chat Endpoint Behavior

Implementation flow:

1. Parse `ChatRequest`.
2. Use provided `session_id`, or generate a new UUID.
3. Read the latest user message from `messages`.
4. Merge page context into the prompt or request metadata.
5. Respect optional `instructions` by adding them to the system prompt.
6. Use `tools` as routing hints:
   - `search` should bias toward retrieval.
   - `github` can be reserved for a future GitHub tool.
7. Call `RAGOrchestrator.answer(...)` with:
   - `user_message`
   - `conversation_id=session_id`
   - configured mode
8. Stream the answer as SSE chunks.
9. End with `data: [DONE]`.

Important note: the provider interface now supports true streaming. The FastAPI chat route should forward provider deltas as SSE events instead of waiting for the completed answer and rechunking it.

## Session And Edit Handling

The widget supports editing a previous user message and sends `edit_index`.

Current gap:

- `InMemoryConversationStore` can append turns, but cannot truncate turns after an edit.

Needed addition:

- Add a safe truncation method to `ConversationState` or `InMemoryConversationStore`.
- Convert widget `edit_index` into a completed-turn count.
- Remove stale turns after that point before calling the orchestrator again.

Without this, edited user messages may still keep old assistant answers in backend memory.

## Search Endpoint Behavior

Implementation flow:

1. Parse `SearchRequest`.
2. Call the configured retriever with the query.
3. Convert each `RetrievalResult` into widget result shape:
   - `title`: `document.title`
   - `url`: `document.metadata["url"]` when available, otherwise a fallback URL
   - `text`: relevant snippet from `document.content`
   - `score`: retrieval score
4. Return `{ "results": [...] }`.

The retriever should eventually point to real Ultralytics docs/content. The in-memory retriever is fine for the first local integration test.

## Feedback Endpoint Behavior

Initial version:

- Validate body.
- Return `{ "ok": true }`.
- Optionally log feedback in memory.

Later version:

- Persist `session_id`, `query_index`, `vote`, timestamp, and possibly trace metadata.
- Connect feedback to answer quality evaluation.

## Dependencies

Add to `pyproject.toml`:

```toml
dependencies = [
  "fastapi",
  "uvicorn",
  "genai-prices",
  "openai",
  "python-dotenv",
]
```

Optional later:

```toml
"pydantic-settings"
```

## Local Run Command

Expected local command after implementation:

```bash
PYTHONPATH=src uvicorn yolorag.api.app:app --reload --host 127.0.0.1 --port 8000
```

The widget can then point to:

```javascript
new UltralyticsChat({
  apiUrl: "http://127.0.0.1:8000/api/chat"
});
```

## Tests To Add

Add API tests with a fake provider so no real model call is made:

- `POST /api/search` returns `200` and `{ "results": [...] }`.
- `POST /api/feedback` returns `200` and `{ "ok": true }`.
- `POST /api/chat` returns:
  - `Content-Type: text/event-stream`
  - `X-Session-ID`
  - at least one `data: {"content": ...}` event
  - final `data: [DONE]`
- `POST /api/chat` with an existing `session_id` reuses the same session.
- `POST /api/chat` with `edit_index` truncates stale conversation turns.

## Suggested Implementation Order

1. Add FastAPI dependency and API package skeleton.
2. Create request/response schemas matching `../llm/docs/API.md`.
3. Add API runtime wiring in `runtime.py`.
4. Implement `/api/search` using existing retriever.
5. Implement `/api/feedback` as a validated no-op.
6. Implement `/api/chat` with provider-token SSE streaming.
7. Add session ID headers and message-count headers.
8. Add conversation truncation for `edit_index`.
9. Add tests with fake provider/runtime.
10. Smoke-test with `../llm/examples/web/demo.html`.

## Future Enhancements

- Real Ultralytics docs index.
- GitHub issue/search tool behind the widget's `github` tool button.
- Persistent sessions instead of in-memory sessions.
- Persistent feedback and trace analytics.
- CORS configuration for deployed widget origins.
- Authentication/rate limiting for public deployments.
