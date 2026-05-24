# yolorag

Provider-agnostic RAG orchestrator prototype for the Ultralytics LLM Engineer take-home challenge.

The first implementation slice focuses on the architecture surfaces that matter most:

- Provider classes normalize vendor-specific responses.
- Usage extractors convert raw provider usage into one `TokenUsage` shape.
- Cost calculation is owned by this project, with local pricing overrides and `genai-prices` as the fallback pricing backend.
- The orchestrator uses lightweight, score-gated retrieval so weak matches do not pollute the prompt.
- The FastAPI chat endpoint exposes the `../llm` widget contract.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Run The Chat API

The FastAPI app exposes the `../llm` widget chat contract at `POST /api/chat/fast`.
`POST /api/chat` remains as a legacy alias for the fast route. Deep agent mode is
available at `POST /api/chat/deep` and returns only the final text response after
the agent finishes any docs or MCP tool calls.

```bash
PYTHONPATH=src uvicorn yolorag.api.app:app --reload --host 127.0.0.1 --port 8000
```

Quick smoke test:

```bash
curl -N http://127.0.0.1:8000/api/chat/fast \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Explain YOLO in one paragraph"}]}'
```

The endpoint returns Server-Sent Events and an `X-Session-ID` header. Send that
same session ID on the next request to reuse the in-memory conversation history.

## Hosted MCP Servers

Deep chat can load tools from MCP servers configured with `YOLORAG_MCP_SERVERS`.
The existing stdio server config still works, and hosted Streamable HTTP servers
can be configured with `type: "http"`.

Hosted GitHub MCP read-only example:

```env
GITHUB_MCP_TOKEN=...
YOLORAG_MCP_SERVERS=[{"name":"github","type":"http","url":"https://api.githubcopilot.com/mcp/","allowed_repositories":["ultralytics/ultralytics"],"headers":{"Authorization":"Bearer ${GITHUB_MCP_TOKEN}","X-MCP-Toolsets":"repos,issues,pull_requests,actions","X-MCP-Readonly":"true"}}]
```

`X-MCP-Readonly` keeps GitHub tool calls read-only, while `X-MCP-Toolsets`
narrows the exposed tools to the issue-troubleshooting surface.
`allowed_repositories` is enforced locally before tool calls leave the app, so
the agent can only use GitHub MCP against the listed repositories.

## Run The Frontend

The local frontend lives in `frontend/`. It uses Vite, React, Tailwind CSS, and the local `../llm/js/chat.js` widget.

```bash
cd frontend
npm install
npm run sync:llm
npm run dev
```

By default, Vite proxies `/api/*` to `http://127.0.0.1:8000`, so the widget can call `/api/chat/fast` while the FastAPI server uses the real configured provider.

Use a real provider for local chat testing:

```bash
YOLORAG_API_PROVIDER=openai \
OPENAI_API_KEY=... \
PYTHONPATH=src uvicorn yolorag.api.app:app --reload --host 127.0.0.1 --port 8000
```

## Model Defaults

The API picks models by provider and mode. Model names can only be configured through environment variables for now. If no env override is present, the built-in defaults are used.

Override order:

1. Mode-specific env vars, such as `YOLORAG_OPENAI_FAST_MODEL`
2. Legacy provider env vars, such as `YOLORAG_OPENAI_MODEL`
3. Built-in defaults

| Provider | Fast mode | Deep/thinking mode |
| --- | --- | --- |
| OpenAI | `gpt-5.4-mini` | `gpt-5.5` |
| DeepSeek | `deepseek-v4-flash` | `deepseek-v4-pro` |

Use environment overrides:

```env
YOLORAG_OPENAI_FAST_MODEL=gpt-5.4-mini
YOLORAG_OPENAI_THINKING_MODEL=gpt-5.5
YOLORAG_DEEPSEEK_FAST_MODEL=deepseek-v4-flash
YOLORAG_DEEPSEEK_THINKING_MODEL=deepseek-v4-pro
```

DeepSeek fast mode explicitly disables thinking. DeepSeek deep mode enables thinking with high reasoning effort.

## Retrieval Tuning

Fast chat uses a small retrieval pass and injects context only when the returned
result clears the configured relevance threshold. Deep chat applies the same
threshold to the `docs_search` tool results. This avoids hard-coded keyword
routing while keeping retrieved context quality-gated.

```env
YOLORAG_CHAT_VECTOR_TOP_K=2
YOLORAG_RETRIEVAL_MIN_SCORE=0.50
```

## Test

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Latency Evals

Run the profile questions against MongoDB retrieval only:

```bash
PYTHONPATH=src python scripts/eval_profile_latency.py --retrieval-only
```

Compare smaller rerank candidate pools:

```bash
PYTHONPATH=src python scripts/eval_profile_latency.py --retrieval-only --rerank-candidates 32
PYTHONPATH=src python scripts/eval_profile_latency.py --retrieval-only --rerank-candidates 16
```

Run the same questions through the full RAG path, including the configured LLM:

```bash
PYTHONPATH=src python scripts/eval_profile_latency.py --top-k 8 --mode fast
```

Each run prints Mongo vector search, reranking, LLM, and app-overhead timings, then writes a JSON report under `evals/runs/`.

## Architecture

`providers/` owns model API calls and normalizes responses into `LLMResponse`.

`usage/` owns provider-specific token extraction and cost calculation.

`retrieval/` owns knowledge-source access.

`review/` owns answer verification and confidence scoring.

`core/` owns routing, conversation state, orchestration, and traces.

The orchestrator should not read raw provider response shapes directly. OpenAI and DeepSeek provider adapters normalize those details before returning.
