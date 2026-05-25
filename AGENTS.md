# AGENTS.md

Instructions for Codex and other coding agents working in this repo.

## Project Scope

- This repo is the YoloRAG FastAPI backend plus a local Vite/React frontend harness.
- The sibling `../llm` folder is an external widget/client contract, not source owned by this repo.
- Keep work focused on the requested change. Do not bundle unrelated cleanup or generated assets.

## Core Product Contracts

- Keep the app API-first. Do not reintroduce CLI runtime surfaces unless explicitly requested.
- Keep model selection configured through environment variables only. Do not add `models.json` or another file-based model config path.
- Preserve the fast/deep chat split:
  - `POST /api/chat/fast` is the normal fast route.
  - `POST /api/chat` is the backwards-compatible fast alias.
  - `POST /api/chat/deep` returns only the final answer text.
  - `POST /api/chat/deep/events` streams typed deep-agent events for the console.
- Fast/deep routing is internal behavior. Do not leak routing labels or implementation mechanics into model-facing prompts unless the product explicitly needs it.
- Keep `/api/chat/fast` latency-first, but allow one bounded hidden tool-selection pass so it can call `docs_search` only when documentation context materially improves the answer.
- Keep fast-chat tool/retrieval events hidden from the public SSE stream; stream only normal content and metrics when requested.
- Keep richer multi-step tool use, GitHub/MCP investigation, and deep review behavior in the deep-agent path.
- Preserve true SSE token streaming for fast chat. Do not fake streaming by chunking a completed response after the fact.
- Retrieval failures must degrade gracefully to LLM-only output instead of breaking chat.

## Retrieval And Agent Rules

- Do not use hard-coded keyword routing for retrieval decisions. Prefer generic confidence or relevance-score gating.
- Use one shared retrieval threshold knob unless asked otherwise: `YOLORAG_RETRIEVAL_MIN_SCORE`, with `0.50` as the documented default.
- Use `raw_user_message` for fast tool selection and retrieval/routing decisions when available, so page context and instructions do not make casual greetings or conversation-local follow-ups look domain-specific.
- For fast chat, skip `docs_search` when the answer is already clear from the current conversation, and retrieve only when docs context is beneficial.
- Let fast and deep paths decide when to call `docs_search`; do not add hidden force-retrieval toggles.
- For public GitHub issue/support replies, first classify the thread type, then gather evidence and draft a concise reply suited to that type.
- Do not use regex, word-filter, phrase-filter, or indicator-word detection as behavior fixes for any functionality. Prefer general prompt policy, behavior design, evidence discipline, and semantic checks over specific-case word detection.

## MCP And GitHub Access

- Prefer hosted GitHub MCP over a self-hosted/Docker GitHub MCP server unless requested.
- Keep hosted GitHub MCP read-only by default.
- Enforce repository access locally with `allowed_repositories`, not only by prompt text or remote server settings.
- When GitHub MCP is enabled, keep access constrained to `ultralytics/ultralytics` unless explicitly requested otherwise.
- Hide or block broad repository discovery tools such as `search_repositories` when they would violate the allowlist.

## Frontend Rules

- Keep the existing deep-agent console wired to `/api/chat/deep/events` unless changing that contract is the requested task.
- Preserve the fast-chat bubble/widget so `/api/chat/fast` can still be tested from the UI.
- `frontend/public/vendor/` is generated/local-only. It mirrors `../llm/js/chat.js` and should stay gitignored.
- Regenerate the widget copy with `npm run sync:llm`; do not treat the copied bundle as a source-owned deliverable.
- `frontend/.env` is optional and only for frontend overrides. Backend/provider credentials belong in the repo root `.env`.
- Follow the existing frontend visual direction. Do not restyle the app unless the task asks for UI changes.

## Local Setup And Commands

Backend setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev]
```

Run the API:

```bash
PYTHONPATH=src uvicorn yolorag.api.app:app --reload --host 127.0.0.1 --port 8000
```

Run backend tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

Run the frontend:

```bash
cd frontend
npm install
npm run sync:llm
npm run dev
```

Build the frontend:

```bash
cd frontend
npm run build
```

Useful smoke checks:

```bash
curl -N http://127.0.0.1:8000/api/chat/fast \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Explain YOLO in one paragraph"}]}'
```

```bash
curl -N http://127.0.0.1:8000/api/chat/deep/events \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Explain how YOLO handles object detection"}]}'
```

## Verification Expectations

- For backend behavior changes, run targeted tests first, then the full unittest suite when the change touches shared runtime, routing, tools, providers, retrieval, or API behavior.
- For frontend changes, run `npm run build` and smoke-test the actual Vite app in a browser.
- For streaming changes, verify the SSE shape at the API layer and through the frontend when feasible.
- For MCP/GitHub changes, verify both an allowed `ultralytics/ultralytics` operation and a blocked outside-repo operation.
- Before reporting readiness, check `git status --short` and make clear which files are changed or untracked.
- In dirty worktrees, stage only the requested files and never revert unrelated user changes.

## Documentation And Config Hygiene

- Keep `.env.example`, `README.md`, and frontend docs aligned with runtime behavior when changing env vars, routes, or setup flow.
- Search for stale references after removing or renaming runtime surfaces.
- Do not commit secrets, PATs, local `.env` files, `node_modules/`, build outputs, or generated widget bundles.
- Prefer repo-grounded answers and real verification over generic advice.
