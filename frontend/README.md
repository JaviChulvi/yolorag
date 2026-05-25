# YoloRAG Frontend

This is a small Vite harness for testing the local YoloRAG FastAPI backend.

The homepage is a native React deep-agent console that posts to
`/api/chat/deep/events` and renders the streamed `status`, `tool_call`,
`tool_result`, `content`, and `done` events.

The widget bundle under `public/vendor/` is intentionally gitignored. It is a local copy of `../llm/js/chat.js`, not source owned by this frontend.

Refresh it after installing dependencies or after changing the sibling widget:

```bash
npm run sync:llm
```

Then run the frontend:

```bash
npm run dev
```
