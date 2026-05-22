# YoloRAG Frontend

This is a small Vite harness for testing the sibling `../llm` chat widget against the local YoloRAG FastAPI backend.

The widget bundle under `public/vendor/` is intentionally gitignored. It is a local copy of `../llm/js/chat.js`, not source owned by this frontend.

Refresh it after installing dependencies or after changing the sibling widget:

```bash
npm run sync:llm
```

Then run the frontend:

```bash
npm run dev
```
