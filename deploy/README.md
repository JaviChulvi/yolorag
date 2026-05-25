# Docker And Coolify Deployment

This deployment runs the demo with both retrieval providers available:

- PostgreSQL/pgvector runs inside the compose stack and is initialized from the
  precomputed embeddings SQL seed.
- MongoDB stays external, using the configured Atlas URI, vector index, and
  MongoDB AI rerank key. Reranking is always enabled.

## Local Smoke Test

Export the current local pgvector embeddings:

```bash
PYTHONPATH=src python -m yolorag.scripts.export_postgres_seed
```

Start the stack:

```bash
SERVICE_PASSWORD_POSTGRES=yolorag docker compose \
  -f docker-compose.coolify.yml \
  -f docker-compose.local.yml \
  up -d --build
```

Open the frontend at `http://127.0.0.1:8080`. The backend is exposed at
`http://127.0.0.1:8000` for direct smoke tests.

Check the seeded Postgres data:

```bash
docker compose -f docker-compose.coolify.yml -f docker-compose.local.yml exec -T postgres \
  psql -U yolorag -d yolorag \
  -c "select count(*), min(embedding_dimensions), max(embedding_dimensions) from docs_chunks;"
```

The current seed contains `4824` rows with `3072`-dimension embeddings.

## Coolify

Use `docker-compose.coolify.yml` as the Docker Compose file. Expose the
`frontend` service on port `8080`; nginx proxies `/api/*` to the internal
`backend` service.

Assign the domain to the `frontend` service only:

```text
https://yolo.chulvi.dev:8080
```

The `:8080` tells Coolify's proxy which container port to route to; users still
visit `https://yolo.chulvi.dev` over normal HTTPS.

Set these required secrets/environment variables in Coolify:

```env
OPENAI_API_KEY=
DEEPSEEK_API_KEY=
YOLORAG_MONGODB_URI=
YOLORAG_MONGODB_AI_API_KEY=
```

`SERVICE_PASSWORD_POSTGRES` is generated automatically by Coolify from the
compose file. You do not need to invent or paste a Postgres password unless you
want to override the generated value.

Optional runtime knobs:

```env
YOLORAG_API_PROVIDER=deepseek
YOLORAG_KNOWLEDGE_PROVIDER=mongodb
YOLORAG_RETRIEVAL_MIN_SCORE=0.50
YOLORAG_FAST_TOOL_TIMEOUT_SECONDS=8
YOLORAG_FAST_RERANK_CANDIDATE_LIMIT=16
GITHUB_MCP_TOKEN=
```

The fast timeout bounds hidden docs-search calls. If MongoDB Atlas vector search
or reranking is slower than the timeout, the backend falls back to an LLM-only
answer and the eval page reports a retrieval error with zero returned docs.

Set `GITHUB_MCP_TOKEN` to enable the built-in hosted GitHub MCP tools in the
deep agent:

```env
GITHUB_MCP_TOKEN=<github-token>
```

The hosted GitHub MCP server URL, read-only headers, toolsets, and
`ultralytics/ultralytics` repository allowlist are configured in code. The
allowlist is enforced by the app before GitHub tool calls leave the backend.

`YOLORAG_POSTGRES_DSN` is set by the compose file to the internal `postgres`
service. The frontend selector can still switch requests between `mongodb` and
`postgresql` at runtime.

The Postgres service builds from `Dockerfile.postgres`, which copies
`deploy/postgres/init/` into `/docker-entrypoint-initdb.d/` inside the image.
This avoids relying on a Coolify bind mount for the seed files.

Postgres init scripts run only when the database volume is empty. To refresh the
seed in an existing Coolify volume, either recreate the volume or import the SQL
dump manually into Postgres. If `/docker-entrypoint-initdb.d` is empty inside
the deployed Postgres container, the deployment is still running the old
bind-mount setup or was built without the seed file in the Docker context.
