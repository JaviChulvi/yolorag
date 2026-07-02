#!/usr/bin/env bash
# Start Qdrant, then run structure-creation scripts once the REST API is ready.
# Qdrant has no native init hook, so this wrapper provides one.
set -u

# Launch the same binary the stock qdrant image runs (workdir /qdrant holds config/ + storage/).
/qdrant/qdrant &
QDRANT_PID=$!

# Run init in the background so a slow/failed init never blocks Qdrant from serving.
(
  for _ in $(seq 1 60); do
    if curl -sf http://localhost:6333/readyz >/dev/null 2>&1 \
       || curl -sf http://localhost:6333/collections >/dev/null 2>&1; then
      for script in /docker-entrypoint-initdb.d/*.sh; do
        [ -e "$script" ] && sh "$script"
      done
      exit 0
    fi
    sleep 1
  done
  echo "yolorag-entrypoint: Qdrant not ready after 60s; skipping baked init" >&2
) &

# Tie the container lifecycle to Qdrant.
wait "$QDRANT_PID"
