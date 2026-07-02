#!/bin/sh
# Structure creation for image embeddings -- the Qdrant mirror of
# deploy/postgres/init/002_image_embeddings.sql and deploy/mongodb/init/001_image_embeddings.js.
#
# Collection "image_embeddings": 128-dim cosine vectors (matching the L2-normalized
# Nomic embeddings), with dataset_id + img_id kept in each point's payload. A keyword
# payload index on dataset_id backs dataset-scoped filtered search.
# Idempotent: skips creation when the collection already exists.

BASE="http://localhost:6333"
COLL="image_embeddings"

if curl -sf "$BASE/collections/$COLL" >/dev/null 2>&1; then
  echo "qdrant init: collection $COLL already exists"
else
  if curl -sf -X PUT "$BASE/collections/$COLL" \
      -H 'Content-Type: application/json' \
      -d '{"vectors":{"size":128,"distance":"Cosine"}}' >/dev/null; then
    echo "qdrant init: created collection $COLL (128-d cosine)"
  else
    echo "qdrant init: failed to create collection $COLL"
  fi
fi

# Payload index on dataset_id (idempotent; harmless if it already exists).
if curl -sf -X PUT "$BASE/collections/$COLL/index?wait=true" \
    -H 'Content-Type: application/json' \
    -d '{"field_name":"dataset_id","field_schema":"keyword"}' >/dev/null 2>&1; then
  echo "qdrant init: ensured dataset_id payload index"
else
  echo "qdrant init: dataset_id index already present or deferred"
fi
