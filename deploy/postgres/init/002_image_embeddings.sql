-- Image embeddings from the Nomic vision model (see yolorag.knowledge.image_embeddings).
-- One 128-dim vector per (dataset_id, img_id). Kept intentionally minimal: no metadata,
-- no timestamps -- just the identifiers and the embedding.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS image_embeddings (
    dataset_id varchar NOT NULL,
    img_id     varchar NOT NULL,
    embedding  vector(128) NOT NULL,
    PRIMARY KEY (dataset_id, img_id)
);

-- The composite PK already indexes the dataset_id prefix for dataset-scoped filtering.
-- Full-precision vector(128) is well under pgvector's 2000-dim HNSW limit, so no halfvec needed.
CREATE INDEX IF NOT EXISTS image_embeddings_embedding_hnsw_idx
    ON image_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
