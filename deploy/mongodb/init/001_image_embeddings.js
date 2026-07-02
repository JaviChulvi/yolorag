// Structure creation for image embeddings -- the MongoDB mirror of
// deploy/postgres/init/002_image_embeddings.sql.
//
// Postgres schema:  image_embeddings(dataset_id varchar, img_id varchar,
//                   embedding vector(128), PRIMARY KEY (dataset_id, img_id))
// Mongo mirror:     one document per (dataset_id, img_id) in db "yolorag",
//                   collection "image_embeddings":
//                     { _id: "<dataset_id>/<img_id>", dataset_id, img_id, embedding: [128 floats] }
//                   + a unique (dataset_id, img_id) index  (== the composite PK)
//                   + a 128-dim cosine $vectorSearch index (== the HNSW cosine index)
//
// Runs once on first container start (empty data dir). The vector index needs the
// search component (mongot) to be up; if it isn't ready yet, creation is best-effort
// here and the app's ensure_schema() creates it idempotently on first ingest.

const database = "yolorag";
const collectionName = "image_embeddings";
const vectorIndexName = "image_embeddings_vector_index";
const dimensions = 128;

const target = db.getSiblingDB(database);

if (!target.getCollectionNames().includes(collectionName)) {
  target.createCollection(collectionName);
  print(`created collection ${database}.${collectionName}`);
}

const collection = target.getCollection(collectionName);

collection.createIndex(
  { dataset_id: 1, img_id: 1 },
  { unique: true, name: "dataset_id_img_id_unique" },
);
print("created unique index dataset_id_img_id_unique");

try {
  collection.createSearchIndex(vectorIndexName, "vectorSearch", {
    fields: [
      { type: "vector", path: "embedding", numDimensions: dimensions, similarity: "cosine" },
      { type: "filter", path: "dataset_id" },
    ],
  });
  print(`created vectorSearch index ${vectorIndexName}`);
} catch (err) {
  print(`vectorSearch index deferred to ensure_schema(): ${err}`);
}
