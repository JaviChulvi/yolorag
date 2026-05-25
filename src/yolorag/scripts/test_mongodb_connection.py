#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB = "yolorag"
DEFAULT_COLLECTION = "docs_chunks"
DEFAULT_DOCUMENT_ID = "setup-test"


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")

    try:
        from bson.json_util import dumps
        from pymongo import MongoClient
        from pymongo.errors import PyMongoError, ServerSelectionTimeoutError
    except ImportError:
        print("Missing MongoDB dependency. Run: python -m pip install -e .")
        return 2

    parser = argparse.ArgumentParser(
        description="Test MongoDB Atlas connectivity and retrieve a known docs chunk document."
    )
    parser.add_argument(
        "--db",
        default=os.getenv("YOLORAG_MONGODB_DB", DEFAULT_DB),
        help="MongoDB database name. Defaults to YOLORAG_MONGODB_DB or yolorag.",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("YOLORAG_MONGODB_CHUNKS_COLLECTION", DEFAULT_COLLECTION),
        help="MongoDB collection name. Defaults to YOLORAG_MONGODB_CHUNKS_COLLECTION or docs_chunks.",
    )
    parser.add_argument(
        "--document-id",
        default=DEFAULT_DOCUMENT_ID,
        help="Document id to retrieve. Defaults to setup-test.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=5000,
        help="Server selection timeout in milliseconds. Defaults to 5000.",
    )
    args = parser.parse_args()

    uri = os.getenv("YOLORAG_MONGODB_URI")
    if not uri:
        print("Missing YOLORAG_MONGODB_URI in .env")
        return 2

    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=args.timeout_ms)
        client.admin.command("ping")
        collection = client[args.db][args.collection]
        document = _find_test_document(collection, args.document_id)
    except ServerSelectionTimeoutError as exc:
        print("Could not connect to MongoDB Atlas before the timeout.")
        print(str(exc))
        return 1
    except PyMongoError as exc:
        print("MongoDB request failed.")
        print(str(exc))
        return 1

    print("MongoDB connection OK")
    print(f"Database: {args.db}")
    print(f"Collection: {args.collection}")

    if document is None:
        print(f"Document not found for id: {args.document_id}")
        print("Checked _id, doc_id, and chunk_id.")
        return 3

    print(f"Retrieved document: {args.document_id}")
    print(dumps(document, indent=2))
    return 0


def _find_test_document(collection: Any, document_id: str) -> dict[str, Any] | None:
    return collection.find_one(
        {
            "$or": [
                {"_id": document_id},
                {"doc_id": document_id},
                {"chunk_id": document_id},
            ]
        }
    )


if __name__ == "__main__":
    raise SystemExit(main())
