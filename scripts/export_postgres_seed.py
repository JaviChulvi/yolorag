#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import os
import re
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from yolorag.knowledge.stores.postgresql import PostgresKnowledgeStoreConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "deploy/postgres/init/010_docs_chunks.sql.gz"


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Export precomputed pgvector docs_chunks embeddings as a SQL seed."
    )
    parser.add_argument(
        "--dsn",
        default=os.getenv("YOLORAG_POSTGRES_DSN", PostgresKnowledgeStoreConfig().dsn),
        help="Source Postgres DSN. Defaults to YOLORAG_POSTGRES_DSN.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination .sql.gz file for Docker/Postgres init.",
    )
    parser.add_argument(
        "--table",
        default="public.docs_chunks",
        help="Table to export. Defaults to public.docs_chunks.",
    )
    args = parser.parse_args()

    raw_dsn = _pg_dump_dsn(args.dsn)
    _verify_source(args.dsn, args.table)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "pg_dump",
        raw_dsn,
        "--data-only",
        "--no-owner",
        "--no-privileges",
        "--table",
        args.table,
    ]
    with subprocess.Popen(command, stdout=subprocess.PIPE) as process:
        assert process.stdout is not None
        with gzip.open(args.output, "wb", compresslevel=9) as output:
            for chunk in iter(lambda: process.stdout.read(1024 * 1024), b""):
                output.write(chunk)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)

    print(f"Wrote {args.output}")
    return 0


def _pg_dump_dsn(dsn: str) -> str:
    url = make_url(dsn)
    if url.drivername.startswith("postgresql+"):
        url = url.set(drivername="postgresql")
    return url.render_as_string(hide_password=False)


def _verify_source(dsn: str, table: str) -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?", table):
        raise ValueError("--table must be a simple table name, optionally schema-qualified.")

    engine = create_engine(dsn)
    with engine.connect() as connection:
        table_name = table.split(".")[-1]
        exists = connection.execute(
            text("select to_regclass(:table_name)"),
            {"table_name": table},
        ).scalar()
        if exists is None:
            raise RuntimeError(f"Missing source table {table!r}.")

        count = connection.execute(text(f"select count(*) from {table}")).scalar_one()
        if count <= 0:
            raise RuntimeError(f"Source table {table!r} has no rows to export.")

        dimensions = connection.execute(
            text(
                f"select min(embedding_dimensions), max(embedding_dimensions) "
                f"from {table}"
            )
        ).one()
        if dimensions != (3072, 3072):
            raise RuntimeError(
                f"Expected 3072-dimensional embeddings in {table_name}, got {dimensions}."
            )

        print(f"Exporting {count} rows from {table} with {dimensions[0]} dimensions.")


if __name__ == "__main__":
    raise SystemExit(main())
