"""End-to-end document ingestion and command-line interface."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Protocol

import typer
from rich.console import Console

from enclave import db
from enclave.ingest.chunking import Chunk, chunk_document
from enclave.ingest.parsers import SUPPORTED_SUFFIXES, parse_document
from enclave.retrieval.hybrid import to_vector_literal


class DocumentEmbedder(Protocol):
    def encode_documents(self, texts: list[str], batch_size: int = 16): ...


@dataclass(slots=True)
class IngestStats:
    files_seen: int = 0
    files_ingested: int = 0
    chunks_written: int = 0
    duplicate_chunks: int = 0
    failures: int = 0


def discover_documents(source: Path) -> list[Path]:
    """Return supported files in stable order."""
    source = source.expanduser()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_file():
        if source.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"unsupported document type: {source.suffix or '<none>'}")
        return [source]
    return sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def _source_path(path: Path, root: Path) -> str:
    if root.is_file():
        return path.name
    return path.relative_to(root).as_posix()


def document_id(source_path: str, owner_id: str | None = None) -> str:
    identity = f"{owner_id}\0{source_path}" if owner_id else source_path
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _write_document(
    conn,
    *,
    doc_id: str,
    source_path: str,
    title: str | None,
    doc_type: str,
    chunks: list[Chunk],
    vectors,
    owner_id: str | None = None,
) -> tuple[int, int]:
    written = 0
    duplicates = 0
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(
            "INSERT INTO documents (doc_id, source_path, title, doc_type, owner_id) "
            "VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (doc_id) DO UPDATE SET "
            "source_path = EXCLUDED.source_path, title = EXCLUDED.title, "
            "doc_type = EXCLUDED.doc_type, ingested_at = now()",
            (doc_id, source_path, title, doc_type, owner_id),
        )
        cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
        for index, chunk in enumerate(chunks):
            vector = None if vectors is None else to_vector_literal(vectors[index])
            cur.execute(
                "INSERT INTO chunks "
                "(doc_id, ordinal, heading_path, content, content_hash, "
                "token_count, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s::vector) "
                "ON CONFLICT (doc_id, content_hash) DO NOTHING",
                (
                    doc_id,
                    chunk.ordinal,
                    chunk.heading_path,
                    chunk.content,
                    chunk.content_hash,
                    chunk.token_count,
                    vector,
                ),
            )
            if cur.rowcount:
                written += 1
            else:
                duplicates += 1
    return written, duplicates


def ingest_path(
    source: Path,
    *,
    conn,
    embedder: DocumentEmbedder | None,
    max_words: int = 220,
    overlap_words: int = 40,
    batch_size: int = 16,
    continue_on_error: bool = False,
    owner_id: str | None = None,
) -> IngestStats:
    """Parse, chunk, embed, and atomically persist every supported file."""
    root = source.expanduser().resolve()
    paths = discover_documents(root)
    stats = IngestStats(files_seen=len(paths))

    prepared = []
    for path in paths:
        try:
            parsed = parse_document(path)
            chunks = chunk_document(
                parsed, max_words=max_words, overlap_words=overlap_words
            )
            prepared.append((path, parsed, chunks))
        except Exception:
            stats.failures += 1
            if not continue_on_error:
                raise

    all_chunks = [chunk for _, _, chunks in prepared for chunk in chunks]
    all_vectors = None
    if embedder is not None and all_chunks:
        all_vectors = embedder.encode_documents(
            [chunk.content for chunk in all_chunks], batch_size=batch_size
        )
        if len(all_vectors) != len(all_chunks):
            raise ValueError("embedder returned the wrong number of vectors")

    vector_offset = 0
    for path, parsed, chunks in prepared:
        try:
            vectors = None
            if all_vectors is not None:
                vectors = all_vectors[vector_offset : vector_offset + len(chunks)]
            vector_offset += len(chunks)
            relative_path = _source_path(path, root)
            written, duplicates = _write_document(
                conn,
                doc_id=document_id(relative_path, owner_id),
                source_path=relative_path,
                title=parsed.title,
                doc_type=parsed.doc_type,
                chunks=chunks,
                vectors=vectors,
                owner_id=owner_id,
            )
            stats.files_ingested += 1
            stats.chunks_written += written
            stats.duplicate_chunks += duplicates
        except Exception:
            stats.failures += 1
            if not continue_on_error:
                raise
    return stats


def _command(
    source: Annotated[Path, typer.Argument(help="A document or directory to ingest.")],
    no_embed: Annotated[
        bool,
        typer.Option("--no-embed", help="Store chunks without loading a model."),
    ] = False,
    max_words: Annotated[int, typer.Option(min=1)] = 220,
    overlap_words: Annotated[int, typer.Option(min=0)] = 40,
    batch_size: Annotated[int, typer.Option(min=1)] = 16,
    continue_on_error: Annotated[bool, typer.Option("--continue-on-error")] = False,
    owner: Annotated[
        str | None,
        typer.Option("--owner", help="Username that owns the imported documents."),
    ] = None,
) -> None:
    """Ingest local documents into Enclave."""
    if overlap_words >= max_words:
        raise typer.BadParameter("must be smaller than max-words", param_hint="overlap")

    embedder = None
    if not no_embed:
        from enclave.models.encoders import get_embedder

        embedder = get_embedder()

    with db.connect() as conn:
        owner_id = None
        if owner:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT user_id FROM users WHERE username=lower(%s) "
                    "AND disabled=false",
                    (owner,),
                )
                row = cur.fetchone()
            if row is None:
                raise typer.BadParameter("unknown username", param_hint="owner")
            owner_id = row[0]
        stats = ingest_path(
            source,
            conn=conn,
            embedder=embedder,
            max_words=max_words,
            overlap_words=overlap_words,
            batch_size=batch_size,
            continue_on_error=continue_on_error,
            owner_id=owner_id,
        )

    Console().print(
        f"Ingested [bold]{stats.files_ingested}/{stats.files_seen}[/bold] files; "
        f"wrote [bold]{stats.chunks_written}[/bold] chunks; "
        f"skipped {stats.duplicate_chunks} duplicates; "
        f"{stats.failures} failures."
    )


def cli() -> None:
    """Package entry point."""
    typer.run(_command)
