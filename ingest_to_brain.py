#!/usr/bin/env python3
"""Unified ingest, Second Brain writer, and local/cloud embedding bridge for chyren-embed-worker.

Supports files (.md, .txt, .pdf, .html, .json, .csv) and .zip archives:
  - local:      extract -> chunk -> write knowledge.db & Second Brain vault -> embed via Ollama -> Qdrant
  - cloud-push: extract -> chunk -> write knowledge.db & Second Brain vault -> push un-embedded chunks to Neon
  - cloud-pull: pull completed vectors from Neon ide_chunks -> upsert into Qdrant
"""
import argparse
import hashlib
import io
import json
import os
import re
import sqlite3
import subprocess
import tempfile
import urllib.request
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup
from pypdf import PdfReader

TEXT_EXTS = {".md", ".txt", ".json", ".csv", ".scad", ".py", ".sh"}
HTML_EXTS = {".html", ".htm"}
PDF_EXTS = {".pdf"}
SUPPORTED_EXTS = TEXT_EXTS | HTML_EXTS | PDF_EXTS

DEFAULT_DB = Path.home() / ".chyren" / "knowledge.db"
DEFAULT_VAULTS = [
    Path.home() / ".chyren" / "vault" / "20_Sources",
    Path.home() / "Chyren" / "2_knowledge" / "Chyren_Second_Brain" / "20_Sources",
]
DEFAULT_QDRANT = os.environ.get("QDRANT_URL", "http://localhost:6333")
DEFAULT_COLLECTIONS = ("chyren_memory_knowledge", "chyren_memory")
DEFAULT_OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
EMBED_BATCH = 20


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return clean[:96] or "document"


def extract_pdf_bytes(raw: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(raw))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(p.strip() for p in pages if p.strip())
        if text:
            return text
    except Exception:
        pass
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(raw)
        tmp.flush()
        proc = subprocess.run(["pdftotext", tmp.name, "-"], capture_output=True, text=True, timeout=60)
        return proc.stdout.strip()


def extract_text_from_bytes(name: str, raw: bytes) -> tuple[str, str]:
    ext = Path(name).suffix.lower()
    if ext in TEXT_EXTS:
        return raw.decode("utf-8", errors="replace").strip(), "document_markdown" if ext == ".md" else "document_text"
    if ext in HTML_EXTS:
        soup = BeautifulSoup(raw.decode("utf-8", errors="replace"), "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return re.sub(r"\n{3,}", "\n\n", soup.get_text("\n")).strip(), "document_html"
    if ext in PDF_EXTS:
        return extract_pdf_bytes(raw), "document_pdf"
    raise ValueError(f"Unsupported extension: {ext}")


def iter_sources(paths: Iterable[Path]) -> Iterable[tuple[str, str, bytes, Path | None]]:
    """Yield (canonical_uri, title, raw_bytes, deletable_local_path) for files, dirs, and zips."""
    for p in paths:
        if p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file() and (child.suffix.lower() in SUPPORTED_EXTS or child.suffix.lower() == ".zip"):
                    yield from iter_sources([child])
            continue
        if not p.is_file():
            continue
        if p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as zf:
                for info in zf.infolist():
                    if info.is_dir() or "__MACOSX" in info.filename:
                        continue
                    if Path(info.filename).suffix.lower() in SUPPORTED_EXTS:
                        yield f"zip://{p}::{info.filename}", Path(info.filename).stem, zf.read(info), None
        elif p.suffix.lower() in SUPPORTED_EXTS:
            yield str(p.resolve()), p.stem, p.read_bytes(), p


def chunk_text(text: str, size: int = 1200, overlap: int = 200) -> list[str]:
    if not text:
        return []
    step = max(1, size - overlap)
    return [text[i : i + size] for i in range(0, len(text), step)]


def write_vault_notes(vault_dirs: list[Path], doc_id: str, title: str, canonical_uri: str, sha: str, text: str) -> list[str]:
    written: list[str] = []
    fname = f"{slugify(title)}_{doc_id[:8]}.md"
    note = (
        f"---\n"
        f"title: {json.dumps(title)}\n"
        f"document_id: {doc_id}\n"
        f"canonical_uri: {json.dumps(canonical_uri)}\n"
        f"content_hash_sha256: {sha}\n"
        f"ingested_by: chyren-embed-worker\n"
        f"created_at: {utc_now_iso()}\n"
        f"---\n\n"
        f"# {title}\n\n{text}\n"
    )
    for vdir in vault_dirs:
        if not vdir.parent.exists():
            continue
        vdir.mkdir(parents=True, exist_ok=True)
        out = vdir / fname
        out.write_text(note, encoding="utf-8")
        written.append(str(out))
    return written


def insert_sqlite_doc(
    db_path: Path,
    doc_id: str,
    run_id: str,
    title: str,
    canonical_uri: str,
    route: str,
    sha: str,
    vault_path: str,
    text: str,
) -> bool:
    """Insert document row if not already present. Returns True if newly inserted."""
    now = utc_now_iso()
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT id FROM documents WHERE canonical_uri = ? AND content_hash_sha256 = ?",
            (canonical_uri, sha),
        )
        if cur.fetchone():
            return False
        cur.execute(
            """INSERT INTO documents (
                id, source, original_path, file_type, title, content, content_length,
                ingested_at, canonical_uri, source_uri, source_title, source_type,
                route, content_hash_sha256, ingest_run_id, vault_path, metadata_json,
                content_text, epistemic_status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                doc_id,
                "chyren-embed-worker",
                canonical_uri,
                Path(canonical_uri.split("::")[-1]).suffix.lstrip("."),
                title,
                text,
                len(text),
                now,
                canonical_uri,
                canonical_uri,
                title,
                "local_document",
                route,
                sha,
                run_id,
                vault_path,
                json.dumps({"ingested_by": "chyren-embed-worker"}),
                text,
                "source_claim",
                now,
                now,
            ),
        )
        con.commit()
        return True
    finally:
        con.close()


def embed_ollama(chunks: list[str], ollama_url: str, model: str) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(chunks), EMBED_BATCH):
        for chunk in chunks[i : i + EMBED_BATCH]:
            req = urllib.request.Request(
                f"{ollama_url.rstrip('/')}/api/embeddings",
                data=json.dumps({"model": model, "prompt": chunk}).encode(),
                headers={"content-type": "application/json"},
                method="POST",
            )
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=180) as resp:
                        vectors.append(json.load(resp)["embedding"])
                    break
                except TimeoutError:
                    if attempt == 2:
                        raise
    return vectors


def upsert_qdrant(qdrant_url: str, collections: Iterable[str], points: list[dict]) -> None:
    if not points:
        return
    for coll in collections:
        req = urllib.request.Request(
            f"{qdrant_url.rstrip('/')}/collections/{coll}/points?wait=true",
            data=json.dumps({"points": points}).encode(),
            headers={"content-type": "application/json"},
            method="PUT",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            if json.load(resp).get("status") != "ok":
                raise RuntimeError(f"Qdrant upsert failed on {coll}")


def push_neon_chunks(neon_url: str, rows: list[tuple[str, str, str, str, str, str, str]]) -> int:
    import psycopg2
    import psycopg2.extras

    con = psycopg2.connect(neon_url)
    con.autocommit = True
    try:
        cur = con.cursor()
        for i in range(0, len(rows), 500):
            psycopg2.extras.execute_values(
                cur,
                "insert into ide_chunks (id, sha, source, source_file, store, chunk_index, text) "
                "values %s on conflict (id) do nothing",
                rows[i : i + 500],
            )
        return len(rows)
    finally:
        con.close()


def pull_neon_vectors(neon_url: str, qdrant_url: str, collections: Iterable[str]) -> int:
    import psycopg2

    con = psycopg2.connect(neon_url)
    con.autocommit = True
    total = 0
    try:
        cur = con.cursor()
        while True:
            cur.execute(
                "select id, sha, source, source_file, store, chunk_index, text, vec, embedded_at "
                "from ide_chunks where vec is not null and pulled_at is null limit 500"
            )
            rows = cur.fetchall()
            if not rows:
                break
            points = [
                {
                    "id": str(cid),
                    "vector": list(vec),
                    "payload": {
                        "text": text,
                        "source": src,
                        "source_file": sfile,
                        "store": store,
                        "chunk_index": cidx,
                        "content_hash_sha256": sha,
                        "ingested_at": emb_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "embedded_by": "chyren-embed-worker (nomic-embed-text-v1.5, FastEmbed)",
                    },
                }
                for cid, sha, src, sfile, store, cidx, text, vec, emb_at in rows
            ]
            upsert_qdrant(qdrant_url, collections, points)
            cur.execute("update ide_chunks set pulled_at = now() where id = any(%s::uuid[])", ([str(r[0]) for r in rows],))
            total += len(rows)
        return total
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest files/zips into knowledge.db, Second Brain vault, and Qdrant (local or Neon cloud).")
    parser.add_argument("inputs", nargs="*", type=Path, help="Files, directories, or .zip archives to ingest.")
    parser.add_argument("--mode", choices=("local", "cloud-push", "cloud-pull"), default="local")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--vault-dir", action="append", type=Path, help="Vault 20_Sources directory (defaults to ~/.chyren/vault/20_Sources and Chyren_Second_Brain/20_Sources).")
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--delete-source", action="store_true", help="Delete local source file after verified ingest (never deletes inside zips).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    neon_url = os.environ.get("NEON_URL") or os.environ.get("CHYREN_DB_URL_NEON_CH3", "")
    if args.mode == "cloud-pull":
        if not neon_url:
            raise SystemExit("Set NEON_URL or CHYREN_DB_URL_NEON_CH3 for cloud-pull.")
        pulled = pull_neon_vectors(neon_url, args.qdrant_url, DEFAULT_COLLECTIONS)
        print(json.dumps({"mode": "cloud-pull", "pulled_points": pulled}, indent=2))
        return

    if not args.inputs:
        parser.error("Provide at least one file, directory, or .zip archive.")

    vault_dirs = args.vault_dir if args.vault_dir else DEFAULT_VAULTS
    run_id = str(uuid.uuid4())
    summary = {"run_id": run_id, "mode": args.mode, "processed": 0, "skipped_duplicate": 0, "chunks": 0, "errors": []}

    for canonical_uri, title, raw, deletable in iter_sources(args.inputs):
        try:
            text, route = extract_text_from_bytes(canonical_uri.split("::")[-1], raw)
            if not text:
                continue
            sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{canonical_uri}:{sha}"))
            chunks = chunk_text(text)
            if args.dry_run:
                print(f"[DRY-RUN] {canonical_uri} -> {len(chunks)} chunks")
                summary["processed"] += 1
                summary["chunks"] += len(chunks)
                continue

            vault_paths = write_vault_notes(vault_dirs, doc_id, title, canonical_uri, sha, text)
            inserted = insert_sqlite_doc(args.db, doc_id, run_id, title, canonical_uri, route, sha, vault_paths[0] if vault_paths else "", text)
            if not inserted:
                summary["skipped_duplicate"] += 1
                if args.delete_source and deletable and deletable.exists():
                    deletable.unlink()
                continue

            if args.mode == "local":
                vecs = embed_ollama(chunks, args.ollama_url, args.model)
                points = [
                    {
                        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:chunk:{i}")),
                        "vector": vec,
                        "payload": {
                            "text": chunk,
                            "document_id": doc_id,
                            "title": title,
                            "source": canonical_uri,
                            "chunk_index": i,
                            "content_hash_sha256": sha,
                        },
                    }
                    for i, (chunk, vec) in enumerate(zip(chunks, vecs))
                ]
                upsert_qdrant(args.qdrant_url, DEFAULT_COLLECTIONS, points)
            elif args.mode == "cloud-push":
                if not neon_url:
                    raise RuntimeError("Set NEON_URL or CHYREN_DB_URL_NEON_CH3 for cloud-push.")
                rows = [
                    (
                        str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:chunk:{i}")),
                        sha,
                        "chyren-embed-worker",
                        canonical_uri,
                        "chyren_memory_knowledge",
                        str(i),
                        chunk.replace("\x00", ""),
                    )
                    for i, chunk in enumerate(chunks)
                ]
                push_neon_chunks(neon_url, rows)

            if args.delete_source and deletable and deletable.exists():
                deletable.unlink()
            summary["processed"] += 1
            summary["chunks"] += len(chunks)
        except Exception as e:
            summary["errors"].append({"source": canonical_uri, "error": str(e)})

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
