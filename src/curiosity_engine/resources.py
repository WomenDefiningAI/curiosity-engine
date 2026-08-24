from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from .db import connect, init_db, jdump, jload, utcnow


class PrivateResourceError(ValueError):
    pass


@dataclass(frozen=True)
class IndexReport:
    collection_id: str
    units: int
    documents: int
    chunks: int
    skipped_documents: int


def discover_private_catalogs(repository_root: str | Path) -> list[Path]:
    private_resources = Path(repository_root).resolve() / "private" / "resources"
    return sorted(path.resolve() for path in private_resources.glob("**/catalog.json") if path.is_file())


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_private_catalog(catalog_path: str | Path, repository_root: str | Path) -> Path:
    catalog = Path(catalog_path).resolve()
    private_root = Path(repository_root).resolve() / "private"
    if not _inside(catalog, private_root):
        raise PrivateResourceError("licensed resource catalogs must live under the ignored private/ tree")
    if not catalog.is_file():
        raise FileNotFoundError(catalog)
    ignore = Path(repository_root) / ".gitignore"
    patterns = ignore.read_text(encoding="utf-8") if ignore.exists() else ""
    if not any(line.strip().rstrip("/") == "private" for line in patterns.splitlines()):
        raise PrivateResourceError(".gitignore must ignore private/ before licensed content can be indexed")
    return catalog


def _chunks(text: str, max_chars: int = 2_400, overlap: int = 240) -> Iterable[tuple[int, str, str | None]]:
    cleaned = text.replace("\x00", "").replace("\r\n", "\n")
    pages = cleaned.split("\f")
    ordinal = 0
    for page_number, page in enumerate(pages, start=1):
        page = re.sub(r"[ \t]+", " ", page)
        page = re.sub(r"\n{4,}", "\n\n", page).strip()
        if not page:
            continue
        start = 0
        while start < len(page):
            end = min(len(page), start + max_chars)
            if end < len(page):
                boundary = max(page.rfind("\n\n", start, end), page.rfind(". ", start, end))
                if boundary > start + max_chars // 2:
                    end = boundary + (2 if page[boundary : boundary + 2] == ". " else 0)
            content = page[start:end].strip()
            if content:
                heading = next((line.strip() for line in content.splitlines() if 3 <= len(line.strip()) <= 120), None)
                yield ordinal, content, f"Page {page_number}: {heading}" if heading else f"Page {page_number}"
                ordinal += 1
            if end >= len(page):
                break
            start = max(start + 1, end - overlap)


def index_collection(
    db_path: str | Path,
    catalog_path: str | Path,
    *,
    repository_root: str | Path,
) -> IndexReport:
    """Index a purchased collection without copying any content into the public tree."""

    catalog_file = assert_private_catalog(catalog_path, repository_root)
    catalog = json.loads(catalog_file.read_text(encoding="utf-8"))
    access = catalog.get("access") or {}
    if access.get("scope") != "family_private" or access.get("redistribution_allowed") is not False:
        raise PrivateResourceError("catalog must declare family_private access and prohibit redistribution")
    init_db(db_path)
    collection_id = str(catalog["id"])
    now = utcnow()
    document_count = 0
    chunk_count = 0
    skipped = 0
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """INSERT INTO resource_collections(id,title,provider,license_scope,source_url,private,indexed_at,metadata_json)
               VALUES(?,?,?,?,?,1,?,?) ON CONFLICT(id) DO UPDATE SET title=excluded.title,provider=excluded.provider,
               license_scope=excluded.license_scope,source_url=excluded.source_url,private=1,indexed_at=excluded.indexed_at,
               metadata_json=excluded.metadata_json""",
            (
                collection_id,
                catalog["title"],
                catalog["provider"],
                access["scope"],
                catalog.get("source_url"),
                now,
                jdump({"access": access, "epistemic_rule": catalog.get("epistemic_rule")}),
            ),
        )
        unit_ids: set[str] = set()
        document_ids: set[str] = set()
        for unit_ordinal, unit in enumerate(catalog.get("units", []), start=1):
            unit_slug = str(unit["id"])
            unit_id = f"{collection_id}:{unit_slug}"
            unit_ids.add(unit_id)
            unit_meta = {
                "summary": unit.get("summary"),
                "week_themes": unit.get("week_themes", []),
                "topic_tags": unit.get("topic_tags", []),
            }
            conn.execute(
                """INSERT INTO resource_units(id,collection_id,title,ordinal,source_url,metadata_json)
                   VALUES(?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET title=excluded.title,ordinal=excluded.ordinal,
                   source_url=excluded.source_url,metadata_json=excluded.metadata_json""",
                (unit_id, collection_id, unit["title"], unit_ordinal, unit.get("source_url"), jdump(unit_meta)),
            )
            unit_dir = catalog_file.parent / unit_slug
            declared = set(unit.get("documents", []))
            available = {path.stem for path in unit_dir.glob("*.txt")}
            missing = declared - available
            if missing:
                raise PrivateResourceError(
                    f"{unit_slug} is missing declared text extracts: {', '.join(sorted(missing))}"
                )
            for doc_slug in sorted(declared):
                text_path = (unit_dir / f"{doc_slug}.txt").resolve()
                pdf_path = (unit_dir / f"{doc_slug}.pdf").resolve()
                if not _inside(text_path, catalog_file.parent) or not _inside(pdf_path, catalog_file.parent):
                    raise PrivateResourceError("resource path escaped collection directory")
                raw = text_path.read_text(encoding="utf-8", errors="replace")
                digest = _sha256_file(pdf_path)
                text_digest = sha256(raw.encode("utf-8")).hexdigest()
                doc_id = f"{unit_id}:{doc_slug}"
                document_ids.add(doc_id)
                document_count += 1
                existing = conn.execute(
                    "SELECT sha256,metadata_json FROM resource_documents WHERE id=?", (doc_id,)
                ).fetchone()
                page_count = max(1, raw.count("\f") + 1)
                conn.execute(
                    """INSERT INTO resource_documents(
                       id,unit_id,title,document_type,source_path,source_url,sha256,page_count,byte_count,
                       content_access,metadata_json,indexed_at
                       ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET title=excluded.title,
                       source_path=excluded.source_path,sha256=excluded.sha256,page_count=excluded.page_count,
                       byte_count=excluded.byte_count,metadata_json=excluded.metadata_json,indexed_at=excluded.indexed_at""",
                    (
                        doc_id,
                        unit_id,
                        doc_slug.replace("-", " ").title(),
                        doc_slug,
                        str(pdf_path),
                        unit.get("source_url"),
                        digest,
                        page_count,
                        pdf_path.stat().st_size,
                        "selected_excerpts",
                        jdump({"text_extract_path": str(text_path), "text_sha256": text_digest}),
                        now,
                    ),
                )
                existing_metadata = jload(existing["metadata_json"]) if existing else {}
                if existing and existing["sha256"] == digest and existing_metadata.get("text_sha256") == text_digest:
                    skipped += 1
                    chunk_count += int(
                        conn.execute("SELECT COUNT(*) FROM resource_chunks WHERE document_id=?", (doc_id,)).fetchone()[
                            0
                        ]
                    )
                    continue
                conn.execute("DELETE FROM resource_chunks WHERE document_id=?", (doc_id,))
                for ordinal, content, heading in _chunks(raw):
                    conn.execute(
                        """INSERT INTO resource_chunks(document_id,ordinal,heading,content,token_estimate,metadata_json)
                           VALUES(?,?,?,?,?,?)""",
                        (doc_id, ordinal, heading, content, max(1, len(content) // 4), jdump({"private": True})),
                    )
                    chunk_count += 1
        if len(unit_ids) != len(catalog.get("units", [])):
            raise PrivateResourceError("unit identifiers are not unique")
        conn.execute(
            "DELETE FROM resource_documents WHERE unit_id IN (SELECT id FROM resource_units WHERE collection_id=?) "
            + (f"AND id NOT IN ({','.join('?' for _ in document_ids)})" if document_ids else ""),
            (collection_id, *sorted(document_ids)),
        )
        conn.execute(
            "DELETE FROM resource_units WHERE collection_id=? "
            + (f"AND id NOT IN ({','.join('?' for _ in unit_ids)})" if unit_ids else ""),
            (collection_id, *sorted(unit_ids)),
        )
    return IndexReport(collection_id, len(unit_ids), document_count, chunk_count, skipped)


def _query_tokens(query: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", query.casefold())
    return list(dict.fromkeys(token for token in tokens if len(token) >= 3))[:12]


def search_resources(
    db_path: str | Path,
    query: str,
    *,
    limit: int = 5,
    include_excerpts: bool = False,
    collection_id: str | None = None,
) -> list[dict[str, Any]]:
    """Search private curriculum. Copyrighted text is omitted unless explicitly requested."""

    tokens = _query_tokens(query)
    if not tokens:
        return []
    fts_query = " OR ".join(f'"{token}"' for token in tokens)
    params: list[Any] = [fts_query]
    collection_clause = ""
    if collection_id:
        collection_clause = " AND c.id=?"
        params.append(collection_id)
    params.append(max(1, min(limit, 20)))
    with connect(db_path) as conn:
        rows = conn.execute(
            """SELECT ch.id AS chunk_id,ch.content,ch.heading,d.id AS document_id,d.title AS document_title,
                      d.document_type,d.page_count,d.content_access,u.id AS unit_id,u.title AS unit_title,
                      u.metadata_json AS unit_metadata,c.id AS collection_id,c.title AS collection_title,
                      bm25(resource_chunks_fts) AS rank
               FROM resource_chunks_fts
               JOIN resource_chunks ch ON ch.id=resource_chunks_fts.rowid
               JOIN resource_documents d ON d.id=ch.document_id
               JOIN resource_units u ON u.id=d.unit_id
               JOIN resource_collections c ON c.id=u.collection_id
               WHERE resource_chunks_fts MATCH ?"""
            + collection_clause
            + " ORDER BY rank LIMIT ?",
            params,
        ).fetchall()
    results: list[dict[str, Any]] = []
    for row in rows:
        meta = jload(row["unit_metadata"])
        item: dict[str, Any] = {
            "collection_id": row["collection_id"],
            "collection_title": row["collection_title"],
            "unit_id": row["unit_id"],
            "unit_title": row["unit_title"],
            "unit_summary": meta.get("summary"),
            "topic_tags": meta.get("topic_tags", []),
            "document_id": row["document_id"],
            "document_title": row["document_title"],
            "document_type": row["document_type"],
            "page_count": row["page_count"],
            "private": True,
        }
        if include_excerpts:
            item["heading"] = row["heading"]
            item["excerpt"] = row["content"][:1_800]
            item["sharing_notice"] = (
                "Purchased family-private excerpt; do not log, redistribute, or place in public eval fixtures."
            )
        results.append(item)
    return results


def resource_inventory(db_path: str | Path) -> dict[str, Any]:
    with connect(db_path) as conn:
        collections = conn.execute(
            "SELECT id,title,provider,license_scope,indexed_at FROM resource_collections ORDER BY title"
        ).fetchall()
        units = conn.execute(
            """SELECT u.id,u.title,u.ordinal,u.metadata_json,c.id AS collection_id,
                      COUNT(DISTINCT d.id) AS documents,COUNT(ch.id) AS chunks
               FROM resource_units u JOIN resource_collections c ON c.id=u.collection_id
               LEFT JOIN resource_documents d ON d.unit_id=u.id
               LEFT JOIN resource_chunks ch ON ch.document_id=d.id
               GROUP BY u.id ORDER BY c.title,u.ordinal"""
        ).fetchall()
    return {
        "collections": [dict(row) for row in collections],
        "units": [
            {**{k: row[k] for k in row.keys() if k != "metadata_json"}, **jload(row["metadata_json"])} for row in units
        ],
        "epistemic_rule": "Availability does not imply exposure, completion, understanding, or interest.",
    }
