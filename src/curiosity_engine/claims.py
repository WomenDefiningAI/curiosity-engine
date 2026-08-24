from __future__ import annotations

from pathlib import Path

from .db import connect, init_db, jdump, jload, utcnow

VALID_CLAIM_STATES = {"hypothesis", "established_pattern", "contradicted", "retired"}


def upsert_claim(
    db_path: str | Path,
    *,
    child_id: str,
    subject: str,
    predicate: str,
    object_: str,
    supporting_evidence_ids: list[int],
    contradicting_evidence_ids: list[int] | None = None,
    requested_status: str = "hypothesis",
    confidence: float = 0.5,
) -> int:
    """Persist an epistemic claim while enforcing evidence thresholds in code."""

    if requested_status not in VALID_CLAIM_STATES:
        raise ValueError(f"invalid claim status: {requested_status}")
    supporting = sorted(set(supporting_evidence_ids))
    contradicting = sorted(set(contradicting_evidence_ids or []))
    if requested_status == "established_pattern" and len(supporting) < 2:
        raise ValueError("established_pattern requires at least two distinct supporting evidence records")
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be 0..1")
    init_db(db_path)
    now = utcnow()
    with connect(db_path) as conn:
        evidence = (
            conn.execute(
                f"SELECT id,child_id,event_id FROM evidence WHERE id IN ({','.join('?' for _ in supporting + contradicting)})",
                (*supporting, *contradicting),
            ).fetchall()
            if supporting or contradicting
            else []
        )
        found = {int(row["id"]): row for row in evidence}
        missing = set(supporting + contradicting) - set(found)
        if missing:
            raise ValueError(f"unknown evidence ids: {sorted(missing)}")
        if any(row["child_id"] not in {None, child_id} for row in evidence):
            raise ValueError("evidence belongs to a different child")
        if requested_status == "established_pattern":
            source_events = {row["event_id"] for row in evidence if row["id"] in supporting and row["event_id"]}
            if len(source_events) < 2:
                raise ValueError("established_pattern requires evidence from at least two distinct events")
        row = conn.execute(
            "SELECT id,supporting_evidence_json,contradicting_evidence_json FROM claims WHERE child_id=? AND subject=? AND predicate=? AND object=?",
            (child_id, subject, predicate, object_),
        ).fetchone()
        if row:
            supporting = sorted(set(jload(row["supporting_evidence_json"], []) + supporting))
            contradicting = sorted(set(jload(row["contradicting_evidence_json"], []) + contradicting))
            status = "contradicted" if contradicting and requested_status == "established_pattern" else requested_status
            conn.execute(
                """UPDATE claims SET status=?,confidence=?,supporting_evidence_json=?,
                   contradicting_evidence_json=?,updated_at=? WHERE id=?""",
                (status, confidence, jdump(supporting), jdump(contradicting), now, row["id"]),
            )
            return int(row["id"])
        cur = conn.execute(
            """INSERT INTO claims(child_id,subject,predicate,object,status,confidence,supporting_evidence_json,
               contradicting_evidence_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                child_id,
                subject,
                predicate,
                object_,
                requested_status,
                confidence,
                jdump(supporting),
                jdump(contradicting),
                now,
                now,
            ),
        )
        return int(cur.lastrowid)
