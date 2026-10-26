"""Remember exposed inputs across runs: a reused case is a regression, never unseen."""
import hashlib
import json
import sqlite3
from pathlib import Path


def register_exposure(path, cases, source_hash, *, purpose, run_id):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with sqlite3.connect(path, timeout=30) as conn:
        path.chmod(0o600)
        conn.execute("CREATE TABLE IF NOT EXISTS exposures (input_hash TEXT PRIMARY KEY, first_run TEXT, source_hash TEXT)")
        conn.execute("BEGIN IMMEDIATE")
        seen = 0
        for case in cases:
            # ID changes and reworded reviewer expectations cannot make an old input fresh.
            digest = hashlib.sha256(json.dumps(case["input"], sort_keys=True).encode()).hexdigest()
            previous = conn.execute("SELECT first_run FROM exposures WHERE input_hash=?", (digest,)).fetchone()
            seen += bool(previous and previous[0] != run_id)
            conn.execute("INSERT OR IGNORE INTO exposures VALUES(?,?,?)", (digest, run_id, source_hash))
    return {"requested": purpose, "previously_exposed_inputs": seen,
            "classification": "regression" if seen else "validation_candidate"
            if purpose == "validation" and all(case.get("split") in {"heldout", "reserved"} for case in cases)
            else "development",
            "limitation": "Novel hashes do not prove unseen concepts. Operator must keep validation inputs out of development."}
