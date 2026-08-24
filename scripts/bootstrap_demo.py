import json
from pathlib import Path

from curiosity_engine.db import init_db
from curiosity_engine.graph import add_child, add_school_signal, upsert_node

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "curiosity.db"
profile = json.loads((ROOT / "examples" / "demo-family" / "profile.json").read_text())

init_db(DB)
for child in profile["children"]:
    add_child(DB, child["id"], child["name"], child.get("birth_year"), child.get("grade"))
    for interest in ["bridges", "birds"] if child["id"] == "demo-a" else ["shapes", "gardening"]:
        upsert_node(DB, child["id"], "interest", interest, confidence=0.7)

add_school_signal(DB, "demo-a", "teacher_language", "Try, notice, revise", source_ref="synthetic-newsletter")
add_school_signal(DB, "demo-a", "math", "repeating patterns", source_ref="synthetic-newsletter")
print(DB)
