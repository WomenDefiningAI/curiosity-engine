"""Optional provider usage ledger with shared, conservative experiment reservations.

Contains no prompts, responses, images, or credentials. Unknown charges retain
their reservation; they never become zero-cost successes. Dollar figures are
estimates from the dated rate card, not an account billing reconciliation.
"""
from __future__ import annotations

import contextvars
import json
import math
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

CASE = contextvars.ContextVar("usage_case", default="")
PRICING_DATE = "2026-09-20"
PRICING_SOURCE = "https://developers.openai.com/api/docs/pricing"
# Standard short-context USD / million tokens: input, cached input, output.
RATES = {"gpt-5.6-terra": (2.0, .2, 12.0), "gpt-5.6-sol": (4.0, .4, 20.0),
         "gpt-5.6-luna": (.2, .02, 1.2)}


class BudgetExceeded(RuntimeError):
    """The experiment cannot reserve another request inside its budget."""


@contextmanager
def usage_case(case_id):
    token = CASE.set(case_id)
    try:
        yield
    finally:
        CASE.reset(token)


def connect_ledger(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    conn = sqlite3.connect(path, timeout=30)
    path.chmod(0o600)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS calls (
        id TEXT PRIMARY KEY, case_id TEXT, phase TEXT, role TEXT, model TEXT,
        status TEXT, reserved_usd REAL, estimated_usd REAL, usage_json TEXT,
        duration_seconds REAL, error_type TEXT, pricing_date TEXT, created_at REAL)""")
    conn.execute("CREATE TABLE IF NOT EXISTS budget (id INTEGER PRIMARY KEY, limit_usd REAL)")
    conn.commit()
    return conn


def estimate_cost(model, usage, *, image=False, web_calls=0):
    if not usage:
        return None
    if model == "jev-1.13.0":
        tokens = usage.get("input_tokens")
        return tokens * .042 / 1e6 if isinstance(tokens, int) and tokens >= 0 else None
    inp, out = usage.get("input_tokens"), usage.get("output_tokens")
    if inp is None or out is None:
        return None
    if image and model == "gpt-image-2":
        details = usage.get("input_tokens_details") or {}
        if "text_tokens" not in details or "image_tokens" not in details:
            return None
        # If image cache modality is absent, use full input rates conservatively.
        return (details["text_tokens"] * 2.5 + details["image_tokens"] * 4 + out * 15) / 1e6
    if model not in RATES:
        return None
    rate, cached_rate, output_rate = RATES[model]
    details = usage.get("input_tokens_details") or {}
    cached = details.get("cached_tokens", 0)
    writes = details.get("cache_write_tokens", 0)
    if min(inp, out, cached, writes) < 0 or cached + writes > inp:
        return None
    if inp > 272_000:
        rate, cached_rate, output_rate = rate * 2, cached_rate * 2, output_rate * 1.5
    # Cache writes are a separate input category billed at 1.25x, not
    # ordinary uncached input. They are included in input_tokens.
    return ((inp - cached - writes) * rate + cached * cached_rate + writes * rate * 1.25 + out * output_rate) / 1e6 + web_calls * .01


def reservation(request, *, image=False):
    model = request["model"]
    if model == "jev-1.13.0":
        return (len(json.dumps(request).encode()) + 4096) * .042 / 1e6
    if request.get("service_tier") not in {None, "auto", "default"}:
        raise BudgetExceeded("A budget requires supported standard service pricing")
    if image:
        if model != "gpt-image-2" or request.get("size") not in {"1024x1024", "1536x1024", "1024x1536"}:
            raise BudgetExceeded("No reviewed image reservation for this model/size")
        # Above high-quality output at these sizes, plus prompt bytes as tokens.
        return .50 * int(request.get("n", 1)) + len(request.get("prompt", "").encode()) * 2.5 / 1e6
    if model not in RATES or not request.get("max_output_tokens"):
        raise BudgetExceeded("Budgeted calls require a known model and bounded output")

    def bound(value):
        if isinstance(value, dict):
            if value.get("type") == "input_image":
                return 25_000  # Conservative high-detail image allowance.
            return sum(len(str(k).encode()) + bound(v) for k, v in value.items())
        if isinstance(value, list):
            return sum(bound(v) for v in value)
        return len(str(value).encode())

    # Include framing and tool observations, at long-context rates. No tokenizer call.
    count = bound(request) + 4096 + (32_000 if request.get("tools") else 0)
    return (count * RATES[model][0] * 2 * 1.25 + request["max_output_tokens"] * RATES[model][2] * 1.5) / 1e6 + (.02 if request.get("tools") else 0)


def metered_call(request, invoke, *, role, image=False):
    path = os.environ.get("CURIOSITY_USAGE_DB")
    if not path:
        return invoke(False)
    limit = float(os.environ.get("CURIOSITY_BUDGET_USD", "0"))
    if not math.isfinite(limit) or limit < 0:
        raise ValueError("Budget must be a finite nonnegative amount")
    with connect_ledger(path) as conn:
        previous = conn.execute("SELECT limit_usd FROM budget WHERE id=1").fetchone()
    if previous and limit != previous[0]:
        raise BudgetExceeded("The existing experiment budget cannot be changed or disabled")
    reserved = reservation(request, image=image) if limit else 0.0
    call_id = uuid4().hex
    with connect_ledger(path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        if limit:
            conn.execute("INSERT OR IGNORE INTO budget VALUES(1,?)", (limit,))
            if conn.execute("SELECT limit_usd FROM budget WHERE id=1").fetchone()[0] != limit:
                raise BudgetExceeded("The existing experiment budget cannot be changed")
            committed = conn.execute("SELECT COALESCE(SUM(COALESCE(estimated_usd,reserved_usd)),0) FROM calls").fetchone()[0]
            if committed + reserved > limit:
                conn.execute("INSERT INTO calls VALUES(?,?,?,?,?,'budget_blocked',0,0,NULL,0,'BudgetExceeded',?,?)",
                             (call_id, CASE.get() or os.environ.get("CURIOSITY_USAGE_CASE", ""),
                              os.environ.get("CURIOSITY_USAGE_PHASE", "runtime"), role, request["model"],
                              PRICING_DATE, time.time()))
                conn.commit()
                raise BudgetExceeded("Experiment spending limit reached before sending a request")
        conn.execute("INSERT INTO calls VALUES(?,?,?,?,?,'running',?,NULL,NULL,NULL,NULL,?,?)",
                     (call_id, CASE.get() or os.environ.get("CURIOSITY_USAGE_CASE", ""),
                      os.environ.get("CURIOSITY_USAGE_PHASE", "runtime"), role, request["model"],
                      reserved, PRICING_DATE, time.time()))
    started = time.monotonic()
    status, error, usage, cost = "failed", None, None, None
    try:
        result = invoke(bool(limit))  # Budgeted calls disable hidden SDK retries.
        raw = getattr(result, "usage", None)
        usage = raw.model_dump() if hasattr(raw, "model_dump") else raw
        web_calls = sum(getattr(item, "type", None) == "web_search_call" for item in getattr(result, "output", []) or [])
        cost = estimate_cost(request["model"], usage, image=image, web_calls=web_calls)
        status = "completed" if cost is not None else "usage_unknown"
        return result
    except Exception as exc:
        error = type(exc).__name__
        raise
    finally:
        with connect_ledger(path) as conn:
            conn.execute("UPDATE calls SET status=?,estimated_usd=?,usage_json=?,duration_seconds=?,error_type=? WHERE id=?",
                         (status, cost, json.dumps(usage), time.monotonic() - started, error, call_id))


def usage_summary(path):
    with connect_ledger(path) as conn:
        rows = [dict(row) for row in conn.execute("SELECT * FROM calls ORDER BY created_at")]
        budget = conn.execute("SELECT limit_usd FROM budget WHERE id=1").fetchone()
    groups = {}
    for row in rows:
        bucket = groups.setdefault(row["phase"], {"calls": 0, "estimated_usd": 0., "unknown_calls": 0})
        bucket["calls"] += 1
        bucket["estimated_usd"] += row["estimated_usd"] or 0
        bucket["unknown_calls"] += row["estimated_usd"] is None
    return {"pricing_date": PRICING_DATE, "pricing_source": PRICING_SOURCE,
            "additional_pricing_sources": {"jev-1.13.0": "https://docs.typesafe.ai/models"}
            if any(row["model"] == "jev-1.13.0" for row in rows) else {}, "phases": groups,
            "budget_usd": budget[0] if budget else None,
            "estimated_usd": sum(r["estimated_usd"] or 0 for r in rows),
            "unknown_calls": sum(r["estimated_usd"] is None for r in rows),
            "reserved_or_spent_usd": sum(r["estimated_usd"] if r["estimated_usd"] is not None else r["reserved_usd"] for r in rows),
            "calls": rows}
