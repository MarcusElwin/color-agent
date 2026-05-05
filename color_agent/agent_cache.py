"""SQLite write-through cache for Tier 4 LLM responses.

Same shape as color_pizza.py's cache. Cache key = (query, model). Skip caching
when temperature is set — that's the consistency path which wants variance.
Repeat queries drop from ~25s to <10ms; the agent's structured payload is JSON
so we can store it as-is."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

TTL_SECONDS = 30 * 24 * 3600  # 30 days

_DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data" / "cache.sqlite"


def _open(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS agent_payloads (
            cache_key TEXT PRIMARY KEY,
            json TEXT NOT NULL,
            fetched_at REAL NOT NULL
        );
    """)
    conn.commit()
    return conn


def _key(query: str, model: str) -> str:
    # query comes in already-normalized from the router; defensive lower() in case
    # the caller skipped normalize.py.
    return f"{model}::{query.strip().lower()}"


def get(query: str, model: str,
        db_path: Path | None = None) -> dict[str, Any] | None:
    db = db_path or _DEFAULT_CACHE
    with _open(db) as conn:
        row = conn.execute(
            "SELECT json, fetched_at FROM agent_payloads WHERE cache_key = ?",
            (_key(query, model),),
        ).fetchone()
    if row is None:
        return None
    payload, fetched_at = row
    if time.time() - fetched_at > TTL_SECONDS:
        return None
    return json.loads(payload)


def put(query: str, model: str, payload: dict[str, Any],
        db_path: Path | None = None) -> None:
    db = db_path or _DEFAULT_CACHE
    with _open(db) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO agent_payloads (cache_key, json, fetched_at) "
            "VALUES (?, ?, ?)",
            (_key(query, model), json.dumps(payload), time.time()),
        )
        conn.commit()


def invalidate_all(db_path: Path | None = None) -> None:
    """For tests / eval --no-cache."""
    db = db_path or _DEFAULT_CACHE
    with _open(db) as conn:
        conn.execute("DELETE FROM agent_payloads")
        conn.commit()
