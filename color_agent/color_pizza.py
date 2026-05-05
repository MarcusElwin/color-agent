"""color.pizza HTTP client with SQLite write-through cache.

API reference (https://github.com/meodai/color-name-api):
  GET https://api.color.pizza/v1/names/?name=<query>&list=default&maxResults=20
    -> {"colors": [{"name", "hex", "rgb", "hsl", "similarity"}, ...]}
       Levenshtein-derived similarity, 0..1, filtered >=0.6, ranked best-first,
       up to maxResults (default 20, max 50).

  GET https://api.color.pizza/v1/?values=<hex>&list=default&noduplicates=true
    -> {"colors": [{"name", "hex", "distance" (CIEDE2000), ...}], "paletteTitle": ...}

Cache:
  - SQLite at data/cache.sqlite, two tables (name_search, hex_lookup).
  - 30-day TTL — color names don't move.
  - Cache miss falls through to HTTP and writes back.
"""

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

import requests

BASE_URL = "https://api.color.pizza/v1"
DEFAULT_LIST = "default"
DEFAULT_MAX_RESULTS = 20
TTL_SECONDS = 30 * 24 * 3600  # 30 days
HTTP_TIMEOUT = 5.0
HEADERS = {"X-Referrer": "color-agent", "Accept-Encoding": "gzip"}

_DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data" / "cache.sqlite"


def _open(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS name_search (
            cache_key TEXT PRIMARY KEY,
            json TEXT NOT NULL,
            fetched_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS hex_lookup (
            cache_key TEXT PRIMARY KEY,
            json TEXT NOT NULL,
            fetched_at REAL NOT NULL
        );
    """)
    conn.commit()
    return conn


class ColorPizzaClient:
    def __init__(self, db_path: Path | None = None,
                 session: requests.Session | None = None):
        self.db_path = db_path or _DEFAULT_CACHE
        self.session = session or requests.Session()

    def _cache_get(self, table: str, key: str) -> dict[str, Any] | None:
        with _open(self.db_path) as conn:
            row = conn.execute(
                f"SELECT json, fetched_at FROM {table} WHERE cache_key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        json_blob, fetched_at = row
        if time.time() - fetched_at > TTL_SECONDS:
            return None
        return json.loads(json_blob)

    def _cache_put(self, table: str, key: str, value: dict[str, Any]) -> None:
        with _open(self.db_path) as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO {table} (cache_key, json, fetched_at) "
                "VALUES (?, ?, ?)",
                (key, json.dumps(value), time.time()),
            )
            conn.commit()

    def name_search(self, query: str, list_: str = DEFAULT_LIST,
                    max_results: int = DEFAULT_MAX_RESULTS) -> dict[str, Any]:
        key = f"{list_}::{max_results}::{query}"
        cached = self._cache_get("name_search", key)
        if cached is not None:
            return cached
        resp = self.session.get(
            f"{BASE_URL}/names/",
            params={"name": query, "list": list_, "maxResults": max_results},
            headers=HEADERS, timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        self._cache_put("name_search", key, data)
        return data

    def hex_lookup(self, hex_: str, list_: str = DEFAULT_LIST,
                   noduplicates: bool = True) -> dict[str, Any]:
        clean = hex_.lstrip("#").lower()
        key = f"{list_}::{noduplicates}::{clean}"
        cached = self._cache_get("hex_lookup", key)
        if cached is not None:
            return cached
        resp = self.session.get(
            f"{BASE_URL}/",
            params={
                "values": clean, "list": list_,
                "noduplicates": "true" if noduplicates else "false",
            },
            headers=HEADERS, timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        self._cache_put("hex_lookup", key, data)
        return data


_default_client: ColorPizzaClient | None = None


def get_client() -> ColorPizzaClient:
    global _default_client
    if _default_client is None:
        _default_client = ColorPizzaClient()
    return _default_client
