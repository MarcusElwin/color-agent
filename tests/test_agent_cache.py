"""Tier 4 SQLite cache: roundtrip + TTL + integration with call_agent."""

import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from color_agent import agent_cache
from color_agent.agent import call_agent


@pytest.fixture
def tmp_db(tmp_path: Path, monkeypatch) -> Path:
    db = tmp_path / "cache.sqlite"
    monkeypatch.setattr(agent_cache, "_DEFAULT_CACHE", db)
    return db


def test_put_and_get_roundtrip(tmp_db):
    payload = {"candidates": [{"hex": "#0047AB"}], "overall_confidence": "high"}
    agent_cache.put("cobalt blue", "claude-sonnet-4-6", payload)
    assert agent_cache.get("cobalt blue", "claude-sonnet-4-6") == payload


def test_get_miss_returns_none(tmp_db):
    assert agent_cache.get("not there", "claude-sonnet-4-6") is None


def test_different_models_dont_share_cache(tmp_db):
    agent_cache.put("cobalt blue", "claude-sonnet-4-6", {"x": "sonnet"})
    agent_cache.put("cobalt blue", "claude-haiku-4-5", {"x": "haiku"})
    assert agent_cache.get("cobalt blue", "claude-sonnet-4-6") == {"x": "sonnet"}
    assert agent_cache.get("cobalt blue", "claude-haiku-4-5") == {"x": "haiku"}


def test_cache_key_is_case_insensitive(tmp_db):
    agent_cache.put("Cobalt Blue", "claude-sonnet-4-6", {"hex": "#0047AB"})
    assert agent_cache.get("cobalt blue", "claude-sonnet-4-6") == {"hex": "#0047AB"}


def test_ttl_expiry(tmp_db):
    agent_cache.put("foo", "m", {"v": 1})
    with sqlite3.connect(tmp_db) as conn:
        conn.execute("UPDATE agent_payloads SET fetched_at = 0")
        conn.commit()
    assert agent_cache.get("foo", "m") is None


def test_invalidate_all(tmp_db):
    agent_cache.put("foo", "m", {"v": 1})
    agent_cache.invalidate_all()
    assert agent_cache.get("foo", "m") is None


# --- integration with call_agent -------------------------------------------

def _block(type_, **kw):
    b = MagicMock(); b.type = type_
    for k, v in kw.items():
        setattr(b, k, v)
    return b


def _resp(content, stop_reason="end_turn"):
    r = MagicMock(); r.content = content; r.stop_reason = stop_reason
    return r


PAYLOAD = {
    "candidates": [{"hex": "#0047AB", "name": "cobalt blue",
                     "confidence": "high", "rationale": "p"}] * 5,
    "overall_confidence": "high",
    "source": "knowledge",
}


def test_call_agent_writes_to_cache_on_first_call(tmp_db):
    cli = MagicMock()
    cli.messages.create.side_effect = [
        _resp([_block("text", text="search summary")]),
        _resp([_block("tool_use", name="return_hex_list", input=PAYLOAD)]),
    ]
    result = call_agent("cobalt blue", model="claude-sonnet-4-6", client=cli)
    assert result["candidates"][0]["hex"] == "#0047AB"
    # Cache should be populated now
    cached = agent_cache.get("cobalt blue", "claude-sonnet-4-6")
    assert cached is not None


def test_call_agent_skips_api_on_cache_hit(tmp_db):
    agent_cache.put("cobalt blue", "claude-sonnet-4-6", PAYLOAD)
    cli = MagicMock()
    result = call_agent("cobalt blue", model="claude-sonnet-4-6", client=cli)
    assert result["_cache_hit"] is True
    assert cli.messages.create.call_count == 0


def test_call_agent_skips_cache_when_temperature_set(tmp_db):
    """Consistency path uses temperature; we MUST hit the API every time."""
    agent_cache.put("cobalt blue", "claude-sonnet-4-6", PAYLOAD)
    cli = MagicMock()
    cli.messages.create.side_effect = [
        _resp([_block("text", text="search")]),
        _resp([_block("tool_use", name="return_hex_list", input=PAYLOAD)]),
    ]
    call_agent("cobalt blue", model="claude-sonnet-4-6",
               temperature=1.0, client=cli)
    assert cli.messages.create.call_count == 2  # API hit despite cache


def test_call_agent_skips_cache_when_use_cache_false(tmp_db):
    agent_cache.put("cobalt blue", "claude-sonnet-4-6", PAYLOAD)
    cli = MagicMock()
    cli.messages.create.side_effect = [
        _resp([_block("text", text="s")]),
        _resp([_block("tool_use", name="return_hex_list", input=PAYLOAD)]),
    ]
    call_agent("cobalt blue", model="claude-sonnet-4-6",
               use_cache=False, client=cli)
    assert cli.messages.create.call_count == 2
