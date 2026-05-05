"""Test isolation: never touch the user's real Tier 4 cache."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_agent_cache(tmp_path: Path, monkeypatch):
    """Redirect the Tier 4 SQLite cache to a per-test tmp dir."""
    from color_agent import agent_cache

    monkeypatch.setattr(agent_cache, "_DEFAULT_CACHE",
                         tmp_path / "agent_cache.sqlite")
    yield
