import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

from color_agent.color_pizza import ColorPizzaClient
from color_agent.tier23 import hex_neighbors, tier2_or_3


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "cache.sqlite"


def _mock_response(payload: dict) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 200
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def test_name_search_caches_result(tmp_db: Path):
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _mock_response({
        "colors": [{"name": "Cobalt Blue", "hex": "#0047ab", "similarity": 1.0}]
    })
    client = ColorPizzaClient(db_path=tmp_db, session=session)

    data1 = client.name_search("cobalt blue")
    data2 = client.name_search("cobalt blue")

    assert data1 == data2
    assert session.get.call_count == 1  # second call hit cache


def test_hex_lookup_caches_result(tmp_db: Path):
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _mock_response({
        "colors": [{"name": "Cobalt", "hex": "#0047ab", "distance": 0.5}]
    })
    client = ColorPizzaClient(db_path=tmp_db, session=session)
    client.hex_lookup("#0047AB")
    client.hex_lookup("0047ab")  # different format, same canonical key
    assert session.get.call_count == 1


def test_tier2_exact_match_confident(tmp_db: Path):
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _mock_response({"colors": [
        {"name": "Cobalt Blue", "hex": "#0047ab", "similarity": 1.0},
        {"name": "Cobalt", "hex": "#0048ac", "similarity": 0.95},
        {"name": "Cobalt Glaze", "hex": "#0050b0", "similarity": 0.92},
        {"name": "Bluebell", "hex": "#0050c0", "similarity": 0.88},
        {"name": "Sapphire", "hex": "#0f52ba", "similarity": 0.85},
    ]})
    client = ColorPizzaClient(db_path=tmp_db, session=session)

    result = tier2_or_3("cobalt blue", k=5, client=client)
    assert result is not None
    cands, tier, confident = result
    assert tier == "2"
    assert confident is True
    assert len(cands) == 5
    assert cands[0].source == "color_pizza_exact"
    assert cands[0].hex == "#0047AB"
    assert cands[0].score == 1.0


def test_tier3_fuzzy_match_high_confidence(tmp_db: Path):
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _mock_response({"colors": [
        {"name": "Cobalt Bluish", "hex": "#0047ab", "similarity": 0.94},
        {"name": "Cobalt", "hex": "#0048ac", "similarity": 0.91},
    ]})
    client = ColorPizzaClient(db_path=tmp_db, session=session)
    result = tier2_or_3("cobalt blu", k=5, client=client)
    assert result is not None
    cands, tier, confident = result
    assert tier == "3"
    assert confident is True
    assert cands[0].source == "color_pizza_fuzzy"


def test_tier3_low_similarity_returns_none(tmp_db: Path):
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _mock_response({"colors": [
        {"name": "Maybe", "hex": "#101010", "similarity": 0.7},
    ]})
    client = ColorPizzaClient(db_path=tmp_db, session=session)
    assert tier2_or_3("xyzzy", k=5, client=client) is None


def test_tier3_empty_returns_none(tmp_db: Path):
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _mock_response({"colors": []})
    client = ColorPizzaClient(db_path=tmp_db, session=session)
    assert tier2_or_3("nothing", k=5, client=client) is None


def test_hex_neighbors_returns_scored_candidates(tmp_db: Path):
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _mock_response({"colors": [
        {"name": "Cobalt Blue", "hex": "#0047ab", "distance": 0.0},
        {"name": "Cobalt", "hex": "#0048ac", "distance": 2.5},
    ]})
    client = ColorPizzaClient(db_path=tmp_db, session=session)
    out = hex_neighbors("#0047AB", k=5, client=client)
    assert out[0].score > out[1].score
    assert out[0].source == "color_pizza_hex"


def test_cache_ttl_expiry(tmp_db: Path):
    """Verify cache rows older than TTL are ignored — manual fetched_at update."""
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _mock_response({"colors": []})
    client = ColorPizzaClient(db_path=tmp_db, session=session)
    client.name_search("foo")
    # Force the cache row's fetched_at to be ancient
    with sqlite3.connect(tmp_db) as conn:
        conn.execute("UPDATE name_search SET fetched_at = 0")
        conn.commit()
    client.name_search("foo")
    assert session.get.call_count == 2  # cache miss after TTL expired
