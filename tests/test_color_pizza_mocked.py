import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

from color_agent.color_pizza import (
    ColorPizzaClient, ColorPizzaPermanentError, ColorPizzaTransientError,
)
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


def test_tier3_below_fuzzy_min_returns_none(tmp_db: Path):
    """FUZZY_MIN is 0.65; anything below that is genuinely "color.pizza
    couldn't find a plausible match" and we should not return it."""
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _mock_response({"colors": [
        {"name": "Maybe", "hex": "#101010", "similarity": 0.55},
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


# --- Retry / error classification ------------------------------------------


def _resp_with_status(status_code: int, payload: dict | None = None) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.reason = f"status_{status_code}"
    resp.json.return_value = payload or {"colors": []}
    return resp


def test_403_retries_then_succeeds(tmp_db: Path):
    """color.pizza occasionally serves 403 from some networks; we retry."""
    session = MagicMock(spec=requests.Session)
    session.get.side_effect = [
        _resp_with_status(403),
        _resp_with_status(200, {"colors": [
            {"name": "Cobalt Blue", "hex": "#0047ab", "similarity": 1.0}
        ]}),
    ]
    client = ColorPizzaClient(db_path=tmp_db, session=session)
    data = client.name_search("cobalt blue")
    assert data["colors"][0]["hex"] == "#0047ab"
    assert session.get.call_count == 2


def test_persistent_5xx_raises_transient(tmp_db: Path):
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _resp_with_status(503)
    client = ColorPizzaClient(db_path=tmp_db, session=session)
    with pytest.raises(ColorPizzaTransientError):
        client.name_search("anything")
    # Initial + 2 retries = 3 attempts total.
    assert session.get.call_count == 3


def test_400_raises_permanent_no_retry(tmp_db: Path):
    """Bad-request style errors are permanent — don't waste retries."""
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _resp_with_status(400)
    client = ColorPizzaClient(db_path=tmp_db, session=session)
    with pytest.raises(ColorPizzaPermanentError):
        client.name_search("anything")
    assert session.get.call_count == 1


def test_connection_error_is_transient(tmp_db: Path):
    session = MagicMock(spec=requests.Session)
    session.get.side_effect = requests.ConnectionError("boom")
    client = ColorPizzaClient(db_path=tmp_db, session=session)
    with pytest.raises(ColorPizzaTransientError):
        client.name_search("anything")
    assert session.get.call_count == 3  # initial + 2 retries


# --- Lowered fuzzy floor + cluster tightness -------------------------------


def test_tier3_fuzzy_in_065_to_085_now_returned(tmp_db: Path):
    """Pre-fix this returned None; now `salmon pink`-like queries land in Tier 3."""
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _mock_response({"colors": [
        {"name": "Salmon Pink", "hex": "#ff91a4", "similarity": 0.78},
        {"name": "Salmon",      "hex": "#fa8072", "similarity": 0.72},
        {"name": "Light Salmon","hex": "#ffa07a", "similarity": 0.68},
    ]})
    client = ColorPizzaClient(db_path=tmp_db, session=session)
    result = tier2_or_3("salmon pink", k=5, client=client)
    assert result is not None
    cands, tier, _ = result
    assert tier == "3"
    assert cands[0].hex == "#FF91A4"


def test_tight_cluster_marks_confident_even_below_threshold(tmp_db: Path):
    """Three close-RGB hits with mediocre individual similarity = confident."""
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _mock_response({"colors": [
        {"name": "Burnt Sienna",   "hex": "#e97451", "similarity": 0.75},
        {"name": "Burnt Sienna 2", "hex": "#ea7651", "similarity": 0.74},
        {"name": "Sienna Burnt",   "hex": "#e57451", "similarity": 0.72},
    ]})
    client = ColorPizzaClient(db_path=tmp_db, session=session)
    result = tier2_or_3("burnt sienna", k=5, client=client)
    assert result is not None
    _, tier, confident = result
    assert tier == "3"
    assert confident is True  # cluster tightness rescued it


def test_scattered_cluster_stays_unconfident(tmp_db: Path):
    """If top-3 disagree on hue, don't claim confidence."""
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _mock_response({"colors": [
        {"name": "Match A", "hex": "#FF0000", "similarity": 0.75},
        {"name": "Match B", "hex": "#00FF00", "similarity": 0.74},
        {"name": "Match C", "hex": "#0000FF", "similarity": 0.72},
    ]})
    client = ColorPizzaClient(db_path=tmp_db, session=session)
    result = tier2_or_3("ambiguous query", k=5, client=client)
    assert result is not None
    _, tier, confident = result
    assert tier == "3"
    assert confident is False
