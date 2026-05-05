from unittest.mock import MagicMock

import pytest

from color_agent.reflect import reflect


def _block(type_, **kw):
    b = MagicMock(); b.type = type_
    for k, v in kw.items():
        setattr(b, k, v)
    return b


def _resp(content):
    r = MagicMock(); r.content = content; r.stop_reason = "tool_use"
    return r


def _payload(first_hex):
    return {
        "candidates": [
            {"hex": first_hex, "name": "burnt sienna", "confidence": "high",
             "rationale": "corrected from green to orange-brown"},
            {"hex": "#E08550", "name": "burnt sienna alt", "confidence": "high",
             "rationale": "alt"},
            {"hex": "#C9583B", "name": "deeper", "confidence": "medium",
             "rationale": "alt"},
            {"hex": "#D67648", "name": "warmer", "confidence": "medium",
             "rationale": "alt"},
            {"hex": "#A0522D", "name": "sienna", "confidence": "low",
             "rationale": "alt"},
        ],
        "overall_confidence": "high",
        "source": "knowledge",
    }


def test_reflect_corrects_wrong_hex():
    cli = MagicMock()
    cli.messages.create.return_value = _resp([
        _block("tool_use", name="return_hex_list", input=_payload("#E97451"))
    ])
    initial = {
        "candidates": [
            {"hex": "#00FF00", "name": "burnt sienna", "confidence": "low",
             "rationale": "guess"},
        ],
        "overall_confidence": "low",
        "source": "knowledge",
    }
    out = reflect("burnt sienna", initial, client=cli)
    assert out["candidates"][0]["hex"] == "#E97451"
    assert "corrected" in out["candidates"][0]["rationale"].lower()


def test_reflect_uses_default_reviewer():
    cli = MagicMock()
    cli.messages.create.return_value = _resp([
        _block("tool_use", name="return_hex_list", input=_payload("#E97451"))
    ])
    reflect("x", {"candidates": []}, client=cli)
    call = cli.messages.create.call_args
    assert call.kwargs["model"] == "claude-sonnet-4-6"
    assert call.kwargs["tool_choice"] == {"type": "tool", "name": "return_hex_list"}


def test_reflect_reviewer_can_be_overridden():
    cli = MagicMock()
    cli.messages.create.return_value = _resp([
        _block("tool_use", name="return_hex_list", input=_payload("#E97451"))
    ])
    reflect("x", {"candidates": []}, reviewer="claude-opus-4-7", client=cli)
    assert cli.messages.create.call_args.kwargs["model"] == "claude-opus-4-7"


def test_reflect_raises_on_missing_tool():
    cli = MagicMock()
    cli.messages.create.return_value = _resp([_block("text", text="nope")])
    with pytest.raises(RuntimeError):
        reflect("x", {"candidates": []}, client=cli)
