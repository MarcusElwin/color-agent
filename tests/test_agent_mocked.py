from unittest.mock import MagicMock

import pytest

from color_agent.agent import (
    call_agent, model_config, to_candidates, RETURN_HEX_LIST_TOOL,
)


def _block(type_, **kw):
    b = MagicMock()
    b.type = type_
    for k, v in kw.items():
        setattr(b, k, v)
    return b


def _resp(content, stop_reason="end_turn"):
    r = MagicMock()
    r.content = content
    r.stop_reason = stop_reason
    return r


def _five_candidates_payload(source="knowledge", overall="high"):
    return {
        "candidates": [
            {"hex": "#0047AB", "name": "cobalt blue", "confidence": "high",
             "rationale": "primary"},
            {"hex": "#0048AC", "name": "cobalt", "confidence": "high",
             "rationale": "near"},
            {"hex": "#0050B0", "name": "cobalt glaze", "confidence": "medium",
             "rationale": "near"},
            {"hex": "#1F4E79", "name": "deep blue", "confidence": "medium",
             "rationale": "alt"},
            {"hex": "#0F52BA", "name": "sapphire", "confidence": "low",
             "rationale": "alt"},
        ],
        "overall_confidence": overall,
        "source": source,
    }


def test_model_config_haiku_uses_basic_web_search():
    cfg = model_config("claude-haiku-4-5")
    assert cfg.web_search_tool["type"] == "web_search_20250305"
    assert cfg.requires_code_execution is False


def test_model_config_sonnet_uses_advanced_web_search():
    cfg = model_config("claude-sonnet-4-6")
    assert cfg.web_search_tool["type"] == "web_search_20260209"
    # code_execution is auto-injected server-side; we don't add it ourselves
    assert cfg.requires_code_execution is False


def test_model_config_opus_uses_advanced_web_search():
    cfg = model_config("claude-opus-4-7")
    assert cfg.web_search_tool["type"] == "web_search_20260209"


def test_return_hex_list_tool_requires_min_5():
    schema = RETURN_HEX_LIST_TOOL["input_schema"]
    assert schema["properties"]["candidates"]["minItems"] == 5


def test_call_agent_two_step_flow_happy_path():
    """Step 1 returns text (search summary). Step 2 returns forced tool_use."""
    cli = MagicMock()
    step1 = _resp([_block("text", text="Cobalt blue is...")], stop_reason="end_turn")
    step2 = _resp([_block("tool_use", name="return_hex_list",
                           input=_five_candidates_payload())])
    cli.messages.create.side_effect = [step1, step2]

    payload = call_agent("cobalt blue", model="claude-sonnet-4-6", client=cli)

    assert cli.messages.create.call_count == 2
    # Step 1 was auto, step 2 was forced
    call1_kwargs = cli.messages.create.call_args_list[0].kwargs
    call2_kwargs = cli.messages.create.call_args_list[1].kwargs
    assert call1_kwargs["tool_choice"] == {"type": "auto"}
    assert call2_kwargs["tool_choice"] == {"type": "tool", "name": "return_hex_list"}
    assert payload["source"] == "knowledge"
    assert len(payload["candidates"]) == 5


def test_call_agent_raises_if_no_tool_use():
    cli = MagicMock()
    cli.messages.create.side_effect = [
        _resp([_block("text", text="hi")], stop_reason="end_turn"),
        _resp([_block("text", text="no tool")], stop_reason="end_turn"),
    ]
    with pytest.raises(RuntimeError, match="did not call return_hex_list"):
        call_agent("foo", client=cli)


def test_to_candidates_uses_confidence_scores():
    cands = to_candidates(_five_candidates_payload(source="web_search"), k=5)
    assert len(cands) == 5
    assert cands[0].source == "llm_web_search"
    assert cands[0].score == 0.9   # high
    assert cands[2].score == 0.7   # medium
    assert cands[4].score == 0.5   # low
    assert cands[0].hex == "#0047AB"
