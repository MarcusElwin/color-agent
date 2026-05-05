import json

from click.testing import CliRunner

from color_agent.cli import cli


def test_cli_human_output_for_css_color():
    result = CliRunner().invoke(cli, ["crimson", "--no-color"])
    assert result.exit_code == 0
    assert "tier=1" in result.output
    assert "#DC143C" in result.output
    assert "confident=True" in result.output


def test_cli_json_output_shape():
    result = CliRunner().invoke(cli, ["crimson", "--json", "--no-color"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["tier"] == "1"
    assert payload["confident"] is True
    assert len(payload["candidates"]) == 5
    assert payload["candidates"][0]["hex"] == "#DC143C"
    assert payload["candidates"][0]["score"] == 1.0


def test_cli_top_k_arg():
    result = CliRunner().invoke(cli, ["crimson", "-k", "3", "--json"])
    payload = json.loads(result.output)
    assert len(payload["candidates"]) == 3


def test_cli_force_tier1_miss():
    result = CliRunner().invoke(cli, ["not-a-color", "--force", "tier1", "--json"])
    payload = json.loads(result.output)
    assert payload["tier"] == "1"
    assert payload["candidates"] == []
    assert payload["confident"] is False


def test_cli_no_query_errors():
    result = CliRunner().invoke(cli, [])
    assert result.exit_code == 2
    assert "QUERY" in result.output or "Usage" in result.output


def test_cli_help_includes_banner_and_fast_flag():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "--fast" in result.output
    # banner contains a recognizable wordmark fragment
    assert "tiered lookup" in result.output


def test_cli_default_model_is_sonnet_46():
    """The --help text should show claude-sonnet-4-6 as the default."""
    result = CliRunner().invoke(cli, ["--help"])
    assert "claude-sonnet-4-6" in result.output


def test_cli_fast_flag_overrides_model(monkeypatch):
    """--fast must route to claude-haiku-4-5 even if --model is also passed."""
    captured: dict = {}

    def fake_to_hex(query, k=5, force=None, model="x", on_progress=None,
                     use_cache=True):
        captured["model"] = model
        from color_agent.types import Candidate, Result
        return Result(query, query, [Candidate("#000000", "x", 1.0, "css")],
                       True, "1", latency_ms=0)

    monkeypatch.setattr("color_agent.cli.to_hex", fake_to_hex)
    result = CliRunner().invoke(cli, ["red", "--fast", "--quiet"])
    assert result.exit_code == 0
    assert captured["model"] == "claude-haiku-4-5"
