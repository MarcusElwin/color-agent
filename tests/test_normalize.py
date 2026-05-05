from color_agent.normalize import normalize, parse_hex


def test_lowercases_and_strips_punct():
    assert normalize("Cobalt Blue!") == "cobalt blue"


def test_collapses_whitespace():
    assert normalize("  cobalt   blue  ") == "cobalt blue"


def test_hyphen_becomes_space():
    assert normalize("cobalt-blue") == "cobalt blue"


def test_grey_aliased_to_gray():
    assert normalize("grey") == "gray"
    assert normalize("dark grey") == "dark gray"
    assert normalize("Slategrey") == "slategray"


def test_grey_substring_not_stripped():
    assert normalize("greypound") == "greypound"


def test_parse_hex_with_hash():
    assert parse_hex("#0047AB") == "#0047AB"


def test_parse_hex_without_hash():
    assert parse_hex("0047ab") == "#0047AB"


def test_parse_hex_rejects_non_hex():
    assert parse_hex("cobalt") is None
    assert parse_hex("#GGGGGG") is None
    assert parse_hex("#FFF") is None  # 3-char short form intentionally not supported
