"""Query normalization. Lowercase, strip punctuation (keep #), collapse ws,
alias British greys to American grays so cache keys collapse properly."""

import re

_PUNCT = re.compile(r"[^\w#\s]+")
_WS = re.compile(r"\s+")
_HEX = re.compile(r"^#?([0-9a-f]{6})$", re.I)

_GREY_ALIASES = {
    "grey": "gray",
    "darkgrey": "darkgray",
    "lightgrey": "lightgray",
    "slategrey": "slategray",
    "darkslategrey": "darkslategray",
    "dimgrey": "dimgray",
    "lightslategrey": "lightslategray",
}


def normalize(query: str) -> str:
    q = query.strip().lower()
    q = _PUNCT.sub(" ", q)
    q = _WS.sub(" ", q).strip()
    parts = [_GREY_ALIASES.get(tok, tok) for tok in q.split(" ")]
    return " ".join(parts)


def parse_hex(query: str) -> str | None:
    """Return canonical '#RRGGBB' if the query is a bare hex, else None."""
    m = _HEX.match(query.strip())
    if not m:
        return None
    return "#" + m.group(1).upper()
