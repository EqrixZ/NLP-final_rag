"""Small text helpers shared by the loader, chunker and pipeline."""

from __future__ import annotations

import re
import unicodedata

_THAI_CHAR = re.compile(r"[฀-๿]")
_LATIN_CHAR = re.compile(r"[A-Za-z]")
# Zero-width space/joiners, BOM and soft hyphen often appear in text copied from the web.
_INVISIBLE = re.compile(r"[​‌‍⁠﻿­]")
_MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_MD_EMPHASIS = re.compile(r"(\*\*|__|\*|`)")
_HTML_TAG = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"[ \t ]+")


def thai_ratio(text: str) -> float:
    """Return the share of Thai letters among all Thai + Latin letters."""
    thai = len(_THAI_CHAR.findall(text))
    latin = len(_LATIN_CHAR.findall(text))
    total = thai + latin
    return thai / total if total else 0.0


def detect_language(text: str) -> str:
    """Very small language detector: ``"th"`` if the text is mostly Thai, else ``"en"``."""
    return "th" if thai_ratio(text) >= 0.3 else "en"


def clean_line(line: str) -> str:
    """Normalise one line: Unicode NFC, strip markup noise and collapse spaces."""
    line = unicodedata.normalize("NFC", line)
    line = _INVISIBLE.sub("", line)
    line = _HTML_TAG.sub("", line)
    line = _MD_LINK.sub(r"\1", line)
    line = _MD_EMPHASIS.sub("", line)
    line = _SPACES.sub(" ", line)
    return line.strip()


def clean_text(text: str) -> str:
    """Clean a whole document while keeping line breaks (needed to find headings).

    Runs of 3+ blank lines are collapsed to a single blank line.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [clean_line(line) for line in text.split("\n")]
    cleaned = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()
