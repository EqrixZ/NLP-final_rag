"""Load knowledge-base files from ``data/`` and parse their metadata header.

Expected file layout::

    title: โรคไหม้ข้าว (Rice Blast)
    topic_th: โรคไหม้ข้าว            # short names used for suggestions / sidebar
    topic_en: rice blast
    category: rice                  # rice | durian | mango | cassava | vegetables | general
    kind: disease                   # disease | pest | practice
    source_name: กรมการข้าว; IRRI
    source_url: https://...; https://...
    date_accessed: 2026-10-07
    language: th
    ---
    # Markdown body ...
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from rag.config import DATA_DIR, SUPPORTED_EXTENSIONS
from rag.textutils import clean_text, detect_language

_HEADER_LINE = re.compile(r"^([A-Za-z_]+)\s*:\s*(.*)$")


@dataclass
class Document:
    """A cleaned source document with its metadata."""

    text: str
    source_file: str
    title: str
    source_name: str = ""
    source_url: str = ""
    language: str = "th"
    topic_th: str = ""
    topic_en: str = ""
    category: str = "general"
    kind: str = "practice"
    metadata: dict[str, str] = field(default_factory=dict)


def parse_header(raw: str) -> tuple[dict[str, str], str]:
    """Split ``raw`` into (metadata dict, body).

    The header is a block of ``key: value`` lines terminated by a line that is
    exactly ``---``. Files without a header return an empty dict and the full text.
    """
    lines = raw.lstrip("﻿").splitlines()
    meta: dict[str, str] = {}
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "---":
            if meta:
                return meta, "\n".join(lines[i + 1 :])
            continue  # leading front-matter delimiter
        match = _HEADER_LINE.match(stripped)
        if match:
            meta[match.group(1).lower()] = match.group(2).strip()
        elif stripped:
            break  # body started without a header terminator
    return {}, raw


def load_document(path: Path) -> Document:
    """Read, parse and clean a single file."""
    raw = path.read_text(encoding="utf-8")
    meta, body = parse_header(raw)
    text = clean_text(body)
    title = meta.get("title") or path.stem.replace("_", " ")
    return Document(
        text=text,
        source_file=path.name,
        title=title,
        source_name=meta.get("source_name", ""),
        source_url=meta.get("source_url", ""),
        language=meta.get("language") or detect_language(text),
        topic_th=meta.get("topic_th") or title,
        topic_en=meta.get("topic_en") or title,
        category=meta.get("category", "general"),
        kind=meta.get("kind", "practice"),
        metadata=meta,
    )


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """Load every supported file in ``data_dir`` (sorted for deterministic chunk ids)."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Knowledge-base folder not found: {data_dir}")
    paths = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"No .md/.txt files found in {data_dir}")
    return [load_document(p) for p in paths]
