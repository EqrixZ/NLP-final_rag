"""Structure-aware, Thai-aware chunking.

Strategy
--------
1. Split each document into sections using Markdown headings. ``###``
   sub-sections are kept together with their ``##`` parent and labelled inline
   (``"ลำต้นและกิ่ง: ..."``) so short sub-sections do not become tiny chunks.
2. Inside a section, break the text into small *units*: lines (bullets,
   paragraphs) -> sentences -> words. Thai has no spaces between words, so
   sentence splitting uses ``pythainlp.sent_tokenize`` and the last-resort word
   split uses ``pythainlp.word_tokenize`` (newmm). A Thai word is never cut.
3. Greedily pack units into chunks of ~``CHUNK_TARGET_CHARS`` (hard cap
   ``CHUNK_MAX_CHARS``) and prepend a word-aligned overlap of up to
   ``CHUNK_OVERLAP_CHARS`` from the previous chunk of the same section.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pythainlp.tokenize import sent_tokenize, word_tokenize

from rag.config import (
    CHUNK_MAX_CHARS,
    CHUNK_MIN_CHARS,
    CHUNK_OVERLAP_CHARS,
    CHUNK_TARGET_CHARS,
    PASSAGE_PREFIX,
)
from rag.loader import Document
from rag.textutils import thai_ratio

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_EN_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+")


@dataclass
class Chunk:
    """A retrievable piece of a document plus the metadata shown to users."""

    chunk_id: str
    text: str
    source_file: str
    doc_title: str
    section: str
    source_url: str
    language: str

    def embedding_text(self) -> str:
        """Text that is actually embedded: title + section give the vector topical context."""
        return f"{PASSAGE_PREFIX}{self.doc_title} | {self.section}\n{self.text}"


@dataclass
class _Unit:
    text: str
    joiner: str  # separator placed *before* this unit when it is appended
    label: str | None = None  # ``###`` sub-section this unit belongs to


@dataclass
class Section:
    """A ``##`` section (``h2``) or one of its ``###`` sub-sections (``sub``)."""

    h2: str
    sub: str | None
    text: str

    @property
    def name(self) -> str:
        return f"{self.h2} › {self.sub}" if self.sub else self.h2


# --------------------------------------------------------------------------
# Splitting helpers
# --------------------------------------------------------------------------
def split_sections(text: str, default_section: str) -> list[Section]:
    """Split ``text`` on Markdown headings.

    ``#`` (document title) is ignored, ``##`` starts a section and ``###``+
    starts a sub-section of the current ``##``. Empty sections are dropped.
    """
    sections: list[Section] = []
    h2, sub = default_section, None
    lines: list[str] = []

    def flush() -> None:
        body = "\n".join(lines).strip()
        if body:
            sections.append(Section(h2, sub, body))

    for line in text.split("\n"):
        match = _HEADING.match(line)
        if not match:
            lines.append(line)
            continue
        flush()
        lines = []
        level, heading = len(match.group(1)), match.group(2).strip()
        if level == 1:
            h2, sub = default_section, None
        elif level == 2:
            h2, sub = heading, None
        else:
            sub = heading
    flush()
    return sections


def _split_sentences(text: str) -> list[str]:
    """Thai-aware sentence split (Thai uses spaces as clause/sentence boundaries)."""
    if thai_ratio(text) >= 0.3:
        parts = sent_tokenize(text, engine="whitespace+newline")
    else:
        parts = _EN_SENTENCE_END.split(text)
    return [p.strip() for p in parts if p.strip()]


def _split_words(text: str, limit: int) -> list[str]:
    """Last resort: pack dictionary words (newmm) into pieces of at most ``limit`` chars."""
    pieces: list[str] = []
    buf = ""
    for token in word_tokenize(text, engine="newmm", keep_whitespace=True):
        if buf and len(buf) + len(token) > limit:
            pieces.append(buf.strip())
            buf = ""
        buf += token
    if buf.strip():
        pieces.append(buf.strip())
    return pieces


def _to_units(section: Section, unit_limit: int) -> list[_Unit]:
    """Break a (sub-)section into units no longer than ``unit_limit`` characters.

    A sub-section's heading is glued to its first line (``"ราก: ..."``) so the
    label travels with the text and can never be left orphaned at a chunk end.
    """
    units: list[_Unit] = []
    lines = [line.strip() for line in section.text.split("\n") if line.strip()]
    if section.sub and lines:
        lines[0] = f"{section.sub}: {lines[0]}"
    for line in lines:
        if len(line) <= unit_limit:
            units.append(_Unit(line, "\n", section.sub))
            continue
        first = True
        for sentence in _split_sentences(line):
            pieces = [sentence] if len(sentence) <= unit_limit else _split_words(sentence, unit_limit)
            for piece in pieces:
                units.append(_Unit(piece, "\n" if first else " ", section.sub))
                first = False
    return units


def _tail_overlap(text: str, limit: int) -> str:
    """Return the last words of ``text`` totalling at most ``limit`` chars (word-aligned)."""
    if limit <= 0:
        return ""
    tokens = word_tokenize(text, engine="newmm", keep_whitespace=True)
    tail: list[str] = []
    length = 0
    for token in reversed(tokens):
        if length + len(token) > limit:
            break
        tail.append(token)
        length += len(token)
    return "".join(reversed(tail)).strip()


def _pack(units: list[_Unit]) -> list[tuple[str, list[str]]]:
    """Greedily pack units into chunks with overlap.

    Returns ``[(chunk_text, sub_section_labels_in_chunk), ...]``.
    """
    chunks: list[tuple[str, list[str]]] = []
    current, labels = "", []

    def add_label(label: str | None) -> None:
        if label and label not in labels:
            labels.append(label)

    for unit in units:
        candidate = f"{current}{unit.joiner}{unit.text}" if current else unit.text
        too_big = len(candidate) > CHUNK_MAX_CHARS
        past_target = len(candidate) > CHUNK_TARGET_CHARS and len(current) >= CHUNK_MIN_CHARS
        if current and (too_big or past_target):
            chunks.append((current, labels))
            overlap = _tail_overlap(current, CHUNK_OVERLAP_CHARS)
            current, labels = (f"{overlap} {unit.text}" if overlap else unit.text), []
        else:
            current = candidate
        add_label(unit.label)
    if current:
        # Merge a tiny trailing piece into the previous chunk when it still fits.
        if chunks and len(current) < CHUNK_MIN_CHARS and len(chunks[-1][0]) + len(current) + 1 <= CHUNK_MAX_CHARS:
            prev_text, prev_labels = chunks[-1]
            chunks[-1] = (f"{prev_text}\n{current}", prev_labels + [l for l in labels if l not in prev_labels])
        else:
            chunks.append((current, labels))
    return chunks


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def chunk_document(doc: Document) -> list[Chunk]:
    """Split one document into ``Chunk`` objects with full metadata."""
    default_section = "ข้อมูลทั่วไป" if doc.language == "th" else "Overview"
    primary_url = doc.source_url.split(";")[0].strip()
    stem = Path(doc.source_file).stem
    # Leave room for the overlap so that overlap + unit never exceeds the hard cap.
    unit_limit = CHUNK_MAX_CHARS - CHUNK_OVERLAP_CHARS - 1

    # Group consecutive (sub-)sections that share the same ``##`` parent.
    groups: list[tuple[str, list[Section]]] = []
    for section in split_sections(doc.text, default_section):
        if groups and groups[-1][0] == section.h2:
            groups[-1][1].append(section)
        else:
            groups.append((section.h2, [section]))

    chunks: list[Chunk] = []
    for h2, sections in groups:
        units = [unit for section in sections for unit in _to_units(section, unit_limit)]
        for text, labels in _pack(units):
            chunks.append(
                Chunk(
                    chunk_id=f"{stem}#{len(chunks):03d}",
                    text=text,
                    source_file=doc.source_file,
                    doc_title=doc.title,
                    section=f"{h2} › {', '.join(labels)}" if labels else h2,
                    source_url=primary_url,
                    language=doc.language,
                )
            )
    return chunks


def chunk_documents(docs: list[Document]) -> list[Chunk]:
    """Chunk every document, preserving document order."""
    return [chunk for doc in docs for chunk in chunk_document(doc)]
