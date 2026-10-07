"""End-to-end RAG pipeline shared by the Streamlit app and ``evaluate.py``.

Flow: detect language → rewrite follow-up (if history) → FAISS retrieval →
grounded generation. "No information" is detected in two layers that produce
the same structured result (``status == "no_info"``):

* Layer 1 (retrieval): best similarity below the threshold → no LLM call.
* Layer 2 (LLM): the model outputs the ``[[NO_INFO]]`` sentinel.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from sentence_transformers import SentenceTransformer

from rag.chunker import chunk_documents
from rag.config import (
    DATA_DIR,
    DEFAULT_TOP_K,
    MAX_CLOSEST_TOPICS,
    NEAR_MATCH_MIN_SCORE,
    NO_INFO_SENTINEL,
    REFUSAL_EN,
    REFUSAL_TH,
    REWRITE_HISTORY_TURNS,
    SIMILARITY_THRESHOLD,
)
from rag.index import RetrievedChunk, VectorIndex
from rag.llm import ChatModel, LLMError
from rag.loader import load_documents
from rag.prompts import build_answer_messages, build_rewrite_messages
from rag.textutils import detect_language
from rag.topics import Topic, build_topics, main_topics, make_suggestions, mentions_chemicals

logger = logging.getLogger(__name__)

Status = Literal["answered", "no_info", "error"]

_CITATION = re.compile(r"\[(\d+)\]")
# Some models write full-width brackets, optionally with line refs (【1】, 【1†L1-L3】); normalise to [1].
_FULLWIDTH_CITATION = re.compile(r"【\s*(\d+)(?:†[^】]*)?\s*】")


@dataclass
class RAGResult:
    """Everything the UI / evaluator needs about one question."""

    question: str
    answer: str  # answer text; for no_info the card headline; for error a friendly message
    language: str
    rewritten_query: str
    status: Status = "answered"
    # Chunks above the threshold — exactly what the LLM saw.
    sources: list[RetrievedChunk] = field(default_factory=list)
    # Chunks that fell below the threshold (shown for transparency, never sent to the LLM).
    near_misses: list[RetrievedChunk] = field(default_factory=list)
    cited_ranks: set[int] = field(default_factory=set)
    # no_info only: up to 3 nearest distinct documents + templated rephrasings.
    closest_sources: list[RetrievedChunk] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    mentions_chemicals: bool = False
    error: str | None = None  # technical detail for logs / eval, never shown to users

    @property
    def refused(self) -> bool:
        return self.status == "no_info"

    @property
    def retrieved(self) -> list[RetrievedChunk]:
        """Every chunk retrieved for this question (above and below the threshold)."""
        return self.sources + self.near_misses

    def to_dict(self) -> dict[str, Any]:
        """Structured, JSON-friendly view (same shape for both no-info layers)."""

        def brief(r: RetrievedChunk) -> dict[str, Any]:
            c = r.chunk
            return {"title": c.doc_title, "section": c.section, "source_file": c.source_file, "score": round(r.score, 4)}

        return {
            "status": self.status,
            "answer": self.answer,
            "language": self.language,
            "rewritten_query": self.rewritten_query,
            "sources": [brief(r) for r in self.sources],
            "closest_sources": [brief(r) for r in self.closest_sources],
            "suggestions": self.suggestions,
            "mentions_chemicals": self.mentions_chemicals,
        }


def refusal_message(language: str) -> str:
    """Headline of the no-answer card for ``language``."""
    return REFUSAL_TH if language == "th" else REFUSAL_EN


def is_no_info(answer: str) -> bool:
    """True if the LLM signalled "no information" (sentinel, or a bare refusal sentence)."""
    text = answer.strip().strip('"').strip()
    if NO_INFO_SENTINEL in text and len(text.replace(NO_INFO_SENTINEL, "").strip()) <= 40:
        return True
    legacy = ("ไม่พบข้อมูลในเอกสารที่มี", "I couldn't find this in the available documents", REFUSAL_EN)
    return any(r.lower() in text.lower() and len(text) <= len(r) + 40 for r in legacy)


# Backwards-compatible alias.
is_refusal = is_no_info


def build_index(embedder: SentenceTransformer, data_dir: Path = DATA_DIR) -> VectorIndex:
    """Load → clean → chunk → embed the whole knowledge base into a FAISS index."""
    docs = load_documents(data_dir)
    chunks = chunk_documents(docs)
    logger.warning("Building FAISS index: %d documents, %d chunks", len(docs), len(chunks))
    return VectorIndex(embedder, chunks, documents=docs)


class RAGPipeline:
    """Retrieval-augmented question answering over a ``VectorIndex``."""

    def __init__(
        self,
        index: VectorIndex,
        llm: ChatModel | None,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> None:
        self.index = index
        self.llm = llm
        self.threshold = threshold
        self.topics: list[Topic] = build_topics(index.documents)

    @property
    def main_topics(self) -> list[Topic]:
        """One representative topic per category (shown when nothing is close)."""
        return main_topics(self.topics)

    # -- steps ---------------------------------------------------------------
    def rewrite_query(self, question: str, history: list[dict[str, str]] | None) -> str:
        """Turn a follow-up into a standalone query using recent history.

        Falls back to the original question when there is no history, no LLM,
        or the rewrite call fails — retrieval must never be blocked by it.
        """
        if not history or self.llm is None:
            return question
        recent = history[-2 * REWRITE_HISTORY_TURNS :]
        try:
            rewritten = self.llm.chat(build_rewrite_messages(question, recent), temperature=0.0, max_tokens=512)
        except LLMError:
            return question
        rewritten = rewritten.strip().strip('"').splitlines()[0].strip() if rewritten.strip() else ""
        return rewritten or question

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
        """Top-k chunks above the similarity threshold."""
        return self.index.search(query, top_k=top_k, threshold=self.threshold)

    def _closest(self, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Up to ``MAX_CLOSEST_TOPICS`` distinct documents that are reasonably close."""
        seen: set[str] = set()
        closest: list[RetrievedChunk] = []
        for r in candidates:
            if r.score < NEAR_MATCH_MIN_SCORE or r.chunk.source_file in seen:
                continue
            seen.add(r.chunk.source_file)
            closest.append(r)
            if len(closest) == MAX_CLOSEST_TOPICS:
                break
        return closest

    def _mark_no_info(self, result: RAGResult) -> RAGResult:
        """Fill the structured no-info fields (shared by both detection layers)."""
        result.status = "no_info"
        result.answer = refusal_message(result.language)
        result.cited_ranks = set()
        result.closest_sources = self._closest(result.retrieved)
        result.suggestions = make_suggestions(
            result.language, [r.chunk.source_file for r in result.closest_sources], self.topics
        )
        return result

    # -- main entry ----------------------------------------------------------
    def answer(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        top_k: int = DEFAULT_TOP_K,
        language: str | None = None,
        on_step: Callable[[str], None] | None = None,
    ) -> RAGResult:
        """Answer ``question`` (optionally a follow-up given ``history``) from the knowledge base.

        ``language`` forces the reply language (``"th"``/``"en"``); by default it
        follows the question. ``on_step`` receives ``"rewriting"``, ``"searching"``
        and ``"composing"`` so a UI can show progress.
        """
        notify = on_step or (lambda _step: None)
        language = language or detect_language(question)
        if history and self.llm is not None:
            notify("rewriting")
        query = self.rewrite_query(question, history)

        notify("searching")
        hits = self.index.search(query, top_k=top_k, threshold=0.0)
        result = RAGResult(
            question=question,
            answer="",
            language=language,
            rewritten_query=query,
            sources=[h for h in hits if h.score >= self.threshold],
            near_misses=[h for h in hits if h.score < self.threshold],
        )

        # Layer 1: best match below the threshold → no LLM call (saves quota, no hallucination).
        if not result.sources:
            return self._mark_no_info(result)

        if self.llm is None:
            result.status = "error"
            result.error = "LLM is not configured (missing GROQ_API_KEY)"
            result.answer = (
                "ระบบยังไม่พร้อมสร้างคำตอบในขณะนี้ แสดงเฉพาะเอกสารที่เกี่ยวข้องด้านล่าง"
                if language == "th"
                else "Answers are unavailable right now. The related documents are listed below."
            )
            return result

        notify("composing")
        try:
            answer = self.llm.chat(build_answer_messages(question, language, result.sources))
        except LLMError as exc:
            result.status = "error"
            result.error = exc.detail or exc.message_en
            result.answer = exc.message(language)
            return result

        # Layer 2: the model says the context lacks the answer.
        if is_no_info(answer):
            return self._mark_no_info(result)

        answer = _FULLWIDTH_CITATION.sub(r"[\1]", answer).replace(NO_INFO_SENTINEL, "").strip()
        result.answer = answer
        result.cited_ranks = {int(n) for n in _CITATION.findall(answer)} & {s.rank for s in result.sources}
        result.mentions_chemicals = mentions_chemicals(answer)
        return result
