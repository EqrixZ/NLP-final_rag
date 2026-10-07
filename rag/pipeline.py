"""End-to-end RAG pipeline shared by the Streamlit app and ``evaluate.py``.

Flow: detect language → rewrite follow-up (if history) → FAISS retrieval with
threshold → (no hits ⇒ fixed refusal, no LLM call) → grounded generation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from sentence_transformers import SentenceTransformer

from rag.chunker import chunk_documents
from rag.config import (
    DATA_DIR,
    DEFAULT_TOP_K,
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

logger = logging.getLogger(__name__)

_CITATION = re.compile(r"\[(\d+)\]")
# Some models write full-width brackets, optionally with line refs (【1】, 【1†L1-L3】); normalise to [1].
_FULLWIDTH_CITATION = re.compile(r"【\s*(\d+)(?:†[^】]*)?\s*】")


@dataclass
class RAGResult:
    """Everything the UI / evaluator needs about one answered question."""

    question: str
    answer: str
    language: str
    rewritten_query: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    cited_ranks: set[int] = field(default_factory=set)
    # Closest chunks that fell below the threshold (shown for transparency, never sent to the LLM).
    near_misses: list[RetrievedChunk] = field(default_factory=list)
    refused: bool = False
    error: str | None = None


def refusal_message(language: str) -> str:
    """The fixed refusal sentence for ``language``."""
    return REFUSAL_TH if language == "th" else REFUSAL_EN


def is_refusal(answer: str) -> bool:
    """True if ``answer`` is (essentially) the fixed refusal sentence."""
    text = answer.strip().strip('"').strip()
    for refusal in (REFUSAL_TH, REFUSAL_EN):
        if refusal.lower().rstrip(".") in text.lower() and len(text) <= len(refusal) + 40:
            return True
    return False


def build_index(embedder: SentenceTransformer, data_dir: Path = DATA_DIR) -> VectorIndex:
    """Load → clean → chunk → embed the whole knowledge base into a FAISS index."""
    docs = load_documents(data_dir)
    chunks = chunk_documents(docs)
    logger.warning("Building FAISS index: %d documents, %d chunks", len(docs), len(chunks))
    return VectorIndex(embedder, chunks)


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

    # -- main entry ----------------------------------------------------------
    def answer(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> RAGResult:
        """Answer ``question`` (optionally a follow-up given ``history``) from the knowledge base."""
        language = detect_language(question)
        query = self.rewrite_query(question, history)
        hits = self.index.search(query, top_k=top_k, threshold=0.0)
        sources = [h for h in hits if h.score >= self.threshold]
        result = RAGResult(
            question=question,
            answer="",
            language=language,
            rewritten_query=query,
            sources=sources,
            near_misses=[h for h in hits if h.score < self.threshold],
        )

        if not sources:
            # Nothing relevant enough: refuse deterministically without calling the LLM.
            result.answer = refusal_message(language)
            result.refused = True
            return result

        if self.llm is None:
            result.error = "LLM is not configured"
            result.answer = (
                "ยังไม่ได้ตั้งค่า GROQ_API_KEY จึงสร้างคำตอบไม่ได้ (แสดงเฉพาะเอกสารที่ค้นพบ)"
                if language == "th"
                else "GROQ_API_KEY is not configured, so no answer can be generated (showing retrieved sources only)."
            )
            return result

        try:
            answer = self.llm.chat(build_answer_messages(question, language, sources))
        except LLMError as exc:
            result.error = exc.detail or exc.message_en
            result.answer = exc.message(language)
            return result

        answer = _FULLWIDTH_CITATION.sub(r"[\1]", answer)
        result.answer = answer
        result.refused = is_refusal(answer)
        if result.refused:
            result.answer = refusal_message(language)
        else:
            valid = {s.rank for s in sources}
            result.cited_ranks = {int(n) for n in _CITATION.findall(answer)} & valid
        return result
