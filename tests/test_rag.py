"""Offline tests for the RAG core (no API key needed — the LLM is faked).

Run with:  pytest -q
"""

from __future__ import annotations

import pytest
from pythainlp.tokenize import word_tokenize

from rag.chunker import _split_words, chunk_documents, split_sections
from rag.config import CHUNK_MAX_CHARS, REFUSAL_EN, REFUSAL_TH
from rag.index import load_embedder
from rag.llm import GroqChatModel, LLMError
from rag.loader import load_documents, parse_header
from rag.pipeline import RAGPipeline, build_index, is_refusal
from rag.textutils import clean_text, detect_language


# --------------------------------------------------------------------------
# Fakes & fixtures
# --------------------------------------------------------------------------
class FakeLLM:
    """Records calls and returns canned replies (rewrite call vs. answer call)."""

    model = "fake-model"

    def __init__(self, answer: str = "โรคไหม้เกิดจากเชื้อรา [1] ⚠️ อ่านฉลาก", rewrite: str = "", error: LLMError | None = None):
        self.answer = answer
        self.rewrite = rewrite
        self.error = error
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages, temperature=0.1, max_tokens=1024):  # noqa: D401 - protocol method
        self.calls.append(messages)
        if self.error:
            raise self.error
        is_rewrite = "Standalone query" in messages[-1]["content"]
        return self.rewrite if is_rewrite else self.answer


@pytest.fixture(scope="module")
def index():
    return build_index(load_embedder())


# --------------------------------------------------------------------------
# Loader & text cleaning
# --------------------------------------------------------------------------
def test_parse_header_extracts_metadata():
    raw = "title: Test Doc\nsource_url: https://example.org\nlanguage: en\n---\n# Test\nBody text"
    meta, body = parse_header(raw)
    assert meta["title"] == "Test Doc"
    assert meta["source_url"] == "https://example.org"
    assert body.strip().startswith("# Test")


def test_clean_text_strips_noise():
    raw = "**ตัวหนา**​  มี   [ลิงก์](https://x.y)\n\n\n\nบรรทัดใหม่"
    assert clean_text(raw) == "ตัวหนา มี ลิงก์\n\nบรรทัดใหม่"


def test_detect_language():
    assert detect_language("โรคไหม้ข้าวป้องกันอย่างไร") == "th"
    assert detect_language("How do I prevent rice blast?") == "en"


def test_knowledge_base_size_and_metadata():
    docs = load_documents()
    assert len(docs) >= 12, "assignment requires at least 10 files (we target 12+)"
    assert sum(len(d.text) for d in docs) >= 30_000
    for d in docs:
        assert d.title and d.source_url.startswith("http"), d.source_file
        assert d.language in {"th", "en"}


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
def test_split_sections_uses_headings():
    text = "# Title\nintro\n## อาการ\nใบเหลือง\n### ระยะกล้า\nกล้าตาย\n## การป้องกัน\nใช้พันธุ์ต้านทาน"
    names = [s.name for s in split_sections(text, "ข้อมูลทั่วไป")]
    assert names == ["ข้อมูลทั่วไป", "อาการ", "อาการ › ระยะกล้า", "การป้องกัน"]


def test_sub_sections_are_merged_and_labelled():
    from rag.loader import Document

    text = "## อาการ\n### ราก\nรากเน่าเปื่อย\n### ลำต้น\nเปลือกมีน้ำเยิ้ม\n## สาเหตุ\nเชื้อรา"
    doc = Document(text=text, source_file="x.md", title="T", source_url="https://e.org", language="th")
    from rag.chunker import chunk_document

    chunks = chunk_document(doc)
    assert [c.section for c in chunks] == ["อาการ › ราก, ลำต้น", "สาเหตุ"]
    assert chunks[0].text == "ราก: รากเน่าเปื่อย\nลำต้น: เปลือกมีน้ำเยิ้ม"


def test_chunks_respect_size_and_have_metadata():
    chunks = chunk_documents(load_documents())
    assert chunks
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk ids must be unique"
    for c in chunks:
        assert 0 < len(c.text) <= CHUNK_MAX_CHARS
        assert c.source_file and c.doc_title and c.section and c.source_url
        assert c.embedding_text().startswith("passage: ")


def test_thai_word_split_never_cuts_words():
    text = "เกษตรกรควรหมั่นสำรวจแปลงนาอย่างสม่ำเสมอเพื่อตรวจหาอาการของโรคไหม้และแมลงศัตรูข้าว" * 6
    pieces = _split_words(text, 100)
    assert all(len(p) <= 100 for p in pieces)
    assert "".join(pieces) == text
    # Every piece boundary must coincide with a newmm word boundary.
    boundaries, pos = {0}, 0
    for tok in word_tokenize(text, engine="newmm"):
        pos += len(tok)
        boundaries.add(pos)
    pos = 0
    for p in pieces:
        pos += len(p)
        assert pos in boundaries


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "query, expected_file",
    [
        ("โรคไหม้ข้าวมีอาการอย่างไร", "01_rice_blast.md"),
        ("เพลี้ยกระโดดสีน้ำตาลทำลายข้าวอย่างไร", "03_rice_brown_planthopper.md"),
        ("โรครากเน่าโคนเน่าทุเรียนเกิดจากเชื้ออะไร", "05_durian_phytophthora.md"),
        ("How do I test a tomato plant for bacterial wilt?", "10_tomato_bacterial_wilt_virus_whitefly.md"),
        ("What is an economic threshold in IPM?", "11_ipm_principles.md"),
    ],
)
def test_retrieval_finds_expected_file(index, query, expected_file):
    results = index.search(query, top_k=4)
    assert expected_file in {r.chunk.source_file for r in results}
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= s <= 1.0001 for s in scores)


# --------------------------------------------------------------------------
# Pipeline (fake LLM)
# --------------------------------------------------------------------------
def test_pipeline_answer_has_sources_and_citations(index):
    llm = FakeLLM()
    result = RAGPipeline(index, llm).answer("โรคไหม้ข้าวเกิดจากอะไร")
    assert result.sources, "every answer must come with sources"
    assert result.cited_ranks == {1}
    assert not result.refused and result.error is None
    system, user = llm.calls[-1]
    assert "ONLY" in system["content"] and "[1]" in user["content"]


def test_pipeline_rewrites_follow_up_with_history(index):
    llm = FakeLLM(rewrite="วิธีป้องกันโรคไหม้ข้าว")
    history = [
        {"role": "user", "content": "โรคไหม้ข้าวมีอาการอย่างไร"},
        {"role": "assistant", "content": "แผลรูปตาบนใบ [1]"},
    ]
    result = RAGPipeline(index, llm).answer("แล้วโรคนี้ป้องกันยังไง", history=history)
    assert result.rewritten_query == "วิธีป้องกันโรคไหม้ข้าว"
    assert len(llm.calls) == 2  # rewrite + answer
    assert any(s.chunk.source_file == "01_rice_blast.md" for s in result.sources)


def test_below_threshold_refuses_without_calling_llm(index):
    llm = FakeLLM()
    result = RAGPipeline(index, llm, threshold=0.999).answer("ราคาทุเรียนวันนี้เท่าไร")
    assert result.refused and result.answer == REFUSAL_TH
    assert not result.sources and result.near_misses
    assert llm.calls == []


def test_fullwidth_citations_are_normalised(index):
    llm = FakeLLM(answer="Rice blast is caused by a fungus【1】【2†L1-L3】.")
    result = RAGPipeline(index, llm).answer("What causes rice blast?")
    assert result.answer == "Rice blast is caused by a fungus[1][2]."
    assert result.cited_ranks == {1, 2}


def test_llm_refusal_is_normalised(index):
    llm = FakeLLM(answer=f'"{REFUSAL_EN}"')
    result = RAGPipeline(index, llm).answer("How do I control rice blast?")
    assert result.refused and result.answer == REFUSAL_EN


def test_llm_error_gives_friendly_message(index):
    llm = FakeLLM(error=LLMError("ข้อความไทย", "English message", "boom"))
    result = RAGPipeline(index, llm).answer("How do I control rice blast?")
    assert result.error == "boom" and result.answer == "English message"


def test_missing_llm_still_returns_sources(index):
    result = RAGPipeline(index, None).answer("โรคไหม้ข้าวป้องกันอย่างไร")
    assert result.error and result.sources


def test_missing_api_key_raises_friendly_error():
    with pytest.raises(LLMError) as exc:
        GroqChatModel(api_key="")
    assert "GROQ_API_KEY" in exc.value.message("en")


def test_is_refusal():
    assert is_refusal(REFUSAL_TH)
    assert is_refusal(REFUSAL_EN.rstrip("."))
    assert not is_refusal("โรคไหม้เกิดจากเชื้อรา [1]")
