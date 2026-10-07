"""Offline tests for the RAG core (no API key needed — the LLM is faked).

Run with:  pytest -q
"""

from __future__ import annotations

import csv

import pytest
from pythainlp.tokenize import word_tokenize

from rag.chunker import _split_words, chunk_documents, split_sections
from rag.config import CHUNK_MAX_CHARS, NO_INFO_SENTINEL, PROJECT_ROOT, REFUSAL_EN, REFUSAL_TH
from rag.index import load_embedder
from rag.llm import GroqChatModel, LLMError
from rag.loader import load_documents, parse_header
from rag.pipeline import RAGPipeline, build_index, is_no_info
from rag.textutils import clean_text, detect_language
from rag.topics import build_topics, make_suggestions, mentions_chemicals


# --------------------------------------------------------------------------
# Fakes & fixtures
# --------------------------------------------------------------------------
class FakeLLM:
    """Records calls and returns canned replies (rewrite call vs. answer call)."""

    model = "fake-model"

    def __init__(self, answer: str = "โรคไหม้เกิดจากเชื้อรา [1]", rewrite: str = "", error: LLMError | None = None):
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
    assert result.status == "answered" and result.error is None
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


def test_layer1_below_threshold_is_no_info_without_llm_call(index):
    llm = FakeLLM()
    result = RAGPipeline(index, llm, threshold=0.999).answer("ราคาทุเรียนวันนี้เท่าไร")
    assert result.status == "no_info" and result.answer == REFUSAL_TH
    assert not result.sources and result.near_misses
    assert llm.calls == [], "layer 1 must not call the LLM"
    assert len(result.suggestions) == 3
    assert not result.cited_ranks


def test_layer2_sentinel_is_no_info(index):
    llm = FakeLLM(answer=NO_INFO_SENTINEL)
    result = RAGPipeline(index, llm).answer("How do I control coffee leaf rust?")
    assert result.status == "no_info" and result.answer == REFUSAL_EN
    assert result.sources, "layer 2 happens after retrieval found chunks"
    assert 1 <= len(result.closest_sources) <= 3
    assert len({r.chunk.source_file for r in result.closest_sources}) == len(result.closest_sources)
    assert result.suggestions and all(s.endswith("?") for s in result.suggestions)


def test_both_no_info_layers_share_the_same_structure(index):
    layer1 = RAGPipeline(index, FakeLLM(), threshold=0.999).answer("ราคาทุเรียนวันนี้")
    layer2 = RAGPipeline(index, FakeLLM(answer=NO_INFO_SENTINEL)).answer("ราคาทุเรียนวันนี้")
    d1, d2 = layer1.to_dict(), layer2.to_dict()
    assert d1.keys() == d2.keys()
    assert d1["status"] == d2["status"] == "no_info"
    assert {"closest_sources", "suggestions"} <= d1.keys()


def test_unanswerable_test_questions_yield_no_info(index):
    """Every answerable=False row of test_questions.csv ends in status == "no_info"
    (layer 1 or, with an LLM that follows the prompt, layer 2)."""
    with (PROJECT_ROOT / "test_questions.csv").open(encoding="utf-8-sig") as f:
        unanswerable = [r for r in csv.DictReader(f) if r["answerable"] == "False"]
    assert len(unanswerable) >= 3
    pipeline = RAGPipeline(index, FakeLLM(answer=NO_INFO_SENTINEL))
    for row in unanswerable:
        result = pipeline.answer(row["question"])
        assert result.status == "no_info", row["question"]
        assert result.answer == (REFUSAL_TH if row["language"] == "th" else REFUSAL_EN)
        assert result.suggestions


def test_fullwidth_citations_are_normalised(index):
    llm = FakeLLM(answer="Rice blast is caused by a fungus【1】【2†L1-L3】.")
    result = RAGPipeline(index, llm).answer("What causes rice blast?")
    assert result.answer == "Rice blast is caused by a fungus[1][2]."
    assert result.cited_ranks == {1, 2}


def test_legacy_refusal_sentence_is_treated_as_no_info(index):
    llm = FakeLLM(answer='"I couldn\'t find this in the available documents."')
    result = RAGPipeline(index, llm).answer("How do I control rice blast?")
    assert result.status == "no_info" and result.answer == REFUSAL_EN


def test_chemical_answers_are_flagged_for_safety_note(index):
    llm = FakeLLM(answer="พ่นสารป้องกันกำจัดเชื้อรา เช่น ไตรไซคลาโซล ตามฉลาก [1]")
    result = RAGPipeline(index, llm).answer("โรคไหม้ข้าวกำจัดอย่างไร")
    assert result.status == "answered" and result.mentions_chemicals
    plain = RAGPipeline(index, FakeLLM(answer="ไถตากดินและกำจัดวัชพืช [1]")).answer("โรคกาบใบแห้งป้องกันอย่างไร")
    assert not plain.mentions_chemicals


def test_language_hint_overrides_detection(index):
    llm = FakeLLM()
    result = RAGPipeline(index, llm).answer("โรคไหม้ข้าวเกิดจากอะไร", language="en")
    assert result.language == "en"
    assert "Answer in English" in llm.calls[-1][-1]["content"]


def test_suggestions_are_templated_from_topics():
    topics = build_topics(load_documents())
    th = make_suggestions("th", ["03_rice_brown_planthopper.md"], topics)
    assert th[0] == "เพลี้ยกระโดดสีน้ำตาลทำลายพืชอย่างไร" and len(th) == 3
    en = make_suggestions("en", [], topics)
    assert len(en) == len(set(en)) == 3
    assert mentions_chemicals("use a fungicide") and not mentions_chemicals("remove weeds")


def test_llm_error_gives_friendly_message(index):
    llm = FakeLLM(error=LLMError("ข้อความไทย", "English message", "boom"))
    result = RAGPipeline(index, llm).answer("How do I control rice blast?")
    assert result.status == "error"
    assert result.error == "boom" and result.answer == "English message"


def test_missing_llm_still_returns_sources(index):
    result = RAGPipeline(index, None).answer("โรคไหม้ข้าวป้องกันอย่างไร")
    assert result.status == "error" and result.error and result.sources
    assert "API" not in result.answer, "user-facing text stays non-technical"


def test_missing_api_key_raises_friendly_error():
    with pytest.raises(LLMError) as exc:
        GroqChatModel(api_key="")
    assert "GROQ_API_KEY" in exc.value.message("en")


def test_is_no_info():
    assert is_no_info(NO_INFO_SENTINEL)
    assert is_no_info(f" {NO_INFO_SENTINEL}\n")
    assert is_no_info(REFUSAL_TH)
    assert not is_no_info("โรคไหม้เกิดจากเชื้อรา [1]")
