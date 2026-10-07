"""Streamlit UI for the Agriculture & Plant Disease Assistant (Thai/English RAG chatbot).

Run locally with:  streamlit run app.py
All retrieval/generation logic lives in the ``rag`` package; this file is UI only.
Styling lives in ``styles.css`` (injected once) and ``.streamlit/config.toml``.
"""

from __future__ import annotations

import html
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import streamlit as st
from pythainlp.corpus import thai_stopwords
from pythainlp.tokenize import word_tokenize

from rag.config import DEFAULT_TOP_K, EMBEDDING_MODEL_NAME, GROQ_MODEL, MAX_TOP_K
from rag.index import RetrievedChunk, VectorIndex, load_embedder
from rag.llm import GroqChatModel, LLMError
from rag.pipeline import RAGPipeline, RAGResult, build_index
from rag.textutils import thai_ratio
from rag.topics import CATEGORY_LABELS

st.set_page_config(
    page_title="ผู้ช่วยด้านการเกษตรและโรคพืช",
    page_icon=":material/eco:",
    layout="centered",
    initial_sidebar_state="auto",
)

STYLES_PATH = Path(__file__).resolve().parent / "styles.css"

# (topic label key, question) — shown as cards before the first message.
EXAMPLES: list[tuple[str, str]] = [
    ("rice", "โรคไหม้ข้าวมีอาการอย่างไร และป้องกันได้อย่างไร"),
    ("rice", "เพลี้ยกระโดดสีน้ำตาลทำลายข้าวอย่างไร"),
    ("durian", "โรครากเน่าโคนเน่าทุเรียนเกิดจากเชื้ออะไร"),
    ("durian", "หลังเก็บเกี่ยวทุเรียนควรดูแลและใส่ปุ๋ยอย่างไร"),
    ("diagnosis", "How can I tell if my tomato plant has bacterial wilt?"),
    ("safety", "ระยะเก็บเกี่ยวปลอดภัย (PHI) คืออะไร"),
]
EXAMPLE_LABELS: dict[str, str] = {
    "rice": "ข้าว · Rice",
    "durian": "ทุเรียน · Durian",
    "diagnosis": "วินิจฉัยอาการ · Diagnosis",
    "safety": "ความปลอดภัย · Safety",
}

LANGUAGE_OPTIONS: dict[str, str] = {"auto": "อัตโนมัติ (ตามภาษาคำถาม)", "th": "ไทย", "en": "English"}

# Per-message UI strings (a message follows the language of its question).
TEXT: dict[str, dict[str, str]] = {
    "th": {
        "assistant": "ผู้ช่วยเกษตร",
        "query": "คำค้นที่ใช้",
        "sources": "แหล่งข้อมูล ({n})",
        "closest_matches": "เอกสารที่ใกล้เคียงที่สุด (ความเกี่ยวข้องต่ำ)",
        "cited": "อ้างอิงในคำตอบ",
        "similarity": "ความคล้าย",
        "noinfo_text": "ผู้ช่วยนี้ตอบจากคลังเอกสารเท่านั้น และจะไม่เดาคำตอบเมื่อไม่มีข้อมูลรองรับ",
        "closest": "หัวข้อใกล้เคียงที่มีในคลังเอกสาร",
        "covered": "หัวข้อหลักที่มีในคลังเอกสาร",
        "try": "ลองถามว่า",
        "noinfo_foot": "สำหรับเรื่องที่อยู่นอกคลังเอกสาร โปรดสอบถามสำนักงานเกษตรอำเภอใกล้บ้าน",
        "safety": "อ่านฉลากผลิตภัณฑ์ก่อนใช้ทุกครั้ง และปรึกษาเจ้าหน้าที่ส่งเสริมการเกษตรในพื้นที่",
        "error_hint": "หากยังพบปัญหา โปรดลองใหม่ภายหลัง",
        "no_retrieved": "ไม่มีเอกสารที่เกี่ยวข้อง",
    },
    "en": {
        "assistant": "Farm assistant",
        "query": "Search query",
        "sources": "Sources ({n})",
        "closest_matches": "Closest matches (low relevance)",
        "cited": "Cited",
        "similarity": "similarity",
        "noinfo_text": "Answers are limited to the knowledge base, and the assistant will not guess.",
        "closest": "Closest topics we do have",
        "covered": "Main topics we cover",
        "try": "Try asking",
        "noinfo_foot": "For topics outside these documents, contact your local agricultural extension office (สำนักงานเกษตรอำเภอ).",
        "safety": "Read the product label and consult your local agricultural officer.",
        "error_hint": "If the problem continues, please try again later.",
        "no_retrieved": "No related documents",
    },
}

STEP_LABELS: dict[str, str] = {
    "rewriting": "กำลังทำความเข้าใจคำถามต่อเนื่อง...",
    "searching": "กำลังค้นหาเอกสาร...",
    "composing": "กำลังเรียบเรียงคำตอบ...",
}

# Small inline SVG icons (no emoji anywhere in the UI).
ICON_LEAF = (
    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 19c0-8 5-13 14-14 0 9-5 14-13 14"/>'
    '<path d="M5 19c3-4 6-7 10-9"/></svg>'
)
ICON_LEAF_SMALL = ICON_LEAF.replace('width="22" height="22"', 'width="15" height="15"')
ICON_SEARCH = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
    'stroke-linecap="round" aria-hidden="true"><circle cx="11" cy="11" r="6.5"/><path d="m16 16 4.5 4.5"/>'
    '<path d="M8.5 11h5"/></svg>'
)
ICON_ALERT = (
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3 2.5 20h19L12 3z"/>'
    '<path d="M12 10v4"/><path d="M12 17.2v.1"/></svg>'
)

_CITATION = re.compile(r"\[(\d+)\]")
_EN_WORD = re.compile(r"[A-Za-z]{4,}")
_EN_STOP = {"what", "which", "when", "where", "does", "with", "that", "this", "from", "have", "could", "should", "about", "plant", "plants"}


# --------------------------------------------------------------------------
# Secrets & cached resources
# --------------------------------------------------------------------------
def get_secret(name: str) -> str | None:
    """Read a secret from ``st.secrets``; fall back to an environment variable for local dev."""
    try:
        value = st.secrets[name]
        if value:
            return str(value)
    except Exception:  # no secrets.toml, or key missing
        pass
    return os.environ.get(name) or None


@st.cache_resource(show_spinner="กำลังเตรียมคลังความรู้ (ครั้งแรกเท่านั้น)...")
def get_index() -> VectorIndex:
    """Load the embedding model and build the FAISS index ONCE per server process."""
    return build_index(load_embedder(EMBEDDING_MODEL_NAME))


@st.cache_resource(show_spinner=False)
def get_llm(api_key: str, model: str) -> GroqChatModel:
    """Create (and reuse) the Groq client for a given key/model."""
    return GroqChatModel(api_key=api_key, model=model)


@st.cache_data(show_spinner=False)
def load_css() -> str:
    return STYLES_PATH.read_text(encoding="utf-8") if STYLES_PATH.exists() else ""


@lru_cache(maxsize=1)
def _thai_stopwords() -> frozenset[str]:
    return frozenset(thai_stopwords())


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def esc(text: str) -> str:
    return html.escape(text, quote=True)


def answer_html(text: str) -> str:
    """Escape model output (it is never trusted as HTML) and turn [n] into citation badges."""
    safe = html.escape(text, quote=False)
    return _CITATION.sub(r'<sup class="ag-cite">\1</sup>', safe)


def query_terms(query: str) -> list[str]:
    """Words to highlight in snippets (Thai via newmm, English words of 4+ letters)."""
    terms: set[str] = set()
    if thai_ratio(query) > 0:
        stop = _thai_stopwords()
        for tok in word_tokenize(query, engine="newmm", keep_whitespace=False):
            if len(tok) >= 2 and tok not in stop and re.search(r"[฀-๿]", tok):
                terms.add(tok)
    terms.update(w.lower() for w in _EN_WORD.findall(query) if w.lower() not in _EN_STOP)
    return sorted(terms, key=len, reverse=True)


def snippet_html(text: str, terms: list[str], width: int = 230) -> str:
    """A short window of ``text`` around the first matched term, with matches highlighted."""
    flat = " ".join(text.split())
    lower = flat.lower()
    positions = [lower.find(t.lower()) for t in terms if t.lower() in lower]
    start = max(0, min(positions) - 60) if positions else 0
    piece = flat[start : start + width]
    prefix = "… " if start > 0 else ""
    suffix = " …" if start + width < len(flat) else ""
    safe = esc(piece)
    if terms:
        # English terms match whole words only ("wilt" must not mark "wilting"); Thai has no word spaces.
        parts = [rf"\b{re.escape(esc(t))}\b" if t.isascii() else re.escape(esc(t)) for t in terms]
        safe = re.compile("|".join(parts), re.IGNORECASE).sub(lambda m: f"<mark>{m.group(0)}</mark>", safe)
    return f"{prefix}{safe}{suffix}"


def relevance_width(score: float) -> int:
    """Map E5 cosine scores (compressed into ~0.70–0.95) onto a 0–100% bar."""
    return int(round(min(1.0, max(0.0, (score - 0.70) / 0.25)) * 100))


def html_block(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def ask(question: str) -> None:
    """Button callback: queue a question for the next run."""
    st.session_state.pending_question = question


def clear_conversation() -> None:
    st.session_state.messages = []
    st.session_state.pending_question = None


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def render_header() -> None:
    html_block(
        f"""<div class="ag-header">
  <div class="ag-mark">{ICON_LEAF}</div>
  <div>
    <div class="ag-title" role="heading" aria-level="1">ผู้ช่วยด้านการเกษตรและโรคพืช</div>
    <p class="ag-subtitle">Agriculture &amp; Plant Disease Assistant · ตอบจากคลังเอกสารวิชาการที่ตรวจสอบได้</p>
  </div>
</div>
<div class="ag-divider"></div>"""
    )


def render_empty_state() -> None:
    html_block(
        '<p class="ag-welcome">ถามเรื่องโรคพืช แมลงศัตรูพืช และการดูแลพืชได้ทั้งภาษาไทยและภาษาอังกฤษ '
        "ทุกคำตอบอ้างอิงจากเอกสารในคลังความรู้ พร้อมแสดงแหล่งที่มา เลือกคำถามตัวอย่างเพื่อเริ่มต้น</p>"
    )
    # One row per pair, so on phones (where columns stack) the topic order is kept.
    for row in range(0, len(EXAMPLES), 2):
        columns = st.columns(2, gap="small")
        for col, (i, (label_key, question)) in zip(columns, enumerate(EXAMPLES[row : row + 2], start=row)):
            with col, st.container(key=f"example_{i}"):
                html_block(f'<span class="ag-topic-label">{esc(EXAMPLE_LABELS[label_key])}</span>')
                st.button(question, key=f"exbtn_{i}", on_click=ask, args=(question,), type="tertiary", width="stretch")


def render_user(text: str) -> None:
    html_block(f'<div class="ag-user-row"><div class="ag-user">{esc(text)}</div></div>')


def render_source_cards(results: list[RetrievedChunk], query: str, cited: set[int], lang: str, low: bool) -> None:
    t = TEXT[lang]
    terms = query_terms(query)
    cards = []
    for i, r in enumerate(results, 1):
        c = r.chunk
        # Words from the document title match almost every chunk of it; highlighting them is noise.
        card_terms = [term for term in terms if term.lower() not in c.doc_title.lower()]
        title = esc(c.doc_title)
        if c.source_url.startswith(("http://", "https://")):
            title = f'<a class="ag-src-title" href="{esc(c.source_url)}" target="_blank" rel="noopener noreferrer">{title}</a>'
        else:
            title = f'<span class="ag-src-title">{title}</span>'
        badge_num = r.rank if not low else i
        cited_tag = f'<span class="ag-src-cited">{t["cited"]}</span>' if (not low and r.rank in cited) else ""
        bar_class = "ag-rel-bar ag-low" if low else "ag-rel-bar"
        cards.append(
            f"""<div class="ag-src">
  <div class="ag-src-head">
    <span class="ag-badge">{badge_num}</span>
    <div class="ag-src-titles">{title}<div class="ag-src-section">{esc(c.section)} · {esc(c.source_file)}</div></div>
    {cited_tag}
  </div>
  <div class="ag-rel"><div class="{bar_class}"><span style="width:{relevance_width(r.score)}%"></span></div>
    <span class="ag-rel-score">{t["similarity"]} {r.score:.3f}</span></div>
  <p class="ag-snippet">{snippet_html(c.text, card_terms)}</p>
</div>"""
        )
    html_block("\n".join(cards))


def render_sources(result: RAGResult, lang: str) -> None:
    """Collapsed expander with the chunks behind an answer (or the closest low-relevance matches)."""
    t = TEXT[lang]
    if result.status == "no_info":
        results, label, low = result.retrieved, t["closest_matches"], True
    else:
        results, label, low = result.sources, t["sources"].format(n=len(result.sources)), False
    with st.expander(label, icon=":material/menu_book:"):
        if result.rewritten_query and result.rewritten_query != result.question:
            html_block(f'<p class="ag-query">{t["query"]}: {esc(result.rewritten_query)}</p>')
        if not results:
            html_block(f'<p class="ag-query">{t["no_retrieved"]}</p>')
            return
        render_source_cards(results, result.rewritten_query, result.cited_ranks, lang, low)


def render_no_info(result: RAGResult, idx: int, pipeline: RAGPipeline) -> None:
    """Calm, helpful card shown for both no-info layers (retrieval threshold or [[NO_INFO]])."""
    lang = result.language
    t = TEXT[lang]
    with st.container(key=f"noinfo_{idx}"):
        html_block(
            f"""<div class="ag-noinfo-head">{ICON_SEARCH}<span class="ag-noinfo-title">{esc(result.answer)}</span></div>
<p class="ag-noinfo-text">{t["noinfo_text"]}</p>"""
        )
        if result.closest_sources:
            names = {tp.source_file: tp.name(lang) for tp in pipeline.topics}
            chips = "".join(
                f'<span class="ag-chip">{esc(names.get(r.chunk.source_file, r.chunk.doc_title))}'
                f' <span class="ag-chip-sub">› {esc(r.chunk.section.split(" › ")[0])}</span></span>'
                for r in result.closest_sources
            )
            html_block(f'<div class="ag-label">{t["closest"]}</div><div class="ag-chips">{chips}</div>')
        else:
            chips = "".join(f'<span class="ag-chip">{esc(tp.name(lang))}</span>' for tp in pipeline.main_topics)
            html_block(f'<div class="ag-label">{t["covered"]}</div><div class="ag-chips">{chips}</div>')
        if result.suggestions:
            html_block(f'<div class="ag-label">{t["try"]}</div>')
            for j, suggestion in enumerate(result.suggestions):
                with st.container(key=f"suggest_{idx}_{j}"):
                    st.button(suggestion, key=f"sgbtn_{idx}_{j}", on_click=ask, args=(suggestion,), type="tertiary")
        html_block(f'<div class="ag-noinfo-foot">{t["noinfo_foot"]}</div>')


def render_assistant(result: RAGResult, idx: int, pipeline: RAGPipeline) -> None:
    """One assistant turn: white card with mark, body by status, and sources."""
    lang = result.language
    t = TEXT[lang]
    with st.container(key=f"assistant_{idx}"):
        html_block(f'<div class="ag-msg-head"><span class="ag-avatar">{ICON_LEAF_SMALL}</span><span class="ag-who">{t["assistant"]}</span></div>')
        if result.status == "answered":
            html_block(answer_html(result.answer))
            if result.mentions_chemicals:
                html_block(f'<div class="ag-safety">{ICON_ALERT}<span>{t["safety"]}</span></div>')
        elif result.status == "no_info":
            render_no_info(result, idx, pipeline)
        else:
            html_block(f'<div class="ag-notice">{esc(result.answer)}<small>{t["error_hint"]}</small></div>')
        render_sources(result, lang)


def render_sidebar(index: VectorIndex, pipeline: RAGPipeline) -> tuple[int, str]:
    """Minimal sidebar: one-line about, settings, knowledge base summary, clear button.

    Longer lists (topics, examples) sit in collapsed expanders to keep it clean.
    """
    with st.sidebar:
        html_block(
            '<div class="ag-side-label ag-first">เกี่ยวกับ</div>'
            '<p class="ag-side-text">ตอบคำถามโรคพืชและการเพาะปลูกจากเอกสารวิชาการ พร้อมแหล่งอ้างอิง</p>'
        )

        html_block('<div class="ag-side-label">การตั้งค่า</div>')
        top_k = st.slider("จำนวนเอกสารอ้างอิง", min_value=1, max_value=MAX_TOP_K, value=DEFAULT_TOP_K)
        language = st.selectbox("ภาษาของคำตอบ", options=list(LANGUAGE_OPTIONS), format_func=LANGUAGE_OPTIONS.get)

        groups: dict[str, list] = {}
        for topic in pipeline.topics:
            groups.setdefault(topic.category, []).append(topic)
        html_block(
            '<div class="ag-side-label">คลังความรู้</div>'
            f'<p class="ag-side-text">{len(pipeline.topics)} เอกสาร ใน {len(groups)} หมวด</p>'
        )
        with st.expander("ดูหัวข้อทั้งหมด"):
            chunk_counts = Counter(c.source_file for c in index.chunks)
            parts = []
            for category, topics in groups.items():
                label = CATEGORY_LABELS.get(category, {"th": category})["th"]
                items = "".join(
                    f'<li>{esc(tp.topic_th)} <span class="ag-kb-n">{chunk_counts[tp.source_file]}</span></li>'
                    for tp in topics
                )
                parts.append(f'<div class="ag-kb-cat">{esc(label)}</div><ul class="ag-kb-list">{items}</ul>')
            html_block("".join(parts))
        with st.expander("ตัวอย่างคำถาม"):
            for i, (_, question) in enumerate(EXAMPLES[:4]):
                with st.container(key=f"sidebar_example_{i}"):
                    st.button(question, key=f"sbbtn_{i}", on_click=ask, args=(question,), type="tertiary")

        st.button("ล้างการสนทนา", icon=":material/delete_sweep:", on_click=clear_conversation, width="stretch")
        html_block('<p class="ag-side-small">เพื่อการศึกษา ไม่ใช้แทนคำแนะนำเจ้าหน้าที่เกษตร</p>')
    return top_k, language


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
def main() -> None:
    html_block(f"<style>{load_css()}</style>")

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending_question", None)

    model_name = get_secret("GROQ_MODEL") or GROQ_MODEL
    api_key = get_secret("GROQ_API_KEY")
    llm: GroqChatModel | None = None
    setup_notice = None
    if api_key:
        try:
            llm = get_llm(api_key, model_name)
        except LLMError as exc:
            setup_notice = exc.message("th")
    else:
        setup_notice = (
            "ขณะนี้ระบบยังไม่ได้เชื่อมต่อบริการสร้างคำตอบ จึงแสดงได้เฉพาะเอกสารที่เกี่ยวข้อง"
            "<small>สำหรับผู้ดูแลระบบ: ตั้งค่า GROQ_API_KEY ในไฟล์ .streamlit/secrets.toml "
            "หรือใน Settings → Secrets ของ Streamlit Community Cloud</small>"
        )

    try:
        index = get_index()
    except Exception:  # surface data/model loading problems without a stack trace
        render_header()
        html_block('<div class="ag-notice">โหลดคลังความรู้ไม่สำเร็จ โปรดรีเฟรชหน้าอีกครั้งในภายหลัง</div>')
        st.stop()

    pipeline = RAGPipeline(index, llm)
    top_k, language_choice = render_sidebar(index, pipeline)

    render_header()
    if setup_notice:
        html_block(f'<div class="ag-notice">{setup_notice}</div>')

    messages: list[dict] = st.session_state.messages
    pending = st.session_state.pending_question
    if not messages and not pending:
        render_empty_state()

    for idx, msg in enumerate(messages):
        if msg["role"] == "user":
            render_user(msg["content"])
        else:
            render_assistant(msg["result"], idx, pipeline)

    typed = st.chat_input("พิมพ์คำถาม · Ask in Thai or English")
    question = typed or pending
    st.session_state.pending_question = None
    if not question:
        return

    history = [{"role": m["role"], "content": m["content"]} for m in messages]
    messages.append({"role": "user", "content": question})
    render_user(question)

    placeholder = st.empty()
    with placeholder.status(STEP_LABELS["searching"], expanded=False) as status:
        result = pipeline.answer(
            question,
            history=history,
            top_k=top_k,
            language=None if language_choice == "auto" else language_choice,
            on_step=lambda step: status.update(label=STEP_LABELS.get(step, STEP_LABELS["searching"])),
        )
    placeholder.empty()

    messages.append({"role": "assistant", "content": result.answer, "result": result})
    render_assistant(result, len(messages) - 1, pipeline)


main()
