"""Streamlit UI for the Agriculture & Plant Disease Expert Assistant (Thai/English RAG chatbot).

Run locally with:  streamlit run app.py
All retrieval/generation logic lives in the ``rag`` package; this file is UI only.
"""

from __future__ import annotations

import os

import streamlit as st

from rag.config import (
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    GROQ_MODEL,
    MAX_TOP_K,
    SIMILARITY_THRESHOLD,
)
from rag.index import RetrievedChunk, VectorIndex, load_embedder
from rag.llm import GroqChatModel, LLMError
from rag.pipeline import RAGPipeline, build_index

st.set_page_config(
    page_title="ผู้ช่วยผู้เชี่ยวชาญด้านการเกษตรและโรคพืช",
    page_icon="🌾",
    layout="wide",
)

EXAMPLE_QUESTIONS: list[str] = [
    "โรคไหม้ข้าวมีอาการอย่างไร และป้องกันได้อย่างไร",
    "เพลี้ยกระโดดสีน้ำตาลทำลายข้าวอย่างไร",
    "โรครากเน่าโคนเน่าทุเรียนเกิดจากเชื้ออะไร",
    "ใช้เชื้อราไตรโคเดอร์มาอย่างไร",
    "How can I tell if my tomato plant has bacterial wilt?",
    "What are the main principles of IPM?",
]


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


@st.cache_resource(show_spinner="กำลังโหลดโมเดล embedding และสร้างดัชนี FAISS (ครั้งแรกเท่านั้น)...")
def get_index() -> VectorIndex:
    """Load the embedding model and build the FAISS index ONCE per server process."""
    return build_index(load_embedder(EMBEDDING_MODEL_NAME))


@st.cache_resource(show_spinner=False)
def get_llm(api_key: str, model: str) -> GroqChatModel:
    """Create (and reuse) the Groq client for a given key/model."""
    return GroqChatModel(api_key=api_key, model=model)


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------
def render_sources(message: dict) -> None:
    """Show the chunks retrieved for an assistant message inside an expander."""
    sources: list[RetrievedChunk] = message.get("sources", [])
    near_misses: list[RetrievedChunk] = message.get("near_misses", [])
    cited: set[int] = message.get("cited", set())
    rewritten = message.get("rewritten_query")

    label = f"📚 แหล่งข้อมูลที่ค้นพบ ({len(sources)})"
    with st.expander(label, expanded=False):
        if rewritten and rewritten != message.get("question"):
            st.caption(f"🔎 คำค้นที่ใช้หลังเขียนคำถามใหม่ (query rewriting): **{rewritten}**")
        if not sources:
            st.info(
                f"ไม่พบเอกสารที่มีความคล้าย (cosine similarity) ≥ {SIMILARITY_THRESHOLD:.2f} "
                "จึงไม่ได้ส่งคำถามให้ LLM และตอบว่าไม่พบข้อมูล"
            )
        for src in sources:
            chunk = src.chunk
            badge = " ✅ อ้างอิงในคำตอบ" if src.rank in cited else ""
            with st.container(border=True):
                st.markdown(f"**[{src.rank}] {chunk.doc_title}** › {chunk.section}{badge}")
                url_part = f" · [ลิงก์แหล่งที่มา]({chunk.source_url})" if chunk.source_url else ""
                st.caption(f"ความคล้าย (similarity): {src.score:.3f} · ไฟล์: `{chunk.source_file}` · `{chunk.chunk_id}`{url_part}")
                st.markdown("\n".join(f"> {line}" for line in chunk.text.splitlines()))
        if near_misses:
            st.caption("เอกสารที่ใกล้เคียงแต่คะแนนต่ำกว่าเกณฑ์ (ไม่ได้ใช้ตอบ):")
            for src in near_misses:
                st.caption(f"· {src.chunk.doc_title} › {src.chunk.section} — {src.score:.3f}")


def render_message(message: dict) -> None:
    """Render one stored chat message (with sources for assistant turns)."""
    with st.chat_message(message["role"], avatar="🧑‍🌾" if message["role"] == "user" else "🌱"):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("show_sources", True):
            render_sources(message)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

with st.sidebar:
    st.header("🌾 เกี่ยวกับแอป")
    st.write(
        "แชตบอต **RAG** ตอบคำถามเรื่องการปลูกพืชและโรค/แมลงศัตรูพืชในประเทศไทย "
        "โดยอ้างอิงจากเอกสารในคลังความรู้เท่านั้น พร้อมแสดงแหล่งที่มาทุกคำตอบ "
        "ถามได้ทั้งภาษาไทยและภาษาอังกฤษ"
    )

    st.subheader("💡 ตัวอย่างคำถาม")
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        if st.button(q, key=f"example_{i}", width="stretch"):
            st.session_state.pending_question = q

    st.subheader("⚙️ การตั้งค่า")
    top_k = st.slider(
        "จำนวนเอกสารที่ค้นคืน (top-k)",
        min_value=1,
        max_value=MAX_TOP_K,
        value=DEFAULT_TOP_K,
        help="จำนวน chunk สูงสุดที่ส่งให้ LLM ใช้ตอบ (เฉพาะที่ผ่านเกณฑ์ความคล้าย)",
    )
    if st.button("🗑️ ล้างการสนทนา", width="stretch"):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    model_name = get_secret("GROQ_MODEL") or GROQ_MODEL
    st.caption(f"LLM: `{model_name}` (Groq) · Embedding: `{EMBEDDING_MODEL_NAME}` · threshold {SIMILARITY_THRESHOLD:.2f}")

    st.subheader("⚠️ ข้อจำกัดความรับผิดชอบ")
    st.caption(
        "ข้อมูลมีไว้เพื่อการศึกษาและเป็นแนวทางเบื้องต้นเท่านั้น ไม่สามารถใช้แทนคำแนะนำของ"
        "เจ้าหน้าที่ส่งเสริมการเกษตรหรือนักวิชาการโรคพืช ก่อนใช้สารเคมีทุกครั้งให้อ่านฉลากและปฏิบัติตามอย่างเคร่งครัด "
        "แอปนี้ไม่รองรับการวินิจฉัยจากรูปภาพ"
    )

# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------
st.title("🌱 ผู้ช่วยผู้เชี่ยวชาญด้านการเกษตรและโรคพืช")
st.caption("Agriculture & Plant Disease Expert Assistant — ถาม-ตอบจากคลังเอกสารด้วย RAG (FAISS + Groq)")

api_key = get_secret("GROQ_API_KEY")
llm: GroqChatModel | None = None
if not api_key:
    st.error(
        "ไม่พบ **GROQ_API_KEY** — แอปจะแสดงเฉพาะเอกสารที่ค้นพบ แต่สร้างคำตอบไม่ได้\n\n"
        "ตั้งค่าโดยสร้างไฟล์ `.streamlit/secrets.toml` (ดูตัวอย่างที่ `.streamlit/secrets.toml.example`) "
        "หรือเพิ่มใน **Settings → Secrets** บน Streamlit Community Cloud",
        icon="🔑",
    )
else:
    try:
        llm = get_llm(api_key, model_name)
    except LLMError as exc:
        st.error(exc.message("th"), icon="🔑")

try:
    index = get_index()
except Exception as exc:  # surface data/model loading problems in the UI instead of a stack trace
    st.error(f"โหลดคลังความรู้ไม่สำเร็จ: {exc}")
    st.stop()

pipeline = RAGPipeline(index, llm)

if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🌱"):
        st.markdown(
            "สวัสดีครับ 👋 ถามเรื่องโรคพืช แมลงศัตรูพืช หรือการดูแลพืช เช่น ข้าว ทุเรียน มะม่วง มันสำปะหลัง พริก มะเขือเทศ "
            "ได้เลยครับ — *Ask me in Thai or English.*"
        )

for msg in st.session_state.messages:
    render_message(msg)

typed = st.chat_input("พิมพ์คำถามเกี่ยวกับพืชหรือโรคพืช... / Ask a question...")
question = typed or st.session_state.pending_question
st.session_state.pending_question = None

if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    user_msg = {"role": "user", "content": question}
    st.session_state.messages.append(user_msg)
    render_message(user_msg)

    with st.chat_message("assistant", avatar="🌱"):
        with st.spinner("กำลังค้นหาเอกสารและเรียบเรียงคำตอบ..."):
            result = pipeline.answer(question, history=history, top_k=top_k)
        assistant_msg = {
            "role": "assistant",
            "content": result.answer,
            "question": question,
            "rewritten_query": result.rewritten_query,
            "sources": result.sources,
            "near_misses": result.near_misses,
            "cited": result.cited_ranks,
        }
        if result.error:
            st.warning(result.answer)
        else:
            st.markdown(result.answer)
        render_sources(assistant_msg)
    st.session_state.messages.append(assistant_msg)
