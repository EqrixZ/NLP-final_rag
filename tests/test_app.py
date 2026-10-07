"""Headless UI tests using Streamlit's AppTest with a faked Groq call (no network, no key)."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import streamlit as st

import rag.llm
import rag.pipeline
from rag.config import NO_INFO_SENTINEL, REFUSAL_TH

APP = str(Path(__file__).resolve().parent.parent / "app.py")
ANSWER = "**อาการ**\n- แผลรูปตาบนใบข้าว [1]\n- ใช้สารป้องกันกำจัดเชื้อราตามฉลาก [2]"


@pytest.fixture
def fake_groq(monkeypatch):
    replies = {"answer": ANSWER}

    def fake_chat(self, messages, temperature=0.1, max_tokens=1024):
        if "Standalone query" in messages[-1]["content"]:
            return "วิธีป้องกันโรคไหม้ข้าว"
        return replies["answer"]

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(rag.llm.GroqChatModel, "chat", fake_chat)
    return replies


def make_app(api_key: str) -> AppTest:
    """AppTest with injected secrets, so a developer's real .streamlit/secrets.toml is never used."""
    at = AppTest.from_file(APP, default_timeout=180)
    at.secrets["GROQ_API_KEY"] = api_key
    return at.run()


def all_markdown(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_empty_state_header_and_example_cards(fake_groq):
    at = make_app("test-placeholder")
    assert not at.exception
    text = all_markdown(at)
    assert "ผู้ช่วยด้านการเกษตรและโรคพืช" in text and "<style>" in text
    assert any(b.key == "exbtn_0" for b in at.button)


def test_example_card_submits_question(fake_groq):
    at = make_app("test-placeholder")
    next(b for b in at.button if b.key == "exbtn_0").click().run()
    assert not at.exception
    assert at.session_state.messages[0]["content"].startswith("โรคไหม้ข้าว")
    assert at.session_state.messages[1]["result"].status == "answered"


def test_answer_renders_citation_badges_sources_and_safety_note(fake_groq):
    at = make_app("test-placeholder")
    at.chat_input[0].set_value("โรคไหม้ข้าวมีอาการอย่างไร").run()
    assert not at.exception
    result = at.session_state.messages[-1]["result"]
    assert result.status == "answered" and result.sources and result.mentions_chemicals
    text = all_markdown(at)
    assert '<sup class="ag-cite">1</sup>' in text
    assert "ag-safety" in text
    assert any(e.label.startswith("แหล่งข้อมูล (") for e in at.expander)
    assert "ag-src" in text, "source cards render inside the expander"

    # Follow-up uses history -> rewritten query is stored.
    at.chat_input[0].set_value("แล้วโรคนี้ป้องกันยังไง").run()
    assert at.session_state.messages[-1]["result"].rewritten_query == "วิธีป้องกันโรคไหม้ข้าว"


def test_no_info_card_with_suggestions(fake_groq):
    fake_groq["answer"] = NO_INFO_SENTINEL
    at = make_app("test-placeholder")
    at.chat_input[0].set_value("โรคใบขาวอ้อยป้องกันอย่างไร").run()
    assert not at.exception
    result = at.session_state.messages[-1]["result"]
    assert result.status == "no_info"
    text = all_markdown(at)
    assert REFUSAL_TH in text and "ag-noinfo-head" in text
    assert NO_INFO_SENTINEL not in text, "the sentinel is never shown to users"
    assert "<sup class=\"ag-cite\">" not in text, "closest topics are never inline citations"
    assert any("ใกล้เคียงที่สุด" in e.label for e in at.expander)
    suggestion_buttons = [b for b in at.button if (b.key or "").startswith("sgbtn_")]
    assert len(suggestion_buttons) == len(result.suggestions) >= 2

    # Clicking a suggestion asks it.
    fake_groq["answer"] = ANSWER
    suggestion_buttons[0].click().run()
    assert at.session_state.messages[-2]["content"] == result.suggestions[0]


def test_app_without_key_shows_friendly_notice(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    at = make_app("")
    assert not at.exception
    assert "GROQ_API_KEY" in all_markdown(at)
    assert not at.error, "missing key is a calm notice, not a red error box"


def test_index_is_built_once_across_reruns(fake_groq, monkeypatch):
    st.cache_resource.clear()
    calls = []
    original = rag.pipeline.build_index

    def counting_build_index(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(rag.pipeline, "build_index", counting_build_index)
    at = make_app("test-placeholder")
    at.chat_input[0].set_value("โรคไหม้ข้าวมีอาการอย่างไร").run()
    at.chat_input[0].set_value("What is IPM?").run()
    assert not at.exception
    assert len(calls) == 1, "FAISS index must be built once and cached with st.cache_resource"
