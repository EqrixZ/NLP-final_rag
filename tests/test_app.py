"""Headless UI test using Streamlit's AppTest with a faked Groq call (no network, no key)."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import streamlit as st

import rag.llm
import rag.pipeline

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def fake_groq(monkeypatch):
    def fake_chat(self, messages, temperature=0.1, max_tokens=1024):
        if "Standalone query" in messages[-1]["content"]:
            return "วิธีป้องกันโรคไหม้ข้าว"
        return "**อาการ**\n- แผลรูปตาบนใบข้าว [1]\n\n⚠️ อ่านฉลากและปฏิบัติตามคำแนะนำอย่างเคร่งครัด"

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(rag.llm.GroqChatModel, "chat", fake_chat)


def make_app(api_key: str) -> AppTest:
    """AppTest with injected secrets, so a developer's real .streamlit/secrets.toml is never used."""
    at = AppTest.from_file(APP, default_timeout=180)
    at.secrets["GROQ_API_KEY"] = api_key
    return at.run()


def test_app_answers_and_shows_sources(fake_groq):
    at = make_app("test-placeholder")
    assert not at.exception
    assert at.title[0].value.startswith("🌱")

    at.chat_input[0].set_value("โรคไหม้ข้าวมีอาการอย่างไร").run()
    assert not at.exception
    assert len(at.session_state.messages) == 2
    assistant = at.session_state.messages[-1]
    assert "[1]" in assistant["content"]
    assert assistant["sources"], "answer must carry retrieved sources"
    assert any("แหล่งข้อมูลที่ค้นพบ" in e.label for e in at.expander)

    # Follow-up uses history -> rewritten query is stored.
    at.chat_input[0].set_value("แล้วโรคนี้ป้องกันยังไง").run()
    assert at.session_state.messages[-1]["rewritten_query"] == "วิธีป้องกันโรคไหม้ข้าว"


def test_app_without_key_shows_friendly_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    at = make_app("")
    assert not at.exception
    assert any("GROQ_API_KEY" in e.value for e in at.error)


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
