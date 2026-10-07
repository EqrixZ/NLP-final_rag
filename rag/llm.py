"""Thin wrapper around the Groq chat-completions API with friendly error messages.

The API key is passed in by the caller (``app.py`` reads it from
``st.secrets``, ``evaluate.py`` from the environment) — it is never stored in
the code.
"""

from __future__ import annotations

from typing import Protocol

import groq

from rag.config import (
    GROQ_MODEL,
    LLM_MAX_RETRIES,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
)


class LLMError(Exception):
    """Raised when the LLM call fails; carries user-friendly Thai and English messages."""

    def __init__(self, message_th: str, message_en: str, detail: str = "") -> None:
        super().__init__(detail or message_en)
        self.message_th = message_th
        self.message_en = message_en
        self.detail = detail

    def message(self, language: str) -> str:
        """Friendly message in the requested language (``"th"`` or ``"en"``)."""
        return self.message_th if language == "th" else self.message_en


class ChatModel(Protocol):
    """Anything that can turn chat messages into a reply (real Groq client or a test fake)."""

    model: str

    def chat(self, messages: list[dict[str, str]], temperature: float = ..., max_tokens: int = ...) -> str:
        ...


class GroqChatModel:
    """Groq chat model with timeout, retries and error translation."""

    def __init__(self, api_key: str, model: str = GROQ_MODEL) -> None:
        if not api_key:
            raise LLMError(
                "ยังไม่ได้ตั้งค่า GROQ_API_KEY",
                "GROQ_API_KEY is not configured.",
            )
        self.model = model
        self._client = groq.Groq(api_key=api_key, timeout=LLM_TIMEOUT_SECONDS, max_retries=LLM_MAX_RETRIES)

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ) -> str:
        """Send ``messages`` and return the assistant text, raising ``LLMError`` on failure."""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except groq.AuthenticationError as exc:
            raise LLMError(
                "API key ไม่ถูกต้อง กรุณาตรวจสอบ GROQ_API_KEY ใน Secrets",
                "Invalid API key. Please check GROQ_API_KEY in Secrets.",
                str(exc),
            ) from exc
        except groq.RateLimitError as exc:
            raise LLMError(
                "มีการใช้งานเกินโควตาชั่วคราว กรุณารอสักครู่แล้วลองใหม่",
                "Rate limit reached. Please wait a moment and try again.",
                str(exc),
            ) from exc
        except groq.APITimeoutError as exc:
            raise LLMError(
                "เซิร์ฟเวอร์ตอบช้าเกินไป (timeout) กรุณาลองใหม่อีกครั้ง",
                "The model took too long to respond (timeout). Please try again.",
                str(exc),
            ) from exc
        except groq.APIConnectionError as exc:
            raise LLMError(
                "ไม่สามารถเชื่อมต่อกับบริการ LLM ได้ กรุณาตรวจสอบอินเทอร์เน็ตแล้วลองใหม่",
                "Could not connect to the LLM service. Please check the connection and retry.",
                str(exc),
            ) from exc
        except groq.NotFoundError as exc:
            raise LLMError(
                f"ไม่พบโมเดล '{self.model}' กรุณาเปลี่ยนชื่อโมเดลใน rag/config.py",
                f"Model '{self.model}' was not found. Please update GROQ_MODEL in rag/config.py.",
                str(exc),
            ) from exc
        except groq.APIError as exc:
            raise LLMError(
                "เกิดข้อผิดพลาดจากบริการ LLM กรุณาลองใหม่อีกครั้ง",
                "The LLM service returned an error. Please try again.",
                str(exc),
            ) from exc

        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise LLMError("โมเดลไม่ได้ส่งคำตอบกลับมา กรุณาลองใหม่", "The model returned an empty reply. Please try again.")
        # Some models emit narrow no-break spaces (U+202F) around Latin words in Thai text.
        return content.replace("\u202f", " ").strip()
