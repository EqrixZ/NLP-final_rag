"""Central configuration for the RAG pipeline.

Every tunable value lives here so the UI (``app.py``) and the CLI
(``evaluate.py``) always share exactly the same settings.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".md", ".txt")

# --- Chunking --------------------------------------------------------------
# Target ~500 characters: big enough for one complete idea (e.g. a symptom
# description), small enough that retrieval stays precise.
CHUNK_TARGET_CHARS: int = 500
CHUNK_MAX_CHARS: int = 600
CHUNK_OVERLAP_CHARS: int = 80
# Chunks shorter than this are merged into a neighbour (avoids heading-only chunks).
CHUNK_MIN_CHARS: int = 80

# --- Embeddings & retrieval -----------------------------------------------
# Small multilingual model (~470 MB) with good Thai support. E5 models expect
# "query: " / "passage: " prefixes.
EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-small"
QUERY_PREFIX: str = "query: "
PASSAGE_PREFIX: str = "passage: "
EMBED_BATCH_SIZE: int = 32

# Chunks of one document score close together, so multi-part questions
# ("cause AND symptoms") need ~6 chunks to cover all relevant sections.
DEFAULT_TOP_K: int = 6
MAX_TOP_K: int = 10
# Cosine-similarity cut-off. E5 scores are compressed (roughly 0.70-0.95), so
# this is a coarse out-of-domain filter tuned on test/probe queries (see README):
# relevant questions scored >= 0.79 (cross-lingual) while off-topic ones scored
# <= 0.78. In-domain questions that the documents do not cover are refused by
# the grounded system prompt instead.
SIMILARITY_THRESHOLD: float = 0.78

# --- LLM (Groq) ------------------------------------------------------------
# Change the model in ONE place. Can be overridden with the GROQ_MODEL secret
# or environment variable.
# "openai/gpt-oss-120b" is a Groq production model with strong Thai support.
# (llama-3.3-70b-versatile is listed in Groq docs but returned model_not_found
# for our API key in Oct 2026 — check https://console.groq.com/docs/models.)
GROQ_MODEL: str = "openai/gpt-oss-120b"
LLM_TEMPERATURE: float = 0.1
# gpt-oss is a reasoning model: its hidden reasoning tokens count toward this limit.
LLM_MAX_TOKENS: int = 2048
LLM_TIMEOUT_SECONDS: float = 30.0
LLM_MAX_RETRIES: int = 2
# Number of previous chat turns (user+assistant pairs) given to the rewriter.
REWRITE_HISTORY_TURNS: int = 3

# --- Fixed refusal messages -----------------------------------------------
REFUSAL_TH: str = "ไม่พบข้อมูลในเอกสารที่มี"
REFUSAL_EN: str = "I couldn't find this in the available documents."
