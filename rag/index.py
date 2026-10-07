"""Sentence-embedding + FAISS vector index.

Vectors are L2-normalised, so inner product (``IndexFlatIP``) equals cosine
similarity. The corpus is small (a few hundred chunks), so an exact flat index
is both the simplest and the fastest option.
"""

from __future__ import annotations

from dataclasses import dataclass

# NOTE: torch (via sentence_transformers) must be imported BEFORE faiss. Both ship
# their own OpenMP runtime and importing faiss first segfaults on macOS during encode().
from sentence_transformers import SentenceTransformer  # isort: skip

import faiss  # isort: skip
import numpy as np

from rag.chunker import Chunk
from rag.loader import Document
from rag.config import EMBED_BATCH_SIZE, EMBEDDING_MODEL_NAME, QUERY_PREFIX


@dataclass
class RetrievedChunk:
    """A chunk returned by a search, with its cosine score and 1-based rank."""

    chunk: Chunk
    score: float
    rank: int


def load_embedder(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """Load the sentence-transformers model on CPU (downloaded once, then cached by HF)."""
    return SentenceTransformer(model_name, device="cpu")


class VectorIndex:
    """FAISS inner-product index over chunk embeddings."""

    def __init__(
        self,
        embedder: SentenceTransformer,
        chunks: list[Chunk],
        documents: list[Document] | None = None,
    ) -> None:
        if not chunks:
            raise ValueError("Cannot build an index with no chunks.")
        self.embedder = embedder
        self.chunks = chunks
        self.documents = documents or []  # source documents (topic catalog, sidebar)
        vectors = self._encode([c.embedding_text() for c in chunks])
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.embedder.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def search(self, query: str, top_k: int, threshold: float = 0.0) -> list[RetrievedChunk]:
        """Return up to ``top_k`` chunks whose cosine similarity is ``>= threshold``."""
        top_k = max(1, min(top_k, len(self.chunks)))
        query_vec = self._encode([f"{QUERY_PREFIX}{query}"])
        scores, ids = self.index.search(query_vec, top_k)
        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or score < threshold:
                continue
            results.append(RetrievedChunk(self.chunks[idx], float(score), len(results) + 1))
        return results

    def __len__(self) -> int:
        return len(self.chunks)
