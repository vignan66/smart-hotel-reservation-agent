from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norm


@dataclass
class EmbeddingRanker:
    model_name: str
    model: SentenceTransformer | None = None
    doc_embeddings: np.ndarray | None = None
    documents: list[str] | None = None

    def load(self) -> "EmbeddingRanker":
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        return self

    def fit(self, documents: list[str], batch_size: int = 64) -> "EmbeddingRanker":
        self.load()
        assert self.model is not None
        self.documents = documents
        passages = [f"passage: {d}" for d in documents]
        emb = self.model.encode(passages, batch_size=batch_size, show_progress_bar=False)
        emb = np.asarray(emb, dtype=np.float32)
        self.doc_embeddings = _l2_normalize(emb)
        return self

    def rank(self, query: str, top_k: int = 15) -> list[tuple[int, float]]:
        if self.doc_embeddings is None or self.documents is None:
            raise RuntimeError("EmbeddingRanker not fitted. Call fit() first.")
        assert self.model is not None
        q_emb = self.model.encode([f"query: {query}"], show_progress_bar=False)
        q_emb = np.asarray(q_emb, dtype=np.float32)
        q_emb = _l2_normalize(q_emb)
        scores = (self.doc_embeddings @ q_emb.T).reshape(-1)
        idx = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in idx]
