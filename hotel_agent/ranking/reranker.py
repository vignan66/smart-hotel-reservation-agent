from __future__ import annotations

from dataclasses import dataclass

from sentence_transformers import CrossEncoder


@dataclass
class CrossEncoderReranker:
    model_name_or_path: str
    model: CrossEncoder | None = None

    def load(self) -> "CrossEncoderReranker":
        if self.model is None:
            self.model = CrossEncoder(self.model_name_or_path)
        return self

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Return a score per document for a single query."""
        self.load()
        assert self.model is not None
        pairs = [[query, d] for d in documents]
        scores = self.model.predict(pairs)
        return [float(s) for s in scores]
