from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


@dataclass
class BM25Ranker:
    bm25: BM25Okapi | None = None
    corpus_tokens: list[list[str]] | None = None

    def fit(self, documents: list[str]) -> "BM25Ranker":
        self.corpus_tokens = [_tokenize(d) for d in documents]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        return self

    def rank(self, query: str, top_k: int = 15) -> list[tuple[int, float]]:
        if self.bm25 is None:
            raise RuntimeError("BM25Ranker not fitted. Call fit() first.")
        q = _tokenize(query)
        scores = self.bm25.get_scores(q)
        ranked = sorted(enumerate(scores), key=lambda x: float(x[1]), reverse=True)
        return [(i, float(s)) for i, s in ranked[:top_k]]
