from __future__ import annotations

import math
from dataclasses import dataclass


def dcg_at_k(rels: list[int], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(rels[:k]):
        if rel <= 0:
            continue
        dcg += (2**rel - 1) / math.log2(i + 2)
    return dcg


def ndcg_at_k(rels: list[int], k: int) -> float:
    actual = dcg_at_k(rels, k)
    ideal = dcg_at_k(sorted(rels, reverse=True), k)
    if ideal == 0.0:
        return 0.0
    return actual / ideal


def mrr(rels: list[int]) -> float:
    for i, rel in enumerate(rels):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(rels: list[int], k: int) -> float:
    total_pos = sum(1 for r in rels if r > 0)
    if total_pos == 0:
        return 0.0
    pos_in_k = sum(1 for r in rels[:k] if r > 0)
    return pos_in_k / total_pos


@dataclass
class RankingMetrics:
    ndcg_10: float
    mrr: float
    recall_10: float


def compute_metrics_over_queries(rels_by_query: list[list[int]], k: int = 10) -> RankingMetrics:
    if not rels_by_query:
        return RankingMetrics(0.0, 0.0, 0.0)

    ndcgs = [ndcg_at_k(r, k) for r in rels_by_query]
    mrrs = [mrr(r) for r in rels_by_query]
    recs = [recall_at_k(r, k) for r in rels_by_query]

    return RankingMetrics(
        ndcg_10=sum(ndcgs) / len(ndcgs),
        mrr=sum(mrrs) / len(mrrs),
        recall_10=sum(recs) / len(recs),
    )
