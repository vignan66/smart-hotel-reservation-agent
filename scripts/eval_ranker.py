from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from hotel_agent.config import get_settings
from hotel_agent.ranking.bm25_ranker import BM25Ranker
from hotel_agent.ranking.embedding_ranker import EmbeddingRanker
from hotel_agent.ranking.metrics import compute_metrics_over_queries
from hotel_agent.ranking.reranker import CrossEncoderReranker


def rank_labels(order: list[int], labels: list[int]) -> list[int]:
    return [labels[i] for i in order]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=str, required=True, help="CSV with columns: query, hotel_card, label_relevant")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    settings = get_settings()

    df = pd.read_csv(Path(args.pairs))
    required = {"query", "hotel_card", "label_relevant"}
    if not required.issubset(df.columns):
        raise ValueError(f"Pairs CSV missing columns {required}. Found: {list(df.columns)}")

    groups = defaultdict(list)
    for _, r in df.iterrows():
        groups[str(r["query"])].append((str(r["hotel_card"]), int(r["label_relevant"])))

    # Load models once
    embed = EmbeddingRanker(model_name=settings.e5_model_name).load()

    local_ft = Path("models/reranker")
    reranker_name = str(local_ft) if local_ft.exists() else settings.cross_encoder_name
    rerank = CrossEncoderReranker(model_name_or_path=reranker_name).load()

    rels_bm25 = []
    rels_embed = []
    rels_rerank = []

    for q, items in groups.items():
        docs = [d for d, _ in items]
        labels = [y for _, y in items]

        # BM25 within candidate set
        bm25 = BM25Ranker().fit(docs)
        bm_scores = bm25.rank(q, top_k=len(docs))
        bm_order = [i for i, _ in bm_scores]
        rels_bm25.append(rank_labels(bm_order, labels))

        # Embedding within candidate set
        embed.fit(docs)
        emb_scores = embed.rank(q, top_k=len(docs))
        emb_order = [i for i, _ in emb_scores]
        rels_embed.append(rank_labels(emb_order, labels))

        # Cross-encoder within candidate set
        ce_scores = rerank.rerank(q, docs)
        ce_order = sorted(range(len(docs)), key=lambda i: float(ce_scores[i]), reverse=True)
        rels_rerank.append(rank_labels(ce_order, labels))

    m_bm25 = compute_metrics_over_queries(rels_bm25, k=args.k)
    m_emb = compute_metrics_over_queries(rels_embed, k=args.k)
    m_ce = compute_metrics_over_queries(rels_rerank, k=args.k)

    print("\n=== Ranking Evaluation (candidate-set) ===")
    print(f"Queries: {len(groups)}")
    print(f"BM25       NDCG@{args.k}: {m_bm25.ndcg_10:.3f} | MRR: {m_bm25.mrr:.3f} | Recall@{args.k}: {m_bm25.recall_10:.3f}")
    print(f"E5 Embed   NDCG@{args.k}: {m_emb.ndcg_10:.3f} | MRR: {m_emb.mrr:.3f} | Recall@{args.k}: {m_emb.recall_10:.3f}")
    print(f"CrossEnc   NDCG@{args.k}: {m_ce.ndcg_10:.3f} | MRR: {m_ce.mrr:.3f} | Recall@{args.k}: {m_ce.recall_10:.3f}")


if __name__ == "__main__":
    main()
