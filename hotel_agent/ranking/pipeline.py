from __future__ import annotations

from dataclasses import dataclass

from hotel_agent.config import Settings
from hotel_agent.data.cards import hotel_to_card_text
from hotel_agent.ranking.bm25_ranker import BM25Ranker
from hotel_agent.ranking.embedding_ranker import EmbeddingRanker
from hotel_agent.ranking.reranker import CrossEncoderReranker
from hotel_agent.schemas import HotelRecord, RankerOutput
from hotel_agent.tools.price_simulator import simulate_offers


@dataclass
class RankingPipeline:
    settings: Settings

    hotels: list[HotelRecord]
    docs: list[str]

    bm25: BM25Ranker
    embed: EmbeddingRanker
    reranker: CrossEncoderReranker

    @classmethod
    def build(cls, settings: Settings, hotels: list[HotelRecord]) -> "RankingPipeline":
        docs = [hotel_to_card_text(h) for h in hotels]
        bm25 = BM25Ranker().fit(docs)
        embed = EmbeddingRanker(model_name=settings.e5_model_name).fit(docs)

        # Prefer a fine-tuned reranker if present
        local_ft = "models/reranker"
        reranker_name = local_ft if __import__("pathlib").Path(local_ft).exists() else settings.cross_encoder_name
        reranker = CrossEncoderReranker(model_name_or_path=reranker_name)

        return cls(settings=settings, hotels=hotels, docs=docs, bm25=bm25, embed=embed, reranker=reranker)

    def rank(self, query: str, mode: str = "rerank", top_k: int | None = None) -> list[RankerOutput]:
        top_k = top_k or self.settings.top_k_results

        if mode == "bm25":
            ranked = self.bm25.rank(query, top_k=top_k)
            outputs = []
            for idx, score in ranked:
                h = self.hotels[idx]
                offers = simulate_offers(h)
                outputs.append(RankerOutput(hotel=h, score=score, offers=offers))
            return outputs

        if mode == "embed":
            ranked = self.embed.rank(query, top_k=top_k)
            outputs = []
            for idx, score in ranked:
                h = self.hotels[idx]
                offers = simulate_offers(h)
                outputs.append(RankerOutput(hotel=h, score=score, offers=offers))
            return outputs

        if mode == "rerank":
            # Stage 1: retrieve more candidates using embeddings
            stage1 = self.embed.rank(query, top_k=max(top_k * 3, 30))
            cand_idx = [i for i, _ in stage1]
            cand_docs = [self.docs[i] for i in cand_idx]

            # Stage 2: cross-encoder score
            ce_scores = self.reranker.rerank(query, cand_docs)
            reranked = sorted(zip(cand_idx, ce_scores), key=lambda x: float(x[1]), reverse=True)[:top_k]

            outputs = []
            for idx, score in reranked:
                h = self.hotels[idx]
                offers = simulate_offers(h)
                outputs.append(RankerOutput(hotel=h, score=float(score), offers=offers))
            return outputs

        raise ValueError(f"Unknown ranking mode: {mode}")
