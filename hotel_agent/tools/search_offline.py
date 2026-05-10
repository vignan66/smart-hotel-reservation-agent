from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz

from hotel_agent.schemas import HotelRecord, SearchQuery


@dataclass
class OfflineSearchConfig:
    max_pool: int = 5000


def _city_match(a: str, b: str) -> bool:
    a2, b2 = (a or "").strip().lower(), (b or "").strip().lower()
    if not a2 or not b2:
        return False
    if a2 == b2:
        return True
    return fuzz.ratio(a2, b2) >= 85


def filter_hotels(hotels: list[HotelRecord], q: SearchQuery, cfg: OfflineSearchConfig | None = None) -> list[HotelRecord]:
    cfg = cfg or OfflineSearchConfig()

    pool = hotels

    # Destination filter (soft)
    if q.destination:
        dest = q.destination
        matched = [h for h in hotels if _city_match(h.city_name, dest)]
        if matched:
            pool = matched

    # Budget filter (keep unknown prices too)
    if q.max_price is not None:
        budget = q.max_price
        pool = [h for h in pool if (h.price is None or h.price <= budget)]

    # If pool is huge, cap it (keeps app responsive)
    if len(pool) > cfg.max_pool:
        pool = pool[: cfg.max_pool]

    return pool
