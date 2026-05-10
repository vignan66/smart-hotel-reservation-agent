from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hotel_agent.data.cards import hotel_to_card_text
from hotel_agent.schemas import HotelRecord


@dataclass
class D3BuildConfig:
    seed: int = 42
    n_queries: int = 200
    candidates_per_query: int = 15
    max_hotels_per_city: int = 5000


AMENITY_POOL = [
    "Free Wifi",
    "Free parking",
    "Pool",
    "Restaurant",
    "Spa",
    "Fitness center",
    "Breakfast included",
    "Airport shuttle",
    "Beachfront",
]


def _normalize(a: str) -> str:
    return a.strip().lower()


def build_d3_pairs(hotels: list[HotelRecord], cfg: D3BuildConfig) -> pd.DataFrame:
    random.seed(cfg.seed)

    # group by city for more realistic queries
    by_city: dict[str, list[HotelRecord]] = {}
    for h in hotels:
        by_city.setdefault(h.city_name, []).append(h)

    cities = [c for c, hs in by_city.items() if len(hs) >= 3]
    if not cities:
        # fallback: treat all as one city bucket
        cities = ["ANY"]
        by_city = {"ANY": hotels}

    rows = []

    for _ in range(cfg.n_queries):
        city = random.choice(cities)
        city_hotels = by_city[city][: cfg.max_hotels_per_city]

        # Query template selection
        q_type = random.choice(["amenity_budget", "rating", "budget"])

        amenity = None
        rating_thr = None
        budget = None

        if q_type == "amenity_budget":
            amenity = random.choice(AMENITY_POOL)
            # choose a budget based on available prices (if any)
            priced = [h.price for h in city_hotels if h.price is not None]
            if priced:
                p = random.choice(priced)
                budget = float(max(60, min(400, p)))
            else:
                budget = float(random.choice([100, 150, 200, 250]))
            query = f"hotels in {city} under ${int(budget)} with {amenity.lower()}"

        elif q_type == "rating":
            rating_thr = float(random.choice([4.0, 4.2, 4.5]))
            query = f"top rated hotels in {city} with rating above {rating_thr:.1f}"

        else:
            priced = [h.price for h in city_hotels if h.price is not None]
            if priced:
                p = random.choice(priced)
                budget = float(max(60, min(400, p)))
            else:
                budget = float(random.choice([100, 150, 200, 250]))
            query = f"budget hotels in {city} under ${int(budget)}"

        # Candidate set
        candidates = city_hotels.copy()
        random.shuffle(candidates)
        candidates = candidates[: cfg.candidates_per_query]
        if len(candidates) < cfg.candidates_per_query:
            # pad from global pool
            pool = hotels.copy()
            random.shuffle(pool)
            candidates.extend(pool[: cfg.candidates_per_query - len(candidates)])

        # Labels
        for h in candidates:
            label = 1
            if city != "ANY" and h.city_name != city:
                label = 0
            if budget is not None and h.price is not None and h.price > budget:
                label = 0
            if rating_thr is not None and h.rating < rating_thr:
                label = 0
            if amenity is not None:
                # amenity must be present (case-insensitive contains)
                has = any(_normalize(amenity) == _normalize(a) for a in h.amenities)
                if not has:
                    label = 0

            rows.append(
                {
                    "query": query,
                    "city": city,
                    "hotel_id": h.hotel_id,
                    "hotel_name": h.hotel_name,
                    "hotel_card": hotel_to_card_text(h),
                    "label_relevant": int(label),
                    "query_type": q_type,
                }
            )

    return pd.DataFrame(rows)


def save_pairs_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
