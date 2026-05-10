from __future__ import annotations

import hashlib
import random

from hotel_agent.schemas import HotelRecord, PriceOffer


BOOKING_SITES = [
    "Booking.com",
    "Agoda",
    "Expedia",
    "Hotels.com",
    "Trivago",
    "Priceline",
    "Trip.com",
    "Kayak",
]


def _seed_from_hotel(h: HotelRecord) -> int:
    digest = hashlib.md5(h.hotel_id.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def _base_price(h: HotelRecord) -> float:
    if h.price is not None and h.price > 0:
        return float(h.price)
    # fallback heuristic: higher rating -> higher typical price
    return 80 + (h.rating * 40)


def simulate_offers(h: HotelRecord, nights: int = 2) -> list[PriceOffer]:
    """Return deterministic mock offers across multiple booking sites.

    This is for the course prototype; in a production system this would be replaced
    by real APIs.
    """

    rnd = random.Random(_seed_from_hotel(h))
    base = _base_price(h)

    offers: list[PriceOffer] = []
    for site in BOOKING_SITES:
        # variation per site
        noise = rnd.uniform(-0.10, 0.12)
        nightly = max(40.0, base * (1.0 + noise))
        total = nightly * nights
        offers.append(
            PriceOffer(
                site=site,
                nightly_price=round(nightly, 2),
                total_price=round(total, 2),
                currency="USD",
                url=None,
            )
        )

    offers.sort(key=lambda o: o.total_price)
    return offers
