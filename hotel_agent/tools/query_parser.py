from __future__ import annotations

import re

from hotel_agent.schemas import SearchQuery


_BUDGET_RE = re.compile(r"\bunder\s*\$?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
_IN_CITY_RE = re.compile(
    r"\bin\s+([A-Za-z][A-Za-z\s\-]+?)(?=\s+under\b|\s+with\b|\s+near\b|\s+for\b|\s*$)",
    re.IGNORECASE,
)


KNOWN_AMENITIES = {
    "free wifi": "Free Wifi",
    "wifi": "Free Wifi",
    "free parking": "Free parking",
    "parking": "Free parking",
    "pool": "Pool",
    "spa": "Spa",
    "gym": "Fitness center",
    "fitness": "Fitness center",
    "restaurant": "Restaurant",
    "breakfast": "Breakfast included",
    "airport shuttle": "Airport shuttle",
    "beach": "Beachfront",
    "beachfront": "Beachfront",
}


def parse_search_query(text: str) -> SearchQuery:
    raw = (text or "").strip()
    destination = None
    max_price = None

    m = _BUDGET_RE.search(raw)
    if m:
        try:
            max_price = float(m.group(1))
        except Exception:
            max_price = None

    m2 = _IN_CITY_RE.search(raw)
    if m2:
        destination = m2.group(1).strip()

    amenities = []
    low = raw.lower()
    for k, canonical in KNOWN_AMENITIES.items():
        if k in low:
            if canonical not in amenities:
                amenities.append(canonical)

    return SearchQuery(raw=raw, destination=destination, max_price=max_price, amenities=amenities)
