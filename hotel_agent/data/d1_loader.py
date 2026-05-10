from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from hotel_agent.schemas import HotelRecord


AMENITY_NOISE = {
    "reserve now",
    "visit website",
    "pay up to",
    "book now",
}


def _stable_hotel_id(hotel_name: str, city: str, country: str) -> str:
    raw = f"{hotel_name}|{city}|{country}".lower().encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:12]


def _extract_amenities(row: pd.Series) -> list[str]:
    vals = []
    for col in row.index:
        if not str(col).lower().startswith("info") and "info_" not in str(col).lower():
            continue
        v = row[col]
        if pd.isna(v):
            continue
        s = str(v).strip()
        if not s:
            continue
        low = s.lower()
        # remove promo/noise
        if any(n in low for n in AMENITY_NOISE):
            continue
        vals.append(s)
    return vals


def load_hotels_csv(path: Path) -> list[HotelRecord]:
    df = pd.read_csv(path)

    # Try to be resilient to slightly different column names
    col_map = {
        "hotel_name": None,
        "city_name": None,
        "country_name": None,
        "rating": None,
        "reviews_count": None,
        "price": None,
    }

    for c in df.columns:
        lc = c.lower()
        if lc in col_map:
            col_map[lc] = c
        elif lc == "reviews":
            col_map["reviews_count"] = c
        elif lc == "price" or lc == "price_usd":
            col_map["price"] = c

    missing = [k for k, v in col_map.items() if v is None and k not in {"price"}]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}. Found: {list(df.columns)}")

    hotels: list[HotelRecord] = []
    for _, r in df.iterrows():
        hotel_name = str(r[col_map["hotel_name"]]).strip()
        city = str(r[col_map["city_name"]]).strip()
        country = str(r[col_map["country_name"]]).strip()
        rating = float(r[col_map["rating"]])
        reviews = int(r[col_map["reviews_count"]])

        price = None
        if col_map.get("price") is not None:
            try:
                pv = r[col_map["price"]]
                if pd.notna(pv):
                    price = float(pv)
            except Exception:
                price = None

        amenities = _extract_amenities(r)
        hid = _stable_hotel_id(hotel_name, city, country)
        hotels.append(
            HotelRecord(
                hotel_id=hid,
                hotel_name=hotel_name,
                city_name=city,
                country_name=country,
                rating=rating,
                reviews_count=reviews,
                price=price,
                amenities=amenities,
            )
        )

    return hotels


def choose_default_d1_path(raw_path: Path, sample_path: Path) -> Path:
    """Prefer the full raw dataset if available; otherwise fall back to sample."""
    return raw_path if raw_path.exists() else sample_path
