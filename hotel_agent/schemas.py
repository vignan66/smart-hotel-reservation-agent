from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class PriceOffer(BaseModel):
    site: str
    nightly_price: float
    total_price: float
    currency: str = "USD"
    url: Optional[str] = None


class HotelRecord(BaseModel):
    hotel_id: str
    hotel_name: str
    city_name: str
    country_name: str
    rating: float = Field(ge=0.0, le=5.0)
    reviews_count: int = Field(ge=0)
    price: Optional[float] = None
    amenities: list[str] = Field(default_factory=list)

    @field_validator("amenities")
    @classmethod
    def normalize_amenities(cls, v: list[str]) -> list[str]:
        clean = []
        for a in v:
            a2 = (a or "").strip()
            if not a2:
                continue
            clean.append(a2)
        # de-dup while preserving order
        seen = set()
        out = []
        for a in clean:
            key = a.lower()
            if key not in seen:
                out.append(a)
                seen.add(key)
        return out


class SearchQuery(BaseModel):
    raw: str
    destination: Optional[str] = None
    max_price: Optional[float] = None
    amenities: list[str] = Field(default_factory=list)


class BookingStep(str, Enum):
    SELECT_HOTEL = "select_hotel"
    ROOM_TYPE = "room_type"
    DATES = "dates"
    GUESTS = "guests"
    CONFIRM = "confirm"
    COMPLETE = "complete"


class BookingState(BaseModel):
    step: BookingStep = BookingStep.SELECT_HOTEL
    selected_hotel: Optional[HotelRecord] = None

    room_type: Optional[str] = None
    check_in: Optional[str] = None  # ISO date string for simplicity
    check_out: Optional[str] = None
    guests: Optional[int] = Field(default=None, ge=1, le=10)

    booking_id: Optional[str] = None
    canceled: bool = False

    def reset(self) -> "BookingState":
        return BookingState()


class RankerOutput(BaseModel):
    hotel: HotelRecord
    score: float
    offers: list[PriceOffer] = Field(default_factory=list)


class EvalRow(BaseModel):
    query: str
    hotel_card: str
    label_relevant: int
    meta: dict[str, Any] = Field(default_factory=dict)
