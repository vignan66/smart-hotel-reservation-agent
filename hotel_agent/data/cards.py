from __future__ import annotations

from hotel_agent.schemas import HotelRecord


def hotel_to_card_text(h: HotelRecord) -> str:
    price_part = f"Price ${h.price:.0f}/night" if h.price is not None else "Price unknown"
    amenities = ", ".join(h.amenities[:12]) if h.amenities else "Unknown"
    return (
        f"{h.hotel_name}. Located in {h.city_name}, {h.country_name}. "
        f"Rating {h.rating:.1f} from {h.reviews_count} reviews. "
        f"Amenities: {amenities}. {price_part}."
    )
