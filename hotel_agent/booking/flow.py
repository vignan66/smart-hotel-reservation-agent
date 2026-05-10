from __future__ import annotations

import re
import uuid

from hotel_agent.schemas import BookingState, BookingStep


_DATE_RANGE_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2})\s*(?:to|-)\s*(\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)


def system_prompt(state: BookingState) -> str:
    if state.canceled:
        return "This booking was canceled. If you want to start over, select a hotel again."

    if state.step == BookingStep.SELECT_HOTEL:
        return "Select a hotel from the search results to begin booking."

    if state.step == BookingStep.ROOM_TYPE:
        return "What room type would you like? (e.g., Standard, Deluxe, Suite)"

    if state.step == BookingStep.DATES:
        return "Please enter your stay dates as YYYY-MM-DD to YYYY-MM-DD (e.g., 2026-05-12 to 2026-05-14)."

    if state.step == BookingStep.GUESTS:
        return "How many guests? (1-10)"

    if state.step == BookingStep.CONFIRM:
        hotel = state.selected_hotel.hotel_name if state.selected_hotel else "(no hotel)"
        return (
            "Please confirm your booking details:\n"
            f"- Hotel: {hotel}\n"
            f"- Room: {state.room_type}\n"
            f"- Dates: {state.check_in} to {state.check_out}\n"
            f"- Guests: {state.guests}\n\n"
            "Type **confirm** to finalize or **cancel** to stop."
        )

    if state.step == BookingStep.COMPLETE:
        return f"Booking complete! Your confirmation ID is **{state.booking_id}**."

    return ""


def handle_user_message(state: BookingState, message: str) -> BookingState:
    text = (message or "").strip()
    low = text.lower()

    if low in {"cancel", "stop", "quit"}:
        state.canceled = True
        return state

    if state.step == BookingStep.ROOM_TYPE:
        if text:
            state.room_type = text
            state.step = BookingStep.DATES
        return state

    if state.step == BookingStep.DATES:
        m = _DATE_RANGE_RE.search(text)
        if m:
            state.check_in = m.group(1)
            state.check_out = m.group(2)
            state.step = BookingStep.GUESTS
        return state

    if state.step == BookingStep.GUESTS:
        try:
            n = int(re.findall(r"\d+", text)[0])
            if 1 <= n <= 10:
                state.guests = n
                state.step = BookingStep.CONFIRM
        except Exception:
            pass
        return state

    if state.step == BookingStep.CONFIRM:
        if low in {"confirm", "yes", "y"}:
            state.booking_id = str(uuid.uuid4())[:8]
            state.step = BookingStep.COMPLETE
        return state

    return state
