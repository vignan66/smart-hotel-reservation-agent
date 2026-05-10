from __future__ import annotations

import streamlit as st

from hotel_agent.booking.flow import handle_user_message, system_prompt
from hotel_agent.config import get_settings
from hotel_agent.data.d1_loader import choose_default_d1_path, load_hotels_csv
from hotel_agent.ranking.pipeline import RankingPipeline
from hotel_agent.tools.query_parser import parse_search_query
from hotel_agent.schemas import BookingStep


st.set_page_config(page_title="Smart Hotel Reservation Agent", layout="wide")


@st.cache_data(show_spinner=False)
def load_hotels_cached(path_str: str):
    return load_hotels_csv(__import__("pathlib").Path(path_str))


@st.cache_resource(show_spinner=True)
def build_pipeline_cached(path_str: str):
    settings = get_settings()
    hotels = load_hotels_cached(path_str)
    return RankingPipeline.build(settings=settings, hotels=hotels)


def init_session_state():
    if "booking_state" not in st.session_state:
        from hotel_agent.schemas import BookingState

        st.session_state.booking_state = BookingState()

    if "chat" not in st.session_state:
        st.session_state.chat = []  # list of (role, message)


def render_sidebar():
    st.sidebar.title("Booking Status")
    bs = st.session_state.booking_state

    st.sidebar.write(f"**Step:** {bs.step.value}")
    if bs.selected_hotel:
        st.sidebar.write(f"**Hotel:** {bs.selected_hotel.hotel_name}")
    if bs.room_type:
        st.sidebar.write(f"**Room:** {bs.room_type}")
    if bs.check_in and bs.check_out:
        st.sidebar.write(f"**Dates:** {bs.check_in} → {bs.check_out}")
    if bs.guests:
        st.sidebar.write(f"**Guests:** {bs.guests}")

    if st.sidebar.button("Reset booking"):
        st.session_state.booking_state = bs.reset()
        st.session_state.chat = []
        st.rerun()


def render_search_tab(pipeline: RankingPipeline):
    st.subheader("Hotel Search")

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Describe what you want",
            value="2 nights in San Diego under $200 near the beach with parking",
        )
    with col2:
        mode = st.selectbox(
            "Ranking mode",
            options=["rerank", "embed", "bm25"],
            format_func=lambda x: {"bm25": "BM25 (baseline)", "embed": "E5 Embeddings", "rerank": "E5 + Cross-Encoder"}[x],
        )

    if st.button("Search"):
        q = parse_search_query(query)
        st.info(
            f"Parsed: destination={q.destination or '—'}, budget={q.max_price or '—'}, amenities={', '.join(q.amenities) or '—'}"
        )

        results = pipeline.rank(query=q.raw, mode=mode)

        # Optional post-filter: budget (keep unknowns but push them down)
        if q.max_price is not None:
            budget = q.max_price
            known_ok = [r for r in results if (r.hotel.price is not None and r.hotel.price <= budget)]
            unknown = [r for r in results if r.hotel.price is None]
            results = known_ok + unknown

        st.session_state.last_results = results

    results = st.session_state.get("last_results", [])
    if not results:
        st.caption("Run a search to see hotels.")
        return

    st.markdown("---")
    st.subheader("Top results")

    for i, r in enumerate(results, start=1):
        h = r.hotel
        best = r.offers[0] if r.offers else None
        header = f"{i}. {h.hotel_name} — {h.city_name} ({h.rating:.1f}⭐, {h.reviews_count} reviews)"
        with st.expander(header, expanded=(i <= 3)):
            left, right = st.columns([2, 1])
            with left:
                st.write(f"**Location:** {h.city_name}, {h.country_name}")
                st.write(f"**Amenities:** {', '.join(h.amenities) if h.amenities else 'Unknown'}")
                st.write(f"**Model score:** {r.score:.4f}")
            with right:
                if best:
                    st.write("**Best deal (simulated):**")
                    st.write(f"{best.site}: ${best.total_price:.2f} total")
                else:
                    st.write("No offers available")

                if st.button("Book this hotel", key=f"book_{h.hotel_id}"):
                    bs = st.session_state.booking_state
                    bs.selected_hotel = h
                    bs.step = BookingStep.ROOM_TYPE
                    st.session_state.booking_state = bs
                    st.session_state.chat = [("assistant", system_prompt(bs))]
                    st.success("Hotel selected! Switch to the Booking tab to continue.")


def render_booking_tab():
    st.subheader("Booking Assistant")

    bs = st.session_state.booking_state

    if not st.session_state.chat:
        st.session_state.chat = [("assistant", system_prompt(bs))]

    # Render chat
    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(msg)

    user_msg = st.chat_input("Type your response...")
    if user_msg:
        st.session_state.chat.append(("user", user_msg))
        bs = handle_user_message(bs, user_msg)
        st.session_state.booking_state = bs
        st.session_state.chat.append(("assistant", system_prompt(bs)))
        st.rerun()


def main():
    init_session_state()
    render_sidebar()

    settings = get_settings()
    d1_path = choose_default_d1_path(settings.d1_raw_path, settings.d1_sample_path)

    st.title("Smart Hotel Reservation Agent")
    st.caption("Offline demo by default; deep-learning ranking (E5 + cross-encoder) supported.")
    st.write(f"Using dataset: `{d1_path}`")

    pipeline = build_pipeline_cached(str(d1_path))

    tab1, tab2 = st.tabs(["Search", "Booking"])
    with tab1:
        render_search_tab(pipeline)
    with tab2:
        render_booking_tab()


if __name__ == "__main__":
    main()
