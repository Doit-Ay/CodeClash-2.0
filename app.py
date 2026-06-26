"""
EVA Guardian — AI-Powered Safety Assistant for Space Stations
=============================================================
Slim entry point.  All logic lives in the ``src`` package.
"""
# Config must be imported first (sets env vars before OpenCV loads)
from src.config import FEEDBACK_IMAGE_DIR  # noqa: F401 — triggers env setup

import os
import streamlit as st

from src.model_manager import get_model_versions, load_model
from src.ui.styles import get_premium_css, render_header, render_history_counter
from src.ui.dashboard import handle_dashboard
from src.ui.about import handle_about_section


# ── Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="EVA Guardian",
    page_icon="EVA",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    # ── Inject Premium CSS ──────────────────────────────────────────
    st.markdown(get_premium_css(), unsafe_allow_html=True)

    # ── Animated Header ─────────────────────────────────────────────
    st.markdown(render_header(), unsafe_allow_html=True)

    # ── State ───────────────────────────────────────────────────────
    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"

    # ── Navigation Buttons ──────────────────────────────────────────
    nav1, nav2 = st.columns(2)
    with nav1:
        if st.button("Dashboard", key="nav_dashboard", use_container_width=True, type="primary" if st.session_state.page == "Dashboard" else "secondary"):
            st.session_state.page = "Dashboard"
    with nav2:
        if st.button("About", key="nav_about", use_container_width=True, type="primary" if st.session_state.page == "About" else "secondary"):
            st.session_state.page = "About"

    # ── Sidebar ─────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Configuration")
        available_models = get_model_versions()

        if not available_models:
            st.error("No trained models found in `runs/detect`. Train a model first.")
            st.stop()

        latest_name = list(available_models.keys())[0]

        if st.session_state.get("show_model_selector", False):
            selected_name = st.selectbox(
                "Model Version",
                options=list(available_models.keys()),
                key="model_select",
            )
        else:
            selected_name = latest_name

        st.info(f"**Active Model:** {selected_name}")

        if st.button("Select Other Model Versions"):
            st.session_state.show_model_selector = not st.session_state.get(
                "show_model_selector", False
            )
            st.rerun()

        run_path = available_models[selected_name]
        model_path = os.path.join(run_path, "weights", "best.pt")

        confidence = st.slider(
            "Confidence Threshold", 0.0, 1.0, 0.45, 0.05,
            help="Minimum probability for a detection to be valid.",
        )

        # Session history counter
        if "detection_history" in st.session_state and st.session_state.detection_history:
            st.markdown(
                render_history_counter(st.session_state.detection_history),
                unsafe_allow_html=True,
            )

    # ── Load Model ──────────────────────────────────────────────────
    model = load_model(model_path)
    if model is None:
        st.error("Failed to load the selected model.")
        st.stop()

    # ── Page Router ─────────────────────────────────────────────────
    if st.session_state.page == "Dashboard":
        handle_dashboard(model, confidence, selected_name, run_path)
    elif st.session_state.page == "About":
        handle_about_section()


if __name__ == "__main__":
    main()
