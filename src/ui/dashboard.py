"""
EVA Guardian — Dashboard Page
Main dashboard with native Streamlit metrics, live demo, and performance tabs.
"""
import streamlit as st
from ultralytics import YOLO

from src.model_manager import load_metrics_from_run
from src.ui.detection import handle_live_demo
from src.ui.performance import handle_performance_analysis
from src.ui.demo import show_demo_gallery, handle_demo_result


def handle_dashboard(
    model: YOLO,
    confidence: float,
    selected_model_name: str,
    run_path: str,
) -> None:
    """Renders the main dashboard: metrics, live demo, and performance tabs."""

    # --- Native Streamlit Metric Cards ---
    metrics = load_metrics_from_run(run_path)
    if metrics:
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "mAP@0.5",
            f"{metrics['mAP50']:.3f}",
            help="Mean Average Precision at 50% IoU. Higher is better.",
        )
        c2.metric(
            "Precision",
            f"{metrics['Precision']:.3f}",
            help="Correct detections / All detections (TP / (TP + FP)).",
        )
        c3.metric(
            "Recall",
            f"{metrics['Recall']:.3f}",
            help="Found objects / All true objects (TP / (TP + FN)).",
        )
    else:
        st.warning("Could not load performance metrics for this model version.")

    st.divider()

    # --- Try Demo Button (prominent, above tabs) ---
    _, center_col, _ = st.columns([2, 1.5, 2])
    with center_col:
        if st.button(
            "Try Demo",
            key="try_demo_btn",
            use_container_width=True,
            type="primary",
            help="Select from pre-loaded test images to see EVA Guardian in action",
        ):
            show_demo_gallery()

    # --- Show Demo Result (if an image was selected) ---
    if st.session_state.get("demo_selected_path"):
        st.divider()
        handle_demo_result(model, confidence)
        st.divider()

    # --- Tab Navigation ---
    tab_demo, tab_perf = st.tabs(["Live Demo", "Performance Deep Dive"])

    with tab_demo:
        handle_live_demo(model, confidence)

    with tab_perf:
        handle_performance_analysis(run_path)
