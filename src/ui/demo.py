"""
EVA Guardian — Try Demo UI
Provides a gallery popup of pre-loaded test images from the local demo_images directory.
Users select an image to instantly see YOLO detection results.
"""
import os
from io import BytesIO
from typing import Optional

import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from src.config import DEMO_IMAGES_DIR
from src.utils import resize_image
from src.ui.gallery import display_object_gallery
from src.ui.risk_report import generate_smart_risk_report


# ── Image Loading ───────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def _load_demo_image(path: str) -> Optional[Image.Image]:
    """Fetches and returns a PIL Image from a local path, or None on failure."""
    if not path or not os.path.exists(path):
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


# ── Demo Gallery Dialog ─────────────────────────────────────────────


@st.dialog("Try Demo — Select a Test Image", width="large")
def show_demo_gallery() -> None:
    """
    Streamlit dialog that displays a grid of demo test images.
    When the user clicks one, it stores the selection in session state
    and closes the dialog.
    """
    st.write("Select any image below to run EVA Guardian's AI detection on it.")

    if not os.path.exists(DEMO_IMAGES_DIR):
        os.makedirs(DEMO_IMAGES_DIR, exist_ok=True)

    # Get available images from the directory
    valid_extensions = (".png", ".jpg", ".jpeg", ".webp")
    available_files = [
        f for f in os.listdir(DEMO_IMAGES_DIR)
        if f.lower().endswith(valid_extensions)
    ]
    
    # Sort files to maintain consistent order
    available_files.sort()

    if not available_files:
        st.warning(
            f"No demo images found. Please copy some test images into the "
            f"`{DEMO_IMAGES_DIR}` directory."
        )
        return

    # Load all thumbnails
    cols_per_row = 4
    for row_start in range(0, len(available_files), cols_per_row):
        row_files = available_files[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)

        for col_idx, filename in enumerate(row_files):
            filepath = os.path.join(DEMO_IMAGES_DIR, filename)
            with cols[col_idx]:
                with st.spinner("Loading..."):
                    pil_img = _load_demo_image(filepath)

                if pil_img is not None:
                    st.image(pil_img, use_container_width=True)
                    st.caption(f"**{filename}**")
                    if st.button(
                        f"Select",
                        key=f"demo_select_{row_start + col_idx}",
                        use_container_width=True,
                        type="primary",
                    ):
                        st.session_state.demo_selected_path = filepath
                        st.session_state.demo_selected_name = filename
                        st.rerun()
                else:
                    st.error(f"Failed to load: {filename}")


# ── Demo Result Display ─────────────────────────────────────────────


def handle_demo_result(model: YOLO, confidence: float) -> None:
    """
    If a demo image was selected, load it and run detection.
    Renders the full result (annotated image, risk report, gallery).
    """
    path = st.session_state.get("demo_selected_path")
    name = st.session_state.get("demo_selected_name", "Demo Image")

    if not path:
        return

    # Clear button to dismiss result and go back
    if st.button("Clear Demo Result", key="clear_demo"):
        st.session_state.pop("demo_selected_path", None)
        st.session_state.pop("demo_selected_name", None)
        st.rerun()
        return

    st.info(f"Showing detection results for: **{name}**")

    with st.spinner("Loading demo image..."):
        image = _load_demo_image(path)

    if image is None:
        st.error(
            f"Could not load the selected demo image from `{path}`. "
            "It may have been moved or deleted."
        )
        return

    image = resize_image(image)

    # --- Run Detection ---
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Original Image")
        st.image(image, use_container_width=True)

    with st.spinner("Analyzing image..."):
        results = model(image, conf=confidence, verbose=False)

    annotated_bgr = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    with col2:
        st.write("#### Detected Image")
        st.image(annotated_rgb, use_container_width=True)
        _, buf = cv2.imencode(".png", annotated_bgr)
        st.download_button(
            "Download Annotated Image",
            data=BytesIO(buf),
            file_name=f"annotated_demo_{name.replace(' ', '_')}.png",
            mime="image/png",
        )

    st.divider()
    generate_smart_risk_report(results[0])
    st.divider()
    display_object_gallery(image, results[0], f"demo_{name}")
