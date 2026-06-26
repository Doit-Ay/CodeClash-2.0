"""
EVA Guardian — Detection UI
Handles image upload, batch processing, webcam, and missed-object correction.
Uses native Streamlit components.
"""
import io
import zipfile
from io import BytesIO
from typing import Any

import cv2
import numpy as np
import plotly.express as px
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from src.utils import resize_image
from src.ui.gallery import display_object_gallery
from src.ui.risk_report import generate_smart_risk_report
from src.feedback.handler import save_missed_object

# ---------- Public handlers ----------
def handle_live_demo(model: YOLO, confidence: float) -> None:
    """Top-level live-demo section with source toggle."""
    tab_img, tab_batch, tab_cam = st.tabs(["Single Image", "Batch Upload", "Webcam"])

    with tab_img:
        _handle_single_image(model, confidence)
    with tab_batch:
        _handle_batch_images(model, confidence)
    with tab_cam:
        _handle_webcam(model, confidence)


# ---------- Core Image Processing ----------
def _process_and_display_image(model: YOLO, confidence: float, source_file, source_name: str, file_id: str) -> None:
    """Processes a single image (from upload or camera) and displays results."""
    if st.session_state.get("current_file_id") != file_id:
        st.session_state.current_file_id = file_id
        st.session_state.reported_falses = set()

    image = Image.open(source_file).convert("RGB")
    image = resize_image(image)

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
            file_name=f"annotated_{source_name}",
            mime="image/png",
            key=f"dl_{file_id}"
        )

    st.divider()
    generate_smart_risk_report(results[0])
    st.divider()
    display_object_gallery(image, results[0], source_name)
    st.divider()
    _handle_missed_correction(model, image, source_file)


# ---------- Single image ----------
def _handle_single_image(model: YOLO, confidence: float) -> None:
    uploaded = st.file_uploader(
        "Upload an Image", type=["jpg", "jpeg", "png"],
        label_visibility="collapsed", key="single_file_uploader",
    )
    if uploaded is None:
        st.session_state.pop("current_file_id", None)
        st.session_state.pop("reported_falses", None)
        return

    file_id = f"upload_{uploaded.name}_{uploaded.size}"
    _process_and_display_image(model, confidence, uploaded, uploaded.name, file_id)


# ---------- Batch processing ----------
def _handle_batch_images(model: YOLO, confidence: float) -> None:
    uploaded_files = st.file_uploader(
        "Upload multiple images", type=["jpg", "jpeg", "png"],
        accept_multiple_files=True, label_visibility="collapsed", key="batch_uploader",
    )
    if not uploaded_files:
        st.info("Upload one or more images for batch analysis.")
        return

    st.write(f"**Processing {len(uploaded_files)} images...**")
    progress = st.progress(0)

    annotated_images = []
    all_counts = {}

    for idx, uf in enumerate(uploaded_files):
        img = Image.open(uf).convert("RGB")
        img = resize_image(img)
        res = model(img, conf=confidence, verbose=False)
        ann_bgr = res[0].plot()

        for box in res[0].boxes:
            cn = res[0].names[int(box.cls[0])]
            all_counts[cn] = all_counts.get(cn, 0) + 1

        annotated_images.append((uf.name, ann_bgr))
        progress.progress((idx + 1) / len(uploaded_files))

    progress.empty()

    # Display grid
    cols = st.columns(min(len(annotated_images), 4))
    for i, (name, ann_bgr) in enumerate(annotated_images):
        with cols[i % len(cols)]:
            rgb = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
            st.image(rgb, caption=name, use_container_width=True)

    # Summary
    if all_counts:
        st.write("**Batch Summary:**")
        summary_cols = st.columns(len(all_counts))
        for i, (cn, cnt) in enumerate(all_counts.items()):
            summary_cols[i].metric(cn, cnt)

    # Download ZIP
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        for name, ann_bgr in annotated_images:
            _, buf = cv2.imencode(".png", ann_bgr)
            zf.writestr(f"annotated_{name}", buf.tobytes())
    zip_buf.seek(0)
    st.download_button("Download All as ZIP", data=zip_buf, file_name="eva_batch_results.zip", mime="application/zip")


# ---------- Webcam ----------
def _handle_webcam(model: YOLO, confidence: float) -> None:
    st.info("Take a picture using your webcam for instant analysis.")
    camera_image = st.camera_input("Camera", label_visibility="collapsed")
    
    if camera_image is None:
        return
        
    file_id = f"camera_{camera_image.size}"
    _process_and_display_image(model, confidence, camera_image, "camera_capture.jpg", file_id)


# ---------- Missed object correction ----------
def _handle_missed_correction(model: YOLO, image: Image.Image, uploaded_file) -> None:
    st.subheader("Correct Missed Detections (False Negatives)")
    st.write("If the model missed an object, select its class, draw a box on the image below, and submit.")

    with st.form(key="missed_object_form"):
        class_names = list(model.names.values())
        selected_class = st.selectbox("1. Select the class of the missed object:", class_names)

        fig = px.imshow(image)
        fig.update_layout(
            dragmode="drawrect",
            newshape_line_color="red",
            title_text="2. Draw a box on the image",
            title_x=0.5,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0"),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        config = {"modeBarButtonsToAdd": ["drawrect", "eraseshape"]}
        st.plotly_chart(fig, use_container_width=True, config=config)

        submitted = st.form_submit_button("3. Submit Missed Object Feedback")
        if submitted:
            save_missed_object(uploaded_file, image, selected_class)
            st.success("Thank you! Your feedback (with simulated coordinates) has been recorded.")
