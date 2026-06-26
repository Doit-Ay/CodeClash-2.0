"""
EVA Guardian — Object Gallery UI
Displays detected objects in a gallery with false-positive feedback.
Uses native Streamlit components.
"""
import time
from typing import Any

import streamlit as st
from PIL import Image

from src.config import CLASS_META
from src.utils import pad_image_to_square
from src.feedback.handler import save_incorrect_detection


def display_object_gallery(image: Image.Image, results: Any, image_name: str) -> None:
    """Renders detected objects in a gallery with feedback buttons."""
    st.subheader("Detected Object Gallery & Feedback")
    st.write("Review each detection. If a detection is incorrect, report it as a **False Positive**.")

    if not results.boxes:
        st.info("No objects were detected in this image.")
        return

    num_cols = 5
    cols = st.columns(num_cols)

    for i, box in enumerate(results.boxes):
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        if not (xyxy[0] < xyxy[2] and xyxy[1] < xyxy[3]):
            continue

        class_name = results.names[int(box.cls[0])]
        confidence = float(box.conf[0])
        meta = CLASS_META.get(class_name, {"icon": "📦", "label": class_name})

        cropped = image.crop(xyxy)
        padded = pad_image_to_square(cropped)

        with cols[i % num_cols]:
            st.image(padded, use_container_width=True)
            st.caption(f"**{meta['icon']} {meta.get('label', class_name)}** (Conf: {confidence:.2f})")

            is_reported = i in st.session_state.get("reported_falses", set())
            if st.button(
                "Reported ✓" if is_reported else "Report Incorrect",
                key=f"report_{image_name}_{i}",
                disabled=is_reported,
                use_container_width=True,
            ):
                save_incorrect_detection(image_name, box, class_name)
                st.session_state.reported_falses.add(i)
                st.toast(f"Reported '{class_name}' as incorrect. Thank you!")
                time.sleep(0.8)
                st.rerun()
