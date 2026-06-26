"""
EVA Guardian — Feedback Handler
Saves user feedback for false positives and missed detections.
"""
import json
import os
import time
from typing import Any

from PIL import Image

from src.config import INCORRECT_FEEDBACK_FILE, MISSED_FEEDBACK_FILE, FEEDBACK_IMAGE_DIR


def _load_feedback_list(filepath: str) -> list:
    """Safely loads a JSON list from *filepath*, returning [] on any error."""
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def save_incorrect_detection(image_name: str, box: Any, class_name: str) -> None:
    """Records a false-positive report (user says the detection is wrong)."""
    all_fb = _load_feedback_list(INCORRECT_FEEDBACK_FILE)
    all_fb.append({
        "source_image": image_name,
        "incorrectly_detected_as": class_name,
        "bounding_box": box.xyxy[0].cpu().numpy().tolist(),
        "confidence": float(box.conf[0]),
        "feedback_type": "false_positive",
        "timestamp": time.time(),
    })
    with open(INCORRECT_FEEDBACK_FILE, "w") as f:
        json.dump(all_fb, f, indent=2)


def save_missed_object(
    uploaded_file: Any,
    image: Image.Image,
    class_name: str,
    box_coords: dict | None = None,
) -> None:
    """
    Records a missed-object report (false negative).

    *box_coords* should be ``{"x0": …, "y0": …, "x1": …, "y1": …}``
    when real coordinates are available, otherwise falls back to simulated.
    """
    all_fb = _load_feedback_list(MISSED_FEEDBACK_FILE)

    os.makedirs(FEEDBACK_IMAGE_DIR, exist_ok=True)
    unique_name = f"{os.path.splitext(uploaded_file.name)[0]}_{int(time.time())}.png"
    image.save(os.path.join(FEEDBACK_IMAGE_DIR, unique_name))

    if box_coords is None:
        # Simulated fallback when the annotation tool cannot capture coords
        box_coords = {"x0": 100, "y0": 100, "x1": 250, "y1": 250}

    all_fb.append({
        "className": class_name,
        "box_coordinates (simulated)": box_coords,
        "source_image": unique_name,
    })
    with open(MISSED_FEEDBACK_FILE, "w") as f:
        json.dump(all_fb, f, indent=2)
