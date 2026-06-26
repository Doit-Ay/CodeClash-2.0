"""
EVA Guardian — About Page
Project overview and features using native Streamlit expanders.
Matches the style of the original app.
"""
import streamlit as st


def handle_about_section() -> None:
    """Renders the About page using native Streamlit components."""
    st.header("What is EVA Guardian?")
    st.markdown("""
    EVA Guardian is an AI-powered safety assistant designed to operate in high-stakes environments like a space station.
    Its mission is to enhance astronaut safety by automatically identifying and assessing critical equipment in real-time.

    The core of the project is a **YOLOv8 object detection model** that we trained exclusively on a *synthetic dataset* provided by Duality AI. This proves that we can build reliable AI for places that are difficult or impossible
    to get real photos from. We didn't just build a model; we built a complete, interactive dashboard that turns AI detections
    into **actionable intelligence**.
    """)

    st.header("Key Features")

    with st.expander("Live Detection and Analysis", expanded=True):
        st.markdown("""
        - Users can upload their own images or use a live webcam feed.
        - The system runs our best-trained model to find and draw boxes around three key objects: **Fire Extinguishers**, **Toolboxes**, and **Oxygen Tanks**.
        """)

    with st.expander("Smart Risk Assessment", expanded=True):
        st.markdown("""
        This is our most advanced feature. The app doesn't just find objects; it prioritizes them by risk.
        It calculates a numerical **"Risk Score"** for each detected object based on a smart formula that considers:
        - **Urgency:** A Fire Extinguisher is treated as more critical than a Toolbox.
        - **Visibility:** An object that is hard to see (low confidence) is flagged as a higher risk.
        - **Proximity:** The size of the object's box is used to guess how close it is.

        It then presents a sorted list of risks and a simple, color-coded summary (e.g., `CRITICAL ALERT` in red), turning the AI into a true decision-support tool.
        """)

    with st.expander("Complete Human-in-the-Loop Feedback System", expanded=True):
        st.markdown("""
        We implemented two types of feedback to demonstrate how the model can be continuously improved.
        - **Detected Object Gallery:** Users can review a gallery of everything the model found. If the model makes a mistake (e.g., calls a random object a "Toolbox"), the user can click "Report Incorrect" to flag this *false positive*.
        - **Correct Missed Objects:** If the model misses an object entirely, the user can use an interactive chart to draw a new box on the image, providing the correct label for a *missed object* (false negative).
        """)

    with st.expander("Dynamic Performance Dashboard", expanded=True):
        st.markdown("""
        - The application automatically finds the latest and best-trained model from all our training runs.
        - It dynamically loads the performance metrics (**mAP, Precision, and Recall**) from the results file of that specific model and displays them.
        - The "Performance Deep Dive" tab provides a full suite of analysis graphs, including the **Confusion Matrix** and **loss curves**, for a complete technical overview.
        """)

    with st.expander("Try Demo — Instant Test Images", expanded=True):
        st.markdown("""
        - Click the **"Try Demo"** button on the Dashboard to open a gallery of pre-loaded test images.
        - These images are hosted on **OneDrive** and can be selected instantly — no upload required.
        - Select any image to see the full detection pipeline in action: annotated output, risk assessment, and object gallery.
        - Perfect for demonstrating the system's capabilities without needing access to the actual dataset.
        """)

    with st.expander("Batch Image Processing", expanded=True):
        st.markdown("""
        - Upload multiple images at once for batch analysis.
        - Each image is processed with a progress bar showing real-time status.
        - Results are displayed in a grid layout with detection summaries.
        - Download all annotated images as a single **ZIP file**.
        """)

