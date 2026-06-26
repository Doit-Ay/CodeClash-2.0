# EVA Guardian: AI-Powered Safety Assistant for Space Stations

Live - https://codeclash-20-7htdzqnw3juzr8xw3m9aqc.streamlit.app/
---

## 1. Project Description

**EVA Guardian** is an AI-powered safety monitoring system designed to enhance operational safety and situational awareness aboard space stations.

At its core is a **YOLOv8 object detection model**, trained on a fully synthetic dataset from Duality AI’s Falcon platform. The model detects three mission-critical assets:

* Fire Extinguishers
* Toolboxes
* Oxygen Tanks

This end-to-end solution includes a training pipeline, an active learning feedback loop, and a modular **Streamlit dashboard** for live image/webcam detection and smart safety reports. 

Our final model achieved an impressive **mAP@0.5 of 0.964**, proving the power of synthetic data for real-world applications where data is scarce or impossible to collect.

---

## 2. Technology Stack

* **Model:** YOLOv8 (`ultralytics`)
* **Frameworks:** PyTorch, OpenCV, NumPy, PIL
* **Frontend:** Streamlit, Plotly, Streamlit-WebRTC
* **Environment:** Python 3.10+
* **Data Platform:** Duality AI Falcon (Synthetic)

---

## 3. Key Features

* **High-Performance Detection:**
  YOLOv8 trained with robust augmentations. Final metrics show strong generalization to unseen data.

* **Interactive Streamlit Dashboard:**
  Upload single images, use the live webcam feed, or upload multiple images for batch processing. Automatically discovers and loads the best trained model.

* **Smart Risk Assessment:**
  Color-coded alerts based on object detection — tailored for space safety applications. It calculates a numerical Risk Score considering object urgency, proximity, and confidence, visualized in an interactive Plotly Radar Chart.

* **Batch Image Processing:**
  Process multiple images simultaneously with real-time progress tracking, grid visualization, and a one-click ZIP download of all annotated results.

* **Human-in-the-Loop Feedback System:**
  Users can report false positives from a detected object gallery, or draw bounding boxes over missed objects directly in the UI. A feedback processor script (`process_feedback.py`) converts these annotations into retrainable YOLO data.

* **Dynamic Training Metrics Viewer:**
  An expandable section to view confusion matrices, loss curves, and precision-recall charts — all loaded dynamically from the model's training run.

---

## 4. Final Model Performance

| Metric        | Score |
| ------------- | ----- |
| **mAP@0.5**   | 0.964 |
| **Precision** | 0.967 |
| **Recall**    | 0.935 |

*Metrics based on the latest validation runs on synthetic data with custom YOLO parameters.*

---

## 5. How to Run the Project

### Step 1: Setup & Installation

**Prerequisites:**
* Python 3.10 or higher
* NVIDIA GPU with CUDA (recommended for training/inference)

```bash
# Clone the repository
git clone https://github.com/your-username/eva-guardian.git
cd eva-guardian

# Create and activate a virtual environment (Conda recommended)
conda create -n eva_env python=3.10
conda activate eva_env

# Install dependencies
pip install -r requirements.txt
```

---

### Step 2: Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

The app auto-loads your latest YOLO model from the `runs/` directory and is accessible at `http://localhost:8501`.

---

### Step 3 (Optional): Re-Train with New Feedback

If you’ve collected new feedback using the app, process it using the feedback script to prepare the dataset:

```bash
python process_feedback.py
```

Then, you can re-train or resume training the model with the new data:

```bash
python train.py --resume
```

---

## 6. Project Architecture

The application has been heavily refactored for production-readiness, separating the UI from the business logic:

```text
.
├── src/                   # Main application package
│   ├── config.py          # Global constants, paths, and model metadata
│   ├── model_manager.py   # Model loading, discovery, and metric parsing
│   ├── utils.py           # Image processing and resizing utilities
│   ├── ui/                # Streamlit UI components
│   │   ├── dashboard.py   # Main dashboard orchestrator
│   │   ├── detection.py   # Image/Batch/Webcam handlers
│   │   ├── gallery.py     # Object gallery with false-positive feedback
│   │   ├── risk_report.py # Risk assessment logic and radar charts
│   │   ├── performance.py # Training metrics visualizer
│   │   └── styles.py      # Core styling and headers
│   └── feedback/          # Human-in-the-loop feedback processing
│       └── handler.py     # JSON and coordinate management for feedback
├── app.py                 # Streamlit entry point (slim router)
├── train.py               # Main YOLOv8 training script
├── predict.py             # CLI inference script
├── visualize.py           # CLI dataset visualizer
├── process_feedback.py    # Script to convert UI feedback to YOLO dataset formats
├── yolo_params.yaml       # YOLOv8 dataset configuration
├── requirements.txt       # Pinned dependencies
└── Readme.md              # This file
```

---

## 7. Future Enhancements

* **Automated Feedback Loop:**
  Auto-retraining using a background worker monitoring the `feedback.json` queue.
* **Advanced Synthetic Generation:**
  Use domain randomization via Duality Falcon to further improve sim-to-real generalization.
* **Containerization:**
  Package the application using Docker for one-click deployments.
