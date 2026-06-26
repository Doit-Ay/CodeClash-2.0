"""
EVA Guardian — Configuration & Constants
All application-wide constants, paths, and configuration values.
"""
import os

# --- Environment Fixes (set BEFORE any OpenCV import) ---
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Directory & File Paths ---
RUNS_DIR = os.path.join("runs", "detect")
WEIGHTS_FILE = os.path.join("weights", "best.pt")
RESULTS_CSV = "results.csv"

FEEDBACK_DIR = "feedback"
INCORRECT_FEEDBACK_FILE = "incorrect_feedback.json"
MISSED_FEEDBACK_FILE = "feedback.json"
FEEDBACK_IMAGE_DIR = os.path.join(FEEDBACK_DIR, "new_user_images")

# --- UI Constants ---
PAD_COLOR = (14, 17, 31)  # Dark space-blue padding for gallery images
MAX_IMAGE_SIZE = (1280, 720)

# --- Model & Risk Assessment ---
DEFAULT_URGENCY = 0.5
URGENCY_WEIGHTS = {
    "FireExtinguisher": 1.0,
    "OxygenTank": 0.8,
    "ToolBox": 0.6,
}

# Risk severity thresholds
RISK_CRITICAL = 1.5
RISK_WARNING = 0.8

# Class display metadata (icon, color)
CLASS_META = {
    "FireExtinguisher": {"icon": "", "color": "#ff4b4b", "label": "Fire Extinguisher"},
    "OxygenTank":       {"icon": "", "color": "#00d4ff", "label": "Oxygen Tank"},
    "ToolBox":          {"icon": "", "color": "#ffc107", "label": "Tool Box"},
}

# --- Demo Images (Local) ---
# Place your test images directly into this directory
DEMO_IMAGES_DIR = "demo_images"

# --- Startup ---
os.makedirs(FEEDBACK_IMAGE_DIR, exist_ok=True)
os.makedirs(DEMO_IMAGES_DIR, exist_ok=True)
