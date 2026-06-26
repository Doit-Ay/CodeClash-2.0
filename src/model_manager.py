"""
EVA Guardian — Model Manager
Handles YOLO model discovery, loading, and performance-metric extraction.
"""
import os
from typing import Dict, Optional

import pandas as pd
import streamlit as st
from ultralytics import YOLO

from src.config import RUNS_DIR, WEIGHTS_FILE, RESULTS_CSV


@st.cache_data
def get_model_versions() -> Dict[str, str]:
    """
    Discovers all completed training runs under RUNS_DIR.
    Returns {display_name: run_directory_path}, sorted newest-first.
    """
    models: Dict[str, str] = {}
    if not os.path.exists(RUNS_DIR):
        return models

    all_dirs = [
        d
        for d in os.listdir(RUNS_DIR)
        if os.path.isdir(os.path.join(RUNS_DIR, d))
    ]
    train_dirs = [
        d
        for d in all_dirs
        if os.path.exists(os.path.join(RUNS_DIR, d, WEIGHTS_FILE))
    ]
    sorted_paths = sorted(
        [os.path.join(RUNS_DIR, d) for d in train_dirs],
        key=os.path.getmtime,
        reverse=True,
    )

    for i, path in enumerate(sorted_paths):
        version_name = f"Version {len(sorted_paths) - i}"
        if i == 0:
            version_name += " (Latest)"
        models[version_name] = path

    return models


@st.cache_data
def load_metrics_from_run(run_path: str) -> Optional[Dict[str, float]]:
    """Loads mAP50 / Precision / Recall from the last epoch of *results.csv*."""
    results_path = os.path.join(run_path, RESULTS_CSV)
    if not os.path.exists(results_path):
        return None
    try:
        df = pd.read_csv(results_path)
        df.columns = df.columns.str.strip()
        latest = df.iloc[-1]
        return {
            "mAP50": latest.get("metrics/mAP50(B)", 0),
            "Precision": latest.get("metrics/precision(B)", 0),
            "Recall": latest.get("metrics/recall(B)", 0),
        }
    except Exception as e:
        st.warning(f"Could not parse metrics file: {e}")
        return None


@st.cache_resource
def load_model(model_path: str) -> Optional[YOLO]:
    """
    Loads a YOLO model with st.cache_resource so it persists
    across script re-runs.
    """
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        return None
