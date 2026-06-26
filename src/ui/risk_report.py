"""
EVA Guardian — Risk Report UI
Generates the smart risk assessment using native Streamlit components.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Any

from src.config import URGENCY_WEIGHTS, DEFAULT_URGENCY, CLASS_META


def generate_smart_risk_report(results: Any) -> None:
    """Analyzes detection results and renders a risk report."""
    st.subheader("Smart Risk Assessment")

    risk_data = []
    counts = {name: 0 for name in results.names.values()}

    if results.boxes:
        img_h, img_w = results.orig_shape
        img_area = img_h * img_w

        for box in results.boxes:
            class_name = results.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            counts[class_name] += 1

            xyxy = box.xyxy[0]
            box_area = float((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))
            proximity = box_area / img_area if img_area > 0 else 0.0
            urgency = URGENCY_WEIGHTS.get(class_name, DEFAULT_URGENCY)
            score = (urgency * (1 + proximity * 5)) / (confidence + 0.1)

            meta = CLASS_META.get(class_name, {"icon": "📦", "label": class_name})
            risk_data.append({
                "Object": f"{meta['icon']} {meta.get('label', class_name)}",
                "Risk Score": score,
                "Confidence": f"{confidence:.2f}",
                "Proximity": f"{proximity:.2%}",
                "class_name": class_name,
                "raw_confidence": confidence,
                "raw_score": score,
            })

    # --- Risk Table ---
    if risk_data:
        risk_data.sort(key=lambda r: r["raw_score"], reverse=True)

        # Display as a clean dataframe
        df = pd.DataFrame(risk_data)[["Object", "Risk Score", "Confidence", "Proximity"]]
        df["Risk Score"] = df["Risk Score"].apply(lambda x: f"{x:.3f}")
        st.write("**Risk-Prioritized List:**")
        st.dataframe(df, use_container_width=True, hide_index=True)

        # --- Plotly Radar Chart ---
        _render_risk_radar(risk_data)
    else:
        st.success("**All Clear:** No objects detected to assess.")

    st.divider()

    # --- Safety Status Summary ---
    st.write("**Safety Status Summary:**")

    if counts.get("FireExtinguisher", 0) == 0:
        st.error("**CRITICAL ALERT:** No Fire Extinguisher detected in the field of view.")
    else:
        st.success(f"**OK:** Found {counts['FireExtinguisher']} Fire Extinguisher(s).")

    if counts.get("ToolBox", 0) > 0:
        st.warning(f"**ACTION REQUIRED:** Found {counts['ToolBox']} ToolBox(es). Please verify stowage and security.")
    else:
        st.success("**OK:** No unsecured ToolBoxes detected.")

    st.info(f"**STATUS:** Found {counts.get('OxygenTank', 0)} Oxygen Tank(s).")

    # Update session history
    if "detection_history" not in st.session_state:
        st.session_state.detection_history = {n: 0 for n in results.names.values()}
    for k, v in counts.items():
        st.session_state.detection_history[k] = st.session_state.detection_history.get(k, 0) + v


def _render_risk_radar(risk_data: list) -> None:
    """Renders a Plotly radar chart summarizing risk dimensions."""
    # Aggregate by class
    agg = {}
    for item in risk_data:
        cn = item["class_name"]
        if cn not in agg:
            agg[cn] = {"score": 0, "count": 0, "conf_sum": 0}
        agg[cn]["score"] = max(agg[cn]["score"], item["raw_score"])
        agg[cn]["count"] += 1
        agg[cn]["conf_sum"] += item["raw_confidence"]

    categories = list(agg.keys())
    if len(categories) < 3:
        return  # radar needs at least 3 axes

    scores = [agg[c]["score"] for c in categories]
    confs = [agg[c]["conf_sum"] / agg[c]["count"] for c in categories]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores + [scores[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name="Risk Score",
        line_color="#ff4b4b",
        fillcolor="rgba(255,75,75,0.15)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=confs + [confs[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name="Avg Confidence",
        line_color="#4da6ff",
        fillcolor="rgba(77,166,255,0.1)",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, max(scores) * 1.2], gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        ),
        showlegend=True,
        legend=dict(font=dict(color="#9ca3af")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        margin=dict(l=60, r=60, t=30, b=30),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)
