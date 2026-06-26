"""
EVA Guardian — Performance Analysis Page
Displays training metrics and graphs using native Streamlit components.
"""
import os

import streamlit as st


def handle_performance_analysis(run_path: str) -> None:
    """Shows detailed performance graphs from a training run."""
    st.write("This dashboard provides a detailed breakdown of the selected model's performance on the validation dataset.")

    st.write("##### **Training Progress**")
    st.write("These charts show how the model's accuracy (mAP, Precision, Recall) and error rate (Loss) improved over each epoch of training.")
    results_graph = os.path.join(run_path, "results.png")
    if os.path.exists(results_graph):
        st.image(results_graph, use_container_width=True)
    else:
        st.warning("Main training results graph (results.png) not found in the selected run directory.")

    st.divider()

    st.write("##### **Class Performance Analysis**")
    st.write("These graphs break down the model's performance for each specific object class.")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Confusion Matrix:** Shows where the model got confused (e.g., misclassifying a Toolbox as an OxygenTank). The diagonal represents correct predictions.")
        path = os.path.join(run_path, "confusion_matrix.png")
        if os.path.exists(path):
            st.image(path, use_container_width=True)
        else:
            st.info("Confusion Matrix graph not found.")

    with col2:
        st.write("**Precision-Recall Curve:** Illustrates the trade-off between precision and recall for different confidence thresholds. A curve closer to the top-right corner indicates better performance.")
        pr = os.path.join(run_path, "PR_curve.png")
        f1 = os.path.join(run_path, "F1_curve.png")
        if os.path.exists(pr):
            st.image(pr, use_container_width=True)
        elif os.path.exists(f1):
            st.image(f1, use_container_width=True)
        else:
            st.info("Precision-Recall curve not found.")
