import streamlit as st
import pandas as pd
import sys
import os

# --- Robust Pathing ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ----------------------

st.set_page_config(layout="wide", page_title="Comparison Matrix")
st.title("⚔️ Head-to-Head Comparison: Classical Stats vs. Machine Learning")

st.subheader("Attribute Comparison Matrix")
comparison_data = pd.DataFrame({
    "Dimension": [
        "Transparency", "Data Requirements", "Assumptions",
        "Interpretability", "Scalability", "Optimization",
        "Implementation Cost", "Auditability", "Proactive Detection"
    ],
    "Classical Stats": [
        "High (e.g., regression coefficients, p-values)", "Low (can work with small datasets)", "Many (e.g., normality, homoscedasticity)",
        "Strong for decision support", "Poor beyond 3-4 variables", "Limited (fixed structure)",
        "Low (Excel, Minitab)", "High", "Limited & Reactive"
    ],
    "Machine Learning": [
        "Often low ('black-box') unless using SHAP/LIME", "High (more data improves generalization)", "Fewer (nonparametric, flexible models)",
        "Requires post-hoc tools (e.g., LIME, SHAP)", "Excellent with high-dimensional data", "Adaptive and dynamic (online learning)",
        "Higher (Python, cloud infra, data pipelines)", "Lower (can be complex to validate)", "Predictive & Proactive"
    ]
})
st.dataframe(comparison_data, use_container_width=True, hide_index=True)

st.subheader("🏁 The Verdict: Which is Better?")
verdict_data = pd.DataFrame({
    "Metric": ["Interpretability", "Scalability", "Accuracy in Complex Systems", "Ease of Implementation", "Proactive Detection", "Auditability & Compliance"],
    "🏆 Winner": ["Classical Stats", "Machine Learning", "Machine Learning", "Classical Stats", "Machine Learning", "Classical Stats"],
    "Rationale": [
        "Models and outputs are easier to explain and defend without extra tools.",
        "Natively handles larger, messier, and higher-dimensional datasets.",
        "Effectively captures nonlinear patterns and complex interactions.",
        "Tools (like Excel/Minitab) and training are widely available and simpler.",
        "Designed to predict future outcomes, not just describe past events.",
        "Methods are standardized and preferred in regulated industries (e.g., FDA, FAA)."
    ]
})
st.dataframe(verdict_data, use_container_width=True, hide_index=True)
