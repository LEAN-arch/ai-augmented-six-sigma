# main_app.py

import streamlit as st
import sys
import os

# --- Robust Pathing ---
# Add the project root to the Python path
# This ensures that 'utils' can be imported from anywhere in the app
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ----------------------

from utils.config import get_custom_css

st.set_page_config(
    page_title="AI-Augmented Six Sigma | Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for professional styling
st.markdown(get_custom_css(), unsafe_allow_html=True)

st.title("🚀 AI-Augmented Six Sigma")
st.subheader("A Professional Dashboard for Integrating Classical Statistics and Machine Learning")

st.markdown("""
Welcome to the definitive guide for integrating **Machine Learning (ML)** into the **Six Sigma DMAIC framework**. This interactive dashboard is designed for engineers, data scientists, and quality leaders who want to leverage the best of both worlds: the structured rigor of classical statistics and the predictive power of modern ML.
""")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.header("🎯 Our Mission")
    st.write("""
    To provide a clear, technically deep, and interactive resource that demystifies the synergy between Six Sigma and AI. We aim to empower professionals to:
    - **Understand** the strengths and weaknesses of each approach.
    - **Identify** the right tool for the right problem.
    - **Implement** hybrid strategies for superior process control and optimization.
    - **Justify** decisions with both statistical rigor and data-driven evidence.
    """)

with col2:
    st.header("🧭 How to Use This Dashboard")
    st.write("""
    Navigate through the DMAIC phases using the sidebar on the left. Each page offers:
    - **Phase Objectives:** A clear statement of goals for each stage.
    - **Tool Deep Dives:** Side-by-side comparisons with mathematical foundations.
    - **Interactive Simulators:** Hands-on visualizations to build intuition.
    - **Expert Verdicts:** Actionable takeaways for practical application.
    """)

st.info("👈 **Select a page from the sidebar to begin your deep dive into a specific DMAIC phase.**")

st.image("https://i.imgur.com/uGAtbMh.png", caption="The evolution from descriptive statistics to predictive and prescriptive AI-driven quality control.")
