import streamlit as st
import pandas as pd
import sys
import os

# --- Robust Pathing ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ----------------------

st.set_page_config(layout="wide", page_title="Hybrid Strategy")
st.title("🧠 The Hybrid Strategy: The Future of Quality")
st.markdown("The most effective approach is not to choose one over the other, but to build an **AI-Augmented Six Sigma** program that leverages the strengths of both.")

st.subheader("Scenario-Based Recommendations")
guidance_data = pd.DataFrame({
    "Scenario": [
        "Validating a change for FDA/FAA compliance", "Monitoring a high-volume semiconductor fab", "Understanding customer churn from reviews",
        "Optimizing a simple, 3-factor physical process", "Building a 'digital twin' of a chemical reactor", "Providing real-time operator guidance"
    ],
    "Recommended Approach": [
        "Classical Stats (Hypothesis Testing, DOE)", "ML + SPC", "ML NLP", "Classical DOE", "Hybrid: ML Model + Bayesian Opt.", "ML (Real-time Prediction)"
    ],
    "Why?": [
        "Methods are traceable, validated, and legally defensible.", "Detects subtle multivariate drifts in sensor data that SPC misses.", "Processes and extracts sentiment/topics from massive unstructured text.",
        "Simple, effective, and provides clear, interpretable results.", "ML builds the accurate simulation; Bayesian Opt. finds the peak efficiency.", "Predicts outcomes based on current settings and suggests adjustments."
    ]
})
st.dataframe(guidance_data, use_container_width=True, hide_index=True)


st.header("A Unified Workflow")
st.image("https://i.imgur.com/rS2Mtn1.png", caption="An integrated workflow where classical and ML tools support each other at every stage.")

st.markdown("""
### 1. **Baseline with Classical Tools**
   - Use SPC and Capability Analysis to understand your current state. This provides the ground truth.
   - Use SIPOC and Process Maps to align stakeholders.

### 2. **Explore & Analyze with ML**
   - Feed historical data into ML models (Random Forest, XGBoost) to perform a wide-ranging analysis.
   - Use Feature Importance and SHAP to identify the previously unknown drivers of variation. This is your new, data-driven Fishbone diagram.

### 3. **Experiment & Optimize with Both**
   - Use the insights from ML to design a highly targeted classical DOE on the few most important variables. This is more efficient than a broad-based DOE.
   - For digital systems or simulations, use ML-based optimization (Bayesian Opt.) to explore the vast parameter space.

### 4. **Control with a Hybrid System**
   - Keep classical SPC charts on the final output (the CTQ) for simple, robust monitoring and compliance.
   - Deploy ML anomaly detection models on the input and process parameters (the Xs) to create an early warning system that predicts when the CTQ is *about to* go out of control.

This hybrid approach gives you the **interpretability and rigor** of Six Sigma combined with the **scalability and predictive power** of Machine Learning, creating a resilient, intelligent, and continuously improving quality system.
""")
