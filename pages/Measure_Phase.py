import streamlit as st
from utils.data_generator import generate_process_data
from utils.plotting import plot_capability_analysis

st.set_page_config(layout="wide", page_title="Measure Phase")
st.title("🔬 Measure Phase: Quantifying Performance")
st.markdown("In the Measure phase, we validate our measurement systems and establish a baseline. ML provides a more nuanced, distributional view of performance compared to single-point classical metrics.")

st.header("Capability Analysis: Classical (Cp, Cpk) vs. ML (KDE)")
st.markdown("""
- **Classical Approach (Cp, Cpk):** Assumes data is normally distributed. Provides simple, powerful indices to compare process spread and centering against specification limits.
- **ML Approach (Kernel Density Estimation - KDE):** Non-parametric method that estimates the probability distribution of the data without assuming a specific form. It reveals the true shape, including multi-modality or skewness.
""")

st.sidebar.header("Interactive Simulation")
lsl = st.sidebar.slider("Lower Specification Limit (LSL)", 80.0, 95.0, 90.0, key="measure_lsl")
usl = st.sidebar.slider("Upper Specification Limit (USL)", 105.0, 120.0, 110.0, key="measure_usl")
process_mean = st.sidebar.slider("Process Mean", 95.0, 105.0, 101.0, key="measure_mean")
process_std = st.sidebar.slider("Process Std Dev", 0.5, 5.0, 2.0, key="measure_std")

data = generate_process_data(process_mean, process_std, 500, lsl, usl)
fig = plot_capability_analysis(data, lsl, usl, "Process Capability: Classical Histogram vs. ML KDE")
st.plotly_chart(fig, use_container_width=True)

st.info("**Try This:** Move the 'Process Mean' slider closer to a spec limit. Notice how Cpk drops significantly, while the KDE plot visually shows the 'tail' of the distribution crossing the limit. The KDE provides a richer visual understanding of the risk.")
st.success("**Hybrid Strategy:** Report Cpk as the standard industry metric, but use the KDE plot internally to understand the *real* process behavior and diagnose non-normality issues.")
