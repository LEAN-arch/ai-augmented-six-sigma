import streamlit as st
from utils.data_generator import generate_process_data
from utils.plotting_pro import plot_capability_analysis_pro

st.set_page_config(layout="wide", page_title="Measure Phase")
st.title("🔬 Measure Phase: Quantifying Process Performance")
st.markdown("""
**Objective:** To validate the measurement system and establish a reliable baseline of the process's current performance. The mantra is "if you can't measure it, you can't improve it."
""")
st.markdown("---")

st.header("Capability Analysis: Indices vs. Full Distributions")
st.markdown("""
Capability analysis assesses whether a process is capable of meeting customer specifications.
""")

st.sidebar.header("Capability Simulator")
st.sidebar.markdown("Adjust the process parameters to see how they affect capability.")
lsl = st.sidebar.slider("Lower Specification Limit (LSL)", 80.0, 95.0, 90.0, key="measure_lsl")
usl = st.sidebar.slider("Upper Specification Limit (USL)", 105.0, 120.0, 110.0, key="measure_usl")
process_mean = st.sidebar.slider("Process Mean (μ)", 95.0, 105.0, 101.5, key="measure_mean")
process_std = st.sidebar.slider("Process Std Dev (σ)", 0.5, 5.0, 2.0, key="measure_std")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Classical: Cp & Cpk")
    st.info("Single-point indices that summarize capability. They are powerful but assume normality.")
    with st.expander("Show Mathematical Formulas"):
        st.markdown("**Process Potential (Cp):** Measures how well the process spread fits within the specification limits, ignoring centering.")
        st.latex(r''' C_p = \frac{USL - LSL}{6\sigma} ''')
        st.markdown("**Process Capability (Cpk):** Adjusts Cp for process centering. It represents the *actual* capability.")
        st.latex(r''' C_{pk} = \min\left(\frac{USL - \mu}{3\sigma}, \frac{\mu - LSL}{3\sigma}\right) ''')
    
    data = generate_process_data(process_mean, process_std, 1000, lsl, usl)
    fig, cp, cpk = plot_capability_analysis_pro(data, lsl, usl)
    
    st.metric("Process Potential (Cp)", f"{cp:.2f}")
    st.metric("Process Capability (Cpk)", f"{cpk:.2f}")

    if cpk < 1.0:
        st.error("Process is not capable.")
    elif 1.0 <= cpk < 1.33:
        st.warning("Process is marginal. Improvement needed.")
    else:
        st.success("Process is capable.")

with col2:
    st.subheader("ML: Distributional View")
    st.info("Non-parametric methods like Kernel Density Estimation (KDE) visualize the *true* shape of the process distribution, revealing skewness or multiple modes that indices hide.")
    st.plotly_chart(fig, use_container_width=True)


st.success("""
**Verdict & Hybrid Strategy:**
- **Report Cpk:** It is the industry standard and required for most quality management systems.
- **Use KDE for Diagnostics:** The KDE plot is your expert diagnostic tool. If the Cpk is low, the KDE plot tells you *why*—is the process off-center? Is the spread too wide? Is the data not normal? This guides your next steps in the Analyze phase.
""")
