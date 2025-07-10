import streamlit as st
from utils.data_generator import generate_control_chart_data
from utils.plotting_pro import plot_control_chart_pro

st.set_page_config(layout="wide", page_title="Control Phase")
st.title("📡 Control Phase: Sustaining and Monitoring Gains")
st.markdown("""
**Objective:** To implement a system to monitor the improved process, ensuring it remains stable and that the gains are sustained over time. This involves moving from detection to prediction.
""")
st.markdown("---")

st.header("Process Monitoring: SPC vs. ML Anomaly Detection")

st.sidebar.header("Control Simulator")
st.sidebar.markdown("Introduce a shift in the process and see which method detects it first.")
shift_point = st.sidebar.slider("Point of Process Shift", min_value=50, max_value=130, value=100, key="control_shift_point")
shift_magnitude = st.sidebar.slider("Magnitude of Shift (in Std Devs)", 0.5, 4.0, 1.8, 0.1, key="control_shift_mag")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Classical: Statistical Process Control (SPC)")
    st.info("SPC uses control charts to monitor a single process variable. It's based on detecting points or patterns that are statistically unlikely.")
    with st.expander("Show Key SPC Rules (Western Electric)"):
        st.markdown("""
        - **Rule 1:** One point outside the ±3σ limits. (Detects large shifts)
        - **Rule 2:** Two out of three consecutive points outside the ±2σ limits. (Detects smaller shifts)
        - **Rule 4:** Eight consecutive points on one side of the center line. (Detects sustained small shifts or bias)
        """)

with col2:
    st.subheader("ML: Anomaly Detection")
    st.info("ML models learn the normal operating *patterns* of a system, including across multiple variables. They flag deviations from this learned normal pattern.")
    with st.expander("How it Works (Rolling Z-Score Example)"):
        st.markdown("""
        A simple ML approach is to calculate a Z-score based on a *rolling* window of recent data, not the entire history. This adapts to local process behavior.
        """)
        st.latex(r''' Z_{rolling} = \frac{x_i - \mu_{window}}{\sigma_{window}} ''')
        st.markdown("More advanced methods like LSTMs or Autoencoders learn complex, multivariate, and temporal patterns.")


chart_data = generate_control_chart_data(shift_point=shift_point, shift_magnitude=shift_magnitude)
fig_control = plot_control_chart_pro(chart_data)
st.plotly_chart(fig_control, use_container_width=True)

st.success("""
**Verdict & Hybrid Strategy:**
- **The Best of Both:** This is the most critical phase for a hybrid approach.
- **Monitor CTQs with SPC:** Keep classical control charts on your final, critical-to-quality (CTQ) outputs. They are simple, robust, and universally understood.
- **Monitor Inputs (Xs) with ML:** Deploy a multivariate ML anomaly detection model on the *input* and *process* parameters. This model acts as an **early warning system**. It will detect deviations in the inputs *before* they cause a defect in the output, allowing you to move from reactive to proactive control.
""")
