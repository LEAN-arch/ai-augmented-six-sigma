import streamlit as st
from utils.data_generator import generate_control_chart_data
from utils.plotting import plot_control_chart_comparison

st.set_page_config(layout="wide", page_title="Control Phase")
st.title("📡 Control Phase: Sustaining the Gains")
st.markdown("The Control phase ensures the process stays in its improved state. Classical Statistical Process Control (SPC) is reactive, detecting shifts after they happen. ML enables a proactive, predictive approach.")

st.header("SPC (Classical) vs. Anomaly Detection (ML)")
st.markdown("""
- **Classical Control Chart (X-bar):** Monitors a single variable over time. Triggers an alarm when a point falls outside the +/- 3 sigma control limits. It's simple, robust, and effective for stable processes.
- **ML Anomaly Detection:** Learns the normal operating patterns of the system, including correlations between many variables. It can flag subtle deviations in the *pattern* of data that may be a precursor to a failure, even if no single variable has gone out of its classical limits.
""")

st.sidebar.header("Interactive Simulation")
shift_point = st.sidebar.slider("Point of Process Shift", min_value=50, max_value=130, value=100, key="control_shift_point")
shift_magnitude = st.sidebar.slider("Magnitude of Shift (in Std Devs)", 0.5, 4.0, 1.5, 0.1, key="control_shift_mag")

# Generate data and plot
chart_data = generate_control_chart_data(shift_point=shift_point, shift_magnitude=shift_magnitude)
fig_control = plot_control_chart_comparison(chart_data)
st.plotly_chart(fig_control, use_container_width=True)

st.info("**Try This:** Set the 'Magnitude of Shift' to a small value like `1.0`. Notice how the ML anomaly detector (purple star) might flag a deviation *before* any point crosses the red SPC limits (red 'x'). This demonstrates the power of ML for early warning.")

st.error("""
**The Multivariate Trap that ML Solves:**
Imagine temperature and pressure are correlated. In a normal state, if temp goes up, pressure goes up.
- A classical system has two separate control charts.
- A failure mode causes temp to go up while pressure stays flat.
- Neither chart might signal a violation, as both values could be within their individual limits.
- An ML model trained on the *relationship* between variables would immediately flag this broken correlation as a major anomaly.
""")
st.success("**Hybrid Strategy:** Keep classical control charts on your critical-to-quality outputs (the 'what'). Deploy multivariate ML anomaly detection models on the input and process parameters (the 'why') to get early warnings and diagnostic insights.")
