import streamlit as st
from utils.data_generator import generate_nonlinear_data
from utils.plotting import plot_regression_comparison, plot_feature_importance

st.set_page_config(layout="wide", page_title="Analyze Phase")
st.title("📈 Analyze Phase: Finding the Root Causes")
st.markdown("The Analyze phase is where ML truly shines. While classical methods test specific hypotheses, ML models can explore thousands of potential relationships in high-dimensional data to uncover the most impactful drivers.")

st.header("Regression: Linear (Classical) vs. Ensemble (ML)")
st.markdown("""
- **Classical Linear Regression:** Assumes a linear relationship between inputs (X) and output (Y). Highly interpretable coefficients but can be inaccurate for complex, non-linear systems.
- **ML Models (e.g., Random Forest):** Capture complex, non-linear patterns and interactions automatically. Far more accurate on complex data but less directly interpretable.
""")

# Generate and plot
df = generate_nonlinear_data()
fig_reg, model, X, y = plot_regression_comparison(df)
st.plotly_chart(fig_reg, use_container_width=True)
st.info("The visualization clearly shows the linear model (red line) failing to capture the underlying curve of the data, while the Random Forest model (green line) adapts to it almost perfectly.")

st.header("Root Cause Analysis: Pareto (Classical) vs. Feature Importance (ML)")
st.markdown("""
- **Pareto Chart:** A simple bar chart ranking causes by frequency, based on the 80/20 rule. It's a fundamental starting point.
- **ML Feature Importance:** Ranks input variables by their predictive power on the output. It can uncover which variable has the biggest *impact*, even if it's not the most frequent cause.
""")

col1, col2 = st.columns([1, 2])
with col1:
    st.info("**Pareto Thinking:** 'We had 100 defects. 80 were from machine A, 20 from machine B. Let's fix machine A.'")
    st.info("**ML Feature Importance Thinking:** 'Shuffling the 'raw material batch' data tanked our model's accuracy, while shuffling 'operator name' did nothing. Raw material is the most critical driver of quality, regardless of which machine it runs on.'")
with col2:
    fig_imp = plot_feature_importance(model, X, y)
    st.plotly_chart(fig_imp, use_container_width=True)
    st.caption("This chart shows that 'Feature_2_Quadratic' and 'Feature_1_Linear' are critical to predicting the output, while 'Feature_3_Noise' has virtually no impact, correctly identifying the true drivers.")

st.success("**Hybrid Strategy:** Start with classical hypothesis tests for simple questions. For complex, multivariate problems, use a powerful ML model to identify the key drivers via feature importance, then use classical DOE to experiment on those vital few variables.")
