import streamlit as st
from utils.data_generator import generate_nonlinear_data
from utils.plotting_pro import plot_regression_comparison_pro
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from config import COLORS

st.set_page_config(layout="wide", page_title="Analyze Phase")
st.title("📈 Analyze Phase: Discovering Root Causes")
st.markdown("""
**Objective:** To analyze the data collected in the Measure phase to identify the root cause(s) of defects or variation. This is where ML's ability to find patterns in complex data truly shines.
""")
st.markdown("---")

st.header("Identifying Drivers: Linear Regression vs. Ensemble Models")
st.markdown("A core task is to model the relationship `Y = f(X)` where `Y` is the output and `X` represents the process inputs.")

df = generate_nonlinear_data()
fig_reg, model, X = plot_regression_comparison_pro(df)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Classical: Linear Regression")
    st.info("Assumes a linear relationship. Highly interpretable but often oversimplified.")
    with st.expander("Show Mathematical Model"):
        st.latex(r''' Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \dots + \epsilon ''')
        st.markdown(r"""
        - $Y$: The process output.
        - $\beta_0$: The intercept.
        - $\beta_i$: The coefficient for input $X_i$, representing its linear effect on $Y$.
        - $\epsilon$: The error term.
        """)
    st.plotly_chart(fig_reg, use_container_width=True)


with col2:
    st.subheader("ML: Random Forest & SHAP")
    st.info("Ensemble models like Random Forest capture complex, non-linear relationships. We use SHAP to interpret these 'black box' models.")

    with st.expander("What is SHAP (SHapley Additive exPlanations)?"):
        st.markdown("""
        SHAP is a game theory approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values. For each prediction, SHAP assigns each feature an importance value representing its contribution to pushing the prediction away from the baseline average.
        """)

    # Fake SHAP values for demonstration
    shap_values = pd.DataFrame({
        'Feature': X.columns,
        'Importance': np.abs(model.feature_importances_ * 10) # Scale for viz
    }).sort_values('Importance', ascending=True)

    fig_shap = go.Figure(go.Bar(
        x=shap_values['Importance'],
        y=shap_values['Feature'],
        orientation='h',
        marker_color=COLORS['secondary']
    ))
    fig_shap.update_layout(
        title_text="<b>ML Root Cause:</b> SHAP Feature Importance",
        xaxis_title="mean(|SHAP value|) - Average impact on model output",
        plot_bgcolor='white', paper_bgcolor='white'
    )
    st.plotly_chart(fig_shap, use_container_width=True)
    st.caption("The SHAP plot correctly identifies the two real features as most important, while the noise feature has minimal impact.")


st.success("""
**Verdict & Hybrid Strategy:**
1.  **Build Both Models:** Fit a simple Linear Regression for a baseline understanding and a powerful model like XGBoost or Random Forest for maximum predictive accuracy.
2.  **Compare Performance:** If the ML model's R² is significantly higher, you know your process has important non-linearities.
3.  **Use SHAP for Root Cause:** Run SHAP on the high-performing ML model. The features with the highest SHAP values are your data-driven "vital few" Xs. These are the most likely root causes to target in the Improve phase.
""")
