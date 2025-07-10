import streamlit as st
from utils.data_generator import generate_doe_data
from utils.plotting import plot_doe_cube
from utils.plotting_pro import plot_bayesian_optimization_interactive
import numpy as np

st.set_page_config(layout="wide", page_title="Improve Phase")

st.title("⚙️ Improve Phase: Discovering Optimal Solutions")
st.markdown("""
**Objective:** To identify, test, and implement solutions to the root causes identified in the Analyze phase. This involves finding the optimal settings for the critical process inputs (Xs).
""")
st.markdown("---")

tab1, tab2 = st.tabs(["Classical: Design of Experiments (DOE)", "ML: Bayesian Optimization"])

with tab1:
    st.header("Classical Approach: Design of Experiments (DOE)")
    st.markdown("""
    DOE is a structured, statistical method for systematically changing process inputs to determine their effect on the output. It's the gold standard for physical experimentation.
    - **Strength:** Statistically rigorous, separates main effects from interaction effects, provides high confidence in results.
    - **Limitation:** Becomes exponentially expensive as the number of factors or levels increases (curse of dimensionality).
    """)
    doe_data = generate_doe_data()
    fig_doe = plot_doe_cube(doe_data)
    st.plotly_chart(fig_doe, use_container_width=True)
    st.info("This 3D cube plot visualizes a 2³ factorial design, showing the measured 'Yield' at each corner (combination of settings). This allows us to find the best combination and understand the effect of each factor.")

with tab2:
    st.header("ML Counterpart: Bayesian Optimization")
    st.markdown("""
    An intelligent search algorithm for finding the optimum of an expensive-to-evaluate function. It's ideal for optimizing complex simulations or digital processes.
    - **Strength:** Extremely sample-efficient in high-dimensional spaces. Balances exploring new areas with exploiting known good ones.
    - **Limitation:** Can get stuck in local optima; performance depends on the choice of the underlying statistical model (surrogate model).
    """)

    # --- Interactive Bayesian Optimization Demo ---
    # Define a "hidden" true function the algorithm will try to optimize
    def true_function(x):
        return (np.sin(x * 0.8) * 15) + (np.cos(x * 2.5)) * 5 - (x/10)**3

    x_range = np.linspace(0, 20, 200)

    st.sidebar.header("Bayesian Opt. Simulator")
    if 'sampled_points' not in st.session_state:
        st.session_state.sampled_points = {'x': [2.0, 15.0], 'y': [true_function(2.0), true_function(15.0)]}

    if st.sidebar.button("Sample Next Best Point"):
        fig, next_point = plot_bayesian_optimization_interactive(true_function, x_range, st.session_state.sampled_points)
        st.session_state.sampled_points['x'].append(next_point)
        st.session_state.sampled_points['y'].append(true_function(next_point))
    
    if st.sidebar.button("Reset Simulation"):
        st.session_state.sampled_points = {'x': [2.0, 15.0], 'y': [true_function(2.0), true_function(15.0)]}

    fig_bo, _ = plot_bayesian_optimization_interactive(true_function, x_range, st.session_state.sampled_points)
    st.plotly_chart(fig_bo, use_container_width=True)
    st.info("Click 'Sample Next Best Point' in the sidebar. The algorithm uses its model (blue line) and uncertainty (blue shading) to intelligently choose the next point to sample (green line), quickly homing in on the true peak without testing every value.")


st.success("""
**Verdict & Hybrid Strategy:**
-   For **physical processes** with 2-5 key factors, use **Classical DOE**. Its rigor is unmatched for real-world experimentation.
-   For **digital processes**, **simulation optimization**, or **ML model hyperparameter tuning**, use **Bayesian Optimization**. It will find a better solution in far fewer iterations than grid search or random search.
-   **Advanced Hybrid:** Use DOE to build a good regression model of your physical process, creating a "digital twin." Then, use Bayesian Optimization on this fast, cheap digital twin to explore the entire design space and find a global optimum.
""")
