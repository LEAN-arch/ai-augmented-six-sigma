import streamlit as st
from utils.data_generator import generate_doe_data
from utils.plotting import plot_doe_cube

st.set_page_config(layout="wide", page_title="Improve Phase")
st.title("⚙️ Improve Phase: Optimizing the Process")
st.markdown("In the Improve phase, we systematically find the optimal settings for our process variables. Classical DOE is the gold standard for physical experiments, while ML optimization techniques excel in vast, digital design spaces.")

st.header("Classical Approach: Design of Experiments (DOE)")
st.markdown("""
DOE is a structured approach to experimentation. A factorial design, like the one visualized below, tests all combinations of factor levels.
- **Strengths:** Statistically rigorous, provides information on main effects and interactions, highly interpretable.
- **Ideal Use Case:** Physical experiments with a manageable number of factors (e.g., 2-5) and levels.
""")

doe_data = generate_doe_data()
fig_doe = plot_doe_cube(doe_data)
st.plotly_chart(fig_doe, use_container_width=True)
st.info("The 3D cube plot visualizes a 2³ factorial design. Each corner represents a unique combination of high/low settings for three factors. The color and label show the resulting yield, helping us identify the optimal corner (e.g., high Temp, high Pressure, low Time).")


st.header("ML Counterpart: Bayesian Optimization")
st.markdown("""
Bayesian Optimization is an intelligent search algorithm for finding the maximum or minimum of a function.
- **How it works:** It builds a probabilistic model of the objective function and uses it to select the most promising point to evaluate next, balancing exploration (trying new areas) and exploitation (refining known good areas).
- **Strengths:** Far more sample-efficient than grid search for high-dimensional problems. Ideal for optimizing complex simulations or ML model hyperparameters where each evaluation is costly (takes time or money).
- **Ideal Use Case:** Tuning the 50 hyperparameters of a neural network, or finding the optimal chemical formulation in a simulated environment.
""")
st.image("https://i.imgur.com/gUhC8pT.gif", caption="Conceptual animation of Bayesian Optimization finding the maximum of an unknown function with minimal samples.")

st.success("**Hybrid Strategy:** Use classical DOE for initial screening of physical process variables. Then, use the resulting data to build a simulation model (a 'digital twin'). Finally, use Bayesian Optimization to explore thousands of settings on the digital twin to find the true global optimum before confirming with a final physical test.")
