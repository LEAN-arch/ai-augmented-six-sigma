import streamlit as st
from utils.plotting import plot_bayesian_network

st.set_page_config(layout="wide", page_title="Define Phase")
st.title("🌀 Define Phase: Scoping with Clarity and Data")

st.markdown("""
The Define phase is about setting the project's foundation. While classical tools excel at structured, qualitative alignment, ML offers powerful ways to understand complex systems from data at the outset.
""")

col1, col2 = st.columns(2)

with col1:
    st.header("Classical Tool: SIPOC")
    st.info("""
    **SIPOC (Suppliers, Inputs, Process, Outputs, Customers)** is a high-level process map.
    - **Strength:** Excellent for team alignment and defining project scope. It's qualitative, structured, and easy to communicate.
    - **Limitation:** Relies on existing process knowledge; may miss hidden interactions or data-driven relationships.
    """)
    st.text("""
    Example SIPOC for a 'Baking a Cake' Process:
    -------------------------------------------------
    S: Grocery Store, Utility Co.
    I: Flour, Sugar, Electricity, Water
    P: Mix, Bake, Cool
    O: A Finished Cake
    C: Family, Friends
    """)

with col2:
    st.header("ML Counterpart: Causal Graphs")
    st.info("""
    **Causal Graphs (e.g., Bayesian Networks)** model probabilistic relationships between variables.
    - **Strength:** Can be learned from data to reveal non-obvious dependencies and potential causal links in complex systems. Ideal for identifying key variables to measure.
    - **Limitation:** Requires significant data and can be computationally intensive. Results are probabilistic, not deterministic.
    """)
    st.graphviz_chart(plot_bayesian_network())
    st.caption("A Bayesian Network showing how raw material quality and machine settings might influence the final defect rate through intermediate variables.")

st.success("**Hybrid Strategy:** Use SIPOC to align the team and define scope, then use causal discovery on historical data to validate assumptions and identify critical-to-quality (CTQ) variables you might have missed.")
