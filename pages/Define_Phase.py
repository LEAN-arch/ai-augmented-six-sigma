import streamlit as st
import sys
import os
import graphviz

# --- Robust Pathing ---
# This ensures that 'utils' can be imported from the 'pages' directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ----------------------

from utils.plotting_pro import plot_voc_nlp_summary

st.set_page_config(layout="wide", page_title="Define Phase")
st.title("🌀 Define Phase: Establishing the Foundation")

st.markdown("""
**Objective:** To clearly articulate the problem, project goals, scope, and high-level process map. This phase ensures the team is aligned and focused on a well-defined business problem.
""")
st.markdown("---")

# --- Tool Comparison ---
st.header("1. Process Mapping: SIPOC vs. Data-Driven Graphs")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Classical Tool: SIPOC")
    st.info("""
    A qualitative, high-level map that defines the scope of an improvement project by identifying **S**uppliers, **I**nputs, **P**rocess, **O**utputs, and **C**ustomers.
    - **Strength:** Simplicity, team alignment, and clear boundary definition.
    - **Limitation:** Relies entirely on existing domain knowledge and may miss latent, data-driven relationships.
    """)

with col2:
    st.subheader("ML Counterpart: Causal Discovery")
    st.info("""
    Algorithms (e.g., PC, FCI) that infer cause-and-effect structures directly from observational data, often visualized as a graph.
    - **Strength:** Objectively discovers potential causal links and complex interactions that human experts might overlook.
    - **Limitation:** Requires large, high-quality datasets and the output is a set of hypotheses that require validation.
    """)

st.header("2. Voice of the Customer (VOC): Surveys vs. NLP")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Classical Tool: Surveys & Interviews")
    st.info("""
    Manual methods of gathering customer feedback, such as structured surveys, focus groups, and one-on-one interviews.
    - **Strength:** Provides deep, contextual insights.
    - **Limitation:** Slow, expensive, small sample size, and prone to researcher bias. Difficult to scale.
    """)

with col2:
    st.subheader("ML Counterpart: Natural Language Processing (NLP)")
    st.info("""
    Using algorithms to analyze vast amounts of unstructured text data (reviews, support tickets, social media) to identify themes, sentiment, and key topics.
    - **Strength:** Scalable, real-time, objective analysis of massive datasets. Uncovers emerging trends instantly.
    - **Limitation:** Requires text data; may miss the 'why' without deeper qualitative follow-up.
    """)
    fig = plot_voc_nlp_summary()
    st.plotly_chart(fig, use_container_width=True)

st.success("""
**Verdict & Hybrid Strategy:**
-   Use **SIPOC** to create the initial project charter and align stakeholders.
-   Use **NLP** on existing customer feedback data to validate the 'Outputs' and 'Customers' sections of the SIPOC and to quantify the business impact of different problems.
-   Use **Causal Discovery** outputs as a data-driven starting point for building more detailed process maps and identifying potential Xs for the Analyze phase.
""")
