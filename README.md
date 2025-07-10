# AI-Augmented Six Sigma Dashboard (Professional Edition)

This is a commercial-grade, interactive Streamlit dashboard designed for a deep, technical exploration of integrating Machine Learning (ML) into the Six Sigma DMAIC framework. It is built for engineers, quality professionals, data scientists, and business leaders who require a robust understanding of both classical and modern process improvement techniques.

## Key Features

-   **Polished User Experience (UX):** A clean, professional interface with custom styling, intuitive navigation, and a logical information hierarchy.
-   **Deep Technical Content:** Each concept is supported by mathematical formulas, rigorous definitions, and clear justifications.
-   **Advanced Interactive Visualizations:** Go beyond basic charts with custom-built Plotly visualizations that allow users to simulate scenarios, observe model behavior, and gain intuitive understanding of complex algorithms.
-   **Modular & Maintainable Code (DX):** The project is structured with a clear separation of concerns (config, data generation, plotting, app pages), making it easy to maintain, extend, or integrate into other systems.
-   **SME-Level Commentary:** The content is written from the perspective of a subject matter expert, providing actionable insights and strategic recommendations.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd ai-augmented-six-sigma-pro
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: You may also need to install Graphviz at the system level for the Define phase visualization. See [Graphviz Download Page](https://graphviz.org/download/).*

4.  **Run the Streamlit application:**
    ```bash
    streamlit run main_app.py
    ```

The application will open in your default web browser, ready for exploration.
