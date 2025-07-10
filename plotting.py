import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from scipy.stats import norm, gaussian_kde
import graphviz

def plot_bayesian_network():
    """Creates a conceptual Bayesian Network for the Define phase."""
    dot = graphviz.Digraph(comment='Process Bayesian Network')
    dot.attr('node', shape='ellipse', style='filled', color='lightblue')
    dot.attr('edge', color='gray40')
    dot.attr(rankdir='LR')
    
    dot.node('A', 'Raw Material\nQuality')
    dot.node('B', 'Operator\nSkill')
    dot.node('C', 'Machine\nCalibration')
    dot.node('D', 'Ambient\nTemp')
    dot.node('E', 'Product\nViscosity')
    dot.node('F', 'Final\nDefect Rate')

    dot.edges(['AC', 'BC', 'CE', 'DE', 'EF'])
    return dot

def plot_capability_analysis(data, lsl, usl, title):
    """Plots a histogram with capability metrics and a KDE curve."""
    mean, std = np.mean(data), np.std(data)
    
    # Classical Cp/Cpk
    cpu = (usl - mean) / (3 * std)
    cpl = (mean - lsl) / (3 * std)
    cpk = min(cpu, cpl)
    cp = (usl - lsl) / (6 * std)

    # ML approach: Kernel Density Estimation
    kde = gaussian_kde(data)
    x_range = np.linspace(min(data), max(data), 500)
    kde_y = kde(x_range)

    fig = go.Figure()

    # Histogram for classical view
    fig.add_trace(go.Histogram(x=data, nbinsx=30, name='Process Data', histnorm='probability density'))

    # KDE for ML view
    fig.add_trace(go.Scatter(x=x_range, y=kde_y, mode='lines', name='ML: Kernel Density Estimate', line=dict(color='red', width=3)))

    # Add specification limits
    fig.add_vline(x=lsl, line_width=3, line_dash="dash", line_color="red", name="LSL")
    fig.add_vline(x=usl, line_width=3, line_dash="dash", line_color="red", name="USL")
    fig.add_vline(x=mean, line_width=2, line_dash="dot", line_color="black", name="Mean")

    fig.update_layout(
        title_text=title,
        xaxis_title="Measurement Value",
        yaxis_title="Density",
        legend_title="Legend",
        annotations=[
            dict(x=lsl, y=0, xref="x", yref="paper", text=f"LSL={lsl}", showarrow=False, xanchor='right'),
            dict(x=usl, y=0, xref="x", yref="paper", text=f"USL={usl}", showarrow=False, xanchor='left'),
            dict(x=0.95, y=0.95, xref='paper', yref='paper',
                 text=f"<b>Classical:</b><br>Cp = {cp:.2f}<br>Cpk = {cpk:.2f}",
                 showarrow=False, align='left', bordercolor='black', borderwidth=1, bgcolor='rgba(255,255,255,0.8)')
        ]
    )
    return fig

def plot_regression_comparison(df):
    """Compares Linear Regression with a Random Forest Regressor."""
    X = df[['Feature_1_Linear', 'Feature_2_Quadratic', 'Feature_3_Noise']]
    y = df['Output']

    # Classical Model
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    y_pred_lin = lin_reg.predict(X)
    
    # ML Model
    rf_reg = RandomForestRegressor(n_estimators=50, random_state=42)
    rf_reg.fit(X, y)
    y_pred_rf = rf_reg.predict(X)

    # Sort for plotting
    sort_indices = X['Feature_1_Linear'].argsort()
    X_sorted = X.iloc[sort_indices]
    y_sorted = y.iloc[sort_indices]
    y_pred_lin_sorted = y_pred_lin[sort_indices]
    y_pred_rf_sorted = y_pred_rf[sort_indices]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_sorted['Feature_1_Linear'], y=y_sorted, mode='markers', name='Actual Data', marker=dict(opacity=0.5)))
    fig.add_trace(go.Scatter(x=X_sorted['Feature_1_Linear'], y=y_pred_lin_sorted, mode='lines', name='Classical: Linear Regression', line=dict(color='red', width=3)))
    fig.add_trace(go.Scatter(x=X_sorted['Feature_1_Linear'], y=y_pred_rf_sorted, mode='lines', name='ML: Random Forest', line=dict(color='green', width=3, dash='dot')))
    
    fig.update_layout(
        title_text="Classical Regression vs. ML Model on Non-Linear Data",
        xaxis_title="Primary Feature Value",
        yaxis_title="Output",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig, rf_reg, X, y

def plot_feature_importance(model, X, y):
    """Plots feature importance for the ML model."""
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    importances = pd.DataFrame(result.importances_mean, index=X.columns, columns=['Importance']).sort_values('Importance', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=importances['Importance'],
        y=importances.index,
        orientation='h'
    ))
    fig.update_layout(
        title_text="ML Feature Importance (Counterpart to Pareto)",
        xaxis_title="Importance (Performance drop when feature is shuffled)",
        yaxis_title="Feature"
    )
    return fig


def plot_control_chart_comparison(df):
    """Plots a control chart with an ML-based anomaly detection layer."""
    mean = df['Value'].iloc[:100].mean()
    std_dev = df['Value'].iloc[:100].std()
    ucl = mean + 3 * std_dev
    lcl = mean - 3 * std_dev
    
    # ML Anomaly Detection (simple rolling z-score for demonstration)
    window_size = 10
    rolling_mean = df['Value'].rolling(window=window_size).mean()
    rolling_std = df['Value'].rolling(window=window_size).std()
    df['z_score'] = (df['Value'] - rolling_mean) / rolling_std
    ml_anomalies = df[df['z_score'].abs() > 3]

    # Classical SPC violation (first point after shift that is outside limits)
    spc_violations = df[(df['Value'] > ucl) | (df['Value'] < lcl)]

    fig = go.Figure()
    # Process data
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Value'], mode='lines+markers', name='Process Data', marker=dict(size=4)))
    
    # Control limits
    fig.add_hline(y=ucl, line_dash="dash", line_color="red", name="UCL")
    fig.add_hline(y=mean, line_dash="dot", line_color="black", name="Center Line")
    fig.add_hline(y=lcl, line_dash="dash", line_color="red", name="LCL")
    
    # Highlight violations
    fig.add_trace(go.Scatter(x=spc_violations['Time'], y=spc_violations['Value'], mode='markers', name='SPC Violation', marker=dict(color='red', size=12, symbol='x')))
    fig.add_trace(go.Scatter(x=ml_anomalies['Time'], y=ml_anomalies['Value'], mode='markers', name='ML Anomaly', marker=dict(color='purple', size=12, symbol='star')))

    fig.update_layout(
        title="Classical SPC vs. ML Anomaly Detection",
        xaxis_title="Sample Number",
        yaxis_title="Measurement Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_doe_cube(df):
    """Creates a 3D cube plot for a 2^3 factorial design."""
    fig = go.Figure(data=[go.Scatter3d(
        x=df['Temp'],
        y=df['Pressure'],
        z=df['Time'],
        mode='markers+text',
        marker=dict(
            size=12,
            color=df['Yield'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Yield')
        ),
        text=[f"{y:.1f}" for y in df['Yield']],
        textposition='top center'
    )])

    # Add lines for the cube edges
    lines = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            # Check if points differ by only one factor
            if np.sum(df.iloc[i, :3] != df.iloc[j, :3]) == 1:
                lines.append(go.Scatter3d(
                    x=[df.iloc[i]['Temp'], df.iloc[j]['Temp']],
                    y=[df.iloc[i]['Pressure'], df.iloc[j]['Pressure']],
                    z=[df.iloc[i]['Time'], df.iloc[j]['Time']],
                    mode='lines',
                    line=dict(color='grey', width=2),
                    showlegend=False
                ))
    fig.add_traces(lines)

    fig.update_layout(
        title="Classical DOE: 2³ Factorial Design Cube Plot",
        scene=dict(
            xaxis_title='Factor A: Temp',
            yaxis_title='Factor B: Pressure',
            zaxis_title='Factor C: Time'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig
