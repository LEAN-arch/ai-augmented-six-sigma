import numpy as np
import pandas as pd

def generate_process_data(mean, std_dev, size, lsl, usl):
    """Generates process data with potential outliers."""
    data = np.random.normal(mean, std_dev, size)
    # Add some outliers for realism
    num_outliers = int(size * 0.02)
    outliers = np.concatenate([
        np.random.uniform(lsl - 3 * std_dev, lsl, num_outliers // 2),
        np.random.uniform(usl, usl + 3 * std_dev, num_outliers // 2)
    ])
    data[:len(outliers)] = outliers
    np.random.shuffle(data)
    return data

def generate_nonlinear_data(size=200):
    """Generates data with a clear non-linear relationship and noise."""
    X = np.linspace(-10, 10, size)
    # Three features, but only X1 and X2 are truly important
    X1 = X
    X2 = X**2
    X3 = np.random.randn(size) * 5 # Noise feature
    
    y = 2 * X1 + 0.5 * X2 + np.random.normal(0, 5, size)
    
    df = pd.DataFrame({'Feature_1_Linear': X1, 'Feature_2_Quadratic': X2, 'Feature_3_Noise': X3, 'Output': y})
    return df

def generate_control_chart_data(mean=100, std_dev=5, size=150, shift_point=100, shift_magnitude=1.5):
    """Generates time-series data for a control chart with a mean shift."""
    in_control = np.random.normal(mean, std_dev, shift_point)
    out_of_control = np.random.normal(mean + shift_magnitude * std_dev, std_dev, size - shift_point)
    data = np.concatenate([in_control, out_of_control])
    time_index = np.arange(size)
    return pd.DataFrame({'Time': time_index, 'Value': data})

def generate_doe_data():
    """Generates data for a 2^3 factorial design cube plot."""
    factors = [-1, 1]
    data = []
    for f1 in factors:
        for f2 in factors:
            for f3 in factors:
                # Example response function
                response = 10 + 2*f1 + 3*f2 - 1.5*f3 + 1*f1*f2 + np.random.randn() * 0.5
                data.append([f1, f2, f3, response])
    return pd.DataFrame(data, columns=['Temp', 'Pressure', 'Time', 'Yield'])
