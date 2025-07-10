# utils/data_generator.py

import numpy as np
import pandas as pd

def generate_process_data(mean, std_dev, size, lsl, usl):
    """
    Generates normally distributed process data with a few realistic outliers.

    This function simulates a standard manufacturing process output. The core data
    follows a normal distribution, but a small percentage of outliers are added
    beyond the specification limits to mimic real-world process failures or
    measurement errors, providing a more robust test for capability analysis.

    Args:
        mean (float): The mean of the process.
        std_dev (float): The standard deviation of the process.
        size (int): The total number of data points to generate.
        lsl (float): The lower specification limit.
        usl (float): The upper specification limit.

    Returns:
        np.ndarray: An array of simulated process data points.
    """
    # Generate the core, normally distributed data
    data = np.random.normal(mean, std_dev, size)

    # Introduce a small percentage of outliers for realism
    num_outliers = int(size * 0.02)  # 2% of data will be outliers
    if num_outliers > 0:
        outliers = np.concatenate([
            np.random.uniform(lsl - 3 * std_dev, lsl, num_outliers // 2), # Outliers below LSL
            np.random.uniform(usl, usl + 3 * std_dev, num_outliers - (num_outliers // 2)) # Outliers above USL
        ])
        # Replace some of the normal data with outliers
        data[:len(outliers)] = outliers
    
    # Shuffle the data to mix in the outliers randomly
    np.random.shuffle(data)
    return data

def generate_nonlinear_data(size=200):
    """
    Generates data with a clear non-linear relationship plus a noise feature.

    This is designed specifically to demonstrate the weakness of linear models
    and the strength of ML models. The output 'y' is a quadratic function of
    two input features, ensuring a non-linear pattern. A third 'noise' feature
    is added with no correlation to the output, which is crucial for testing
    the feature importance capabilities of ML models.

    Args:
        size (int): The number of data points to generate.

    Returns:
        pd.DataFrame: A DataFrame with three input features and one output variable.
    """
    # Generate the primary feature
    X1 = np.linspace(-10, 10, size)
    
    # Create a second feature with a quadratic relationship
    X2 = X1**2
    
    # Create a third feature that is pure noise and irrelevant to the output
    X3 = np.random.randn(size) * 5
    
    # Define the output 'y' as a function of X1 and X2, with some random error
    y = 2 * X1 + 0.5 * X2 + np.random.normal(0, 5, size)
    
    df = pd.DataFrame({
        'Feature_1_Linear': X1,
        'Feature_2_Quadratic': X2,
        'Feature_3_Noise': X3,
        'Output': y
    })
    return df

def generate_control_chart_data(mean=100, std_dev=5, size=150, shift_point=100, shift_magnitude=1.5):
    """
    Generates time-series data simulating a process with a mean shift.

    This function creates a dataset ideal for comparing SPC and ML anomaly detection.
    The first part of the series represents a stable, "in-control" process. At a
    specified 'shift_point', the process mean shifts by a given magnitude,
    simulating a process change or degradation.

    Args:
        mean (float): The initial in-control process mean.
        std_dev (float): The process standard deviation.
        size (int): The total number of samples in the time series.
        shift_point (int): The index at which the process mean shifts.
        shift_magnitude (float): The size of the shift in terms of standard deviations.

    Returns:
        pd.DataFrame: A DataFrame with 'Time' and 'Value' columns.
    """
    # Generate the in-control part of the process
    in_control_data = np.random.normal(mean, std_dev, shift_point)
    
    # Generate the out-of-control part with the shifted mean
    out_of_control_data = np.random.normal(mean + shift_magnitude * std_dev, std_dev, size - shift_point)
    
    # Concatenate the two parts to form the full time series
    full_process_data = np.concatenate([in_control_data, out_of_control_data])
    time_index = np.arange(size)
    
    return pd.DataFrame({'Time': time_index, 'Value': full_process_data})

def generate_doe_data():
    """
    Generates data for a classic 2^3 factorial Design of Experiments (DOE).

    This simulates the results of a physical experiment with three factors
    (e.g., Temperature, Pressure, Time), each tested at a low (-1) and high (+1)
    level. The response function is designed to have significant main effects and
    at least one interaction effect, which is a key concept in DOE.

    Returns:
        pd.DataFrame: A DataFrame with factor settings and the resulting 'Yield'.
    """
    factors = [-1, 1]  # Low and High levels
    data = []
    
    # Iterate through all 2*2*2 = 8 combinations
    for f1 in factors:
        for f2 in factors:
            for f3 in factors:
                # A realistic response function with main effects, an interaction, and noise
                response = 10 + 2*f1 + 3*f2 - 1.5*f3 + 1.2*f1*f2 + np.random.randn() * 0.5
                data.append([f1, f2, f3, response])
                
    return pd.DataFrame(data, columns=['Temp', 'Pressure', 'Time', 'Yield'])
