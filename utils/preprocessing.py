"""
Data Preprocessing Module
Handles all data preprocessing operations for time series forecasting
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
from .technical_indicators import add_all_indicators, get_feature_columns


def load_and_prepare_data(filepath, date_column='Date', target_column='Close', use_technical_indicators=False):
    """
    Load and prepare stock data from CSV file
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file containing stock data
    date_column : str
        Name of the date column
    target_column : str
        Name of the target column to predict (default: 'Close')
    use_technical_indicators : bool
        Whether to add technical indicators (default: False)
    
    Returns:
    --------
    pd.DataFrame
        Cleaned and sorted dataframe
    """
    # Load data
    df = pd.read_csv(filepath)
    
    # Convert date column to datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Sort by date (critical for time series)
    df = df.sort_values(by=date_column)
    
    # Handle missing values using forward fill
    df = df.fillna(method='ffill')
    
    # Add technical indicators if requested
    if use_technical_indicators:
        df = add_all_indicators(df)
        print("Added technical indicators")
    
    # Remove any remaining NaN values
    df = df.dropna()
    
    # Reset index
    df = df.reset_index(drop=True)
    
    print(f"Loaded {len(df)} records from {filepath}")
    print(f"Date range: {df[date_column].min()} to {df[date_column].max()}")
    
    return df


def scale_data(data, scaler=None, feature_range=(0, 1)):
    """
    Normalize data to specified range using MinMaxScaler
    
    Parameters:
    -----------
    data : np.array or pd.Series
        Data to scale
    scaler : MinMaxScaler or None
        If provided, use existing scaler (for test data)
        If None, create and fit new scaler (for train data)
    feature_range : tuple
        Range to scale data to (default: (0, 1))
    
    Returns:
    --------
    tuple
        (scaled_data, scaler)
    """
    # Reshape if 1D array
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    # Create new scaler or use existing
    if scaler is None:
        scaler = MinMaxScaler(feature_range=feature_range)
        scaled_data = scaler.fit_transform(data)
    else:
        scaled_data = scaler.transform(data)
    
    return scaled_data, scaler


def create_sliding_window(data, target=None, window_size=60):
    """
    Convert time series to supervised learning format using sliding window
    
    Parameters:
    -----------
    data : np.array
        Input features (scaled)
    target : np.array or None
        Target values (scaled). If None, assumes data contains target.
    window_size : int
        Number of previous time steps to use as input features (lookback period)
    
    Returns:
    --------
    tuple
        (X, y) where X is input features and y is target values
        X shape: (samples, window_size, features)
        y shape: (samples,)
    """
    X, y = [], []
    
    # If target is not provided, assume univariate and data is the target
    if target is None:
        target = data
    
    # Ensure data and target are numpy arrays
    data = np.array(data)
    target = np.array(target)
    
    for i in range(window_size, len(data)):
        # Features: previous window_size values
        X.append(data[i-window_size:i])
        # Target: current value
        y.append(target[i])
    
    X, y = np.array(X), np.array(y)
    
    # Reshape X for LSTM input if it's 2D (samples, window_size) -> (samples, window_size, 1)
    if len(X.shape) == 2:
        X = X.reshape(X.shape[0], X.shape[1], 1)
    
    print(f"Created {X.shape[0]} sequences with window size {window_size}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y


def prepare_multivariate_data(df, target_col='Close', window_size=60):
    """
    Prepare multivariate data for training (scaling + sliding window)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with features
    target_col : str
        Target column name
    window_size : int
        Window size for sliding window
        
    Returns:
    --------
    tuple
        (X, y, feature_scaler, target_scaler)
    """
    # Get feature columns
    feature_cols = get_feature_columns()
    
    # Ensure all feature columns exist, if not fall back to univariate
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}. Falling back to univariate.")
        feature_cols = [target_col]
        
    # Extract features and target
    features = df[feature_cols].values
    target = df[target_col].values.reshape(-1, 1)
    
    # Scale features
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = feature_scaler.fit_transform(features)
    
    # Scale target independently (for easier inversion)
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_target = target_scaler.fit_transform(target)
    
    # Create sliding window
    X, y = create_sliding_window(scaled_features, scaled_target, window_size)
    
    return X, y, feature_scaler, target_scaler


def time_aware_split(X, y, train_ratio=0.8):
    """
    Split time series data into train and test sets
    IMPORTANT: Does NOT shuffle data to preserve temporal order
    
    Parameters:
    -----------
    X : np.array
        Input features
    y : np.array
        Target values
    train_ratio : float
        Proportion of data to use for training (default: 0.8)
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    # Calculate split index
    split_idx = int(len(X) * train_ratio)
    
    # Split data chronologically
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Train/Test ratio: {train_ratio:.0%}/{1-train_ratio:.0%}")
    
    return X_train, X_test, y_train, y_test


def inverse_scale_data(scaled_data, scaler):
    """
    Convert scaled data back to original scale
    
    Parameters:
    -----------
    scaled_data : np.array
        Scaled data
    scaler : MinMaxScaler
        Fitted scaler used for scaling
    
    Returns:
    --------
    np.array
        Data in original scale
    """
    # Reshape if necessary
    if len(scaled_data.shape) == 1:
        scaled_data = scaled_data.reshape(-1, 1)
    
    return scaler.inverse_transform(scaled_data)


def save_scaler(scaler, filepath='models/scaler.pkl'):
    """
    Save fitted scaler to disk
    
    Parameters:
    -----------
    scaler : MinMaxScaler
        Fitted scaler to save
    filepath : str
        Path to save the scaler
    """
    with open(filepath, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {filepath}")


def load_scaler(filepath='models/scaler.pkl'):
    """
    Load fitted scaler from disk
    
    Parameters:
    -----------
    filepath : str
        Path to saved scaler
    
    Returns:
    --------
    MinMaxScaler
        Loaded scaler
    """
    with open(filepath, 'rb') as f:
        scaler = pickle.load(f)
    print(f"Scaler loaded from {filepath}")
    return scaler


if __name__ == "__main__":
    # Example usage
    print("=== Testing Preprocessing Module ===\n")
    
    # Create sample data
    sample_data = np.array([100, 105, 103, 108, 112, 115, 113, 118, 120, 122]).reshape(-1, 1)
    print(f"Original data shape: {sample_data.shape}")
    print(f"Original data: {sample_data.flatten()}")
    
    # Scale data
    scaled_data, scaler = scale_data(sample_data)
    print(f"\nScaled data: {scaled_data.flatten()}")
    
    # Create sequences
    X, y = create_sliding_window(scaled_data, window_size=3)
    print(f"\nSequences created:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = time_aware_split(X, y, train_ratio=0.7)
