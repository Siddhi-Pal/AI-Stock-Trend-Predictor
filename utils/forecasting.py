"""
Forecasting Module
Implements baseline and LSTM-based forecasting models
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import pandas as pd
from .technical_indicators import add_all_indicators, get_feature_columns


class NaiveForecaster:
    """
    Naive baseline forecaster
    Predicts next value as the last observed value
    """
    
    def __init__(self):
        self.last_value = None
    
    def fit(self, data):
        """
        Fit the naive model by storing the last observed value
        
        Parameters:
        -----------
        data : np.array
            Training data
        """
        self.last_value = data[-1]
        print(f"Naive forecaster fitted. Last value: {self.last_value}")
    
    def predict(self, steps=1):
        """
        Predict future values (always returns last observed value)
        
        Parameters:
        -----------
        steps : int
            Number of steps to predict
        
        Returns:
        --------
        np.array
            Predictions (all equal to last observed value)
        """
        return np.array([self.last_value] * steps)


def build_lstm_model(window_size, n_features=1, lstm_units=[50, 50], dropout=0.2, learning_rate=0.001):
    """
    Build LSTM model architecture for time series forecasting
    
    Parameters:
    -----------
    window_size : int
        Number of previous time steps (lookback period)
    n_features : int
        Number of features per time step (default: 1 for univariate)
    lstm_units : list
        Number of units in each LSTM layer
    dropout : float
        Dropout rate for regularization
    learning_rate : float
        Learning rate for Adam optimizer
    
    Returns:
    --------
    keras.Model
        Compiled LSTM model
    
    Architecture:
    ------------
    Input → LSTM(50) → Dropout(0.2) → LSTM(50) → Dropout(0.2) → Dense(1)
    """
    model = Sequential()
    
    # First LSTM layer with return sequences
    model.add(LSTM(units=lstm_units[0], 
                   return_sequences=True if len(lstm_units) > 1 else False,
                   input_shape=(window_size, n_features)))
    model.add(Dropout(dropout))
    
    # Additional LSTM layers
    for i in range(1, len(lstm_units)):
        return_seq = i < len(lstm_units) - 1
        model.add(LSTM(units=lstm_units[i], return_sequences=return_seq))
        model.add(Dropout(dropout))
    
    # Output layer
    model.add(Dense(units=1))
    
    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    
    print("\n=== LSTM Model Architecture ===")
    model.summary()
    
    return model


def build_enhanced_lstm_model(window_size, n_features=8, learning_rate=0.001):
    """
    Build Balanced LSTM Model (Optimized for Stability)
    
    Reverted to a simpler architecture similar to the original successful model,
    but adapted for multivariate input.
    
    Architecture:
    ------------
    Input (window_size, n_features)
        ↓
    LSTM(50, return_sequences=True) → Dropout(0.2)
        ↓
    LSTM(50, return_sequences=False) → Dropout(0.2)
        ↓
    Dense(25) → Dropout(0.1)
        ↓
    Dense(1) → Output
    
    Parameters:
    -----------
    window_size : int
        Number of previous time steps
    n_features : int
        Number of features
    learning_rate : float
        Learning rate
    
    Returns:
    --------
    keras.Model
        Compiled model
    """
    model = Sequential(name='Balanced_LSTM')
    
    # First LSTM layer
    model.add(LSTM(50, return_sequences=True, name='LSTM_1', 
                   input_shape=(window_size, n_features)))
    model.add(Dropout(0.2, name='Dropout_1'))
    
    # Second LSTM layer
    model.add(LSTM(50, return_sequences=False, name='LSTM_2'))
    model.add(Dropout(0.2, name='Dropout_2'))
    
    # Dense layer for feature extraction
    model.add(Dense(25, activation='relu', name='Dense_Features'))
    model.add(Dropout(0.1, name='Dropout_3'))
    
    # Output layer
    model.add(Dense(1, name='Output'))
    
    # Compile with Adam optimizer
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mae', 'mape']
    )
    
    print("\n" + "="*70)
    print("BALANCED LSTM MODEL (reverting to proven architecture)")
    print("="*70)
    model.summary()
    return model


def train_lstm(model, X_train, y_train, X_val, y_val, 
               epochs=100, batch_size=32, patience=10, 
               model_path='models/lstm_model.keras'):
    """
    Train LSTM model with early stopping and model checkpointing
    
    Parameters:
    -----------
    model : keras.Model
        LSTM model to train
    X_train, y_train : np.array
        Training data
    X_val, y_val : np.array
        Validation data
    epochs : int
        Maximum number of training epochs
    batch_size : int
        Batch size for training
    patience : int
        Early stopping patience (epochs without improvement)
    model_path : str
        Path to save best model
    
    Returns:
    --------
    keras.callbacks.History
        Training history
    """
    # Ensure model directory exists
    import os
    if os.path.dirname(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Early stopping callback
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    
    # Model checkpoint callback
    checkpoint = ModelCheckpoint(
        filepath=model_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Learning rate reduction callback (NEW - improves convergence)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,              # Reduce LR by 50%
        patience=5,              # After 5 epochs without improvement
        min_lr=0.00001,          # Minimum learning rate
        verbose=1
    )
    
    print(f"\n=== Training LSTM Model ===")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}")
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, checkpoint, reduce_lr],  # Added reduce_lr
        verbose=1
    )
    
    print(f"\nModel saved to {model_path}")
    
    return history


def recursive_forecast(model, last_window, n_steps, scaler):
    """
    Generate multi-step forecast using recursive prediction
    
    Parameters:
    -----------
    model : keras.Model
        Trained LSTM model
    last_window : np.array
        Last window_size values to start prediction
    n_steps : int
        Number of future steps to predict
    scaler : MinMaxScaler
        Fitted scaler to inverse transform predictions
    
    Returns:
    --------
    np.array
        Future predictions in original scale
    
    Process:
    --------
    1. Predict next value using last window
    2. Add prediction to window
    3. Remove oldest value from window
    4. Repeat for n_steps
    """
    # Initialize predictions list
    predictions = []
    
    # Current window for prediction (copy to avoid modifying original)
    current_window = last_window.copy()
    
    # Generate predictions recursively
    for step in range(n_steps):
        # Reshape for model input: (1, window_size, 1)
        current_input = current_window.reshape(1, current_window.shape[0], 1)
        
        # Predict next value
        next_pred = model.predict(current_input, verbose=0)[0, 0]
        
        # Store prediction
        predictions.append(next_pred)
        
        # Update window: remove oldest, add newest
        current_window = np.append(current_window[1:], next_pred)
    
    # Convert to numpy array and reshape
    predictions = np.array(predictions).reshape(-1, 1)
    
    # Inverse transform to original scale
    predictions_original = scaler.inverse_transform(predictions)
    
    return predictions_original.flatten()


def multivariate_recursive_forecast(model, df, n_steps, feature_scaler, target_scaler, window_size=60, is_return_target=False):
    """
    Generate multivariate forecast by iteratively predicting and updating indicators
    
    Process:
    1. Append prediction to history
    2. Recalculate indicators (MA, RSI, etc.)
    3. Scale features
    4. Predict next step
    5. Repeat
    
    Parameters:
    -----------
    model : keras.Model
        Trained LSTM model
    df : pd.DataFrame
        Historical data (Open, High, Low, Close, Volume, Date)
    n_steps : int
        Number of steps to predict
    feature_scaler : MinMaxScaler
        Scaler for input features
    target_scaler : MinMaxScaler
        Scaler for target output (Close)
    window_size : int
        Lookback window size
    is_return_target : bool
        If True, model predicts Returns instead of Price
        
    Returns:
    --------
    np.array
        Predicted values (original scale)
    """
    import pandas as pd
    import numpy as np
    
    # Make a copy to avoid modifying original
    df_forecast = df.copy()
    
    predictions = []
    
    # Get last date
    last_date = df_forecast['Date'].iloc[-1]
    
    # Generate future business dates
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_steps)
    
    print(f"Generating forecast for {n_steps} days (Returns Target: {is_return_target})...")
    
    for i, future_date in enumerate(future_dates):
        # 1. Update indicators on current history
        df_indicators = add_all_indicators(df_forecast)
        
        # 2. Get feature columns
        feature_cols = get_feature_columns()
        
        # 3. Get last window of data
        last_window_data = df_indicators.iloc[-window_size:][feature_cols].values
        
        # 4. Scale features
        # Note: transform expects (n_samples, n_features)
        last_window_scaled = feature_scaler.transform(last_window_data)
        
        # 5. Reshape for LSTM: (1, window_size, n_features)
        input_seq = last_window_scaled.reshape(1, window_size, -1)
        
        # 6. Predict next scaled value
        pred_scaled = model.predict(input_seq, verbose=0)
        
        # 7. Inverse scale to get raw prediction (Price or Return)
        pred_raw = target_scaler.inverse_transform(pred_scaled)[0, 0]
        
        # 8. Handle Return-to-Price conversion if needed
        if is_return_target:
            # pred_raw is a Return (e.g., 0.01)
            # Clip return to avoid explosion
            pred_raw = np.clip(pred_raw, -0.2, 0.2)
            
            # Reconstruct Price
            last_close = df_forecast['Close'].iloc[-1]
            pred_close = last_close * (1 + pred_raw)
        else:
            # pred_raw is Price
            pred_close = pred_raw
        
        # 9. Store prediction (always store PRICE)
        predictions.append(pred_close)
        
        # 10. Append new row to dataframe for next iteration
        # We assume Open/High/Low are same as Close for simplicity in forecast
        # Volume is carried forward from last known value
        new_row = pd.DataFrame([{
            'Date': future_date,
            'Close': pred_close,
            'Open': pred_close,
            'High': pred_close,
            'Low': pred_close,
            'Volume': df_forecast['Volume'].iloc[-1]
        }])
        
        df_forecast = pd.concat([df_forecast, new_row], ignore_index=True)
        
    return np.array(predictions)


def load_trained_model(model_path='models/lstm_model.keras'):
    """
    Load trained LSTM model from disk
    
    Parameters:
    -----------
    model_path : str
        Path to saved model
    
    Returns:
    --------
    keras.Model
        Loaded model
    """
    model = load_model(model_path)
    print(f"Model loaded from {model_path}")
    return model


if __name__ == "__main__":
    # Example usage
    print("=== Testing Forecasting Module ===\n")
    
    # Test Naive Forecaster
    print("--- Naive Forecaster ---")
    sample_data = np.array([100, 105, 103, 108, 112])
    naive_model = NaiveForecaster()
    naive_model.fit(sample_data)
    naive_preds = naive_model.predict(steps=5)
    print(f"Naive predictions: {naive_preds}\n")
    
    # Test LSTM Model Building
    print("--- LSTM Model ---")
    model = build_lstm_model(window_size=60, lstm_units=[50, 50], dropout=0.2)
