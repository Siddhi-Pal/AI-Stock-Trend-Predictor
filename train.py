"""
Training Script
Orchestrates end-to-end model training pipeline for stock price forecasting
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from utils.data_fetcher import fetch_stock_data, save_stock_data
from utils.preprocessing import (
    load_and_prepare_data, prepare_multivariate_data, 
    time_aware_split, save_scaler, inverse_scale_data
)
from utils.forecasting import (
    NaiveForecaster, build_enhanced_lstm_model, train_lstm
)
from utils.evaluation import (
    calculate_metrics, compare_with_baseline, print_metrics,
    print_comparison, plot_predictions, plot_training_history
)

# Set random seeds for reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)


def main(stock_symbol='RELIANCE.NS', window_size=60, epochs=100, batch_size=32):
    """
    Main training pipeline
    
    Parameters:
    -----------
    stock_symbol : str
        Stock ticker to train on
    window_size : int
        Sliding window size (lookback period)
    epochs : int
        Maximum training epochs
    batch_size : int
        Training batch size
    """
    
    print("\n" + "="*80)
    print("TIME SERIES FORECASTING - PRODUCTION TRAINING PIPELINE (ENHANCED)")
    print("="*80)
    print(f"\nStock Symbol: {stock_symbol}")
    print(f"Window Size: {window_size} days")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print("="*80 + "\n")
    
    # ========================
    # STEP 1: Fetch Stock Data
    # ========================
    print("\n[STEP 1/7] Fetching Stock Data (Max History)...")
    print("-" * 80)
    
    # Fetch maximum available data for better training
    stock_data = fetch_stock_data(stock_symbol, period='max', interval='1d')
    
    if stock_data is None:
        print("Error: Failed to fetch stock data. Exiting.")
        return
    
    # Save stock data
    data_path = save_stock_data(stock_symbol, stock_data)
    
    # ========================
    # STEP 2: Preprocess Data
    # ========================
    print("\n[STEP 2/7] Preprocessing Data & Adding Technical Indicators...")
    print("-" * 80)
    
    # Load and prepare with technical indicators
    df = load_and_prepare_data(data_path, target_column='Close', use_technical_indicators=True)
    
    print(f"Features: {df.columns.tolist()}")
    
    # Prepare multivariate data - PREDICTING RETURNS NOT PRICE
    # Returns X (sequences), y (targets), and fitted scalers
    # Note: We use 'Returns' as target which is stationary (-0.1 to 0.1)
    X, y, feature_scaler, target_scaler = prepare_multivariate_data(
        df, target_col='Returns', window_size=window_size
    )
    
    # Time-aware split (80% train, 20% test)
    X_train, X_test, y_train, y_test = time_aware_split(X, y, train_ratio=0.8)
    
    # Further split training data for validation (80% train, 20% validation)
    split_idx = int(len(X_train) * 0.8)
    X_train_final = X_train[:split_idx]
    X_val = X_train[split_idx:]
    y_train_final = y_train[:split_idx]
    y_val = y_train[split_idx:]
    
    print(f"Final training set: {len(X_train_final)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Save scalers for later use
    # Clean symbol for filenames
    clean_symbol = stock_symbol.replace('^', '').replace('.', '_')
    
    feature_scaler_path = f'models/{clean_symbol}_feature_scaler.pkl'
    target_scaler_path = f'models/{clean_symbol}_target_scaler.pkl'
    
    save_scaler(feature_scaler, feature_scaler_path)
    save_scaler(target_scaler, target_scaler_path)
    print(f"Saved scalers to {feature_scaler_path} and {target_scaler_path}")
    
    # Also save as 'latest' for backward compatibility or default use
    save_scaler(feature_scaler, 'models/feature_scaler.pkl')
    save_scaler(target_scaler, 'models/target_scaler.pkl')
    
    # ========================
    # STEP 3: Baseline Model
    # ========================
    print("\n[STEP 3/7] Training Baseline Model...")
    print("-" * 80)
    
    # Train naive forecaster on returns
    naive_model = NaiveForecaster()
    
    # Inverse scale to get actual returns
    train_returns = target_scaler.inverse_transform(y_train.reshape(-1, 1))
    naive_model.fit(train_returns)
    
    # Baseline predictions (Returns)
    baseline_pred_returns = naive_model.predict(steps=len(y_test))
    
    # RECONSTRUCT PRICES for Evaluation
    # We need the price just before the test set starts
    split_index = int(len(df) * 0.8)
    # The first text index is split_index + window_size
    # Wait, create_sliding_window cuts off the first window_size
    # Let's align indices carefully
    
    # Get actual prices corresponding to test set
    # The test set targets correspond to df['Returns'][split_index+window_size:]
    # So we need prices from that same range
    test_indices = range(len(df) - len(y_test), len(df))
    test_prices_original = df['Close'].iloc[test_indices].values
    
    # To reconstruct, we need Price_{t-1}. 
    # For the first prediction, we need Price_{start-1}
    last_known_price = df['Close'].iloc[test_indices[0]-1]
    
    # Reconstruct Baseline Prices
    baseline_prices = [last_known_price]
    for r in baseline_pred_returns:
        next_p = baseline_prices[-1] * (1 + r)
        baseline_prices.append(next_p)
    baseline_predictions = np.array(baseline_prices[1:])
    
    # Calculate baseline metrics on PRICES
    baseline_metrics = calculate_metrics(test_prices_original, baseline_predictions)
    print_metrics(baseline_metrics, "Baseline Model (Price Reconstructed)")
    
    # ========================
    # STEP 4: Build Enhanced LSTM Model
    # ========================
    print("\n[STEP 4/7] Building Enhanced LSTM Model (Production Ready)...")
    print("-" * 80)
    
    # Determine number of features from input data
    n_features = X.shape[2]
    print(f"Input features per time step: {n_features}")
    
    lstm_model = build_enhanced_lstm_model(
        window_size=window_size,
        n_features=n_features,
        learning_rate=0.001
    )
    
    # ========================
    # STEP 5: Train LSTM Model
    # ========================
    print("\n[STEP 5/7] Training LSTM Model (Target: Returns)...")
    print("-" * 80)
    
    history = train_lstm(
        model=lstm_model,
        X_train=X_train_final,
        y_train=y_train_final,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        patience=15,
        model_path='models/enhanced_lstm.keras'
    )
    
    # Plot training history
    plot_training_history(history, save_path='data/plots/training_history.png')
    
    # ========================
    # STEP 6: Evaluate LSTM Model
    # ========================
    print("\n[STEP 6/7] Evaluating LSTM Model...")
    print("-" * 80)
    
    # Make predictions (Returns)
    lstm_pred_scaled = lstm_model.predict(X_test, verbose=0)
    lstm_pred_returns = target_scaler.inverse_transform(lstm_pred_scaled).flatten()
    
    # Reconstruct LSTM Prices
    lstm_prices = [last_known_price]
    for r in lstm_pred_returns:
        # Constrain return to realistic bounds (-20% to +20%) to avoid explosion
        r = np.clip(r, -0.2, 0.2)
        next_p = lstm_prices[-1] * (1 + r)
        lstm_prices.append(next_p)
    lstm_predictions = np.array(lstm_prices[1:])
    
    # Calculate LSTM metrics on PRICES
    lstm_metrics = calculate_metrics(test_prices_original, lstm_predictions)
    print_metrics(lstm_metrics, "Enhanced LSTM Model (Price Reconstructed)")
    
    # ========================
    # STEP 7: Compare Models
    # ========================
    print("\n[STEP 7/7] Comparing Models...")
    print("-" * 80)
    
    comparison = compare_with_baseline(lstm_metrics, baseline_metrics)
    print_comparison(comparison)
    
    # Plot predictions
    plot_predictions(
        y_true=test_prices_original,
        y_pred=lstm_predictions,
        title=f"{stock_symbol} - LSTM Predictions (Reconstructed from Returns)",
        save_path='data/plots/lstm_predictions.png'
    )
    
    # Plot baseline predictions too
    plot_predictions(
        y_true=test_prices_original,
        y_pred=baseline_predictions,
        title=f"{stock_symbol} - Baseline Predictions (Reconstructed)",
        save_path='data/plots/baseline_predictions.png'
    )
    
    # ========================
    # Summary
    # ========================
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\n[OK] Model saved to: models/enhanced_lstm.keras")
    print(f"[OK] Feature Scaler saved to: models/feature_scaler.pkl")
    print(f"[OK] Target Scaler saved to: models/target_scaler.pkl")
    print(f"[OK] Plots saved to: data/plots/")
    print(f"\nLSTM RMSE: {lstm_metrics['RMSE']:.4f}")
    print(f"Baseline RMSE: {baseline_metrics['RMSE']:.4f}")
    print(f"Improvement: {comparison['RMSE']['improvement_%']:.2f}%")
    print("\n" + "="*80 + "\n")
    
    return lstm_model, feature_scaler, target_scaler, history


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Enhanced LSTM model for stock price forecasting')
    parser.add_argument('--stock', type=str, default='RELIANCE.NS',
                        help='Stock symbol (default: RELIANCE.NS)')
    parser.add_argument('--window', type=int, default=60,
                        help='Window size (default: 60)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs (default: 100)')
    parser.add_argument('--batch', type=int, default=32,
                        help='Batch size (default: 32)')
    
    args = parser.parse_args()
    
    # Run training
    main(
        stock_symbol=args.stock,
        window_size=args.window,
        epochs=args.epochs,
        batch_size=args.batch
    )
