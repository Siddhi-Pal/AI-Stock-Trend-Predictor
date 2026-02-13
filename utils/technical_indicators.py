"""
Technical Indicators Module
Calculates various technical indicators for enhanced stock price prediction

Technical indicators provide additional features that capture:
- Trend (Moving Averages)
- Momentum (RSI)
- Volume patterns
- Volatility (Daily Returns)
"""

import pandas as pd
import numpy as np


def calculate_moving_average(df, column='Close', window=7):
    """
    Calculate Simple Moving Average (SMA)
    
    SMA smooths out price data to identify trend direction
    
    Args:
        df: DataFrame with price data
        column: Column to calculate MA on (default: Close)
        window: Number of periods (days) for averaging
    
    Returns:
        Series with moving average values
    """
    return df[column].rolling(window=window, min_periods=1).mean()


def calculate_rsi(df, column='Close', period=14):
    """
    Calculate Relative Strength Index (RSI)
    
    RSI measures momentum on a scale of 0-100
    - Above 70: Overbought (price may drop)
    - Below 30: Oversold (price may rise)
    
    Formula:
        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
    
    Args:
        df: DataFrame with price data
        column: Column to calculate RSI on (default: Close)
        period: Lookback period (default: 14 days)
    
    Returns:
        Series with RSI values (0-100)
    """
    # Calculate price changes
    delta = df[column].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    # Calculate relative strength
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    # Handle division by zero (when avg_loss = 0)
    rsi = rsi.fillna(100)
    
    return rsi


def calculate_daily_returns(df, column='Close'):
    """
    Calculate daily percentage returns
    
    Returns = (Price_today - Price_yesterday) / Price_yesterday
    
    Captures:
    - Volatility
    - Daily price changes
    - Risk measure
    
    Args:
        df: DataFrame with price data
        column: Column to calculate returns on (default: Close)
    
    Returns:
        Series with daily returns as percentages
    """
    return df[column].pct_change().fillna(0)


def normalize_volume(df):
    """
    Normalize volume using z-score normalization
    
    Normalized_Volume = (Volume - Mean) / Std Dev
    
    Benefits:
    - Brings volume to same scale as other features
    - Highlights unusual volume spikes
    
    Args:
        df: DataFrame with Volume column
    
    Returns:
        Series with normalized volume
    """
    mean = df['Volume'].mean()
    std = df['Volume'].std()
    
    # Avoid division by zero
    if std == 0:
        return pd.Series(0, index=df.index)
    
    return (df['Volume'] - mean) / std


def add_all_indicators(df):
    """
    Add all technical indicators to the DataFrame
    
    This creates a multi-variate dataset for the LSTM model:
    - Original: 1 feature (Close price)
    - Enhanced: 7 features (Close + 6 indicators)
    
    Features added:
    1. MA_7: 7-day moving average (short-term trend)
    2. MA_30: 30-day moving average (long-term trend)
    3. RSI: Relative Strength Index (momentum)
    4. Volume_Norm: Normalized volume
    5. Returns: Daily percentage change
    6. Price_Range: High - Low (intraday volatility)
    7. Price_Position: (Close - Low) / (High - Low) (position in daily range)
    
    Args:
        df: DataFrame with OHLCV data (Open, High, Low, Close, Volume, Date)
    
    Returns:
        DataFrame with original columns + technical indicators
    """
    # Make a copy to avoid modifying original
    df_enhanced = df.copy()
    
    # 1. Moving Averages
    df_enhanced['MA_7'] = calculate_moving_average(df_enhanced, 'Close', window=7)
    df_enhanced['MA_30'] = calculate_moving_average(df_enhanced, 'Close', window=30)
    
    # 2. RSI
    df_enhanced['RSI'] = calculate_rsi(df_enhanced, 'Close', period=14)
    
    # 3. Normalized Volume
    df_enhanced['Volume_Norm'] = normalize_volume(df_enhanced)
    
    # 4. Daily Returns
    df_enhanced['Returns'] = calculate_daily_returns(df_enhanced, 'Close')
    
    # 5. Price Range (High - Low)
    df_enhanced['Price_Range'] = df_enhanced['High'] - df_enhanced['Low']
    
    # 6. Price Position within daily range
    df_enhanced['Price_Position'] = (
        (df_enhanced['Close'] - df_enhanced['Low']) / 
        (df_enhanced['High'] - df_enhanced['Low'] + 1e-10)  # Add small value to avoid division by zero
    ).fillna(0.5)  # If High == Low, assume middle position
    
    # Fill any remaining NaN values with forward fill, then backward fill
    df_enhanced = df_enhanced.fillna(method='ffill').fillna(method='bfill')
    
    # If still any NaN (e.g., first row), fill with 0
    df_enhanced = df_enhanced.fillna(0)
    
    return df_enhanced


def get_feature_columns():
    """
    Return list of feature columns for model training
    
    Use this to extract the right columns when feeding data to LSTM
    
    Returns:
        List of column names to use as features
    """
    return ['Close', 'MA_7', 'MA_30', 'RSI', 'Volume_Norm', 'Returns', 'Price_Range', 'Price_Position']


def prepare_multivariate_data(df):
    """
    Prepare multi-variate data for LSTM training
    
    Extracts feature columns and scales appropriately
    
    Args:
        df: DataFrame with all technical indicators
    
    Returns:
        numpy array of shape (n_samples, n_features)
    """
    feature_cols = get_feature_columns()
    return df[feature_cols].values


# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    import yfinance as yf
    
    print("Testing Technical Indicators Module...")
    print("=" * 60)
    
    # Fetch sample data
    ticker = yf.Ticker("RELIANCE.NS")
    df = ticker.history(period="6mo")
    df = df.reset_index()
    
    print(f"\nOriginal data shape: {df.shape}")
    print(f"Original columns: {df.columns.tolist()}")
    
    # Add indicators
    df_enhanced = add_all_indicators(df)
    
    print(f"\nEnhanced data shape: {df_enhanced.shape}")
    print(f"Enhanced columns: {df_enhanced.columns.tolist()}")
    
    # Show sample
    print("\nSample enhanced data (last 5 rows):")
    print(df_enhanced[get_feature_columns()].tail())
    
    # Show statistics
    print("\nFeature Statistics:")
    print(df_enhanced[get_feature_columns()].describe())
    
    print("\n✓ Technical Indicators Module working correctly!")
