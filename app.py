"""
Streamlit App for Time Series Forecasting
AI-Powered Stock Forecasting System for Indian & Global Markets
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from utils.data_fetcher import (
    get_indian_stock_symbols, get_global_stock_symbols,
    fetch_stock_data, get_stock_info
)
from utils.preprocessing import (
    load_and_prepare_data, scale_data, create_sliding_window,
    load_scaler, inverse_scale_data
)
from utils.forecasting import (
    NaiveForecaster, load_trained_model, multivariate_recursive_forecast, build_enhanced_lstm_model
)
from utils.evaluation import calculate_metrics
from utils.technical_indicators import add_all_indicators, get_feature_columns

# Page configuration
st.set_page_config(
    page_title="AI Stock Forecasting System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .disclaimer {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
        margin-top: 1rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Title and header
st.markdown('<div class="main-header">📈 AI-Powered Stock Forecasting System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time predictions for Indian & Global markets using Enhanced LSTM Neural Networks</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Get stock symbols
indian_stocks = get_indian_stock_symbols()
global_stocks = get_global_stock_symbols()
all_stocks = {**indian_stocks, **global_stocks}

# Stock selection

# Stock selection
market_type = st.sidebar.radio("Select Market", ["🇮🇳 Indian Stocks", "🌍 Global Stocks", "🔍 Custom Search"])

if market_type == "🇮🇳 Indian Stocks":
    stock_options = indian_stocks
    st.sidebar.info("💡 Indian stocks from NSE/BSE")
    currency_symbol = "₹"
    stock_display = {f"{symbol} - {name}": symbol for symbol, name in stock_options.items()}
    selected_display = st.sidebar.selectbox("Select Stock", options=list(stock_display.keys()))
    selected_stock = stock_display[selected_display]
    stock_name = stock_options[selected_stock]

elif market_type == "🌍 Global Stocks":
    stock_options = global_stocks
    st.sidebar.info("💡 Major global stocks and indices")
    currency_symbol = "$"
    stock_display = {f"{symbol} - {name}": symbol for symbol, name in stock_options.items()}
    selected_display = st.sidebar.selectbox("Select Stock", options=list(stock_display.keys()))
    selected_stock = stock_display[selected_display]
    stock_name = stock_options[selected_stock]

else: # Custom Search
    st.sidebar.markdown("### 🔍 Smart Stock Search")
    
    # Exchange Selector
    exchange = st.sidebar.radio("Select Exchange", ["🇺🇸 Global / USA", "🇮🇳 NSE (Natl Stock Exch)", "🇮🇳 BSE (Bombay Stock Exch)"])
    
    if exchange == "🇺🇸 Global / USA":
        user_input = st.sidebar.text_input("Enter Symbol", value="AAPL", help="e.g. AAPL, TSLA, BTC-USD").upper()
        selected_stock = user_input
        currency_symbol = "$"
        stock_name = selected_stock
        
    elif exchange == "🇮🇳 NSE (Natl Stock Exch)":
        user_input = st.sidebar.text_input("Enter Stock Name", value="RELIANCE", help="Type name only (e.g. TATASTEEL)").upper()
        # Auto-append suffix if missing
        if not user_input.endswith(".NS"):
            selected_stock = f"{user_input}.NS"
        else:
            selected_stock = user_input
        currency_symbol = "₹"
        stock_name = f"{user_input} (NSE)"
        
    else: # BSE
        user_input = st.sidebar.text_input("Enter Stock Name", value="SARLAPOLY", help="Type name only (e.g. SARLAPOLY)").upper()
        # Auto-append suffix if missing
        if not user_input.endswith(".BO"):
            selected_stock = f"{user_input}.BO"
        else:
            selected_stock = user_input
        currency_symbol = "₹"
        stock_name = f"{user_input} (BSE)"

# Clean symbol for filenames
clean_symbol = selected_stock.replace('^', '').replace('.', '_')

# Forecast configuration
st.sidebar.subheader("🔮 Forecast Settings")
forecast_horizon = st.sidebar.slider("Forecast Horizon (days)", 7, 90, 30)
window_size = st.sidebar.slider("Window Size (days)", 30, 120, 60)

# Model Status & Training
st.sidebar.markdown("---")
st.sidebar.subheader("🧠 AI Model Status")

# Check for specific model
specific_model_path = f'models/{clean_symbol}_lstm.keras'
generic_model_path = 'models/enhanced_lstm.keras'
using_specific_model = os.path.exists(specific_model_path)

if using_specific_model:
    st.sidebar.success(f"✅ Trained Model Found for {selected_stock}")
    active_model_path = specific_model_path
else:
    st.sidebar.warning(f"⚠️ No specific model for {selected_stock}")
    st.sidebar.info(f"Using Generic Model (Result may vary)")
    active_model_path = generic_model_path

# Model selection
model_type = st.sidebar.selectbox("Model Type", ["Enhanced LSTM (Multivariate)", "Naive Baseline"])

# Train Button
if st.sidebar.button("🏋️ Train New Model for " + selected_stock):
    with st.spinner(f"Training specialized model for {selected_stock}... (This may take a minute)"):
        import subprocess
        import sys
        
        # Construct command
        cmd = [sys.executable, "train.py", "--stock", selected_stock, "--epochs", "50"]
        
        # Run training
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            st.sidebar.success("Training Complete!")
            # Rename generic outputs to specific locally to keep them safe
            # train.py currently saves to 'enhanced_lstm.keras'
            # We need to rename it to specific_model_path
            # Actually, train.py overwrites 'enhanced_lstm.keras'
            # So we rename it NOW
            import shutil
            try:
                shutil.copy('models/enhanced_lstm.keras', specific_model_path)
                st.sidebar.success(f"Model saved: {specific_model_path}")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error moving model: {e}")
        else:
            st.sidebar.error("Training Failed")
            st.sidebar.code(stderr)

# Download data button
if st.sidebar.button("📥 Fetch Latest Data", type="primary"):
    with st.spinner(f"Fetching latest data for {selected_stock}..."):
        data = fetch_stock_data(selected_stock, period='max', interval='1d')
        if data is not None:
            # Save to data/stocks
            data_path = f"data/stocks/{clean_symbol}.csv"
            os.makedirs("data/stocks", exist_ok=True)
            data.to_csv(data_path, index=False)
            st.sidebar.success(f"✓ Data fetched successfully!")
        else:
            st.sidebar.error("Failed to fetch data")

# Main content
st.markdown("---")

# Load data and make predictions
data_path = f"data/stocks/{clean_symbol}.csv"  # Use clean_symbol here

# Auto-fetch if data doesn't exist for custom ticker
if not os.path.exists(data_path):
    with st.spinner(f"First time seeing {selected_stock}. Auto-fetching data..."):
         data = fetch_stock_data(selected_stock, period='max', interval='1d')
         if data is not None:
            os.makedirs("data/stocks", exist_ok=True)
            data.to_csv(data_path, index=False)
         else:
            st.error(f"Could not find data for symbol '{selected_stock}'. Please check the ticker.")
            st.stop()

try:
    # Load data with standard preparation
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Display stock info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Stock", selected_stock)
    with col2:
        st.metric("Latest Price", f"{currency_symbol}{df['Close'].iloc[-1]:.2f}")
    with col3:
        st.metric("52W High", f"{currency_symbol}{df['Close'].max():.2f}")
    with col4:
        st.metric("52W Low", f"{currency_symbol}{df['Close'].min():.2f}")
    
    st.markdown("---")
    
    # Historical data visualization
    st.subheader("📊 Historical Stock Price (Last 5 Years)")
    
    # Create candlestick chart
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=('Price', 'Volume')
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Volume
    fig.add_trace(
        go.Bar(
            x=df['Date'],
            y=df['Volume'],
            name='Volume',
            marker_color='#1f77b4'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        title_text=f"{stock_name} ({selected_stock}) Historical Data",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Make predictions
    st.markdown("---")
    st.subheader("🤖 AI Price Prediction")
    
    with st.spinner("Generating predictions..."):
        # 1. Prepare Data with Indicators
        df_prepared = add_all_indicators(df)
        
        # 2. Scale Features
        feature_cols = get_feature_columns()
        
        # Determine which scaler to load
        # If we are using a specific model, we MUST use its specific scaler
        # If we are using generic model, we use generic scaler
        
        specific_feature_scaler_path = f'models/{clean_symbol}_feature_scaler.pkl'
        specific_target_scaler_path = f'models/{clean_symbol}_target_scaler.pkl'
        
        # Check if specific scalers exist (Preferred)
        if os.path.exists(specific_feature_scaler_path) and os.path.exists(specific_target_scaler_path):
            feature_scaler_path = specific_feature_scaler_path
            target_scaler_path = specific_target_scaler_path
            st.sidebar.success(f"✅ Using specific scalers for {selected_stock}")
        elif os.path.exists('models/feature_scaler.pkl') and os.path.exists('models/target_scaler.pkl'):
            # Fallback to generic
            feature_scaler_path = 'models/feature_scaler.pkl'
            target_scaler_path = 'models/target_scaler.pkl'
            st.sidebar.info(f"ℹ️ Using generic scalers (Warning: Accuracy may be lower)")
        else:
            st.error("No scalers found. Please train a model first.")
            st.stop()
            
        feature_scaler = load_scaler(feature_scaler_path)
        target_scaler = load_scaler(target_scaler_path)
            
        features = df_prepared[feature_cols].values
        scaled_features = feature_scaler.transform(features)
        
        # 3. Create sequences for historical validation
        # We assume we want to predict Close price
        X, y = create_sliding_window(scaled_features, target=None, window_size=window_size)
        
        # X shape: (samples, window_size, 8)
        # y shape: (samples, 8) -> we only care about Close (index 0 usually, but depends on get_feature_columns)
        
        # Determine index of 'Close' in feature columns
        close_idx = feature_cols.index('Close')
        # Actually, create_sliding_window returns y from 'data' if passed.
        # So y contains all features.
        # But we want to inverse scale 'Close'.
        # Target scaler was fitted on 'Close' only.
        
        # Correct approach for validation data:
        # Get actual Close prices corresponding to prediction targets
        # The first prediction target is at index `window_size`
        actual_close = df_prepared['Close'].iloc[window_size:].values
        dates_for_pred = df_prepared['Date'].iloc[window_size:].values
        
        if model_type == "Enhanced LSTM (Multivariate)":
            if os.path.exists(active_model_path):
                model = load_trained_model(active_model_path)
                
                # Predict on all sequences (Model outputs Returns)
                predictions_scaled = model.predict(X, verbose=0)
                pred_returns = target_scaler.inverse_transform(predictions_scaled).flatten()
                
                # RECONSTRUCT PRICES (One-Step-Ahead)
                # Pred_Price[t] = Actual_Price[t-1] * (1 + Pred_Return[t])
                # We need Actual Prices corresponding to the step BEFORE the target
                # Targets start at df[window_size]. So "Previous" starts at df[window_size-1]
                prev_indices = range(window_size - 1, len(df_prepared) - 1)
                # Ensure length matches predictions (train_split might affect X generation in creating_sliding_window?)
                # create_sliding_window returns len(data) - window_size samples.
                # So we need len(predictions) previous prices.
                
                start_prev_idx = window_size - 1
                end_prev_idx = start_prev_idx + len(pred_returns)
                
                prev_prices = df_prepared['Close'].iloc[start_prev_idx:end_prev_idx].values
                
                # Calculate Predicted Prices
                # Clip returns to prevent explosion artifacts in visualization
                pred_returns = np.clip(pred_returns, -0.2, 0.2) 
                predictions = prev_prices * (1 + pred_returns)
                
                # Align lengths for metrics
                min_len = min(len(actual_close), len(predictions))
                actual = actual_close[:min_len]
                preds = predictions[:min_len]
                pred_dates = dates_for_pred[:min_len]
                
                # Metrics
                metrics = calculate_metrics(actual, preds)
                
            else:
                st.error(f"Model not found at {active_model_path}. Please train the model using the sidebar button.")
                st.stop()
                
        else: # Naive Baseline
            # Predict last value
            preds = actual_close[:-1] # Shift by 1
            # Prepend first known value to match length?
            # Simplest: Just shift actual - Naive is P_t = P_{t-1}
            # This is exactly what we have in prev_prices, but let's do it simply
            preds = df_prepared['Close'].iloc[window_size-1 : -1].values
            actual = actual_close
            # Align
            min_len = min(len(actual), len(preds))
            actual = actual[:min_len]
            preds = preds[:min_len]
            pred_dates = dates_for_pred[:min_len]
            
            metrics = calculate_metrics(actual, preds)

        # Display Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE (Root Mean Sq Error)", f"{currency_symbol}{metrics['RMSE']:.2f}", 
                   delta_color="inverse", help="Lower is better")
        col2.metric("MAE (Mean Absolute Error)", f"{currency_symbol}{metrics['MAE']:.2f}",
                   delta_color="inverse", help="Lower is better")
        col3.metric("MAPE (Mean Abs % Error)", f"{metrics['MAPE']:.2f}%",
                   delta_color="inverse", help="Lower is better")
        
        # Plot validation
        st.subheader("📉 Prediction Performance (Test on History)")
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=pred_dates,
            y=actual,
            name='Actual Price',
            line=dict(color='#2E86AB', width=2)
        ))
        
        fig_pred.add_trace(go.Scatter(
            x=pred_dates,
            y=preds,
            name='Predicted Price',
            line=dict(color='#C73E1D', width=1, dash='dot')
        ))
        
        fig_pred.update_layout(
            title="Model Validation: Actual vs Predicted Prices",
            xaxis_title="Date",
            yaxis_title=f"Stock Price ({currency_symbol})",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig_pred, width='stretch')
        
        # Future Forecast
        st.markdown("---")
        st.subheader(f"🔮 {forecast_horizon}-Day Future Forecast")
        
        if model_type == "Enhanced LSTM (Multivariate)":
            # Generate future predictions using multivariate recursive forecast
            # PASS is_return_target=True
            future_pred = multivariate_recursive_forecast(
                model, df_prepared, forecast_horizon, feature_scaler, target_scaler, window_size,
                is_return_target=True
            )
            
            # Generate dates
            last_date = df['Date'].iloc[-1]
            future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon)
            
            # Plot
            fig_forecast = go.Figure()
            
            # Historical tail
            tail_days = 90
            fig_forecast.add_trace(go.Scatter(
                x=df['Date'].iloc[-tail_days:],
                y=df['Close'].iloc[-tail_days:],
                name='Historical',
                line=dict(color='#2E86AB', width=2)
            ))
            
            # Forecast
            fig_forecast.add_trace(go.Scatter(
                x=future_dates,
                y=future_pred,
                name='Forecast',
                line=dict(color='#28a745', width=2, dash='dot')
            ))
            
            fig_forecast.update_layout(
                title=f"Future Price Forecast ({forecast_horizon} Days)",
                xaxis_title="Date",
                yaxis_title=f"Stock Price ({currency_symbol})",
                height=500,
                hovermode='x unified'
            )
            st.plotly_chart(fig_forecast, width='stretch')
            
            # Forecast Table
            st.subheader("📋 Forecast Data")
            forecast_df = pd.DataFrame({
                'Date': future_dates.strftime('%Y-%m-%d'),
                'Day': range(1, forecast_horizon + 1),
                f'Predicted Price ({currency_symbol})': [f"{price:.2f}" for price in future_pred]
            })
            st.dataframe(forecast_df, width='stretch', height=300)
            
            # Download button
            csv = forecast_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Forecast as CSV",
                data=csv,
                file_name=f"{selected_stock}_forecast_{forecast_horizon}days.csv",
                mime="text/csv"
            )
        else:
            st.info("Select 'Enhanced LSTM' to see future forecasts.")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.code(str(e))
    st.exception(e)

# Disclaimer
st.markdown("---")
st.markdown("""
<div class="disclaimer">
    <strong>⚠️ Investment Disclaimer:</strong><br>
    This tool is for educational and research purposes only. Stock price predictions are inherently uncertain 
    and should NOT be used as the sole basis for investment decisions. Past performance does not guarantee 
    future results. Always consult with a qualified financial advisor before making investment decisions.
    <br><br>
    <strong>Limitations:</strong>
    <ul>
        <li>Stock prices are influenced by many unpredictable factors (news, events, market sentiment)</li>
        <li>LSTM models can only learn from historical patterns</li>
        <li>Forecast accuracy decreases with longer horizons</li>
        <li>Black swan events cannot be predicted</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Built using TensorFlow, Keras, and Streamlit</p>
    <p>Data Source: Yahoo Finance via yfinance API</p>
</div>
""", unsafe_allow_html=True)
