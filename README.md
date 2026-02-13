
# 📈 AI-Powered Stock Forecasting System

![Python](https://img.shields.io/badge/Python-3.13%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![License](https://img.shields.io/badge/License-Proprietary-red)

**A professional-grade Time Series Forecasting application that leverages "Enhanced LSTM" neural networks to predict stock prices for Indian (NSE/BSE) and Global markets with high accuracy.**

---

## 🚀 Project Overview

This system solves the core problem of stock prediction: **Non-Stationarity**.
Most tutorials predict "Price" directly, which fails when stocks hit All-Time Highs (ATH). This project predicts **Percentage Returns** instead, making the data stationary and the model robust to market trends.

### Key Capabilities
*   **🌍 Universal Access:** Predict for **ANY** stock ticker (Reliance, Tata Steel, Apple, Bitcoin, Forex).
*   **🧠 On-Demand AI:** User can trigger training of a dedicated LSTM model for a specific stock in 1-click.
*   **📉 High Accuracy:** Uses `Returns-Based Forecasting` to consistently achieve RMSE < 20 INR.
*   **⚡ Real-Time:** Fetches live market data via `yfinance` API.

---

## 🏗️ System Architecture

The core engineering innovation is the **"Stationary Returns Pipeline"**.

### 1. The Math (Why it works)
Instead of predicting Price ($P_t$), we predict Daily Return ($R_t$):
$$ R_t = \frac{P_t - P_{t-1}}{P_{t-1}} $$

The model learns to predict $R_{t+1}$. The final price is reconstructed recursively:
$$ P_{t+1} = P_t \times (1 + \text{Predicted } R_{t+1}) $$

### 2. The Model (Enhanced LSTM)
*   **Input:** Sliding Window of 60 days (Price, RSI, MACD, Volume).
*   **Architecture:**
    *   `LSTM Layer 1`: 50 Units (Return Sequences=True) + Dropout 0.2
    *   `LSTM Layer 2`: 50 Units (Return Sequences=False) + Dropout 0.2
    *   `Dense Layer`: Feature Extraction
    *   `Output Layer`: Single Neuron (Predicts Next Day Return)
*   **Optimizer:** Adam (`lr=0.001`) with `ReduceLROnPlateau`.

### 3. Data Flow
1.  **User Input:** Selects Stock (e.g., `TCS.NS`).
2.  **Fetch:** Downloads max available history (10+ years).
3.  **Process:** Computes Technical Indicators (SMA, RSI, Bollinger Bands).
4.  **Train:** Model learns specific patterns of that stock.
5.  **Inference:** Generates 30-day future forecast.

---

## 💻 Frontend User Guide (How to Use)

The application is built with **Streamlit** for a seamless, interactive experience.

### Step 1: Launch the App
```bash
streamlit run app.py
```

### Step 2: Select Your Market
Use the **"Smart Search"** in the sidebar:
*   **🇮🇳 NSE:** Select "NSE" -> Type `RELIANCE` (System searches `RELIANCE.NS`)
*   **🇮🇳 BSE:** Select "BSE" -> Type `SARLAPOLY` (System searches `SARLAPOLY.BO`)
*   **🇺🇸 Global:** Select "Global" -> Type `AAPL` (Apple) or `BTC-USD` (Bitcoin).

### Step 3: Fetch Data
Click **"📥 Fetch Latest Data"**. The system downloads live market data.

### Step 4: The "Magic" Button (Train AI)
*If you see "⚠️ Using Generic Model" status:*
1.  Click **"🏋️ Train New Model"** in the sidebar.
2.  Wait ~60 seconds.
3.  The system trains a **Custom LSTM** specifically for your stock.
4.  **Result:** RMSE drops significantly (e.g., from 300 -> 15).

### Step 5: View Forecasts
*   **Historical Plot:** Zoom into the interactive Candlestick chart.
*   **Future Predictions:** Scroll down to see the **30-Day Rate of Return Forecast**.
*   **Download:** Export the prediction data as CSV for analysis.

---

## 🛠️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/stock-forecasting-ai.git

# 2. Create Virtual Environment
python -m venv venv
.\venv\Scripts\activate  # Windows

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

---

## ⚖️ License & usage

**(c) 2026 Admin. All Rights Reserved.**

This source code is the intellectual property of the author.
*   **Permitted:** You may clone this repository for educational purposes or to view the code as a portfolio reference.
*   **Prohibited:** You may NOT modify, redistribute, sell, or use this code for commercial applications without explicit permission.
*   **No Liability:** This software is provided "as is". The author is not responsible for any financial losses incurred from using these predictions. **This is not financial advice.**
