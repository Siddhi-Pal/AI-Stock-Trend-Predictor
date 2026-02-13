"""
Stock Data Fetcher Module
Fetches real-time stock data from Yahoo Finance using yfinance API
Supports global stocks and Indian NSE/BSE stocks
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime


def get_indian_stock_symbols():
    """
    Returns list of major Indian NSE stocks
    NSE stocks use .NS suffix, BSE stocks use .BO suffix
    """
    indian_stocks = {
        # Banking & Financial Services
        'HDFCBANK.NS': 'HDFC Bank',
        'ICICIBANK.NS': 'ICICI Bank',
        'KOTAKBANK.NS': 'Kotak Mahindra Bank',
        'SBIN.NS': 'State Bank of India',
        'AXISBANK.NS': 'Axis Bank',
        
        # IT Services
        'TCS.NS': 'Tata Consultancy Services',
        'INFY.NS': 'Infosys',
        'WIPRO.NS': 'Wipro',
        'HCLTECH.NS': 'HCL Technologies',
        'TECHM.NS': 'Tech Mahindra',
        
        # Energy & Oil
        'RELIANCE.NS': 'Reliance Industries',
        'ONGC.NS': 'Oil and Natural Gas Corporation',
        'BPCL.NS': 'Bharat Petroleum',
        
        # Automobiles
        'MARUTI.NS': 'Maruti Suzuki',
        'TATAMOTORS.NS': 'Tata Motors',
        'M&M.NS': 'Mahindra & Mahindra',
        
        # Pharmaceuticals
        'SUNPHARMA.NS': 'Sun Pharmaceutical',
        'DRREDDY.NS': 'Dr. Reddy\'s Laboratories',
        'CIPLA.NS': 'Cipla',
        
        # FMCG
        'HINDUNILVR.NS': 'Hindustan Unilever',
        'ITC.NS': 'ITC Limited',
        'NESTLEIND.NS': 'Nestle India',
        
        # Metals & Mining
        'TATASTEEL.NS': 'Tata Steel',
        'HINDALCO.NS': 'Hindalco Industries',
        
        # Telecom
        'BHARTIARTL.NS': 'Bharti Airtel',
    }
    return indian_stocks


def get_global_stock_symbols():
    """
    Returns list of major global stocks
    """
    global_stocks = {
        'AAPL': 'Apple Inc.',
        'GOOGL': 'Alphabet Inc.',
        'MSFT': 'Microsoft Corporation',
        'AMZN': 'Amazon.com Inc.',
        'TSLA': 'Tesla Inc.',
        'META': 'Meta Platforms Inc.',
        'NVDA': 'NVIDIA Corporation',
        '^GSPC': 'S&P 500 Index',
        '^DJI': 'Dow Jones Industrial Average',
        '^NSEI': 'NIFTY 50 Index',
    }
    return global_stocks


def fetch_stock_data(symbol, period='5y', interval='1d'):
    """
    Download historical stock data using yfinance
    
    Parameters:
    -----------
    symbol : str
        Stock ticker symbol (e.g., 'RELIANCE.NS', 'AAPL')
    period : str
        Historical period to fetch ('1y', '2y', '5y', 'max')
    interval : str
        Data granularity ('1d', '1wk', '1mo')
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: Date, Open, High, Low, Close, Volume
    """
    try:
        print(f"Fetching data for {symbol}...")
        stock = yf.Ticker(symbol)
        
        # Download historical data
        df = stock.history(period=period, interval=interval)
        
        if df.empty:
            print(f"No data found for {symbol}")
            return None
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Keep only relevant columns
        columns_to_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = df[[col for col in columns_to_keep if col in df.columns]]
        
        print(f"Successfully fetched {len(df)} records for {symbol}")
        return df
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return None


def save_stock_data(symbol, dataframe, path='data/stocks/'):
    """
    Save fetched stock data to CSV
    
    Parameters:
    -----------
    symbol : str
        Stock ticker symbol
    dataframe : pd.DataFrame
        Stock data to save
    path : str
        Directory path to save the CSV file
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Clean symbol for filename (remove special characters)
        clean_symbol = symbol.replace('^', '').replace('.', '_')
        filepath = os.path.join(path, f"{clean_symbol}.csv")
        
        # Save to CSV
        dataframe.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Error saving data: {str(e)}")
        return None


def get_stock_info(symbol):
    """
    Get additional stock information
    
    Parameters:
    -----------
    symbol : str
        Stock ticker symbol
    
    Returns:
    --------
    dict
        Dictionary with stock information
    """
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        return {
            'name': info.get('longName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'current_price': info.get('currentPrice', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            '52_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
            '52_week_low': info.get('fiftyTwoWeekLow', 'N/A'),
        }
    except:
        return {}


if __name__ == "__main__":
    # Example usage
    print("=== Testing Stock Data Fetcher ===\n")
    
    # Fetch sample Indian stock
    indian_data = fetch_stock_data('RELIANCE.NS', period='1y')
    if indian_data is not None:
        print(f"\nRELIANCE.NS Data Shape: {indian_data.shape}")
        print(indian_data.head())
        save_stock_data('RELIANCE.NS', indian_data)
    
    # Fetch sample global stock
    global_data = fetch_stock_data('AAPL', period='1y')
    if global_data is not None:
        print(f"\nAAPL Data Shape: {global_data.shape}")
        print(global_data.head())
        save_stock_data('AAPL', global_data)
