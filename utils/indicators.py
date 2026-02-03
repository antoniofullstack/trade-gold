import numpy as np
import pandas as pd


def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_sma(prices, window):
    """Calculate Simple Moving Average"""
    return prices.rolling(window=window).mean()


def calculate_returns(prices):
    """Calculate percentage returns"""
    return prices.pct_change()


def add_technical_indicators(df, price_col='Close'):
    """Add all technical indicators to dataframe"""
    df = df.copy()
    
    # Returns
    df['Return'] = calculate_returns(df[price_col])
    
    # Simple Moving Averages
    df['SMA_fast'] = calculate_sma(df[price_col], window=10)
    df['SMA_slow'] = calculate_sma(df[price_col], window=30)
    
    # RSI
    df['RSI'] = calculate_rsi(df[price_col], window=14)
    
    # Drop NaN values
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df


def validate_data(df):
    """Validate that dataframe has required columns"""
    required_columns = ['Close', 'Return', 'SMA_fast', 'SMA_slow', 'RSI']
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if len(df) == 0:
        raise ValueError("Dataframe is empty after processing")
    
    return True
