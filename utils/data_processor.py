import pandas as pd
import numpy as np
from .indicators import add_technical_indicators, validate_data


def load_and_prepare_data(filepath, price_col='Close', separator=';', date_format='%Y.%m.%d %H:%M:%S'):
    """
    Load data from CSV file and prepare with technical indicators
    
    Args:
        filepath (str): Path to CSV file
        price_col (str): Name of price column
        separator (str): CSV separator (default: ';')
        date_format (str): Date format string
    
    Returns:
        pd.DataFrame: Prepared dataframe with indicators
    """
    try:
        # Load data with proper separator
        df = pd.read_csv(filepath, sep=separator)
        
        # Handle different column names
        if 'Time' in df.columns and 'Date' not in df.columns:
            df = df.rename(columns={'Time': 'Date'})
        
        # Ensure we have the price column
        if price_col not in df.columns:
            raise ValueError(f"Price column '{price_col}' not found in data")
        
        # Parse dates if Date column exists
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], format=date_format, errors='coerce')
            df = df.dropna(subset=['Date'])
            df = df.set_index('Date')
        
        # Add technical indicators
        df = add_technical_indicators(df, price_col)
        
        # Validate data
        validate_data(df)
        
        print(f"Data loaded successfully: {len(df)} rows")
        if hasattr(df.index, 'min') and hasattr(df.index, 'max'):
            print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def create_sample_data(n_samples=5000, start_price=2000):
    """
    Create sample gold price data for testing
    
    Args:
        n_samples (int): Number of samples to generate
        start_price (float): Starting price
    
    Returns:
        pd.DataFrame: Sample data with technical indicators
    """
    np.random.seed(42)
    
    # Generate realistic price movement
    returns = np.random.normal(0, 0.01, n_samples)
    prices = [start_price]
    
    for ret in returns:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    # Create dataframe
    df = pd.DataFrame({
        'Close': prices[1:]  # Remove first price as we don't have its return
    })
    
    # Add technical indicators
    df = add_technical_indicators(df, 'Close')
    
    print(f"Sample data created: {len(df)} rows")
    return df


def split_data(df, train_ratio=0.8):
    """
    Split data into train and test sets
    
    Args:
        df (pd.DataFrame): Input dataframe
        train_ratio (float): Ratio for training data
    
    Returns:
        tuple: (train_df, test_df)
    """
    split_idx = int(len(df) * train_ratio)
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"Train data: {len(train_df)} rows")
    print(f"Test data: {len(test_df)} rows")
    
    return train_df, test_df


def save_processed_data(df, filepath):
    """
    Save processed dataframe to CSV
    
    Args:
        df (pd.DataFrame): Dataframe to save
        filepath (str): Output filepath
    """
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")


def load_processed_data(filepath):
    """
    Load processed dataframe from CSV
    
    Args:
        filepath (str): Input filepath
    
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    df = pd.read_csv(filepath)
    validate_data(df)
    print(f"Data loaded from {filepath}: {len(df)} rows")
    return df
