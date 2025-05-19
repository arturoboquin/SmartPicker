# /home/ubuntu/etf_stock_picker_app/backend/feature_engineering.py

import pandas as pd
import numpy as np

# Optional: Attempt to import pandas_ta, if not available, some features might need manual implementation or will be skipped.
PANDAS_TA_AVAILABLE = False
print("pandas_ta import disabled due to compatibility issues. Some TA features (RSI, MACD, OBV, ATR) will be unavailable.")

def calculate_returns(series: pd.Series, n_days: int):
    """Calculates n-day percentage returns."""
    return series.pct_change(n_days)

def calculate_sma(series: pd.Series, window: int):
    """Calculates Simple Moving Average (SMA)."""
    return series.rolling(window=window, min_periods=1).mean()

def calculate_ema(series: pd.Series, window: int):
    """Calculates Exponential Moving Average (EMA)."""
    return series.ewm(span=window, adjust=False, min_periods=1).mean()

def calculate_rolling_volatility(series: pd.Series, window: int):
    """Calculates rolling standard deviation of daily returns."""
    daily_returns = series.pct_change()
    return daily_returns.rolling(window=window, min_periods=1).std() * np.sqrt(window) # Annualized if window is year

def calculate_distance_from_high_low(series: pd.Series, window: int = 252):
    """Calculates distance from N-day high and low."""
    rolling_high = series.rolling(window=window, min_periods=1).max()
    rolling_low = series.rolling(window=window, min_periods=1).min()
    distance_from_high = (series - rolling_high) / rolling_high
    distance_from_low = (series - rolling_low) / (rolling_low + 1e-9) # Add epsilon to avoid division by zero if low is 0
    return distance_from_high, distance_from_low

def engineer_features_for_ml(daily_history_df: pd.DataFrame, lookback_days: int = 252):
    """
    Engineers features from daily historical price data for machine learning.
    Uses the last `lookback_days` of data for calculations where appropriate.
    
    Args:
        daily_history_df (pd.DataFrame): DataFrame with daily prices, must include 
                                         at least 'Open', 'High', 'Low', 'Close', 'Volume'.
                                         Index should be DatetimeIndex.
        lookback_days (int): The number of trading days (typically 1 year = 252 days) 
                             to use for calculating rolling features.

    Returns:
        pd.DataFrame: DataFrame with original data and engineered features.
    """
    if not isinstance(daily_history_df, pd.DataFrame) or daily_history_df.empty:
        return pd.DataFrame()
    if not all(col in daily_history_df.columns for col in ["Open", "High", "Low", "Close", "Volume"]):
        print("Error: daily_history_df must contain Open, High, Low, Close, Volume columns.")
        return pd.DataFrame()
    
    features_df = daily_history_df.copy()
    close_prices = features_df["Close"]
    volume = features_df["Volume"]

    # 1. Momentum: 20, 50, 100-day % returns
    for n in [20, 50, 100]:
        features_df[f"return_{n}d"] = calculate_returns(close_prices, n)

    # 2. SMA/EMA: 20, 50, 200 days
    for n in [20, 50, 200]:
        features_df[f"sma_{n}d"] = calculate_sma(close_prices, n)
        features_df[f"ema_{n}d"] = calculate_ema(close_prices, n)

    # 3. Rolling Volatility: 30-day std dev of daily returns
    # The spec says 30-day std dev. If it means annualized, multiply by sqrt(30).
    # If it means just the std dev of daily returns over 30 days, then no sqrt(30).
    # Assuming non-annualized for now based on typical TA feature usage.
    features_df["volatility_30d"] = close_prices.pct_change().rolling(window=30, min_periods=1).std()

    # 4. RSI (14), MACD, OBV, ATR (using pandas_ta if available)
    if PANDAS_TA_AVAILABLE:
        # RSI (14)
        features_df.ta.rsi(length=14, append=True) # Appends RSI_14
        # MACD (standard 12, 26, 9)
        features_df.ta.macd(append=True) # Appends MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        # OBV
        features_df.ta.obv(append=True) # Appends OBV
        # ATR (14)
        # ATR needs High, Low, Close. pandas_ta handles this automatically.
        features_df.ta.atr(length=14, append=True) # Appends ATR_14
    else:
        # Manual implementation or placeholders if pandas_ta is not available
        features_df["RSI_14"] = np.nan # Placeholder
        features_df["MACD_12_26_9"] = np.nan # Placeholder
        features_df["MACDh_12_26_9"] = np.nan # Placeholder
        features_df["MACDs_12_26_9"] = np.nan # Placeholder
        features_df["OBV"] = np.nan # Placeholder
        features_df["ATR_14"] = np.nan # Placeholder
        print("Skipping RSI, MACD, OBV, ATR due to missing pandas_ta.")

    # 5. Volume Trends: recent vs historical average (e.g., 20-day avg volume vs 200-day avg volume)
    sma_vol_20 = calculate_sma(volume, 20)
    sma_vol_200 = calculate_sma(volume, 200)
    features_df["volume_trend_20_vs_200"] = sma_vol_20 / (sma_vol_200 + 1e-9) # Ratio, add epsilon

    # 6. Distance from 52-week high/low (using lookback_days, typically 252 for 1 year)
    dist_high, dist_low = calculate_distance_from_high_low(close_prices, window=lookback_days)
    features_df[f"distance_from_{lookback_days}d_high"] = dist_high
    features_df[f"distance_from_{lookback_days}d_low"] = dist_low
    
    # The spec mentions "Use the last 252 trading days (1 year) to calculate"
    # This implies that for a given day, features are calculated based on the preceding 'lookback_days'.
    # Most rolling functions inherently do this. For single point calculations like distance from high/low,
    # the window parameter handles this.

    # Drop rows with NaNs created by rolling windows, especially at the beginning
    # Or, the calling function can decide how to handle NaNs (e.g., for training data prep)
    # For now, returning with NaNs, as the amount to drop depends on the longest window.
    # The ML training step will need to handle this (e.g., df.dropna()).
    
    return features_df

if __name__ == '__main__':
    # Create a sample DataFrame for testing (mimicking yfinance history output)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='B') # Business days
    data_size = len(dates)
    sample_data = pd.DataFrame({
        'Open': np.random.rand(data_size) * 100 + 50,
        'High': np.random.rand(data_size) * 10 + 150, # Ensure High > Open/Low/Close
        'Low': np.random.rand(data_size) * 10 + 40,  # Ensure Low < Open/High/Close
        'Close': np.random.rand(data_size) * 100 + 50,
        'Volume': np.random.randint(100000, 1000000, size=data_size)
    }, index=dates)
    sample_data['High'] = sample_data[['Open', 'Close']].max(axis=1) + np.random.rand(data_size) * 5
    sample_data['Low'] = sample_data[['Open', 'Close']].min(axis=1) - np.random.rand(data_size) * 5

    print("Sample Daily History Data (first 5 rows):")
    print(sample_data.head())

    print(f"\nAttempting to use pandas_ta: {PANDAS_TA_AVAILABLE}")
    if not PANDAS_TA_AVAILABLE:
        print("Please install pandas_ta for full feature set: pip install pandas_ta")

    print("\nEngineering features...")
    # Use a smaller lookback for test if data is not very long, or ensure sample_data is long enough
    # The sample_data is ~4 years, so 252 is fine.
    engineered_df = engineer_features_for_ml(sample_data, lookback_days=252)

    print("\nEngineered Features DataFrame (last 5 rows with features):")
    # Print columns to see what was added
    print("Columns:", engineered_df.columns.tolist())
    # Print tail, which should have fewer NaNs
    print(engineered_df.tail())

    # Check for NaNs in the last few rows of key features
    print("\nNaN check for last 5 rows of key features:")
    key_feature_cols = ['return_20d', 'sma_20d', 'volatility_30d']
    if PANDAS_TA_AVAILABLE:
        key_feature_cols.extend(['RSI_14', 'MACD_12_26_9', 'OBV', 'ATR_14'])
    else:
        key_feature_cols.extend(['RSI_14']) # Check placeholder if pandas_ta not available
    
    # Filter out columns that might not exist if pandas_ta failed or for other reasons
    existing_key_features = [col for col in key_feature_cols if col in engineered_df.columns]
    print(engineered_df[existing_key_features].tail().isnull().sum())


