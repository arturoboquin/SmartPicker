# /home/ubuntu/etf_stock_picker_app/backend/labeling_strategy.py

import pandas as pd
import numpy as np

def assign_labels(daily_history_df: pd.DataFrame, forward_days: int = 126, buy_threshold: float = 0.15, sell_threshold: float = -0.15):
    """
    Calculates 6-month (126 trading days) forward return and assigns labels (Buy/Sell/Hold).

    Args:
        daily_history_df (pd.DataFrame): DataFrame with daily prices, must include 'Close'.
                                         Index should be DatetimeIndex.
        forward_days (int): Number of trading days to look forward for return calculation (default 126 for ~6 months).
        buy_threshold (float): Percentage increase for a "Buy" label (e.g., 0.15 for +15%).
        sell_threshold (float): Percentage decrease for a "Sell" label (e.g., -0.15 for -15%).

    Returns:
        pd.DataFrame: Original DataFrame with added 'forward_return' and 'label' columns.
    """
    if not isinstance(daily_history_df, pd.DataFrame) or daily_history_df.empty:
        return pd.DataFrame()
    if "Close" not in daily_history_df.columns:
        print("Error: daily_history_df must contain 'Close' column.")
        return pd.DataFrame()

    labeled_df = daily_history_df.copy()
    close_prices = labeled_df["Close"]

    # Calculate 6-month forward return
    # (df["Close"].shift(-126) / df["Close"]) - 1
    labeled_df["forward_return"] = (close_prices.shift(-forward_days) / close_prices) - 1

    # Assign labels based on forward return
    conditions = [
        (labeled_df["forward_return"] > buy_threshold),
        (labeled_df["forward_return"] < sell_threshold)
    ]
    choices = ["Buy", "Sell"]
    labeled_df["label"] = np.select(conditions, choices, default="Hold")

    # Rows where forward_return is NaN (typically at the end of the series due to shift)
    # will have their label determined by the default "Hold" if not explicitly handled.
    # This is acceptable as we cannot determine Buy/Sell for the most recent data points.
    # Alternatively, one could set label to NaN as well for these rows:
    # labeled_df.loc[labeled_df["forward_return"].isnull(), "label"] = np.nan

    return labeled_df

if __name__ == '__main__':
    # Create a sample DataFrame for testing
    dates = pd.date_range(start='2022-01-01', periods=300, freq='B') # Approx 1 year + 2 months of data
    data_size = len(dates)
    sample_data = pd.DataFrame({
        'Close': np.random.rand(data_size) * 100 + 50
    }, index=dates)
    
    # Simulate some trends for more interesting labels
    trend = np.linspace(0, 20, data_size) # Upward trend
    noise = np.random.normal(0, 5, data_size)
    sample_data['Close'] = sample_data['Close'] + trend + noise
    sample_data['Close'] = sample_data['Close'].clip(lower=10) # Ensure prices are positive

    print("Sample Daily History Data (first 5 rows):")
    print(sample_data.head())

    print("\nAssigning labels...")
    labeled_df = assign_labels(sample_data, forward_days=126, buy_threshold=0.15, sell_threshold=-0.15)

    print("\nLabeled DataFrame (showing some rows with actual forward returns and labels):")
    # Display rows where forward_return is not NaN to see actual labels
    # The last `forward_days` rows will have NaN for forward_return and 'Hold' label by default or NaN if changed
    print(labeled_df.head(10))
    print("\nLabeled DataFrame (last 10 rows to show NaNs in forward_return):")
    print(labeled_df.tail(10))

    print("\nLabel distribution:")
    print(labeled_df["label"].value_counts())

    # Test with a shorter DataFrame to see more NaNs
    short_dates = pd.date_range(start='2023-01-01', periods=150, freq='B')
    short_data = pd.DataFrame({'Close': np.random.rand(len(short_dates)) * 50 + 20}, index=short_dates)
    print("\nAssigning labels to shorter dataframe...")
    labeled_short_df = assign_labels(short_data)
    print(labeled_short_df.tail())
    print("Label distribution for short df:")
    print(labeled_short_df["label"].value_counts())

