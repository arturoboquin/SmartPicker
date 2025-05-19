# /home/ubuntu/etf_stock_picker_app/backend/analysis_logic.py

import pandas as pd
import numpy as np

def prepare_price_chart_data(history_df: pd.DataFrame):
    """
    Prepares historical price data for charting.
    Ensures DateTime index and includes Open, High, Low, Close, Volume.
    Args:
        history_df (pd.DataFrame): DataFrame from yfinance ticker.history().
    Returns:
        pd.DataFrame: Processed DataFrame suitable for charting, or None if input is invalid.
    """
    if not isinstance(history_df, pd.DataFrame) or history_df.empty:
        return None
    
    chart_data = history_df.copy()
    # Ensure index is DatetimeIndex
    if not isinstance(chart_data.index, pd.DatetimeIndex):
        chart_data.index = pd.to_datetime(chart_data.index)
    
    # Select relevant columns, ensure they exist
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    cols_to_use = [col for col in required_cols if col in chart_data.columns]
    
    return chart_data[cols_to_use]

def calculate_rolling_dividend_yield(history_df: pd.DataFrame, dividends_df: pd.DataFrame, window_days: int = 252):
    """
    Calculates rolling annualized dividend yield.
    Assumes dividends_df has DatetimeIndex and a column named 'Dividends'.
    Assumes history_df has DatetimeIndex and a 'Close' price column.

    Args:
        history_df (pd.DataFrame): Daily price history with 'Close' prices.
        dividends_df (pd.DataFrame): Dividend payment history.
        window_days (int): The window in trading days for rolling sum of dividends (default 1 year).

    Returns:
        pd.Series: Rolling dividend yield, or None if data is insufficient.
    """
    if not all(isinstance(df, pd.DataFrame) for df in [history_df, dividends_df]):
        return None
    if history_df.empty or dividends_df.empty or 'Close' not in history_df.columns or 'Dividends' not in dividends_df.columns:
        return None

    # Ensure datetime indices
    history_df.index = pd.to_datetime(history_df.index)
    dividends_df.index = pd.to_datetime(dividends_df.index)

    # Align dividends to trading days and sum them over a rolling window
    # Reindex dividends to history's dates, fill NaNs with 0, then calculate rolling sum
    # This gives the sum of dividends in the past 'window_days' for each trading day
    dividends_on_history_dates = dividends_df["Dividends"].reindex(history_df.index, fill_value=0)
    rolling_dividends_sum = dividends_on_history_dates.rolling(window=window_days, min_periods=1).sum()

    # Calculate annualized dividend yield: (Rolling Sum of Dividends / Current Price)
    # Note: This is a simplified annualization. More precise methods might consider payment frequency.
    # For this implementation, we assume the rolling sum over a year approximates annual dividends.
    rolling_yield = (rolling_dividends_sum / history_df["Close"]) * 100 # As percentage
    rolling_yield.name = "RollingDividendYield"
    return rolling_yield.replace([np.inf, -np.inf], np.nan) # Handle potential division by zero if price is 0

def extract_key_metrics(info_dict: dict, history_df: pd.DataFrame = None):
    """
    Extracts P/E, Market Cap, Beta, and other relevant metrics from the 'info' dictionary.
    Beta might require history_df if not directly available or if a custom calculation is desired.

    Args:
        info_dict (dict): Ticker info dictionary from yfinance.
        history_df (pd.DataFrame, optional): Historical price data, potentially for beta calculation.

    Returns:
        dict: A dictionary of key metrics.
    """
    if not isinstance(info_dict, dict):
        return {}

    metrics = {}
    # P/E Ratio - several fields might provide this, e.g., 'trailingPE', 'forwardPE'
    metrics["trailingPE"] = info_dict.get("trailingPE")
    metrics["forwardPE"] = info_dict.get("forwardPE")
    
    metrics["marketCap"] = info_dict.get("marketCap")
    metrics["beta"] = info_dict.get("beta") # yfinance often provides this directly
    
    # Other useful metrics from 'info'
    metrics["sector"] = info_dict.get("sector")
    metrics["industry"] = info_dict.get("industry")
    metrics["previousClose"] = info_dict.get("previousClose")
    metrics["fiftyTwoWeekHigh"] = info_dict.get("fiftyTwoWeekHigh")
    metrics["fiftyTwoWeekLow"] = info_dict.get("fiftyTwoWeekLow")
    metrics["dividendYield"] = info_dict.get("dividendYield") # This is current yield, not rolling
    metrics["payoutRatio"] = info_dict.get("payoutRatio")
    metrics["volume"] = info_dict.get("volume")
    metrics["averageVolume"] = info_dict.get("averageVolume")

    # If beta is not available and history_df is provided, one could calculate it here
    # This would require a benchmark series (e.g., S&P 500) and is more involved.
    # For now, we rely on yfinance's provided beta.

    return {k: v for k, v in metrics.items() if v is not None} # Clean out None values

def summarize_analyst_sentiment(recommendations_df: pd.DataFrame):
    """
    Summarizes analyst sentiment from the recommendations DataFrame.
    Counts recommendation types (e.g., Buy, Hold, Sell).

    Args:
        recommendations_df (pd.DataFrame): DataFrame of analyst recommendations from yfinance.
                                          Expected to have a 'To Grade' or similar column.

    Returns:
        dict: A summary of analyst sentiment (e.g., counts of Buy/Sell/Hold), or None.
    """
    if not isinstance(recommendations_df, pd.DataFrame) or recommendations_df.empty:
        return None

    # Common column names for recommendations are 'To Grade', 'Action'
    # Let's check for common ones. The spec mentions 'recommendations' -> 'Last 10 recommendations'
    # yfinance 'recommendations' df typically has 'Firm', 'To Grade', 'From Grade', 'Action'
    grade_column = None
    if 'To Grade' in recommendations_df.columns:
        grade_column = 'To Grade'
    elif 'Action' in recommendations_df.columns: # Some APIs might use 'Action'
        grade_column = 'Action'
    
    if not grade_column:
        return {"error": "Relevant recommendation column not found."}

    # Standardize common recommendations
    # This mapping can be expanded based on observed values
    def standardize_grade(grade):
        grade_lower = str(grade).lower()
        if any(buy_term in grade_lower for buy_term in ["buy", "outperform", "overweight", "strong buy", "accumulate"]):
            return "Buy"
        elif any(sell_term in grade_lower for sell_term in ["sell", "underperform", "underweight", "strong sell", "reduce"]):
            return "Sell"
        elif any(hold_term in grade_lower for hold_term in ["hold", "neutral", "equal-weight", "market perform"]):
            return "Hold"
        return "Other" # For grades not easily categorized

    recommendations_df["StandardizedGrade"] = recommendations_df[grade_column].apply(standardize_grade)
    sentiment_summary = recommendations_df["StandardizedGrade"].value_counts().to_dict()
    
    # Include latest recommendation date if possible
    if isinstance(recommendations_df.index, pd.DatetimeIndex) and not recommendations_df.index.empty:
        sentiment_summary["latestRecommendationDate"] = recommendations_df.index.max().strftime('%Y-%m-%d')
    
    return sentiment_summary


if __name__ == '__main__':
    # --- Dummy Data for Testing ---
    # Sample history_df
    dates_hist = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
    data_hist = {'Open': [100, 101, 102, 103, 104], 'High': [102, 103, 104, 105, 106],
                 'Low': [99, 100, 101, 102, 103], 'Close': [101, 102, 103, 104, 105],
                 'Volume': [1000, 1100, 1200, 1300, 1400]}
    sample_history_df = pd.DataFrame(data_hist, index=dates_hist)

    # Sample dividends_df
    dates_div = pd.to_datetime(['2023-01-02', '2023-04-01'])
    data_div = {'Dividends': [0.5, 0.6]}
    sample_dividends_df = pd.DataFrame(data_div, index=dates_div)

    # Sample info_dict
    sample_info_dict = {
        "symbol": "TEST", "sector": "Technology", "industry": "Software",
        "trailingPE": 25.5, "forwardPE": 22.0, "marketCap": 1e12, "beta": 1.1,
        "dividendYield": 0.015, "payoutRatio": 0.35, "previousClose": 150.0,
        "fiftyTwoWeekHigh": 180.0, "fiftyTwoWeekLow": 120.0, "volume": 5e6, "averageVolume": 4.5e6
    }

    # Sample recommendations_df
    dates_rec = pd.to_datetime(['2023-01-01', '2023-02-15', '2023-03-20'])
    data_rec = {'Firm': ['Alpha', 'Beta', 'Gamma'], 'To Grade': ['Buy', 'Hold', 'Outperform']}
    sample_recommendations_df = pd.DataFrame(data_rec, index=dates_rec)

    # --- Test Functions ---
    print("--- Testing prepare_price_chart_data ---")
    chart_data = prepare_price_chart_data(sample_history_df)
    print(chart_data.head() if chart_data is not None else "None")

    print("\n--- Testing calculate_rolling_dividend_yield ---")
    rolling_yield = calculate_rolling_dividend_yield(sample_history_df, sample_dividends_df, window_days=3) # Short window for test
    print(rolling_yield if rolling_yield is not None else "None")

    print("\n--- Testing extract_key_metrics ---")
    key_metrics = extract_key_metrics(sample_info_dict)
    print(key_metrics)

    print("\n--- Testing summarize_analyst_sentiment ---")
    sentiment = summarize_analyst_sentiment(sample_recommendations_df)
    print(sentiment)

    # Test with empty recommendations
    empty_recs = pd.DataFrame(columns=['Firm', 'To Grade'])
    sentiment_empty = summarize_analyst_sentiment(empty_recs)
    print("Sentiment with empty recs:", sentiment_empty)

