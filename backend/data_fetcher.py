# /home/ubuntu/etf_stock_picker_app/backend/data_fetcher.py

import yfinance as yf
import pandas as pd

# Define a function to fetch all required data for a given ticker
def fetch_ticker_data(ticker_symbol: str):
    """
    Fetches various financial data for a given stock ticker symbol using yfinance.

    Args:
        ticker_symbol (str): The stock ticker symbol (e.g., "AAPL").

    Returns:
        dict: A dictionary containing different dataframes and info for the ticker.
              Keys are: "history", "info", "dividends", "financials", 
              "balance_sheet", "cashflow", "recommendations", "calendar".
              Returns None for a specific key if data is not available or an error occurs.
    """
    ticker = yf.Ticker(ticker_symbol)
    data = {}

    try:
        # Price History (10 years, daily)
        hist = ticker.history(period="10y", interval="1d")
        data["history"] = hist if not hist.empty else None
    except Exception as e:
        print(f"Error fetching history for {ticker_symbol}: {e}")
        data["history"] = None

    try:
        # Basic Info
        info = ticker.info
        data["info"] = info if info else None
    except Exception as e:
        print(f"Error fetching info for {ticker_symbol}: {e}")
        data["info"] = None

    try:
        # Dividends
        dividends = ticker.dividends
        data["dividends"] = dividends if not dividends.empty else None
    except Exception as e:
        print(f"Error fetching dividends for {ticker_symbol}: {e}")
        data["dividends"] = None

    # Financials, Balance Sheet, Cashflow (typically for stocks)
    # These might not be available for all tickers (e.g., ETFs)
    try:
        financials = ticker.financials
        data["financials"] = financials if not financials.empty else None
    except Exception as e:
        print(f"Error fetching financials for {ticker_symbol}: {e}")
        data["financials"] = None

    try:
        balance_sheet = ticker.balance_sheet
        data["balance_sheet"] = balance_sheet if not balance_sheet.empty else None
    except Exception as e:
        print(f"Error fetching balance_sheet for {ticker_symbol}: {e}")
        data["balance_sheet"] = None

    try:
        cashflow = ticker.cashflow
        data["cashflow"] = cashflow if not cashflow.empty else None
    except Exception as e:
        print(f"Error fetching cashflow for {ticker_symbol}: {e}")
        data["cashflow"] = None

    try:
        # Analyst Recommendations (last 10)
        recommendations = ticker.recommendations
        if recommendations is not None and not recommendations.empty:
            data["recommendations"] = recommendations.tail(10)
        else:
            data["recommendations"] = None
    except Exception as e:
        print(f"Error fetching recommendations for {ticker_symbol}: {e}")
        data["recommendations"] = None

    try:
        # Earnings Calendar
        calendar = ticker.calendar
        data["calendar"] = calendar if calendar is not None and not calendar.empty else None # calendar is a dataframe
    except Exception as e:
        print(f"Error fetching calendar for {ticker_symbol}: {e}")
        data["calendar"] = None
        
    return data

if __name__ == '__main__':
    # Example usage:
    sample_ticker = "MSFT" # Try with a stock
    # sample_ticker = "VOO" # Try with an ETF
    
    print(f"Fetching data for {sample_ticker}...")
    ticker_data = fetch_ticker_data(sample_ticker)

    if ticker_data:
        for key, value in ticker_data.items():
            print(f"\n--- {key.upper()} ---")
            if isinstance(value, pd.DataFrame):
                print(value.head() if not value.empty else "No data")
            elif isinstance(value, dict):
                # Print first 5 items for brevity if it's a large dict (like 'info')
                print(dict(list(value.items())[:5])) 
            else:
                print(value if value is not None else "No data")
    else:
        print(f"Could not retrieve data for {sample_ticker}")

