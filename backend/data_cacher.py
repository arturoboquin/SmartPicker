# /home/ubuntu/etf_stock_picker_app/backend/data_cacher.py

import os
import pickle
import pandas as pd
from datetime import datetime

DATA_CACHE_DIR = "/home/ubuntu/etf_stock_picker_app/data_cache"

def save_ticker_data_to_cache(ticker_symbol: str, data_to_cache: dict):
    """
    Saves the fetched ticker data to a dedicated directory for the ticker.
    Each data item (e.g., history, info) is saved as a separate pickle file.
    An 'updated_on.txt' file is created/updated with the current timestamp.

    Args:
        ticker_symbol (str): The stock ticker symbol.
        data_to_cache (dict): The dictionary of data fetched by data_fetcher.fetch_ticker_data.
    """
    ticker_cache_path = os.path.join(DATA_CACHE_DIR, ticker_symbol.upper())
    os.makedirs(ticker_cache_path, exist_ok=True)

    for key, data_item in data_to_cache.items():
        if data_item is not None:
            file_path = os.path.join(ticker_cache_path, f"{key}.pkl")
            try:
                with open(file_path, "wb") as f:
                    pickle.dump(data_item, f)
            except Exception as e:
                print(f"Error saving {key} for {ticker_symbol} to cache: {e}")
        else:
            # If a data item is None, we can optionally remove an old cached file if it exists
            # or simply not write anything, which is the current behavior.
            pass 

    # Update the timestamp file
    timestamp_file = os.path.join(ticker_cache_path, "updated_on.txt")
    with open(timestamp_file, "w") as f:
        f.write(datetime.utcnow().isoformat())
    print(f"Data for {ticker_symbol} cached successfully at {ticker_cache_path}")

def load_ticker_data_from_cache(ticker_symbol: str):
    """
    Loads all cached data for a given ticker symbol from its dedicated directory.
    Also returns the last updated timestamp.

    Args:
        ticker_symbol (str): The stock ticker symbol.

    Returns:
        tuple: (dict, datetime | None)
               A dictionary containing the cached data items (or None if not found/error).
               The last updated timestamp (datetime object) or None if not found.
    """
    ticker_cache_path = os.path.join(DATA_CACHE_DIR, ticker_symbol.upper())
    cached_data = {}
    last_updated = None

    if not os.path.isdir(ticker_cache_path):
        return None, None # No cache directory for this ticker

    # Expected data keys based on data_fetcher output
    expected_keys = ["history", "info", "dividends", "financials", 
                     "balance_sheet", "cashflow", "recommendations", "calendar"]

    for key in expected_keys:
        file_path = os.path.join(ticker_cache_path, f"{key}.pkl")
        if os.path.exists(file_path):
            try:
                with open(file_path, "rb") as f:
                    cached_data[key] = pickle.load(f)
            except Exception as e:
                print(f"Error loading {key} for {ticker_symbol} from cache: {e}")
                cached_data[key] = None
        else:
            cached_data[key] = None # File not found in cache

    timestamp_file = os.path.join(ticker_cache_path, "updated_on.txt")
    if os.path.exists(timestamp_file):
        with open(timestamp_file, "r") as f:
            try:
                last_updated = datetime.fromisoformat(f.read().strip())
            except ValueError:
                last_updated = None # Invalid timestamp format
    
    # If no data was loaded at all, treat as if cache doesn't exist for this key set
    if all(value is None for value in cached_data.values()) and last_updated is None:
        return None, None
        
    return cached_data, last_updated

if __name__ == '__main__':
    # Example Usage (requires data_fetcher.py in the same directory or PYTHONPATH)
    # from data_fetcher import fetch_ticker_data

    # Create dummy data for testing if data_fetcher is not available
    def fetch_ticker_data(ticker_symbol):
        print(f"Simulating fetch for {ticker_symbol}")
        return {
            "history": pd.DataFrame({"Close": [1,2,3]}),
            "info": {"symbol": ticker_symbol, "sector": "Tech"},
            "dividends": pd.DataFrame({"Dividends": [0.1,0.1,0.12]}),
            "financials": None, # Simulate some data not being available
            "balance_sheet": pd.DataFrame({"Assets": [1000]}),
            "cashflow": pd.DataFrame({"OperatingCashflow": [200]}),
            "recommendations": pd.DataFrame({"Firm": ["MS"], "To Grade": ["Buy"]}),
            "calendar": pd.DataFrame({"Earnings Date": ["2025-07-01"]})
        }

    sample_ticker_to_cache = "GOOG"
    print(f"\n--- Testing Cache Saving for {sample_ticker_to_cache} ---")
    data_to_save = fetch_ticker_data(sample_ticker_to_cache)
    if data_to_save:
        save_ticker_data_to_cache(sample_ticker_to_cache, data_to_save)
    
    print(f"\n--- Testing Cache Loading for {sample_ticker_to_cache} ---")
    loaded_data, updated_time = load_ticker_data_from_cache(sample_ticker_to_cache)

    if loaded_data:
        print(f"Data loaded for {sample_ticker_to_cache}, last updated: {updated_time}")
        for key, value in loaded_data.items():
            print(f"  {key}: {'DataFrame' if isinstance(value, pd.DataFrame) else type(value)} {'(empty)' if isinstance(value, pd.DataFrame) and value.empty else ''} {'(None)' if value is None else ''}")
            if isinstance(value, pd.DataFrame) and not value.empty:
                print(value.head(2))
            elif isinstance(value, dict):
                 print(dict(list(value.items())[:2]))

    else:
        print(f"No cached data found for {sample_ticker_to_cache} or cache is empty.")

    # Test loading a non-existent ticker
    non_existent_ticker = "NONEXISTENTTICKER"
    print(f"\n--- Testing Cache Loading for {non_existent_ticker} ---")
    loaded_data_ne, updated_time_ne = load_ticker_data_from_cache(non_existent_ticker)
    if loaded_data_ne:
        print(f"Data loaded for {non_existent_ticker}, last updated: {updated_time_ne}")
    else:
        print(f"No cached data found for {non_existent_ticker}.")

