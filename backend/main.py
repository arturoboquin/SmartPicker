# /home/ubuntu/etf_stock_picker_app/backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from enum import Enum
import io
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime, timedelta
import pandas as pd

# Changed relative imports to absolute imports
from data_fetcher import fetch_ticker_data
from data_cacher import load_ticker_data_from_cache, save_ticker_data_to_cache

app = FastAPI(
    title="Intelligent ETF and Stock Picker API",
    description="API for analyzing ETFs/stocks, optimizing portfolios, and predicting price movements.",
    version="0.1.0"
)

# CORS Configuration
origins = [
    "https://hmhkcwcb.manus.space", # New deployed frontend
    # "https://qxhvhrck.manus.space", # Old deployed frontend
    "http://localhost:8080",      # Local frontend testing
    "http://localhost:3000",      # Common React dev port
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SymbolsRequest(BaseModel):
    symbols: List[str]

# Helper to convert pandas DataFrames to dict for JSON response if needed
def convert_data_for_json(data):
    if isinstance(data, pd.DataFrame):
        return data.reset_index().to_dict(orient="records") 
    elif isinstance(data, pd.Series):
        return data.reset_index().to_dict(orient="records")
    elif isinstance(data, dict):
        return {k: convert_data_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_data_for_json(i) for i in data]
    elif isinstance(data, (datetime, pd.Timestamp)):
        return data.isoformat()
    return data

@app.get("/health", tags=["General"])
async def health_check():
    """Check the health of the API."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/analyze", tags=["Analysis"])
async def analyze_tickers(request: SymbolsRequest) -> Dict[str, Any]:
    results = {}
    cache_expiry_hours = 24
    for symbol in request.symbols:
        symbol_upper = symbol.upper()
        cached_data, last_updated = load_ticker_data_from_cache(symbol_upper)
        data_to_return = None
        if cached_data and last_updated:
            if datetime.utcnow() - last_updated < timedelta(hours=cache_expiry_hours):
                data_to_return = cached_data
        if data_to_return is None:
            fetched_data = fetch_ticker_data(symbol_upper)
            if fetched_data:
                save_ticker_data_to_cache(symbol_upper, fetched_data)
                data_to_return = fetched_data
            else:
                results[symbol_upper] = {"error": f"Could not fetch data for {symbol_upper}"}
                continue
        if data_to_return:
            response_data = {}
            response_data["fundamentals"] = data_to_return.get("info")
            response_data["historical_prices"] = data_to_return.get("history")
            response_data["dividend_data"] = data_to_return.get("dividends")
            results[symbol_upper] = convert_data_for_json(response_data)
        else:
            results[symbol_upper] = {"error": f"Data processing failed for {symbol_upper}"}
    return results

import joblib
from feature_engineering import engineer_features_for_ml
from model_trainer import MODEL_PATH
import os

class PredictRequest(BaseModel):
    symbol: str
    lookback_days: int = 252

class PredictResponse(BaseModel):
    label: str
    confidence: float | None = None
    features_used: Dict[str, Any] | None = None
    error: str | None = None

PREDICTION_MODEL = None
FEATURE_COLUMNS_FOR_MODEL = None

if os.path.exists(MODEL_PATH):
    try:
        model_payload = joblib.load(MODEL_PATH)
        # Check if the model is saved as a dictionary with 'model' and 'feature_columns' keys
        if isinstance(model_payload, dict) and 'model' in model_payload and 'feature_columns' in model_payload:
            PREDICTION_MODEL = model_payload['model']
            FEATURE_COLUMNS_FOR_MODEL = model_payload['feature_columns']
            print(f"Loaded model and feature columns from {MODEL_PATH}")
        # Fallback for older model format
        elif hasattr(model_payload, 'predict'):
            PREDICTION_MODEL = model_payload
            if hasattr(PREDICTION_MODEL, 'feature_names_in_'):
                FEATURE_COLUMNS_FOR_MODEL = PREDICTION_MODEL.feature_names_in_
            else:
                print("Warning: Model does not have feature_names_in_.")
        else:
            print("Warning: Invalid model format. Expected dictionary with 'model' key or a model object.")
            PREDICTION_MODEL = None
    except Exception as e:
        print(f"Error loading prediction model: {e}")
        PREDICTION_MODEL = None
else:
    print(f"Warning: Prediction model not found at {MODEL_PATH}.")

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_ticker_movement(request: PredictRequest) -> PredictResponse:
    global FEATURE_COLUMNS_FOR_MODEL
    if PREDICTION_MODEL is None:
        return PredictResponse(label="Error", error="Model not loaded.")
    symbol_upper = request.symbol.upper()
    ticker_data_full = fetch_ticker_data(symbol_upper)
    if not ticker_data_full or ticker_data_full.get("history") is None or ticker_data_full["history"].empty:
        return PredictResponse(label="Error", error=f"Could not fetch historical data for {symbol_upper}")
    historical_prices_df = ticker_data_full["history"]
    features_df_full = engineer_features_for_ml(historical_prices_df, lookback_days=request.lookback_days)
    if features_df_full.empty:
        return PredictResponse(label="Error", error=f"Feature engineering failed for {symbol_upper}")
    latest_features_series = features_df_full.iloc[-1].copy()
    current_feature_columns = FEATURE_COLUMNS_FOR_MODEL
    if FEATURE_COLUMNS_FOR_MODEL is None:
        original_cols = set(historical_prices_df.columns)
        potential_feature_cols = [col for col in features_df_full.columns if col not in original_cols and col not in ["label", "forward_return"]]
        current_feature_columns = features_df_full[potential_feature_cols].dropna(axis=1, how='all').columns.tolist()
    missing_cols = [col for col in current_feature_columns if col not in latest_features_series.index]
    if missing_cols:
        return PredictResponse(label="Error", error=f"Missing required feature columns: {missing_cols}")
    features_for_prediction_df = pd.DataFrame([latest_features_series[current_feature_columns]])
    if features_for_prediction_df.isnull().values.any():
        features_for_prediction_df.fillna(0, inplace=True)
    try:
        prediction_label = PREDICTION_MODEL.predict(features_for_prediction_df)[0]
        prediction_proba = PREDICTION_MODEL.predict_proba(features_for_prediction_df)[0]
        confidence = None
        if hasattr(PREDICTION_MODEL, 'classes_'):
            class_index = list(PREDICTION_MODEL.classes_).index(prediction_label)
            confidence = float(prediction_proba[class_index])
        else:
            confidence = float(max(prediction_proba))
    except Exception as e:
        return PredictResponse(label="Error", error=f"Error during model prediction: {e}")
    features_used_dict = latest_features_series[current_feature_columns].to_dict()
    for key, value in features_used_dict.items():
        if isinstance(value, (np.generic, pd.Timestamp)):
            features_used_dict[key] = convert_data_for_json(value)
    return PredictResponse(label=str(prediction_label), confidence=confidence, features_used=features_used_dict)

from portfolio_optimizer import optimize_portfolio_mpt
import numpy as np

class OptimizePortfolioRequest(BaseModel):
    symbols: List[str]
    risk_free_rate: float = 0.02

class OptimizePortfolioResponse(BaseModel):
    weights: Dict[str, float] | None = None
    expected_return: float | None = None
    volatility: float | None = None
    sharpe_ratio: float | None = None
    error: str | None = None

@app.post("/optimize", response_model=OptimizePortfolioResponse, tags=["Optimization"])
async def optimize_portfolio(request: OptimizePortfolioRequest) -> OptimizePortfolioResponse:
    price_history_for_optimization = {}
    min_data_points_for_opt = 60
    for symbol in request.symbols:
        symbol_upper = symbol.upper()
        ticker_data_full = fetch_ticker_data(symbol_upper)
        if not ticker_data_full or ticker_data_full.get("history") is None or ticker_data_full["history"].empty:
            return OptimizePortfolioResponse(error=f"Could not fetch historical price data for {symbol_upper}")
        history_df = ticker_data_full["history"]["Close"]
        if len(history_df) < min_data_points_for_opt:
            return OptimizePortfolioResponse(error=f"Insufficient historical data for {symbol_upper}")
        price_history_for_optimization[symbol_upper] = history_df
    if not price_history_for_optimization or len(price_history_for_optimization) < 1:
        return OptimizePortfolioResponse(error="No valid price histories collected.")
    optimal_weights, expected_return, volatility, sharpe_ratio = optimize_portfolio_mpt(
        price_history_dict=price_history_for_optimization, 
        risk_free_rate=request.risk_free_rate
    )
    if optimal_weights is None:
        return OptimizePortfolioResponse(error="Portfolio optimization failed.")
    serialized_weights = {k: float(v) if isinstance(v, np.float64) else v for k, v in optimal_weights.items()}
    return OptimizePortfolioResponse(
        weights=serialized_weights,
        expected_return=float(expected_return) if expected_return is not None else None,
        volatility=float(volatility) if volatility is not None else None,
        sharpe_ratio=float(sharpe_ratio) if sharpe_ratio is not None else None
    )

class ReportType(str, Enum):
    ANALYSIS = "analysis"
    PREDICTION = "prediction"
    OPTIMIZATION = "optimization"

@app.post("/download/{report_type}", tags=["Data Export"])
async def download_report(report_type: ReportType, request_body: Dict[str, Any]):
    """Generate and download reports in CSV format."""
    df = pd.DataFrame()
    filename = f"{report_type.value}_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"

    if report_type == ReportType.ANALYSIS:
        symbols_request = SymbolsRequest(**request_body)
        analysis_data = await analyze_tickers(symbols_request)
        # Flatten the analysis data for CSV
        records = []
        for symbol, data in analysis_data.items():
            if "error" not in data:
                record = {"symbol": symbol}
                # Add fundamentals, handling potential missing keys or complex structures
                fundamentals = data.get("fundamentals", {})
                if isinstance(fundamentals, dict):
                    for k, v in fundamentals.items():
                        if not isinstance(v, (dict, list)):
                            record[f"fundamental_{k}"] = v
                # For simplicity, we're not deeply flattening historical_prices or dividend_data here
                # but one might choose to export e.g., latest price or summary stats
                records.append(record)
        df = pd.DataFrame(records)

    elif report_type == ReportType.PREDICTION:
        predict_request = PredictRequest(**request_body)
        prediction_data = await predict_ticker_movement(predict_request)
        if prediction_data.error:
            return {"error": prediction_data.error}
        data_dict = prediction_data.dict()
        # Remove features_used if it's too complex for a single CSV row, or flatten it
        if 'features_used' in data_dict and isinstance(data_dict['features_used'], dict):
            features_flat = {f"feature_{k}": v for k,v in data_dict['features_used'].items()}
            del data_dict['features_used']
            data_dict.update(features_flat)
        df = pd.DataFrame([data_dict])

    elif report_type == ReportType.OPTIMIZATION:
        optimize_request = OptimizePortfolioRequest(**request_body)
        optimization_data = await optimize_portfolio(optimize_request)
        if optimization_data.error:
            return {"error": optimization_data.error}
        # Prepare data for DataFrame
        opt_data_dict = optimization_data.dict()
        if opt_data_dict.get('weights'):
            weights_df = pd.DataFrame(list(opt_data_dict['weights'].items()), columns=['symbol', 'weight'])
            summary_df = pd.DataFrame([{
                'expected_return': opt_data_dict.get('expected_return'),
                'volatility': opt_data_dict.get('volatility'),
                'sharpe_ratio': opt_data_dict.get('sharpe_ratio')
            }])
            # For simplicity, returning weights. Could combine with summary.
            df = weights_df 
        else:
            df = pd.DataFrame([opt_data_dict])

    if df.empty:
        return {"message": f"No data to export for {report_type.value} with the given parameters."}

    stream = io.StringIO()
    df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    return response

if __name__ == "__main__":
    pass

