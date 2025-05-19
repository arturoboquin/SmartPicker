# /home/ubuntu/etf_stock_picker_app/backend/model_trainer.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import sys

# Adjust Python path to include the current directory for module imports when run as a script
# This allows 'from feature_engineering import ...' to work if 'backend' is the CWD.
# More robust solutions involve running as a module (python -m backend.model_trainer) or proper packaging.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from feature_engineering import engineer_features_for_ml
from labeling_strategy import assign_labels

MODEL_DIR = "/home/ubuntu/etf_stock_picker_app/models"
MODEL_FILENAME = "long_term_prediction_model.joblib"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

def train_prediction_model(historical_data_10y: pd.DataFrame, 
                             feature_lookback_days: int = 252, 
                             label_forward_days: int = 126, 
                             test_size: float = 0.2, 
                             random_state: int = 42):
    """
    Trains a Random Forest classifier to predict long-term (6-month) stock price movement.
    """
    if not isinstance(historical_data_10y, pd.DataFrame) or historical_data_10y.empty:
        print("Error: Historical data is empty or not a DataFrame.")
        return None, None, None, None, None
    
    print("Step 1: Engineering features...")
    features_df = engineer_features_for_ml(historical_data_10y, lookback_days=feature_lookback_days)
    if features_df.empty:
        print("Error: Feature engineering resulted in an empty DataFrame.")
        return None, None, None, None, None

    print("Step 2: Assigning labels...")
    labeled_df = assign_labels(features_df, forward_days=label_forward_days)
    if labeled_df.empty:
        print("Error: Labeling resulted in an empty DataFrame.")
        return None, None, None, None, None

    original_cols = set(historical_data_10y.columns)
    # These are columns that were *added* by feature engineering
    engineered_feature_names = [col for col in features_df.columns if col not in original_cols]
    
    # We need to ensure that the columns selected for X are only the engineered features
    # and do not include future info like 'label' or 'forward_return'
    # Also, ensure these features actually exist in labeled_df (they should, as it's based on features_df)
    final_feature_columns = [col for col in engineered_feature_names if col in labeled_df.columns and col not in ["label", "forward_return"]]

    df_for_training = labeled_df[final_feature_columns + ["label"]].copy()
    print(f"Before NaN handling - Shape: {df_for_training.shape}")
    print(f"NaN counts per column: {df_for_training.isna().sum().sum()}")
    print(f"Inf counts: {np.isinf(df_for_training.select_dtypes(include=[np.number])).sum().sum()}")
    
    # Replace inf values with NaN
    df_for_training.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Instead of dropping all NaN rows, fill NaNs with appropriate values
    # For numeric columns, fill with median
    numeric_cols = df_for_training.select_dtypes(include=[np.number]).columns
    # Check if we have any non-NaN values to calculate medians
    if not df_for_training[numeric_cols].notna().any().any():
        print("Error: All numeric values are NaN. Using zeros instead.")
        df_for_training[numeric_cols] = df_for_training[numeric_cols].fillna(0)
    else:
        # Fill NaNs with column medians where possible
        for col in numeric_cols:
            if df_for_training[col].notna().any():
                df_for_training[col] = df_for_training[col].fillna(df_for_training[col].median())
            else:
                df_for_training[col] = df_for_training[col].fillna(0)
    
    # For any remaining NaNs (non-numeric columns), drop those rows
    df_for_training.dropna(inplace=True)
    
    print(f"After NaN handling - Shape: {df_for_training.shape}")

    if df_for_training.empty:
        print("Error: DataFrame is empty after dropping NaNs. Not enough data or too many NaNs in features/labels.")
        return None, None, None, None, None
    
    X = df_for_training[final_feature_columns]
    y = df_for_training["label"]

    if X.empty or y.empty:
        print("Error: Feature set X or target y is empty after processing.")
        return None, None, None, None, None
    if len(y.unique()) < 2:
        print(f"Error: Target variable 'label' has only {len(y.unique())} unique classes. Needs at least 2 for classification. Labels found: {y.unique()}")
        return None, None, None, None, None

    print(f"Step 3: Splitting data... Features shape: {X.shape}, Target shape: {y.shape}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    print("Step 4: Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced')
    
    print("Step 5: Performing 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"Cross-validation accuracy scores: {cv_scores}")
    print(f"Mean CV accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    model.fit(X_train, y_train)

    # Save feature names with the model if possible (sklearn >= 0.24 stores feature_names_in_)
    # This is automatically handled by scikit-learn if X_train is a Pandas DataFrame.

    print("Step 6: Evaluating model on test set...")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred, labels=model.classes_)
    
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", pd.DataFrame(conf_matrix, index=model.classes_, columns=model.classes_))

    print("Step 7: Saving model...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    # Save the model and the list of feature columns it was trained on
    model_payload = {
        "model": model,
        "feature_columns": final_feature_columns # Save the exact feature names
    }
    joblib.dump(model_payload, MODEL_PATH)
    print(f"Model and feature columns saved to {MODEL_PATH}")

    return model, final_feature_columns, report, conf_matrix, cv_scores

if __name__ == '__main__':
    print("--- Model Training Example ---")
    num_days_data = 252 * 10 + 200 
    dates = pd.date_range(start='2010-01-01', periods=num_days_data, freq='B')
    data_size = len(dates)
    
    ohlcv_data = pd.DataFrame({
        'Open': np.random.rand(data_size) * 100 + 50,
        'High': np.random.rand(data_size) * 100 + 55, 
        'Low': np.random.rand(data_size) * 100 + 45,
        'Close': np.random.rand(data_size) * 100 + 50,
        'Volume': np.random.randint(100000, 1000000, size=data_size)
    }, index=dates)
    ohlcv_data['High'] = ohlcv_data[['Open', 'Close']].max(axis=1) + np.random.uniform(0, 5, size=data_size)
    ohlcv_data['Low'] = ohlcv_data[['Open', 'Close']].min(axis=1) - np.random.uniform(0, 5, size=data_size)
    ohlcv_data = ohlcv_data.clip(lower=0.01)

    print(f"Generated sample historical data with {len(ohlcv_data)} rows.")

    # Skip pandas_ta dependency for now since we're having compatibility issues
    print("Skipping pandas_ta dependency check to avoid import errors.")

    # Ensure feature_engineering and labeling_strategy can be imported
    # This is handled by sys.path.append(SCRIPT_DIR) at the top
    trained_model, features_used, cl_report, c_matrix, cv_s = train_prediction_model(ohlcv_data)

    if trained_model:
        print("\n--- Training Summary ---")
        print("Model training completed successfully.")
        print(f"Features used for training: {features_used}")
    else:
        print("\nModel training failed.")

