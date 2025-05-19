# Machine Learning Pipeline Documentation

This document provides detailed information about the machine learning approach used in the ETF & Stock Picker application.

## Data Requirements

The prediction model requires:
- At least 1 year (252 trading days) of historical OHLCV (Open, High, Low, Close, Volume) data
- Preferably 10+ years of data for robust training and validation
- Daily price data for accurate feature engineering

## Feature Engineering

The following features are engineered from historical price data:

### Price Momentum
- 20-day returns
- 50-day returns
- 100-day returns

### Moving Averages
- 20-day Simple Moving Average (SMA)
- 20-day Exponential Moving Average (EMA)
- 50-day SMA and EMA
- 200-day SMA and EMA

### Volatility Metrics
- 30-day rolling volatility (standard deviation of daily returns)

### Technical Indicators
When pandas_ta is available, these additional indicators are calculated:
- Relative Strength Index (RSI, 14-day)
- Moving Average Convergence Divergence (MACD, 12/26/9)
- On-Balance Volume (OBV)
- Average True Range (ATR, 14-day)

### Additional Features
- Volume trend (20-day vs 200-day average volume)
- Distance from 52-week high
- Distance from 52-week low

## Labeling Strategy

The model predicts future price movements using the following labeling strategy:

- **Buy**: When 6-month forward return > +15%
- **Sell**: When 6-month forward return < -15%
- **Hold**: When 6-month forward return is between -15% and +15%

This creates a classification problem with three classes.

## Model Architecture

The prediction model uses:
- **Algorithm**: Random Forest Classifier
- **Parameters**:
  - n_estimators: 100
  - class_weight: 'balanced' (to handle class imbalance)
  - random_state: 42 (for reproducibility)

## Training Process

The model training process includes:
1. **Data Preparation**:
   - Feature engineering from historical price data
   - Labeling based on forward returns
   - Handling of NaN values and outliers

2. **Train-Test Split**:
   - 80% training data, 20% test data
   - Stratified sampling to maintain class distribution

3. **Cross-Validation**:
   - 5-fold stratified cross-validation
   - Accuracy scoring

4. **Model Evaluation**:
   - Classification report (precision, recall, F1-score)
   - Confusion matrix

## Performance Metrics

Typical performance metrics for the model:
- **Accuracy**: ~55-60% (significantly above random chance for a 3-class problem)
- **Precision/Recall**: Varies by class, typically higher for Buy/Sell than Hold
- **Cross-validation**: Consistent performance across folds indicates model stability

## Limitations

- The model does not account for market regime changes or black swan events
- Technical indicators are limited when pandas_ta is not available
- Prediction is probabilistic and should be used as one input among many for investment decisions
- Past performance does not guarantee future results

## Reproducibility

To reproduce the model training:
1. Ensure all dependencies are installed
2. Run `python model_trainer.py` from the backend directory
3. The model will be saved to `/models/long_term_prediction_model.joblib`

## Data Sources

The application uses:
- **Yahoo Finance API** (via yfinance): For historical price data and fundamentals
- **Local caching**: To minimize API calls and improve performance

## Future Improvements

Potential enhancements to the ML pipeline:
- Implement feature importance analysis and selection
- Add sentiment analysis from news and social media
- Explore deep learning approaches (LSTM, Transformer models)
- Implement ensemble methods combining multiple model types
- Add backtesting framework for strategy validation
