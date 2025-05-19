# Intelligent ETF & Stock Picker

A comprehensive application for analyzing ETFs and stocks, predicting price movements, and optimizing portfolios using machine learning.

## Features

- **Data Analysis**: Fetch and analyze historical price data, fundamentals, and dividends for any ticker symbol
- **Machine Learning Predictions**: Predict future price movements (Buy/Hold/Sell) using a trained Random Forest classifier
- **Portfolio Optimization**: Calculate optimal portfolio weights to maximize Sharpe ratio
- **Data Export**: Download analysis, prediction, and optimization results as CSV files
- **Interactive UI**: User-friendly interface for configuring and visualizing results

## Architecture

The application consists of two main components:

1. **Backend (FastAPI)**: Handles data fetching, caching, analysis, prediction, and optimization
2. **Frontend (React)**: Provides an intuitive user interface for interacting with the backend

## Installation

### Prerequisites

- Python 3.11+
- Node.js 16+
- pip and npm/yarn

### Backend Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/etf-stock-picker.git
   cd etf-stock-picker
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install backend dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

4. (Optional) Install additional dependencies for enhanced features:
   ```bash
   pip install pandas_ta cvxpy[ECOS]
   ```

### Frontend Setup

1. Install frontend dependencies:
   ```bash
   cd ../frontend
   npm install  # or: yarn install
   ```

## Usage

### Running the Backend

1. Start the FastAPI server:
   ```bash
   cd backend
   python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. Access the API documentation:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Running the Frontend

1. Start the development server:
   ```bash
   cd frontend
   npm run dev  # or: yarn dev
   ```

2. Access the application in your browser:
   - http://localhost:5173 (or the port shown in your terminal)

### Training the ML Model

The application requires a trained machine learning model to enable prediction functionality. To generate this model:

```bash
cd backend
python model_trainer.py
```

This will:
- Generate sample data (or you can modify to use real data)
- Engineer features from historical price data
- Train a Random Forest classifier
- Save the model to `/models/long_term_prediction_model.joblib`

## API Endpoints

Once the backend is running, the following endpoints are available:

- `GET /health`: Check API health status
- `POST /analyze`: Analyze ticker symbols and return historical data and fundamentals
- `POST /predict`: Predict future price movement (Buy/Hold/Sell) for a given ticker
- `POST /optimize`: Calculate optimal portfolio weights for a set of tickers
- `POST /download/{report_type}`: Generate and download CSV reports

## Data Pipeline

The application follows this data pipeline:

1. **Data Fetching**: Historical price data is fetched from Yahoo Finance API
2. **Data Caching**: Fetched data is cached to minimize API calls
3. **Feature Engineering**: Technical indicators and other features are calculated
4. **Prediction**: ML model predicts future price movements
5. **Optimization**: Modern Portfolio Theory is applied for portfolio optimization

## Machine Learning Approach

The prediction model:
- Uses a Random Forest classifier
- Trained on 10+ years of historical data
- Features include momentum indicators, moving averages, volatility metrics, and technical indicators
- Predicts 6-month forward price movement (Buy/Hold/Sell)
- Achieves approximately 60% accuracy on test data

## Deployment

### Local Deployment

Follow the installation and usage instructions above.

### Production Deployment

For production deployment:

1. Build the frontend:
   ```bash
   cd frontend
   npm run build  # or: yarn build
   ```

2. Serve the static files and backend using a production WSGI server:
   ```bash
   cd backend
   gunicorn -k uvicorn.workers.UvicornWorker main:app
   ```

3. Use a reverse proxy (Nginx, Apache) to serve the frontend static files and route API requests to the backend.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Yahoo Finance API](https://pypi.org/project/yfinance/) for financial data
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
- [React](https://reactjs.org/) for the frontend framework
- [scikit-learn](https://scikit-learn.org/) for machine learning capabilities
