import { useState } from 'react';
import './App.css';

// Define interfaces for API responses for better type checking
interface FundamentalData {
  // Define based on actual yFinance info structure, e.g.,
  shortName?: string;
  symbol?: string;
  marketCap?: number;
  beta?: number;
  trailingPE?: number;
  forwardPE?: number;
  dividendYield?: number;
  // Add other relevant fields
}

interface PricePoint {
  Date: string; // Or Date object, ensure consistent parsing
  Open: number;
  High: number;
  Low: number;
  Close: number;
  Volume: number;
  Dividends: number;
  Stock_Splits: number;
}

interface DividendPoint {
    Date: string; // Or Date object
    Dividends: number;
}

interface AnalysisResult {
  fundamentals: FundamentalData | null;
  historical_prices: PricePoint[] | null;
  dividend_data: DividendPoint[] | null;
  error?: string;
}

interface PredictionResult {
  label: string;
  confidence?: number;
  features_used?: Record<string, any>;
  error?: string;
}

interface OptimizationResult {
  weights?: Record<string, number>;
  expected_return?: number;
  volatility?: number;
  sharpe_ratio?: number;
  error?: string;
}

const API_BASE_URL = "https://8000-icrhswvk4rbzcacj3zpwp-db60917a.manus.computer";

function App() {
  const [tickers, setTickers] = useState<string>('AAPL,MSFT,GOOG');
  const [predictionHorizon, setPredictionHorizon] = useState<number>(252);
  const [optimizePortfolio, setOptimizePortfolio] = useState<boolean>(false);
  const [analysisResults, setAnalysisResults] = useState<Record<string, AnalysisResult> | null>(null);
  const [predictionResults, setPredictionResults] = useState<Record<string, PredictionResult>>({});
  const [optimizationResult, setOptimizationResult] = useState<OptimizationResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);
    setAnalysisResults(null);
    setPredictionResults({});
    setOptimizationResult(null);

    const symbolsArray = tickers.split(',').map(s => s.trim()).filter(s => s);
    if (symbolsArray.length === 0) {
      setError('Please enter at least one ticker symbol.');
      setLoading(false);
      return;
    }

    try {
      // 1. Fetch Analysis Data
      const analyzeResponse = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbols: symbolsArray }),
      });
      if (!analyzeResponse.ok) throw new Error(`Analysis API Error: ${analyzeResponse.statusText}`);
      const analysisData = await analyzeResponse.json();
      setAnalysisResults(analysisData);

      // 2. Fetch Predictions for each ticker
      const newPredictionResults: Record<string, PredictionResult> = {};
      for (const symbol of symbolsArray) {
        try {
            const predictResponse = await fetch(`${API_BASE_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symbol: symbol, lookback_days: predictionHorizon }),
            });
            if (!predictResponse.ok) {
                const errData = await predictResponse.json();
                newPredictionResults[symbol] = { label: 'Error', error: errData.detail || `Prediction API Error for ${symbol}: ${predictResponse.statusText}` };
                continue;
            }
            newPredictionResults[symbol] = await predictResponse.json();
        } catch (predError: any) {
            newPredictionResults[symbol] = { label: 'Error', error: `Failed to get prediction for ${symbol}: ${predError.message}` };
        }
      }
      setPredictionResults(newPredictionResults);

      // 3. Fetch Optimization Data (if checked)
      if (optimizePortfolio && symbolsArray.length > 0) {
        const optimizeResponse = await fetch(`${API_BASE_URL}/optimize`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ symbols: symbolsArray, risk_free_rate: 0.02 }), // Assuming default risk_free_rate
        });
        if (!optimizeResponse.ok) throw new Error(`Optimization API Error: ${optimizeResponse.statusText}`);
        const optimizationData = await optimizeResponse.json();
        setOptimizationResult(optimizationData);
      }

    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async (reportType: string) => {
    const symbolsArray = tickers.split(',').map(s => s.trim()).filter(s => s);
    let requestBody = {};
    let endpoint = `${API_BASE_URL}/download/${reportType}`;

    if (reportType === 'analysis') {
        if (symbolsArray.length === 0) { alert('Please enter symbols for analysis report.'); return; }
        requestBody = { symbols: symbolsArray };
    } else if (reportType === 'prediction') {
        if (symbolsArray.length === 0) { alert('Please enter a symbol for prediction report.'); return; } // Assuming prediction download is for first symbol for simplicity
        if (symbolsArray.length > 1) { alert('Prediction download is for a single symbol. Using the first one: ' + symbolsArray[0]);}
        requestBody = { symbol: symbolsArray[0], lookback_days: predictionHorizon };
    } else if (reportType === 'optimization') {
        if (!optimizePortfolio || symbolsArray.length === 0) { alert('Please run optimization first or enter symbols.'); return; }
        requestBody = { symbols: symbolsArray, risk_free_rate: 0.02 };
    }

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody),
        });
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `Download failed: ${response.statusText}`);
        }
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${reportType}_report.csv`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
    } catch (err: any) {
        setError(`Download error: ${err.message}`);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Intelligent ETF & Stock Picker</h1>
      </header>
      <main>
        <div className="input-panel">
          <h2>Configuration</h2>
          <div>
            <label htmlFor="tickers">Tickers (comma-separated):</label>
            <input 
              type="text" 
              id="tickers" 
              value={tickers} 
              onChange={(e) => setTickers(e.target.value)} 
            />
          </div>
          <div>
            <label htmlFor="predictionHorizon">Prediction Lookback (days):</label>
            <input 
              type="number" 
              id="predictionHorizon" 
              value={predictionHorizon} 
              onChange={(e) => setPredictionHorizon(parseInt(e.target.value, 10))} 
            />
          </div>
          <div>
            <input 
              type="checkbox" 
              id="optimizePortfolio" 
              checked={optimizePortfolio} 
              onChange={(e) => setOptimizePortfolio(e.target.checked)} 
            />
            <label htmlFor="optimizePortfolio">Optimize Portfolio (Max Sharpe Ratio)</label>
          </div>
          <button onClick={handleAnalyze} disabled={loading}>
            {loading ? 'Analyzing...' : 'Analyze & Predict'}
          </button>
        </div>

        {error && <div className="error-message">Error: {error}</div>}

        <div className="download-buttons">
            <h3>Download Reports (CSV)</h3>
            <button onClick={() => handleDownload('analysis')} disabled={loading || !analysisResults}>Download Analysis Data</button>
            <button onClick={() => handleDownload('prediction')} disabled={loading || Object.keys(predictionResults).length === 0}>Download Prediction Data</button>
            {optimizePortfolio && (
                <button onClick={() => handleDownload('optimization')} disabled={loading || !optimizationResult}>Download Optimization Data</button>
            )}
        </div>

        {analysisResults && (
          <div className="results-panel">
            <h2>Analysis Results</h2>
            {Object.entries(analysisResults).map(([symbol, data]) => (
              <div key={symbol} className="ticker-result">
                <h3>{symbol.toUpperCase()}</h3>
                {data.error ? (
                    <p className="error-message">Error: {data.error}</p>
                ) : (
                    <>
                        <h4>Fundamentals</h4>
                        {data.fundamentals ? (
                            <ul>
                                {Object.entries(data.fundamentals).map(([key, value]) => {
                                    if (typeof value === 'string' || typeof value === 'number') {
                                        return <li key={key}><strong>{key}:</strong> {value}</li>;
                                    }
                                    return null;
                                })}
                            </ul>
                        ) : <p>No fundamental data available.</p>}
                        
                        {predictionResults[symbol] && (
                            <>
                                <h4>Prediction</h4>
                                {predictionResults[symbol].error ? (
                                    <p className="error-message">Prediction Error: {predictionResults[symbol].error}</p>
                                ) : (
                                    <p>
                                        <strong>Label:</strong> {predictionResults[symbol].label} 
                                        {predictionResults[symbol].confidence && ` (Confidence: ${(predictionResults[symbol].confidence! * 100).toFixed(2)}%)`}
                                    </p>
                                )}
                            </>
                        )}
                        {/* Placeholder for charts - would use a charting library like Recharts or Chart.js */}
                        {data.historical_prices && <p><em>Price chart would be here. ({data.historical_prices.length} data points)</em></p>}
                        {data.dividend_data && <p><em>Dividend chart would be here. ({data.dividend_data.length} data points)</em></p>}
                    </>
                )}
              </div>
            ))}
          </div>
        )}

        {optimizePortfolio && optimizationResult && (
          <div className="results-panel">
            <h2>Portfolio Optimization Result</h2>
            {optimizationResult.error ? (
                <p className="error-message">Error: {optimizationResult.error}</p>
            ) : (
                <>
                    <p><strong>Expected Annual Return:</strong> {(optimizationResult.expected_return! * 100).toFixed(2)}%</p>
                    <p><strong>Annual Volatility:</strong> {(optimizationResult.volatility! * 100).toFixed(2)}%</p>
                    <p><strong>Sharpe Ratio:</strong> {optimizationResult.sharpe_ratio!.toFixed(4)}</p>
                    <h4>Optimal Weights:</h4>
                    <ul>
                        {optimizationResult.weights && Object.entries(optimizationResult.weights).map(([symbol, weight]) => (
                            <li key={symbol}><strong>{symbol}:</strong> {(weight * 100).toFixed(2)}%</li>
                        ))}
                    </ul>
                </>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;

