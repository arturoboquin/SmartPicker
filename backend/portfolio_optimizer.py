# /home/ubuntu/etf_stock_picker_app/backend/portfolio_optimizer.py

import pandas as pd
import numpy as np
import cvxpy as cp

# Assuming data_fetcher.py is available to get historical prices
# from .data_fetcher import fetch_ticker_data # For standalone testing

def calculate_portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate):
    """Calculates portfolio return, volatility, and Sharpe ratio."""
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / (portfolio_volatility + 1e-9) # Add epsilon for stability
    return portfolio_return, portfolio_volatility, sharpe_ratio

def optimize_portfolio_mpt(price_history_dict: dict, risk_free_rate: float = 0.02):
    """
    Optimizes asset allocation using Modern Portfolio Theory (MPT) to maximize Sharpe ratio.

    Args:
        price_history_dict (dict): A dictionary where keys are ticker symbols and 
                                   values are pandas Series or DataFrames containing 
                                   historical 'Close' prices for each ticker.
                                   The DataFrames must have a DatetimeIndex.
        risk_free_rate (float): The risk-free rate of return (annualized, e.g., 0.02 for 2%).

    Returns:
        tuple: (optimal_weights_dict, expected_return, volatility, sharpe_ratio) or (None, None, None, None) if error.
    """
    if not isinstance(price_history_dict, dict) or not price_history_dict:
        print("Error: price_history_dict must be a non-empty dictionary.")
        return None, None, None, None

    symbols = list(price_history_dict.keys())
    num_assets = len(symbols)

    if num_assets == 0:
        print("Error: No assets provided for optimization.")
        return None, None, None, None
    
    # Combine historical close prices into a single DataFrame
    # Ensure all series have the same DatetimeIndex and are aligned
    all_prices_list = []
    for symbol in symbols:
        if not isinstance(price_history_dict[symbol], (pd.Series, pd.DataFrame)):
            print(f"Error: Price history for {symbol} is not a Series or DataFrame.")
            return None, None, None, None
        
        prices = price_history_dict[symbol]
        if isinstance(prices, pd.DataFrame):
            if 'Close' not in prices.columns:
                print(f"Error: 'Close' column missing in price history for {symbol}.")
                return None, None, None, None
            prices = prices['Close']
        
        prices.name = symbol # Ensure series has a name for joining
        all_prices_list.append(prices)
    
    try:
        # Concatenate all price series, aligning by index (date)
        # Using outer join to keep all dates, then forward fill and back fill NaNs
        # This assumes that missing data for one stock on a day it didn't trade can be filled
        # A more robust approach might involve more sophisticated NaN handling or ensuring input data is clean.
        combined_prices_df = pd.concat(all_prices_list, axis=1, join='outer')
        combined_prices_df.ffill(inplace=True)
        combined_prices_df.bfill(inplace=True)
        combined_prices_df.dropna(inplace=True) # Drop any remaining NaNs (e.g., if a stock has no data at all)
    except Exception as e:
        print(f"Error concatenating price data: {e}")
        return None, None, None, None

    if combined_prices_df.empty or combined_prices_df.shape[0] < 2: # Need at least 2 data points for returns
        print("Error: Combined price data is empty or has insufficient history after processing.")
        return None, None, None, None

    # Calculate daily returns
    daily_returns = combined_prices_df.pct_change().dropna()

    if daily_returns.empty or daily_returns.shape[0] < 2:
        print("Error: Daily returns data is empty or insufficient after processing.")
        return None, None, None, None

    # Calculate mean daily returns and covariance matrix
    # Annualize them (assuming 252 trading days in a year)
    mean_daily_returns = daily_returns.mean()
    annualized_mean_returns = mean_daily_returns * 252
    
    cov_matrix_daily = daily_returns.cov()
    annualized_cov_matrix = cov_matrix_daily * 252

    # Define optimization problem using cvxpy
    weights = cp.Variable(num_assets)
    portfolio_return_expr = annualized_mean_returns.values @ weights
    portfolio_risk_expr = cp.quad_form(weights, annualized_cov_matrix.values)

    # Objective: Maximize Sharpe Ratio = (Return - RiskFreeRate) / Risk
    # This is equivalent to maximizing (Return - RiskFreeRate) for a given risk, or minimizing risk for a given return.
    # To directly maximize Sharpe, it's often reformulated or solved iteratively.
    # A common approach is to fix portfolio variance (risk) and maximize risk-adjusted return,
    # or fix return and minimize variance.
    # For maximizing Sharpe directly with cvxpy, we can use a trick:
    # Maximize: y.T @ mu - rf * k
    # Subject to: y.T @ Sigma @ y <= 1 (or == 1)
    #             y.T @ 1 == k
    #             y >= 0, k >= 0
    #             And then weights = y / k
    # This is a bit more complex. A simpler approach for this context might be to minimize variance for a target return,
    # or maximize return for a target variance, and then iterate to find max Sharpe, or use a direct solver if available.

    # Simpler: Minimize volatility subject to constraints (classic Markowitz)
    # For maximizing Sharpe, we can iterate over target returns or use a solver that handles this objective.
    # cvxpy can handle maximizing (mu.T @ w - rf) / cp.sqrt(cp.quad_form(w, Sigma)) if the problem is DCP after transformation.
    # However, the direct Sharpe ratio maximization is non-convex. 
    # We will maximize (expected_return - risk_free_rate) / volatility by minimizing volatility for a range of returns
    # and picking the one with the highest Sharpe, or use a common reformulation.

    # For this implementation, we'll use the formulation to maximize (return - rf) / std_dev
    # which can be done by minimizing risk (variance) for a given expected return.
    # We will target maximizing the Sharpe Ratio directly if possible, or use a common QP formulation.
    
    # Maximize Sharpe Ratio: (portfolio_return - risk_free_rate) / portfolio_volatility
    # This is a non-convex problem. We can solve it by transforming it.
    # Let K = 1 / portfolio_volatility and y = weights * K.
    # Maximize: annualized_mean_returns.values @ y - risk_free_rate * K
    # Subject to: cp.quad_form(y, annualized_cov_matrix.values) <= 1 (or == 1)
    #             cp.sum(y) == K
    #             y >= 0
    #             K >= 0 (or K > 0)
    # The actual weights are y / K.

    # Using a simpler QP: Minimize risk for a set of target returns and find max Sharpe
    # Or, more directly for max Sharpe (if solver supports it or using a common trick):
    # We will use the common approach of minimizing variance (0.5 * w.T * Sigma * w)
    # subject to sum(w) = 1, w >= 0, and optionally a target return constraint.
    # To maximize Sharpe, we can iterate or use a specific solver setup.
    
    # Let's use the formulation for maximizing Sharpe ratio by minimizing risk, assuming a fixed risk aversion or iterating.
    # For simplicity and directness, we'll try to maximize the portfolio return for a given level of risk (variance).
    # Or, more standardly, minimize variance for a target return.
    # The problem statement asks for maximizing Sharpe. A common way is to solve: 
    # min 0.5 * w' * Sigma * w  s.t. mu' * w = target_ret, sum(w) = 1, w >= 0.
    # Then iterate target_ret to find max Sharpe.

    # A direct QP for max Sharpe (often used): 
    # Maximize mu.T * w - lambda_risk_aversion * 0.5 * w.T * Sigma * w
    # This requires choosing lambda. 

    # We will use the formulation to maximize (portfolio_return - risk_free_rate) subject to a fixed volatility, 
    # or minimize volatility subject to a minimum return. 
    # The most straightforward for max Sharpe with cvxpy without complex transformations is often to iterate or use a solver that handles it.

    # Let's try a direct formulation if cvxpy handles it, or a common alternative.
    # The problem is to maximize (mu.T @ w - rf) / sqrt(w.T @ Sigma @ w)
    # This is not DCP. We use the variable transformation y = w / (sum of w_abs) and maximize (mu.T @ y - rf) / sqrt(y.T @ Sigma @ y)
    # subject to sum(y) = 1, y_i >= 0.
    # This is still tricky. Let's use a standard MVO approach: minimize variance for a target return.
    # Then we can iterate through target returns to find the one that maximizes Sharpe. 
    # Or, a simpler approach for this exercise: just minimize variance. Then calculate Sharpe.
    # The spec says "maximize (mean_returns - risk_free_rate) / portfolio_std"

    # Maximize Sharpe Ratio using cvxpy (common formulation for QP solvers)
    # Let x_i be the amount invested in asset i. Then sum(x_i) = Total Budget (e.g., 1)
    # Maximize (r_bar.T @ x - r_f) / sqrt(x.T @ Sigma @ x)
    # This is equivalent to: Min x.T @ Sigma @ x  s.t. r_bar.T @ x - r_f = 1 (or some constant > 0)
    # and sum(x) = K (another variable), x_i >= 0. Then weights = x / K.

    # Using the formulation from CVXPY examples for maximizing Sharpe ratio:
    # Maximize mu.T * x - rf
    # Subject to: sum(x) = 1, x >= 0, cp.quad_form(x, Sigma) == 1 (this fixes volatility to 1, then scale)
    # This is not quite right. 

    # Let's use the standard approach: minimize variance for a range of returns, then pick max Sharpe.
    # For a single optimization pass as requested by the spec (maximize Sharpe directly):
    # We can use a trick: Maximize w^T * mu / sqrt(w^T * Sigma * w)
    # This is equivalent to minimizing w^T * Sigma * w subject to w^T * mu = C (constant)
    # and sum(w) = 1, w >= 0. Iterate C. 

    # A more direct (but potentially complex) cvxpy formulation for max Sharpe:
    # Let y = w / ( (mu-rf)^T w ). Maximize 1 / sqrt(y^T Sigma y) subject to (mu-rf)^T y = 1, sum(y) = K, y_i >=0.
    # This is also not straightforward. 

    # Simplest for now: Minimize portfolio variance. Then calculate Sharpe. This is not Max Sharpe.
    # To truly maximize Sharpe: We need a specific solver or iterate.
    # Let's use a common QP formulation that maximizes Sharpe by transforming variables.
    # (From Wikipedia / common finance texts for MPT with short-selling allowed, then adapt)
    # For no short-selling: 
    #   y = cp.Variable(num_assets)
    #   k = cp.Variable()
    #   objective = cp.Maximize(annualized_mean_returns.values @ y - risk_free_rate * k)
    #   constraints = [cp.sum(y) == k, cp.quad_form(y, annualized_cov_matrix.values) <= 1, k >= 0, y >= 0]
    #   prob = cp.Problem(objective, constraints)
    #   prob.solve()
    #   if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
    #       optimal_k = k.value
    #       optimal_y = y.value
    #       optimal_weights = optimal_y / optimal_k if optimal_k > 1e-6 else np.zeros(num_assets)
    #   else: # Error or infeasible
    #       print(f"Optimization failed or was infeasible. Status: {prob.status}")
    #       return None, None, None, None
    # This formulation is for maximizing the Sharpe ratio directly.

    # Let's use the above formulation for maximizing Sharpe Ratio.
    y_var = cp.Variable(num_assets, name="y_allocations")
    k_var = cp.Variable(name="k_scaling_factor")

    objective = cp.Maximize(annualized_mean_returns.values @ y_var - risk_free_rate * k_var)
    constraints = [
        cp.sum(y_var) == k_var, 
        cp.quad_form(y_var, annualized_cov_matrix.values) <= 1.0, # Portfolio variance constraint related to y
        k_var >= 1e-9, # k must be positive
        y_var >= 0  # No short selling
    ]
    problem = cp.Problem(objective, constraints)
    
    try:
        # Some solvers are better than others. ECOS is good for this type of SOCP.
        # CVXPY will pick one. Default is often OSQP or SCS.
        problem.solve(solver=cp.ECOS) # Try ECOS, or let CVXPY choose
    except Exception as e:
        print(f"Solver error during optimization: {e}")
        # Fallback to default solver if ECOS fails or is not available
        try:
            problem.solve()
        except Exception as e2:
            print(f"Fallback solver error: {e2}")
            return None, None, None, None

    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        if k_var.value is not None and k_var.value > 1e-9:
            optimal_weights_array = y_var.value / k_var.value
        else: # Should not happen if k_var >= 1e-9 is enforced and solution is optimal
            print("Warning: k_var is too small or None after optimization. Setting weights to zero.")
            optimal_weights_array = np.zeros(num_assets)
        
        # Ensure weights sum to 1 (they should by formulation, but floating point errors)
        optimal_weights_array = optimal_weights_array / np.sum(optimal_weights_array)
        optimal_weights_array = np.clip(optimal_weights_array, 0, 1) # Clip to handle tiny negatives from solver
        optimal_weights_array = optimal_weights_array / np.sum(optimal_weights_array) # Re-normalize

        opt_return, opt_volatility, opt_sharpe = calculate_portfolio_performance(
            optimal_weights_array, annualized_mean_returns.values, annualized_cov_matrix.values, risk_free_rate
        )
        optimal_weights_dict = {symbols[i]: optimal_weights_array[i] for i in range(num_assets)}
        return optimal_weights_dict, opt_return, opt_volatility, opt_sharpe
    else:
        print(f"Optimization failed. Problem status: {problem.status}")
        print(f"k_var value: {k_var.value if k_var is not None else 'None'}")
        print(f"y_var value: {y_var.value if y_var is not None else 'None'}")
        return None, None, None, None

if __name__ == '__main__':
    # --- Dummy Data for Testing ---
    # This requires a data_fetcher or manually created price history dict
    # For example:
    date_rng = pd.date_range(start='2020-01-01', end='2023-01-01', freq='B')
    np.random.seed(42)
    prices_a = pd.Series(100 + np.random.randn(len(date_rng)).cumsum(), index=date_rng, name='StockA')
    prices_b = pd.Series(150 + np.random.randn(len(date_rng)).cumsum(), index=date_rng, name='StockB')
    prices_c = pd.Series(80 + np.random.randn(len(date_rng)).cumsum(), index=date_rng, name='StockC')
    
    # Ensure all prices are positive
    prices_a = prices_a.clip(lower=0.01)
    prices_b = prices_b.clip(lower=0.01)
    prices_c = prices_c.clip(lower=0.01)

    sample_price_history = {
        'StockA': prices_a,
        'StockB': prices_b,
        'StockC': prices_c
    }

    print("--- Testing Portfolio Optimization (MPT) ---")
    risk_free = 0.01 # 1%
    weights, exp_ret, vol, sharpe = optimize_portfolio_mpt(sample_price_history, risk_free_rate=risk_free)

    if weights:
        print("\nOptimal Weights:")
        for stock, weight in weights.items():
            print(f"  {stock}: {weight:.4f}")
        print(f"Sum of Weights: {sum(weights.values()):.4f}") # Should be close to 1.0
        print(f"\nExpected Annual Return: {exp_ret:.4f}")
        print(f"Annual Volatility (Std Dev): {vol:.4f}")
        print(f"Sharpe Ratio: {sharpe:.4f}")
    else:
        print("\nPortfolio optimization failed.")

    # Test with single asset
    print("\n--- Testing with a single asset ---")
    single_asset_history = {'StockA': prices_a}
    s_weights, s_exp_ret, s_vol, s_sharpe = optimize_portfolio_mpt(single_asset_history, risk_free_rate=risk_free)
    if s_weights:
        print("Optimal Weights (Single Asset):")
        for stock, weight in s_weights.items():
            print(f"  {stock}: {weight:.4f}") # Should be 1.0
        print(f"Expected Return: {s_exp_ret:.4f}")
        print(f"Volatility: {s_vol:.4f}")
        print(f"Sharpe Ratio: {s_sharpe:.4f}")
    else:
        print("Single asset optimization failed.")

