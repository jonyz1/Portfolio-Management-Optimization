import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns
import matplotlib.pyplot as plt
import os

# Configuration from Task 1
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TICKERS = ["TSLA", "BND", "SPY"]
TRADING_DAYS = 252

# Load cleaned data for all tickers from Task 1
data = {}
for ticker in TICKERS:
    df = pd.read_csv(os.path.join(OUTPUT_DIR, f'{ticker}_cleaned_data.csv'), index_col='Date', parse_dates=True)
    data[ticker] = df['Adj Close']

prices = pd.DataFrame(data)

# Calculate daily returns
returns = prices.pct_change().dropna()

# Expected returns
# For TSLA: Use forecasted return (assume from Task 3; placeholder: 20% annualized based on forecast growth)
tsla_forecast_growth = 0.20  # Replace with actual from Task 3 (e.g., (forecast_mean / last_price - 1) * TRADING_DAYS)
exp_returns = expected_returns.mean_historical_return(prices, frequency=TRADING_DAYS)
exp_returns['TSLA'] = tsla_forecast_growth

# Covariance matrix
cov_matrix = risk_models.sample_cov(returns, frequency=TRADING_DAYS)

# Efficient Frontier
ef = EfficientFrontier(exp_returns, cov_matrix)

# Max Sharpe Portfolio
weights_max_sharpe = ef.max_sharpe()
perf_max_sharpe = ef.portfolio_performance(verbose=False)

# Min Volatility Portfolio
weights_min_vol = ef.min_volatility()
perf_min_vol = ef.portfolio_performance(verbose=False)

# Generate frontier points
target_returns = np.linspace(perf_min_vol[0], perf_max_sharpe[0] * 1.2, 50)
risks = []
for ret in target_returns:
    ef.efficient_return(ret)
    perf = ef.portfolio_performance(verbose=False)
    risks.append(perf[1])

# Plot
plt.figure(figsize=(10, 6))
plt.plot(risks, target_returns, label='Efficient Frontier')
plt.scatter(perf_max_sharpe[1], perf_max_sharpe[0], marker='*', color='r', s=200, label='Max Sharpe')
plt.scatter(perf_min_vol[1], perf_min_vol[0], marker='*', color='g', s=200, label='Min Vol')
plt.xlabel('Volatility')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier')
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, 'task4_efficient_frontier.png'))
plt.close()

# Recommend Max Sharpe
rec_weights = weights_max_sharpe
rec_perf = perf_max_sharpe
summary = f"Weights: {rec_weights}\nExp Return: {rec_perf[0]:.4f}\nVol: {rec_perf[1]:.4f}\nSharpe: {rec_perf[2]:.4f}\nJustification: Prioritizes risk-adjusted returns."
with open(os.path.join(OUTPUT_DIR, 'task4_portfolio.txt'), 'w') as f:
    f.write(summary)

print("Task 4 completed. Plot saved to outputs/task4_efficient_frontier.png. Portfolio details to outputs/task4_portfolio.txt.")