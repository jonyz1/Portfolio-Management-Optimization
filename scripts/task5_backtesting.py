import pandas as pd
import numpy as np
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

# Backtesting period: Last year (2024-08-01 to 2025-07-31)
backtest_prices = prices['2024-08-01':'2025-07-31']
backtest_returns = backtest_prices.pct_change().dropna()

# Optimal weights from Task 4 (placeholder; replace with actual from task4_portfolio.txt)
optimal_weights = {'TSLA': 0.3, 'BND': 0.4, 'SPY': 0.3}  # Example; load from Task 4 output

# Benchmark: 60% SPY, 40% BND
benchmark_weights = {'SPY': 0.6, 'BND': 0.4, 'TSLA': 0.0}

# Portfolio returns
strategy_returns = (backtest_returns * pd.Series(optimal_weights)).sum(axis=1)
benchmark_returns = (backtest_returns * pd.Series(benchmark_weights)).sum(axis=1)

# Cumulative returns
strategy_cum = (1 + strategy_returns).cumprod()
benchmark_cum = (1 + benchmark_returns).cumprod()

# Sharpe ratios (risk-free rate from Task 1)
risk_free = 0.02 / TRADING_DAYS
strategy_sharpe = (strategy_returns.mean() - risk_free) / strategy_returns.std() * np.sqrt(TRADING_DAYS)
benchmark_sharpe = (benchmark_returns.mean() - risk_free) / benchmark_returns.std() * np.sqrt(TRADING_DAYS)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(strategy_cum.index, strategy_cum, label='Strategy')
plt.plot(benchmark_cum.index, benchmark_cum, label='Benchmark')
plt.title('Backtest: Strategy vs Benchmark')
plt.ylabel('Cumulative Return')
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, 'task5_backtest.png'))
plt.close()

# Save results
results = f"Strategy Total Return: {strategy_cum.iloc[-1] - 1:.4f}\nBenchmark Total Return: {benchmark_cum.iloc[-1] - 1:.4f}\nStrategy Sharpe: {strategy_sharpe:.4f}\nBenchmark Sharpe: {benchmark_sharpe:.4f}\nSummary: Strategy likely outperforms if model-driven weights are effective."
with open(os.path.join(OUTPUT_DIR, 'task5_results.txt'), 'w') as f:
    f.write(results)

print("Task 5 completed. Plot saved to outputs/task5_backtest.png. Results to outputs/task5_results.txt.")