
import os
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

# ---------------------------
# Configuration
# ---------------------------
TICKERS = ["TSLA", "BND", "SPY"]
START_DATE = "2015-07-01"
END_DATE = "2025-07-31"  # inclusive intent
OUTPUT_DIR = "outputs"
ROLL_WINDOW = 30  # rolling window size for volatility
RISK_FREE_RATE = 0.02  # annual risk-free rate used for Sharpe calculation
TRADING_DAYS = 252
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Utility functions
# ---------------------------

def download_data(tickers, start, end):
    dfs = {}
    for t in tickers:
        print(f"Downloading {t}...")
        df = yf.download(t, start=start, end=end, progress=False)
        if df.empty:
            raise RuntimeError(f"No data returned for {t}. Check ticker or date range.")
        dfs[t] = df
    return dfs


def clean_and_engineer(df, roll_window=30):
    df = df.copy()
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Sort index
    df.sort_index(inplace=True)

    # Forward fill then back fill to handle missing market-days
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Feature engineering
    df['Adj Close'] = df.get('Adj Close', df['Close'])
    df['Daily_Return'] = df['Adj Close'].pct_change()
    df['Log_Return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    df['Rolling_Volatility'] = df['Daily_Return'].rolling(window=roll_window).std()
    df['Rolling_Mean'] = df['Adj Close'].rolling(window=roll_window).mean()

    return df


def plot_price_series(dfs, output_path=None):
    plt.figure(figsize=(12,6))
    for t, df in dfs.items():
        plt.plot(df.index, df['Adj Close'], label=t)
    plt.title('Adjusted Close Price (2015-07-01 to 2025-07-31)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    plt.show()


def plot_daily_returns_hist(dfs, output_path=None):
    plt.figure(figsize=(12,6))
    for t, df in dfs.items():
        sns.kdeplot(df['Daily_Return'].dropna(), label=t, fill=False)
    plt.title('Daily Returns KDE (2015-2025)')
    plt.xlabel('Daily Return')
    plt.legend()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    plt.show()


def plot_rolling_volatility(dfs, output_path=None):
    plt.figure(figsize=(12,6))
    for t, df in dfs.items():
        plt.plot(df.index, df['Rolling_Volatility'], label=t)
    plt.title(f'{ROLL_WINDOW}-Day Rolling Volatility')
    plt.xlabel('Date')
    plt.ylabel('Volatility (std of daily returns)')
    plt.legend()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    plt.show()


def detect_outliers(df, k=3):
    # flag returns beyond k * std as outliers
    thr = k * df['Daily_Return'].std()
    out = df[np.abs(df['Daily_Return']) > thr]
    return out


def adf_test(series, name='series'):
    series = series.dropna()
    if len(series) < 10:
        print(f"Not enough observations for ADF on {name}")
        return None
    result = adfuller(series, autolag='AIC')
    output = {
        'ADF Statistic': result[0],
        'p-value': result[1],
        'usedlag': result[2],
        'nobs': result[3],
        'crit_vals': result[4]
    }
    print(f"ADF test for {name} -> ADF={output['ADF Statistic']:.4f}, p={output['p-value']:.4f}")
    for k, v in output['crit_vals'].items():
        print(f"   {k} : {v:.4f}")
    if output['p-value'] <= 0.05:
        print('   => Likely stationary (reject H0)')
    else:
        print('   => Likely non-stationary (fail to reject H0)')
    print('')
    return output


def compute_risk_metrics(df, risk_free_rate=0.02):
    dr = df['Daily_Return'].dropna()
    if dr.empty:
        return {'VaR_95': np.nan, 'Sharpe': np.nan}
    var_95 = np.percentile(dr, 5)
    ann_mean = dr.mean() * TRADING_DAYS
    ann_std = dr.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ann_mean - risk_free_rate) / (ann_std + 1e-12)
    return {'VaR_95': var_95, 'Ann_Mean_Return': ann_mean, 'Ann_Std': ann_std, 'Sharpe': sharpe}


# ---------------------------
# Main flow
# ---------------------------

def main():
    # 1) Download
    raw = download_data(TICKERS, START_DATE, END_DATE)

    # 2) Clean & Feature engineer
    processed = {}
    for t, df in raw.items():
        df2 = clean_and_engineer(df, roll_window=ROLL_WINDOW)
        processed[t] = df2
        # save cleaned csv
        csv_path = os.path.join(OUTPUT_DIR, f"{t}_clean.csv")
        df2.to_csv(csv_path)
        print(f"Saved cleaned CSV -> {csv_path}")

    # 3) Basic stats
    for t, df in processed.items():
        print(f"\n--- {t} Summary ---")
        print(df[['Adj Close', 'Daily_Return', 'Rolling_Volatility']].describe().T)

    # 4) Plots
    plot_price_series(processed, output_path=os.path.join(OUTPUT_DIR, 'prices_all.png'))
    plot_daily_returns_hist(processed, output_path=os.path.join(OUTPUT_DIR, 'returns_kde.png'))
    plot_rolling_volatility(processed, output_path=os.path.join(OUTPUT_DIR, 'rolling_volatility.png'))

    # 5) Outlier detection
    for t, df in processed.items():
        out = detect_outliers(df, k=3)
        print(f"{t}: Found {len(out)} extreme return days (>|3 sigma|). Sample: ")
        print(out[['Adj Close', 'Daily_Return']].head(3))

    # 6) ADF tests
    for t, df in processed.items():
        print(f"\n--- ADF Tests for {t} ---")
        adf_test(df['Adj Close'], name=f"{t} Adj Close")
        adf_test(df['Daily_Return'], name=f"{t} Daily Return")

    # 7) Risk metrics
    metrics = {}
    for t, df in processed.items():
        metrics[t] = compute_risk_metrics(df, risk_free_rate=RISK_FREE_RATE)
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'risk_metrics_summary.csv'))
    print('\nRisk metrics saved to outputs/risk_metrics_summary.csv')
    print(metrics_df)

    print('\nAll done. Check the outputs/ folder for CSVs and plots.')


if __name__ == '__main__':
    main()