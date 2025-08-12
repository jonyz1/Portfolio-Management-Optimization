import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from datetime import timedelta
import os

# Configuration from Task 1
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load cleaned TSLA data from Task 1
tsla_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'TSLA_cleaned_data.csv'), index_col='Date', parse_dates=True)
data = tsla_df['Adj Close']

# Assume LSTM is the best model; retrain on full data for future forecast
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(scaled_data, seq_length)

# Build and train LSTM on full data
lstm_model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
    LSTM(50, activation='relu'),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X, y, epochs=20, batch_size=32, verbose=1)

# Forecast 12 months (approx 365 days) into the future
forecast_days = 365
last_date = data.index[-1]
forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='D')
forecast = []
current_batch = scaled_data[-seq_length:].reshape((1, seq_length, 1))

for _ in range(forecast_days):
    pred = lstm_model.predict(current_batch, verbose=0)
    forecast.append(pred[0, 0])
    current_batch = np.append(current_batch[:, 1:, :], [[[pred[0, 0]]]], axis=1)

forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()

# Simulate confidence intervals using historical volatility (from Task 1 rolling std)
historical_vol = tsla_df['Rolling_Std_Return'].mean() * np.sqrt(252)  # Annualized approx
ci_upper = forecast * (1 + historical_vol * 1.96 / np.sqrt(forecast_days))  # Simplified 95% CI
ci_lower = forecast * (1 - historical_vol * 1.96 / np.sqrt(forecast_days))

# Plot
plt.figure(figsize=(12, 6))
plt.plot(data.index[-365:], data.values[-365:], label='Historical')
plt.plot(forecast_dates, forecast, label='Forecast')
plt.fill_between(forecast_dates, ci_lower, ci_upper, alpha=0.2, label='95% CI')
plt.title('TSLA 12-Month Future Forecast')
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, 'task3_forecast.png'))
plt.close()

# Save analysis
trend = "upward" if forecast[-1] > forecast[0] else "downward" if forecast[-1] < forecast[0] else "stable"
analysis = f"Trend: {trend}\nAverage Forecast Price: {np.mean(forecast):.2f}\nCI Width at End: {(ci_upper[-1] - ci_lower[-1]):.2f}\nImplications: Widening CI indicates increasing uncertainty over time.\nOpportunities: Growth potential if upward.\nRisks: High volatility."
with open(os.path.join(OUTPUT_DIR, 'task3_analysis.txt'), 'w') as f:
    f.write(analysis)

print("Task 3 completed. Forecast plot saved to outputs/task3_forecast.png. Analysis saved to outputs/task3_analysis.txt.")