import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import os

# Configuration from Task 1
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load cleaned TSLA data from Task 1
tsla_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'TSLA_cleaned_data.csv'), index_col='Date', parse_dates=True)
data = tsla_df['Adj Close']

# Split data chronologically: 2015-2023 train, 2024-2025 test
train_data = data['2015-07-01':'2023-12-31']
test_data = data['2024-01-01':'2025-07-31']

# ARIMA Model (using auto_arima for parameter optimization)
arima_model = auto_arima(train_data, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
arima_fit = arima_model.fit(train_data)
arima_forecast = arima_fit.predict(n_periods=len(test_data))

# LSTM Model
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))
test_scaled = scaler.transform(test_data.values.reshape(-1, 1))

def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 60
X_train, y_train = create_sequences(train_scaled, seq_length)
X_test, y_test = create_sequences(test_scaled, seq_length)

# Build and train LSTM
lstm_model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
    LSTM(50, activation='relu'),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# Forecast with LSTM (walk-forward)
lstm_forecast = []
current_batch = train_scaled[-seq_length:].reshape((1, seq_length, 1))
for i in range(len(test_data)):
    pred = lstm_model.predict(current_batch, verbose=0)
    lstm_forecast.append(pred[0, 0])
    current_batch = np.append(current_batch[:, 1:, :], [[[pred[0, 0]]]], axis=1)
lstm_forecast = scaler.inverse_transform(np.array(lstm_forecast).reshape(-1, 1)).flatten()

# Evaluate models
metrics = {
    'ARIMA': {
        'MAE': mean_absolute_error(test_data, arima_forecast),
        'RMSE': np.sqrt(mean_squared_error(test_data, arima_forecast)),
        'MAPE': mean_absolute_percentage_error(test_data, arima_forecast)
    },
    'LSTM': {
        'MAE': mean_absolute_error(test_data, lstm_forecast),
        'RMSE': np.sqrt(mean_squared_error(test_data, lstm_forecast)),
        'MAPE': mean_absolute_percentage_error(test_data, lstm_forecast)
    }
}

# Save metrics
pd.DataFrame(metrics).to_csv(os.path.join(OUTPUT_DIR, 'task2_metrics.csv'))

# Plot forecasts
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data, label='Actual')
plt.plot(test_data.index, arima_forecast, label='ARIMA Forecast')
plt.plot(test_data.index, lstm_forecast, label='LSTM Forecast')
plt.title('TSLA Price Forecast Comparison')
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, 'task2_forecast_comparison.png'))
plt.close()

print("Task 2 completed. Metrics saved to outputs/task2_metrics.csv. Plot saved to outputs/task2_forecast_comparison.png.")