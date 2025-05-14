import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['open_time'] = pd.to_datetime(data['open_time'])
    data.set_index('open_time', inplace=True)
    data = data[['open', 'high', 'low', 'close', 'volume']]
    data.dropna(inplace=True)
    return data

# Function to resample data to different timeframes
def resample_data(data, timeframe):
    tf_data = data.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    return tf_data

# Function to add technical indicators
def add_technical_indicators(data):
    data['SMA'] = data['close'].rolling(window=14).mean()
    data['RSI'] = 100 - (100 / (1 + (data['close'].diff().gt(0).rolling(window=14).sum() / data['close'].diff().lt(0).abs().rolling(window=14).sum())))
    data['Bollinger_Upper'] = data['close'].rolling(window=20).mean() + 2 * data['close'].rolling(window=20).std()
    data.dropna(inplace=True)
    return data

# Function to normalize data
def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(data)
    return normalized_data, scaler

# Creating Sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Function to build the LSTM model
def build_lstm_model(input_shape):
    lstm_model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, activation='relu', return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    return lstm_model

# Function to build the GRU model
def build_gru_model(input_shape):
    gru_model = Sequential([
        GRU(64, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(64, activation='relu', return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    gru_model.compile(optimizer='adam', loss='mse')
    return gru_model

# Function to train the model
def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )

    return model, history

def evaluate_model(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)
    predictions_rescaled = scaler.inverse_transform(
        np.concatenate((predictions.reshape(-1, 1), np.zeros((predictions.shape[0], 7))), axis=1)
    )[:, 0]  # Get the 'close' column

    y_test_rescaled = scaler.inverse_transform(
        np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 7))), axis=1)
    )[:, 0]  # Get the 'close' column

    mse = mean_squared_error(y_test_rescaled, predictions_rescaled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
    r2 = r2_score(y_test_rescaled, predictions_rescaled)
    next_predicted_price = predictions_rescaled[-1]

    return y_test_rescaled, predictions_rescaled, mse, rmse, mae, r2, next_predicted_price

# Function to save the model
def save_model(model, model_type, timeframe):
    model.save(f'model_{model_type}_{timeframe}.h5')
    print(f"Model for {model_type} at {timeframe} saved as model_{model_type}_{timeframe}.h5")

# Function to plot results for each timeframe
def plot_results(y_test, predictions, model_type, timeframe, next_predicted_price, dates):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_test, label='Real Prices')
    plt.plot(dates, predictions, label='Predicted Prices', linestyle='--')
    plt.title(f'{model_type} Model - {timeframe} Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    plt.savefig(f'{model_type}_{timeframe}_prediction.png')  # Save plot
    plt.show()

# Function to plot comparison of evaluation metrics
def plot_metrics_comparison(results):
    timeframes = list(results.keys())
    metrics = ['MSE', 'RMSE', 'MAE', 'R²']
    models = ['LSTM', 'GRU']  # Get the model names

    metric_values = {metric: [[results[tf][model][metric] for tf in timeframes] for model in models] for metric in metrics}
    # Iterate through models and timeframes to get metrics

    plt.figure(figsize=(14, 10))
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)

        # Adjust bar plotting for multiple models
        bar_width = 0.35
        x_pos = np.arange(len(timeframes))

        plt.bar(x_pos - bar_width/2, metric_values[metric][0], bar_width, label='LSTM', color='blue')  # LSTM bars
        plt.bar(x_pos + bar_width/2, metric_values[metric][1], bar_width, label='GRU', color='green')  # GRU bars

        plt.title(f'{metric} Comparison')
        plt.xlabel('Timeframe')
        plt.ylabel(metric)
        plt.xticks(x_pos, timeframes)  # Set x-axis labels
        plt.legend()

        # Adjust text position for multiple models
        for j, value in enumerate(metric_values[metric][0]):  # LSTM values
            plt.text(j - bar_width/2, value, f'{value:.2f}', ha='center', va='bottom')
        for j, value in enumerate(metric_values[metric][1]):  # GRU values
            plt.text(j + bar_width/2, value, f'{value:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('metrics_comparison.png')
    plt.show()

# Main function to execute all steps
def main():
    file_path = '/content/drive/MyDrive/Colab Notebooks/binance_data_20180101_to_20241229.csv'
    timeframes = {'1H': '1 Hour', '4H': '4 Hours', '1D': '1 Day'}
    seq_length = 50
    results = {}

    data = load_and_preprocess_data(file_path)

    for tf, tf_name in timeframes.items():
        print(f"\nProcessing timeframe: {tf_name}")

        tf_data = resample_data(data, tf)
        tf_data = add_technical_indicators(tf_data)
        features = tf_data[['close', 'open', 'high', 'low', 'volume', 'SMA', 'RSI', 'Bollinger_Upper']].values
         # Normalize data
        normalized_features, scaler = normalize_data(features)
        X, y = create_sequences(normalized_features, seq_length)

        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Extract date index for plotting
        dates = tf_data.index[train_size + seq_length:]

        # Train and evaluate LSTM model
        lstm_model = build_lstm_model((X.shape[1], X.shape[2]))
        lstm_model, lstm_history = train_model(lstm_model, X_train, y_train, X_test, y_test, epochs=100)
        y_test_rescaled_lstm, predictions_rescaled_lstm, mse_lstm, rmse_lstm, mae_lstm, r2_lstm, next_predicted_price_lstm = evaluate_model(lstm_model, X_test, y_test, scaler)

        # Save LSTM model and plot results
        results[tf_name] = {'LSTM': {'MSE': mse_lstm, 'RMSE': rmse_lstm, 'MAE': mae_lstm, 'R²': r2_lstm, 'Next Predicted Price': next_predicted_price_lstm}}
        save_model(lstm_model, 'LSTM', tf)
        plot_results(y_test_rescaled_lstm, predictions_rescaled_lstm, 'LSTM', tf_name, next_predicted_price_lstm,dates)

        # Train and evaluate GRU model
        gru_model = build_gru_model((X.shape[1], X.shape[2]))
        gru_model, gru_history = train_model(gru_model, X_train, y_train, X_test, y_test, epochs=100)
        y_test_rescaled_gru, predictions_rescaled_gru, mse_gru, rmse_gru, mae_gru, r2_gru, next_predicted_price_gru = evaluate_model(gru_model, X_test, y_test, scaler)

        # Save GRU model and plot results
        results[tf_name]['GRU'] = {'MSE': mse_gru, 'RMSE': rmse_gru, 'MAE': mae_gru, 'R²': r2_gru, 'Next Predicted Price': next_predicted_price_gru}
        save_model(gru_model, 'GRU', tf)
        plot_results(y_test_rescaled_gru, predictions_rescaled_gru, 'GRU', tf_name, next_predicted_price_gru,dates)

    # Display evaluation metrics and next predicted prices for different timeframes
    print("\nComparison of evaluation metrics and next predicted prices for different timeframes:")
    for tf_name, metrics in results.items():
        print(f"\n{tf_name}:")
        for model_name, metrics_values in metrics.items():
            print(f"Model: {model_name}")
            print(f"MSE: {metrics_values['MSE']:.2f}")
            print(f"RMSE: {metrics_values['RMSE']:.2f}")
            print(f"MAE: {metrics_values['MAE']:.2f}")
            print(f"R²: {metrics_values['R²']:.2f}")
            print(f"Next Predicted Price: {metrics_values['Next Predicted Price']:.2f}")

    # Plot metrics comparison
    plot_metrics_comparison(results)

# Execute the main function
main()