# Summary of Installed Package Versions on Windows 10
This document provides a summary of the package versions used in the project and their status.

## Installed Package Versions

### Python
- **Version**: 3.9.0

### TensorFlow
- **Version**: 2.12.0
- **Summary**: TensorFlow is an open-source framework for machine learning.

### NumPy
- **Version**: 1.23.5
- **Summary**: NumPy is a fundamental library for array computations in Python.

### Keras
- **Version**: 2.12.0 (Part of TensorFlow)
- **Summary**: Keras is an API for building and training deep learning models.

### SciPy
- **Version**: 1.13.1
- **Summary**: SciPy is a library for scientific and numerical computations.

### Pandas
- **Version**: 2.2.3
- **Summary**: Pandas is a library for data manipulation and analysis.

### Matplotlib
- **Version**: (To be installed as needed for data visualization)
- **Summary**: Matplotlib is a library for data plotting and visualization.

## Important Notes

- **Version Conflict**: Note that NumPy version 2.0.2 is not compatible with TensorFlow version 2.12.0. It is recommended to downgrade NumPy to version 1.23.5 for compatibility with TensorFlow.
- **Virtual Environment**: Always ensure that you are working in a virtual environment (`myenv`) to avoid conflicts with other packages.
-------------------------------------------------------------
# Work Report: Financial Time Series Prediction with LSTM and GRU Models

**Prepared by [Nahid Shahab] on [2025-01-28]**

## 1. Introduction

The objective of this project is to predict financial time series data using advanced deep learning techniques. The focus is on leveraging Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks to capture the temporal dependencies in the data. The project specifically deals with cryptocurrency price prediction, using historical data to forecast future trends. Python, TensorFlow, and related libraries were employed to implement the models.

---

## 2. Data Loading and Preprocessing

### 2.1 Data Loading
- The historical cryptocurrency price data was loaded from a CSV file using Pandas.
- Relevant columns (`open_time`, `open`, `high`, `low`, `close`, and `volume`) were extracted.

### 2.2 Data Transformation
- The `open_time` column was converted to datetime format and set as the index for better time series handling.
- Data was resampled to different timeframes (e.g., hourly, 4-hourly, daily) to accommodate different analysis resolutions.

### 2.3 Feature Engineering
- Technical indicators were added to enrich the feature set:
  - **Simple Moving Average (SMA):** Calculated using a rolling window.
  - **Relative Strength Index (RSI):** Computed to capture momentum in the price movements.
  - **Bollinger Bands:** Added to indicate price volatility.

---

## 3. Data Normalization and Sequence Creation

### 3.1 Normalization
- All features were normalized using `MinMaxScaler` from the `sklearn` library to scale values between 0 and 1, ensuring stable model training.

### 3.2 Sequence Creation
- Historical sequences of 50 time steps were created to train the models.
- A sliding window approach was used to form input-output pairs where each sequence corresponds to the next time step's prediction.

---

## 4. Model Architecture

### 4.1 LSTM Model
- **Layers:**
  - Two LSTM layers with 64 units each and ReLU activation.
  - Dropout layers (rate = 0.2) to mitigate overfitting.
  - Dense layers for final predictions.
- The model was compiled with the Adam optimizer and Mean Squared Error (MSE) loss function.

### 4.2 GRU Model
- **Layers:**
  - Two GRU layers with 64 units each and ReLU activation.
  - Dropout layers (rate = 0.2).
  - Dense layers for output generation.
- Similar compilation settings as the LSTM model.

---

## 5. Model Training and Evaluation

### 5.1 Training Process
- Data was split into training (80%) and testing (20%) sets.
- Models were trained using Early Stopping to monitor validation loss and halt training when no improvement was observed.
- Batch size and epochs were tuned for optimal performance.

### 5.2 Evaluation Metrics
The models were evaluated on the testing set using the following metrics:
- **Mean Squared Error (MSE):** Measures the average squared difference between predictions and actual values.
- **Root Mean Squared Error (RMSE):** Square root of MSE for interpretability.
- **Mean Absolute Error (MAE):** Average absolute difference between predictions and actual values.
- **R-squared (R²):** Indicates the proportion of variance explained by the model.

---

## 6. Results and Analysis

### 6.1 Quantitative Results
Comparison of evaluation metrics and next predicted prices for different timeframes:

#### 1 Hour:
- **Model: LSTM**
  - MSE: 52,597,412.63
  - RMSE: 7,252.41
  - MAE: 6,706.79
  - R²: 0.86
  - Next Predicted Price: 84,193.25
- **Model: GRU**
  - MSE: 97,053,193.73
  - RMSE: 9,851.56
  - MAE: 8,549.18
  - R²: 0.75
  - Next Predicted Price: 76,525.74

#### 4 Hours:
- **Model: LSTM**
  - MSE: 5,087,262.81
  - RMSE: 2,255.50
  - MAE: 1,669.39
  - R²: 0.99
  - Next Predicted Price: 98,266.94
- **Model: GRU**
  - MSE: 49,097,389.30
  - RMSE: 7,006.95
  - MAE: 6,526.16
  - R²: 0.87
  - Next Predicted Price: 84,181.45

#### 1 Day:
- **Model: LSTM**
  - MSE: 11,292,165.31
  - RMSE: 3,360.38
  - MAE: 2,412.14
  - R²: 0.97
  - Next Predicted Price: 96,741.01
- **Model: GRU**
  - MSE: 14,843,537.47
  - RMSE: 3,852.73
  - MAE: 2,776.86
  - R²: 0.96
  - Next Predicted Price: 94,420.26

### 6.2 Visualization
- Plots were generated to compare predicted vs. actual prices for different timeframes.
- Loss curves for training and validation were analyzed to ensure convergence.

### 6.3 Model Comparison
- LSTM demonstrated better performance on long-term dependencies, while GRU was faster to train and computationally efficient.

---

## 7. Conclusion

This project successfully implemented LSTM and GRU models for cryptocurrency price prediction. Key takeaways include:
- LSTM performed better for capturing long-term dependencies.
- Feature engineering significantly improved model accuracy.
- Early Stopping and dropout effectively mitigated overfitting.

**Future Work:**
- Experiment with hybrid models combining LSTM and GRU.
- Incorporate external factors (e.g., market news) as additional features.
- Optimize hyperparameters using grid search or Bayesian optimization.

---

## 8. Appendix

### Sample Code Snippet

```python
# Example of LSTM model implementation
# Function to build the LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

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
```

---

This document provides a comprehensive overview of the project, detailing each step and its outcomes.
