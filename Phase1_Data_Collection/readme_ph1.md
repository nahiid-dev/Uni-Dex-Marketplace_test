# README: Data Collection Script

This script facilitates downloading and updating cryptocurrency trading data from Binance using its API. The collected data is saved as CSV files for further analysis.

## Features
1. **Download Historical Data**: Fetch historical candlestick data (OHLCV) for a specific cryptocurrency.
2. **Update Existing Data**: Automatically updates data by checking for new entries from the last available date.
3. **Retry Mechanism**: Ensures data retrieval with retries in case of network or API issues.
4. **Customizable Parameters**: Specify the symbol, interval, and date range for data collection.

## Dependencies
The script requires the following Python libraries:
- `requests`  
- `pandas`  
- `datetime`  
- `tenacity`  
- `os`

Install the required libraries using:
```bash
pip install requests pandas tenacity
