import requests
import pandas as pd
import datetime
from tenacity import retry, stop_after_attempt, wait_fixed
import os

@retry(stop=stop_after_attempt(5), wait=wait_fixed(10))
def get_binance_data(symbol, interval, start, end):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": int(start.timestamp() * 1000),
        "endTime": int(end.timestamp() * 1000),
        "limit": 1000
    }

    data = []
    while True:
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            klines = response.json()
            if not klines:
                break
            data.extend(klines)
            params["startTime"] = klines[-1][0] + 1
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            raise

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time", 
        "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", 
        "taker_buy_quote_asset_volume", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit='ms')
    df.set_index("open_time", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

def save_binance_data_in_chunks(symbol, interval, start_year, end_year, months_per_chunk=6):
    current_start = datetime.datetime(start_year, 1, 1)
    end = datetime.datetime(end_year, 12, 31)
    if not os.path.exists('binance_data_chunks'):
        os.makedirs('binance_data_chunks')

    while current_start < end:
        current_end = current_start + datetime.timedelta(days=months_per_chunk*30)
        if current_end > end:
            current_end = end

        try:
            df = get_binance_data(symbol, interval, current_start, current_end)
            file_name = f'binance_data_chunks/binance_data_{current_start.strftime("%Y%m%d")}_to_{current_end.strftime("%Y%m%d")}.csv'
            df.to_csv(file_name)
            print(f"Data from {current_start} to {current_end} saved to {file_name}.")
        except Exception as e:
            print(f"An error occurred for the period {current_start} to {current_end}: {e}")

        current_start = current_end

symbol = "BTCUSDT"
interval = "15m"
start_year = 2018
end_year = datetime.datetime.now().year

save_binance_data_in_chunks(symbol, interval, start_year, end_year)
