import requests
import pandas as pd
import datetime
from tenacity import retry, stop_after_attempt, wait_fixed
from pathlib import Path

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
    while params["startTime"] < int(end.timestamp() * 1000):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            klines = response.json()
            if not klines:
                break
            data.extend(klines)
            params["startTime"] = klines[-1][0] + 1  # ادامه از آخرین timestamp
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

def get_last_saved_date(symbol):
    files = sorted(Path(".").glob(f'binance_data_{symbol}_*.csv'))
    if not files:
        return None
    
    last_file = files[-1]
    try:
        df = pd.read_csv(last_file, parse_dates=["open_time"])
        return df.index[-1]  # آخرین داده ذخیره‌شده
    except Exception as e:
        print(f"Error reading {last_file}: {e}")
        return None

def download_new_data(symbol, interval, start_date, end_date):
    df = get_binance_data(symbol, interval, start_date, end_date)
    if df.empty:
        print("No new data available.")
        return None

    new_file_name = f'binance_data_{symbol}_{start_date.strftime("%Y%m%d")}_to_{end_date.strftime("%Y%m%d")}.csv'
    df.to_csv(new_file_name)
    print(f"Data saved: {new_file_name}")
    return new_file_name

def update_data(symbol, interval):
    today = datetime.datetime.now()
    last_saved_date = get_last_saved_date(symbol)

    if last_saved_date is None:
        last_saved_date = datetime.datetime(2018, 1, 1)  # تاریخ پیش‌فرض

    if today.date() > last_saved_date.date():
        download_new_data(symbol, interval, last_saved_date, today)
    else:
        print("Data is already up-to-date.")

# مشخصات
symbol = "ETHUSDT"
interval = "1h"

# به‌روزرسانی داده‌ها
update_data(symbol, interval)
