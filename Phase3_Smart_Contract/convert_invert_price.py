import pandas as pd
import numpy as np

# خواندن فایل CSV
input_file = 'position_results_predictive.csv'
df = pd.read_csv(input_file)

def to_float(val):
    try:
        return float(val)
    except:
        return np.nan

def sqrtPriceX96_to_price(sqrtPriceX96):
    try:
        sqrtPriceX96 = float(sqrtPriceX96)
        price = (sqrtPriceX96 / 2**96) ** 2
        return price
    except:
        return np.nan

def invert(val):
    try:
        return 1/float(val) if float(val) != 0 else np.nan
    except:
        return np.nan

# تبدیل ستون sqrtPriceX96_pool به قیمت (USDC/ETH)
df['sqrtPriceX96_pool_decimal'] = df['sqrtPriceX96_pool'].apply(to_float)
df['usdc_per_eth'] = df['sqrtPriceX96_pool_decimal'].apply(sqrtPriceX96_to_price)
# معکوس کردن برای بدست آوردن ETH/USDC
df['eth_per_usdc'] = df['usdc_per_eth'].apply(invert)

# نمایش 5 مقدار آخر برای بررسی
print(df[['sqrtPriceX96_pool','usdc_per_eth','eth_per_usdc']].tail())

# ذخیره خروجی
output_file = 'position_results_predictive_with_invert.csv'
df.to_csv(output_file, index=False)
print(f"نتیجه در فایل {output_file} ذخیره شد.")
