import pandas as pd
import numpy as np
from decimal import Decimal, getcontext
from pathlib import Path
import sys

# --- Constants ---
getcontext().prec = 78
TOKEN0_NAME = "USDC"
TOKEN1_NAME = "WETH"
TOKEN0_DECIMALS = 6
TOKEN1_DECIMALS = 18

# --- Conversion Functions ---
def tick_to_usd_price(tick_val: float | str | Decimal,
                      token0_decimals: int = TOKEN0_DECIMALS,
                      token1_decimals: int = TOKEN1_DECIMALS) -> float | None:
    if not pd.notna(tick_val) or str(tick_val).strip() == "" or str(tick_val).strip().lower() == 'nan':
        return np.nan
    try:
        tick = Decimal(str(tick_val))
        numerator = Decimal('10') ** (token1_decimals - token0_decimals)
        denominator = Decimal('1.0001') ** tick
        if denominator == Decimal(0): return np.nan
        return float(numerator / denominator)
    except Exception:
        return np.nan

def amount_to_readable(amount_val: float | str | Decimal, decimals: int) -> float | None:
    if not pd.notna(amount_val) or str(amount_val).strip() == "" or str(amount_val).strip().lower() == 'nan':
        return np.nan
    try:
        amount = Decimal(str(amount_val))
        return float(amount / (Decimal('10') ** decimals))
    except Exception:
        return np.nan

def format_and_round(df_series: pd.Series, round_digits: int) -> pd.Series:
    s = pd.to_numeric(df_series.astype(str).str.replace(',', '', regex=False), errors='coerce')
    if pd.api.types.is_numeric_dtype(s):
        return s.round(round_digits)
    return df_series

def process_file(input_filepath: Path, output_filepath: Path):
    print(f"Processing {input_filepath} -> {output_filepath}...")
    try:
        df = pd.read_csv(input_filepath)
    except Exception as e:
        print(f"  ❌ Error reading {input_filepath}: {e}")
        return

    df_out = df.copy()

    # Columns to convert from tick to USD price - List has been revised
    tick_columns_to_convert = [
        'currentTick_pool',
        'finalTickLower_contract',
        'finalTickUpper_contract',
        'predictedTick_calculated' # Kept for Predictive strategy analysis
        # 'targetTickLower_calculated', # Removed as per user feedback
        # 'targetTickUpper_calculated', # Removed as per user feedback
        # 'targetTickLower_offchain',   # Removed as per user feedback (assuming implied)
        # 'targetTickUpper_offchain'    # Removed as per user feedback (assuming implied)
    ]

    for col in tick_columns_to_convert:
        if col in df_out.columns:
            new_col_name = f'{col}_{TOKEN1_NAME}{TOKEN0_NAME}_price'
            df_out[new_col_name] = df_out[col].apply(tick_to_usd_price)
            df_out[new_col_name] = format_and_round(df_out[new_col_name], 4)

    # Convert token amounts to readable amounts with specific token names in column
    token0_amount_cols = {
        'amount0_provided_to_mint': f'amount_{TOKEN0_NAME}_minted',
        'initial_contract_balance_token0': f'initial_balance_{TOKEN0_NAME}',
        'fees_collected_token0': f'fees_{TOKEN0_NAME}_collected'
    }
    token1_amount_cols = {
        'amount1_provided_to_mint': f'amount_{TOKEN1_NAME}_minted',
        'initial_contract_balance_token1': f'initial_balance_{TOKEN1_NAME}',
        'fees_collected_token1': f'fees_{TOKEN1_NAME}_collected'
    }

    for col, new_base_name in token0_amount_cols.items():
        if col in df_out.columns:
            new_col_name = f'{new_base_name}_readable'
            df_out[new_col_name] = df_out[col].apply(lambda x: amount_to_readable(x, TOKEN0_DECIMALS))
            df_out[new_col_name] = format_and_round(df_out[new_col_name], 4)

    for col, new_base_name in token1_amount_cols.items():
        if col in df_out.columns:
            new_col_name = f'{new_base_name}_readable'
            df_out[new_col_name] = df_out[col].apply(lambda x: amount_to_readable(x, TOKEN1_DECIMALS))
            df_out[new_col_name] = format_and_round(df_out[new_col_name], 4)

    price_cols_to_round = ['actualPrice_pool', 'predictedPrice_api']
    for col in price_cols_to_round:
        if col in df_out.columns:
            df_out[col] = pd.to_numeric(df_out[col], errors='coerce') # Ensure numeric before rounding
            df_out[col] = df_out[col].round(4)
            
    if 'gas_cost_eth' in df_out.columns:
        df_out['gas_cost_eth'] = pd.to_numeric(df_out['gas_cost_eth'], errors='coerce').round(6)
    if 'gas_used' in df_out.columns:
         df_out['gas_used'] = pd.to_numeric(df_out['gas_used'], errors='coerce').fillna(0).astype('Int64')

    liquidity_cols = ['liquidity_contract', 'finalLiquidity_contract', 'currentLiquidity_contract']
    for col in liquidity_cols:
        if col in df_out.columns:
            s_numeric = pd.to_numeric(df_out[col], errors='coerce').fillna(0)
            try: df_out[col] = s_numeric.astype('Int64')
            except Exception: df_out[col] = s_numeric.astype(float).astype(int)

    try:
        df_out.to_csv(output_filepath, index=False, encoding='utf-8')
        print(f"  ✅ Successfully created {output_filepath}")
    except Exception as e:
        print(f"  ❌ Error writing {output_filepath}: {e}")

if __name__ == "__main__":
    try:
        script_path = Path(__file__).resolve()
        script_dir = script_path.parent
        csv_base_dir = script_dir.parent 
    except NameError:
        script_dir = Path('.').resolve()
        csv_base_dir = script_dir.parent 
        print(f"Warning: __file__ not defined. Using CWD for script_dir: {script_dir}, CSV base: {csv_base_dir}")

    input_files_info = [
        {"original": "position_results_predictive.csv", "converted": "predictive_readable.csv"},
        {"original": "position_results_baseline.csv", "converted": "baseline_readable.csv"}
    ]

    output_dir = script_dir / "plots_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_info in input_files_info:
        original_file_path = csv_base_dir / file_info["original"]
        converted_file_path = output_dir / file_info["converted"]
        
        if original_file_path.exists():
            process_file(original_file_path, converted_file_path)
        else:
            print(f"  ❌ Input file not found: {original_file_path}")
    
    print("\nConversion process finished.")