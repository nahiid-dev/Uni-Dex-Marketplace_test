import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
from pathlib import Path
import sys
from decimal import Decimal, getcontext

# --- 1. Data Loading ---
PREDICTIVE_FILE_NAME = "position_results_predictive.csv"
BASELINE_FILE_NAME = "position_results_baseline.csv"
try:
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    csv_dir = script_dir.parent
except NameError:
    print("Warning: __file__ variable not found. Using the current working directory.")
    script_dir = Path('.').resolve()
    csv_dir = script_dir.parent
    print(f"Assumed script path: {script_dir}")

predictive_path = csv_dir / PREDICTIVE_FILE_NAME
baseline_path = csv_dir / BASELINE_FILE_NAME
print(f"Script path: {script_dir}")
print(f"CSV search path: {csv_dir}")
print(f"Attempting to read file: {predictive_path}")
print(f"Attempting to read file: {baseline_path}")
try:
    df_pred = pd.read_csv(predictive_path)
    df_base = pd.read_csv(baseline_path)
    print("✅ Files loaded successfully.")
except FileNotFoundError:
    print("❌ Error: One or both CSV files were not found.")
    sys.exit(1)
except Exception as e:
    print(f"❌ An unexpected error occurred while reading the files: {e}")
    sys.exit(1)

# --- 2. Data Preprocessing & Style ---
def preprocess(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    numeric_cols = [
        'predictedPrice_api', 'actualPrice_pool', 'currentTick_pool',
        'finalTickLower_contract', 'finalTickUpper_contract',
        'gas_cost_eth', 'liquidity_contract', 'finalLiquidity_contract',
        'amount0_provided_to_mint', 'amount1_provided_to_mint',
        'initial_contract_balance_token0', 'initial_contract_balance_token1',
        'fees_collected_token0', 'fees_collected_token1'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if 'liquidity_contract' not in df.columns and 'finalLiquidity_contract' in df.columns:
        df['liquidity_contract'] = df['finalLiquidity_contract']
    
    cols_to_fillna = numeric_cols + ['liquidity_contract']
    for col in cols_to_fillna:
        if col in df.columns:
            df[col] = df[col].fillna(0)
        else: # اگر ستونی اصلاً وجود ندارد، آن را با صفر ایجاد کنید
            df[col] = 0
            
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

df_pred = preprocess(df_pred.copy())
df_base = preprocess(df_base.copy())

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# --- 3. Helper & Metric Functions ---
getcontext().prec = 78

def tick_to_price(tick_val):
    try:
        # Ensure tick_val is a valid number before exponentiation
        return 1.0001 ** float(tick_val)
    except (ValueError, TypeError):
        # Return NaN or some indicator if conversion fails
        return np.nan


def calculate_errors(y_true, y_pred):
    errors = y_true - y_pred
    mae = np.mean(np.abs(errors))
    y_true_safe = np.where(y_true == 0, 1e-9, y_true) # Avoid division by zero
    mape = np.mean(np.abs(errors / y_true_safe)) * 100
    return errors, mae, mape

def get_amounts_in_lp(liquidity, current_tick, lower_tick, upper_tick):
    L = Decimal(str(liquidity))
    current_tick_dec = Decimal(str(current_tick))
    lower_tick_dec = Decimal(str(lower_tick))
    upper_tick_dec = Decimal(str(upper_tick))

    # Handle potential invalid tick values before calculation
    if not (pd.notna(current_tick) and pd.notna(lower_tick) and pd.notna(upper_tick)):
        return 0,0
    if L == 0: return 0, 0
    if lower_tick_dec >= upper_tick_dec: return 0,0 # Invalid range

    try:
        sqrt_P_current = Decimal('1.0001')**(current_tick_dec / Decimal('2'))
        sqrt_P_lower = Decimal('1.0001')**(lower_tick_dec / Decimal('2'))
        sqrt_P_upper = Decimal('1.0001')**(upper_tick_dec / Decimal('2'))
    except Exception as e:
        # print(f"Error in sqrt calculation: {e} with ticks: C={current_tick}, L={lower_tick}, U={upper_tick}")
        return 0,0


    amount0_in_lp = Decimal(0)
    amount1_in_lp = Decimal(0)

    if current_tick_dec < lower_tick_dec:
        amount0_in_lp = L * (Decimal(1) / sqrt_P_lower - Decimal(1) / sqrt_P_upper)
    elif current_tick_dec >= upper_tick_dec:
        amount1_in_lp = L * (sqrt_P_upper - sqrt_P_lower)
    else: # In range
        amount0_in_lp = L * (Decimal(1) / sqrt_P_current - Decimal(1) / sqrt_P_upper)
        amount1_in_lp = L * (sqrt_P_current - sqrt_P_lower)
    
    return int(amount0_in_lp), int(amount1_in_lp)


def analyze_data(df, contract_type):
    results = {}
    
    # Time in Range calculation
    if 'finalTickLower_contract' in df.columns and 'finalTickUpper_contract' in df.columns and \
       'currentTick_pool' in df.columns and \
       not (df['finalTickLower_contract'].eq(0) & df['finalTickUpper_contract'].eq(0)).all():
        df['is_in_range'] = (df['currentTick_pool'] >= df['finalTickLower_contract']) & \
                           (df['currentTick_pool'] <= df['finalTickUpper_contract'])
        time_in_range_pct = df['is_in_range'].mean() * 100 if not df['is_in_range'].empty else 0
    else:
        df['is_in_range'] = False; time_in_range_pct = 0
    results['time_in_range_pct'] = time_in_range_pct

    # Error metrics for Predictive
    if contract_type == 'Predictive':
        if 'actualPrice_pool' in df.columns and 'predictedPrice_api' in df.columns and not df.empty:
            # Ensure columns are numeric and NaNs handled (should be done in preprocess)
            df['actualPrice_pool'] = pd.to_numeric(df['actualPrice_pool'], errors='coerce').fillna(0)
            df['predictedPrice_api'] = pd.to_numeric(df['predictedPrice_api'], errors='coerce').fillna(0)
            
            valid_actual_prices = df['actualPrice_pool'][df['actualPrice_pool'] != 0]
            if not valid_actual_prices.empty:
                errors, mae, mape = calculate_errors(df['actualPrice_pool'], df['predictedPrice_api'])
                df['prediction_error_pct'] = np.where(df['actualPrice_pool'] != 0, (errors / df['actualPrice_pool']) * 100, 0.0)
                results['mae'] = mae; results['mape'] = mape
            else:
                df['prediction_error_pct'] = 0.0; results['mae'] = 0.0; results['mape'] = 0.0
        else:
            df['prediction_error_pct'] = 0.0; results['mae'] = 'N/A'; results['mape'] = 'N/A'

    token0_decimals = 6
    token1_decimals = 18
    
    # Initialize P&L columns
    pnl_cols = ['amount0_in_lp', 'amount1_in_lp', 'v_lp_usd', 'v_remaining_usd', 
                'v_current_total_usd', 'v_hodl_usd', 'il_usd', 'v_fees_usd', 
                'v_gas_usd', 'pnl_vs_hodl_usd', 'period_actual_pnl_usd',
                'value_lp_at_period_end_usd', 'pnl_lp_holding_period_usd',
                'position_i_ended_in_range', 'net_pnl_period_usd']
    for col in pnl_cols:
        if col == 'position_i_ended_in_range':
             df[col] = pd.NA # Boolean can use pd.NA
        else:
            df[col] = 0.0 # Default to float for numeric P&L columns

    for index, row in df.iterrows():
        try:
            L_val_final = Decimal(str(row.get('finalLiquidity_contract', 0)))
            L_val_current = Decimal(str(row.get('liquidity_contract', 0)))
            L_val = L_val_final if L_val_final != Decimal(0) else L_val_current
            
            current_tick_val = row['currentTick_pool']
            lower_tick_val = row['finalTickLower_contract']
            upper_tick_val = row['finalTickUpper_contract']
            eth_price_usd_val = row['actualPrice_pool']

            current_tick = Decimal(str(current_tick_val))
            lower_tick = Decimal(str(lower_tick_val))
            upper_tick = Decimal(str(upper_tick_val))
            eth_price_usd = Decimal(str(eth_price_usd_val)) if pd.notna(eth_price_usd_val) and eth_price_usd_val!=0 else Decimal(0)

            amt0_lp, amt1_lp = 0,0
            if L_val > 0 and lower_tick < upper_tick:
                amt0_lp, amt1_lp = get_amounts_in_lp(L_val, current_tick, lower_tick, upper_tick)
            df.loc[index, 'amount0_in_lp'] = amt0_lp
            df.loc[index, 'amount1_in_lp'] = amt1_lp
            
            v_lp_token0_usd = (Decimal(amt0_lp) / (Decimal(10)**token0_decimals))
            v_lp_token1_usd = (Decimal(amt1_lp) / (Decimal(10)**token1_decimals)) * eth_price_usd
            v_lp_usd_calc = v_lp_token0_usd + v_lp_token1_usd
            df.loc[index, 'v_lp_usd'] = float(v_lp_usd_calc)

            bal0_before_mint = Decimal(str(row['initial_contract_balance_token0']))
            bal1_before_mint = Decimal(str(row['initial_contract_balance_token1']))
            amt0_minted = Decimal(str(row['amount0_provided_to_mint']))
            amt1_minted = Decimal(str(row['amount1_provided_to_mint']))
            remaining_token0_contract = bal0_before_mint - amt0_minted
            remaining_token1_contract = bal1_before_mint - amt1_minted
            v_remaining_usd_calc = (remaining_token0_contract / (Decimal(10)**token0_decimals)) + \
                                   (remaining_token1_contract / (Decimal(10)**token1_decimals)) * eth_price_usd
            df.loc[index, 'v_remaining_usd'] = float(v_remaining_usd_calc)
            v_current_total_usd_calc = v_lp_usd_calc + v_remaining_usd_calc
            df.loc[index, 'v_current_total_usd'] = float(v_current_total_usd_calc)
            v_hodl_usd_calc = (bal0_before_mint / (Decimal(10)**token0_decimals)) + \
                              (bal1_before_mint / (Decimal(10)**token1_decimals)) * eth_price_usd
            df.loc[index, 'v_hodl_usd'] = float(v_hodl_usd_calc)
            il_usd_calc = v_hodl_usd_calc - v_current_total_usd_calc
            df.loc[index, 'il_usd'] = float(il_usd_calc)
            fees0 = Decimal(str(row['fees_collected_token0'])); fees1 = Decimal(str(row['fees_collected_token1']))
            v_fees_usd_calc = (fees0 / (Decimal(10)**token0_decimals)) + \
                              (fees1 / (Decimal(10)**token1_decimals)) * eth_price_usd
            df.loc[index, 'v_fees_usd'] = float(v_fees_usd_calc)
            gas_eth = Decimal(str(row['gas_cost_eth']))
            v_gas_usd_calc = gas_eth * eth_price_usd
            df.loc[index, 'v_gas_usd'] = float(v_gas_usd_calc)
            pnl_vs_hodl_usd_calc = (v_current_total_usd_calc + v_fees_usd_calc) - v_hodl_usd_calc - v_gas_usd_calc
            df.loc[index, 'pnl_vs_hodl_usd'] = float(pnl_vs_hodl_usd_calc)

            if index == 0:
                df.loc[index, 'period_actual_pnl_usd'] = float(pnl_vs_hodl_usd_calc)
            else:
                v_current_total_usd_prev = Decimal(str(df.loc[index-1, 'v_current_total_usd']))
                current_gas_usd = Decimal(str(df.loc[index, 'v_gas_usd'])) # Gas for current op
                current_fees_usd = Decimal(str(df.loc[index, 'v_fees_usd'])) # Fees for current op (likely 0)
                df.loc[index, 'period_actual_pnl_usd'] = float(v_current_total_usd_calc - v_current_total_usd_prev - current_gas_usd + current_fees_usd)
        except Exception as e:
            print(f"Error in main P&L loop for row {index}, contract {contract_type}: {e}")
            # Set problematic columns to NaN or 0 to avoid breaking downstream calcs like cumsum
            for pnl_col_item in ['v_lp_usd', 'v_remaining_usd', 'v_current_total_usd', 'v_hodl_usd', 'il_usd', 'v_fees_usd', 'v_gas_usd', 'pnl_vs_hodl_usd', 'period_actual_pnl_usd']:
                df.loc[index, pnl_col_item] = 0.0


    if len(df) > 1:
        for i in range(len(df) - 1):
            try:
                L_i = Decimal(str(df.loc[i, 'finalLiquidity_contract']))
                if L_i == Decimal(0):
                    df.loc[i, 'value_lp_at_period_end_usd'] = 0.0
                    df.loc[i, 'pnl_lp_holding_period_usd'] = 0.0
                    df.loc[i, 'position_i_ended_in_range'] = False
                    df.loc[i, 'net_pnl_period_usd'] = float(-Decimal(str(df.loc[i, 'v_gas_usd'])))
                    continue

                lower_tick_i_val = df.loc[i, 'finalTickLower_contract']
                upper_tick_i_val = df.loc[i, 'finalTickUpper_contract']
                v_lp_start_i_usd_val = df.loc[i, 'v_lp_usd']
                gas_cost_entry_i_usd_val = df.loc[i, 'v_gas_usd']

                current_tick_at_end_of_i_val = df.loc[i+1, 'currentTick_pool']
                eth_price_at_end_of_i_val = df.loc[i+1, 'actualPrice_pool']
                
                lower_tick_i = Decimal(str(lower_tick_i_val))
                upper_tick_i = Decimal(str(upper_tick_i_val))
                v_lp_start_i_usd = Decimal(str(v_lp_start_i_usd_val))
                gas_cost_entry_i_usd = Decimal(str(gas_cost_entry_i_usd_val))
                current_tick_at_end_of_i = Decimal(str(current_tick_at_end_of_i_val))
                eth_price_at_end_of_i = Decimal(str(eth_price_at_end_of_i_val)) if pd.notna(eth_price_at_end_of_i_val) and eth_price_at_end_of_i_val!=0 else Decimal(0)

                amt0_lp_end_i, amt1_lp_end_i = 0,0
                if L_i > 0 and lower_tick_i < upper_tick_i:
                    amt0_lp_end_i, amt1_lp_end_i = get_amounts_in_lp(L_i, current_tick_at_end_of_i, lower_tick_i, upper_tick_i)
                
                v_lp_end_i_token0_usd = (Decimal(amt0_lp_end_i) / (Decimal(10)**token0_decimals))
                v_lp_end_i_token1_usd = (Decimal(amt1_lp_end_i) / (Decimal(10)**token1_decimals)) * eth_price_at_end_of_i
                v_lp_end_i_usd_calc = v_lp_end_i_token0_usd + v_lp_end_i_token1_usd
                df.loc[i, 'value_lp_at_period_end_usd'] = float(v_lp_end_i_usd_calc)
                
                pnl_lp_holding_calc = v_lp_end_i_usd_calc - v_lp_start_i_usd
                df.loc[i, 'pnl_lp_holding_period_usd'] = float(pnl_lp_holding_calc)
                df.loc[i, 'net_pnl_period_usd'] = float(pnl_lp_holding_calc - gas_cost_entry_i_usd)

                ended_in_range = (current_tick_at_end_of_i >= lower_tick_i) and \
                                 (current_tick_at_end_of_i < upper_tick_i)
                df.loc[i, 'position_i_ended_in_range'] = ended_in_range
            except Exception as e:
                print(f"Error in holding period P&L for row {i}, contract {contract_type}: {e}")
                df.loc[i, 'value_lp_at_period_end_usd'] = 0.0
                df.loc[i, 'pnl_lp_holding_period_usd'] = 0.0
                df.loc[i, 'position_i_ended_in_range'] = False
                df.loc[i, 'net_pnl_period_usd'] = float(-Decimal(str(df.loc[i, 'v_gas_usd']))) if pd.notna(df.loc[i, 'v_gas_usd']) else 0.0


    if not df.empty:
        df.loc[len(df)-1, 'net_pnl_period_usd'] = float(df.loc[len(df)-1, 'period_actual_pnl_usd']) if pd.notna(df.loc[len(df)-1, 'period_actual_pnl_usd']) else 0.0
        df.loc[len(df)-1, 'pnl_lp_holding_period_usd'] = 0.0 # Cannot be calculated for the last row
        df.loc[len(df)-1, 'value_lp_at_period_end_usd'] = float(df.loc[len(df)-1, 'v_lp_usd']) # Value at end is its current value
        df.loc[len(df)-1, 'position_i_ended_in_range'] = pd.NA


    df['period_actual_pnl_usd'] = pd.to_numeric(df['period_actual_pnl_usd'], errors='coerce').fillna(0.0)
    df['net_pnl_period_usd'] = pd.to_numeric(df['net_pnl_period_usd'], errors='coerce').fillna(0.0)

    df['cumulative_actual_pnl_usd'] = df['period_actual_pnl_usd'].cumsum()
    df['cumulative_net_pnl_usd'] = df['net_pnl_period_usd'].cumsum()

    results['total_gas_cost_eth'] = df['gas_cost_eth'].sum()
    df['cumulative_gas_eth'] = df['gas_cost_eth'].cumsum()
    df['rolling_tir'] = df['is_in_range'].rolling(window=5, min_periods=1).mean() * 100
    
    results['avg_v_lp_usd'] = df['v_lp_usd'].mean() if not df['v_lp_usd'].dropna().empty else 0.0
    results['final_v_lp_usd'] = df['v_lp_usd'].iloc[-1] if not df.empty and not df['v_lp_usd'].dropna().empty else 0.0
    results['total_il_usd'] = df['il_usd'].sum()
    results['avg_il_usd'] = df['il_usd'].mean() if not df['il_usd'].dropna().empty else 0.0
    results['total_fees_usd'] = df['v_fees_usd'].sum()
    results['total_gas_usd'] = df['v_gas_usd'].sum()
    results['total_pnl_vs_hodl_usd'] = df['pnl_vs_hodl_usd'].sum()
    results['final_cumulative_actual_pnl_usd'] = df['cumulative_actual_pnl_usd'].iloc[-1] if not df.empty and not df['cumulative_actual_pnl_usd'].dropna().empty else 0.0
    results['final_cumulative_net_pnl_usd'] = df['cumulative_net_pnl_usd'].iloc[-1] if not df.empty and not df['cumulative_net_pnl_usd'].dropna().empty else 0.0

    if len(df) > 1 and not df['position_i_ended_in_range'].iloc[:-1].dropna().empty:
        results['percent_periods_ended_in_range'] = df['position_i_ended_in_range'].iloc[:-1].mean() * 100
    else:
        results['percent_periods_ended_in_range'] = pd.NA

    print(f"\n--- {contract_type} DataFrame Head with new P&L columns (Showing subset for brevity) ---")
    print(df[['timestamp', 'v_lp_usd', 'net_pnl_period_usd', 'cumulative_net_pnl_usd', 'position_i_ended_in_range']].head())
    print("-----------------------------------------------------\n")

    return results, df

# --- (بقیه کد برای ایجاد جدول خلاصه و نمودارها مانند قبل، با اصلاحات لازم برای نمایش معیارهای جدید) ---
# ... (فراخوانی analyze_data)
results_pred, df_pred = analyze_data(df_pred.copy(), 'Predictive')
results_base, df_base = analyze_data(df_base.copy(), 'Baseline')

# --- 4. Create Summary DataFrame ---
summary_data = {
    'Metric': [
        'Time In Range (%)', 'Total Gas Cost (ETH)', 'Mean Absolute Error (MAE)',
        'Mean Absolute Percentage Error (MAPE %)', 'Average LP Value (USD)',
        'Final LP Value (USD)', 'Total Impermanent Loss (USD)',
        'Average Impermanent Loss (USD)', 'Total Fees Earned (USD)',
        'Total Gas Cost (USD)', 'Net P&L vs HODL (USD)',
        'Final Cumulative Actual P&L (USD)',
        '% Periods Ended In Range', 
        'Final Cumulative Net Holding P&L (USD)' # تغییر نام برای وضوح
    ],
    'Predictive Strategy': [
        f"{results_pred.get('time_in_range_pct', 0):.2f}",
        f"{results_pred.get('total_gas_cost_eth', 0):.6f}",
        f"{results_pred.get('mae', 'N/A') if isinstance(results_pred.get('mae'), (int, float)) else 'N/A'}",
        f"{results_pred.get('mape', 'N/A') if isinstance(results_pred.get('mape'), (int, float)) else 'N/A'}",
        f"{results_pred.get('avg_v_lp_usd', 0):.2f}",
        f"{results_pred.get('final_v_lp_usd', 0):.2f}",
        f"{results_pred.get('total_il_usd', 0):.2f}",
        f"{results_pred.get('avg_il_usd', 0):.2f}",
        f"{results_pred.get('total_fees_usd', 0):.2f}",
        f"{results_pred.get('total_gas_usd', 0):.2f}",
        f"{results_pred.get('total_pnl_vs_hodl_usd', 0):.2f}",
        f"{results_pred.get('final_cumulative_actual_pnl_usd', 0):.2f}",
        f"{results_pred.get('percent_periods_ended_in_range', 'N/A') if pd.notna(results_pred.get('percent_periods_ended_in_range')) else 'N/A':.2f}",
        f"{results_pred.get('final_cumulative_net_pnl_usd', 0):.2f}"
    ],
    'Baseline Strategy': [
        f"{results_base.get('time_in_range_pct', 0):.2f}",
        f"{results_base.get('total_gas_cost_eth', 0):.6f}",
        'N/A', 'N/A',
        f"{results_base.get('avg_v_lp_usd', 0):.2f}",
        f"{results_base.get('final_v_lp_usd', 0):.2f}",
        f"{results_base.get('total_il_usd', 0):.2f}",
        f"{results_base.get('avg_il_usd', 0):.2f}",
        f"{results_base.get('total_fees_usd', 0):.2f}",
        f"{results_base.get('total_gas_usd', 0):.2f}",
        f"{results_base.get('total_pnl_vs_hodl_usd', 0):.2f}",
        f"{results_base.get('final_cumulative_actual_pnl_usd', 0):.2f}",
        f"{results_base.get('percent_periods_ended_in_range', 'N/A') if pd.notna(results_base.get('percent_periods_ended_in_range')) else 'N/A':.2f}",
        f"{results_base.get('final_cumulative_net_pnl_usd', 0):.2f}"
    ]
}
summary_df = pd.DataFrame(summary_data)
print("--- Results Summary Table ---")
print(summary_df.to_string(index=False))
print("\nN/A = Not Applicable")
print("Warning: Ensure you are using your full dataset for final results.")

# --- 5. Plotting for Thesis ---
fig = plt.figure(figsize=(20, 42)) 
gs = fig.add_gridspec(7, 2)

ax1 = fig.add_subplot(gs[0, 0]) 
if not df_pred.empty:
    sns.lineplot(data=df_pred, x='timestamp', y='actualPrice_pool', label='Actual Price', marker='o', ax=ax1, color='b')
    if 'predictedPrice_api' in df_pred.columns:
        sns.lineplot(data=df_pred, x='timestamp', y='predictedPrice_api', label='Predicted Price', marker='x', linestyle='--', ax=ax1, color='g')
    for i, row in df_pred.iterrows():
        if row.get('finalTickLower_contract', 0) != 0 or row.get('finalTickUpper_contract', 0) != 0:
            lower_price = tick_to_price(row['finalTickLower_contract'])
            upper_price = tick_to_price(row['finalTickUpper_contract'])
            if pd.notna(lower_price) and pd.notna(upper_price) and lower_price < upper_price:
                 ax1.fill_between(df_pred['timestamp'], lower_price, upper_price,
                                 alpha=0.15, color='g', label='Predictive Range' if i == 0 else "")
ax1.set_title('Predictive Strategy: Price & Liquidity Range', weight='bold')
ax1.set_xlabel('Time'); ax1.set_ylabel('ETH Price (USD)')
ax1.legend(); ax1.tick_params(axis='x', rotation=30)

ax2 = fig.add_subplot(gs[0, 1]) 
if not df_base.empty:
    sns.lineplot(data=df_base, x='timestamp', y='actualPrice_pool', label='Actual Price', marker='o', ax=ax2, color='b')
    for i, row in df_base.iterrows():
        if row.get('finalTickLower_contract', 0) != 0 or row.get('finalTickUpper_contract', 0) != 0:
            lower_price = tick_to_price(row['finalTickLower_contract'])
            upper_price = tick_to_price(row['finalTickUpper_contract'])
            if pd.notna(lower_price) and pd.notna(upper_price) and lower_price < upper_price:
                ax2.fill_between(df_base['timestamp'], lower_price, upper_price,
                                 alpha=0.2, color='orange', label='Baseline Range' if i == 0 else "")
ax2.set_title('Baseline Strategy: Price & Liquidity Range', weight='bold')
ax2.set_xlabel('Time'); ax2.set_ylabel('ETH Price (USD)')
ax2.legend(); ax2.tick_params(axis='x', rotation=30)

ax3 = fig.add_subplot(gs[1, 0]) 
if 'prediction_error_pct' in df_pred.columns and not df_pred['prediction_error_pct'].dropna().empty:
    sns.lineplot(data=df_pred, x='timestamp', y='prediction_error_pct', ax=ax3, color='r', label='Instantaneous Error (%)')
    ax3.axhline(0, color='grey', linestyle='--')
    mape_val = results_pred.get('mape', 'N/A')
    mape_display_str = f"{mape_val:.2f}%" if isinstance(mape_val, (int, float)) else "N/A" 
    ax3.set_title(f"Prediction Error Over Time (MAPE: {mape_display_str})", weight='bold') 
    ax3.tick_params(axis='x', rotation=30)
    if not df_pred['prediction_error_pct'].dropna().empty:
        ax3_hist = ax3.inset_axes([0.65, 0.65, 0.3, 0.3])
        sns.histplot(df_pred['prediction_error_pct'].dropna(), bins=10, kde=False, ax=ax3_hist, color='r', alpha=0.6)
        ax3_hist.set_title('Error Distribution', fontsize=9)
else:
    ax3.text(0.5, 0.5, "Prediction error data not available", ha='center', va='center', transform=ax3.transAxes)
ax3.set_xlabel('Time'); ax3.set_ylabel('Prediction Error (%)')

ax4 = fig.add_subplot(gs[1, 1]) 
if 'actualPrice_pool' in df_pred.columns and 'predictedPrice_api' in df_pred.columns and \
   not df_pred[['actualPrice_pool', 'predictedPrice_api']].dropna().empty:
    sns.scatterplot(data=df_pred, x='actualPrice_pool', y='predictedPrice_api', ax=ax4, s=100, alpha=0.7, label='Predictions')
    min_val = np.nanmin([df_pred['actualPrice_pool'].dropna().min(), df_pred['predictedPrice_api'].dropna().min()])
    max_val = np.nanmax([df_pred['actualPrice_pool'].dropna().max(), df_pred['predictedPrice_api'].dropna().max()])
    if pd.notna(min_val) and pd.notna(max_val) and min_val != max_val : # اضافه کردن شرط برای جلوگیری از خطای یکسان بودن min و max
        lims = [min_val, max_val]
        ax4.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='Ideal Prediction')
    elif pd.notna(min_val) and pd.notna(max_val) and min_val == max_val: # اگر همه نقاط یکسان باشند
        ax4.plot([min_val-1,max_val+1], [min_val-1,max_val+1], 'k--', alpha=0.75, zorder=0, label='Ideal Prediction')


else:
    ax4.text(0.5, 0.5, "Price data not available for scatter plot", ha='center', va='center', transform=ax4.transAxes)
ax4.set_title('Actual Price vs. Predicted Price', weight='bold')
ax4.set_xlabel('Actual Price (USD)'); ax4.set_ylabel('Predicted Price (USD)')
ax4.legend(); ax4.set_aspect('equal', adjustable='box')

ax5 = fig.add_subplot(gs[2, 0]) 
if 'rolling_tir' in df_pred.columns and not df_pred.empty :
    sns.lineplot(data=df_pred, x='timestamp', y='rolling_tir', label='Predictive', marker='.', ax=ax5)
if 'rolling_tir' in df_base.columns and not df_base.empty:
    sns.lineplot(data=df_base, x='timestamp', y='rolling_tir', label='Baseline', marker='.', ax=ax5)
ax5.axhline(results_pred.get('time_in_range_pct', 0), color='blue', linestyle=':', label=f"Predictive Avg ({results_pred.get('time_in_range_pct', 0):.1f}%)")
ax5.axhline(results_base.get('time_in_range_pct', 0), color='orange', linestyle=':', label=f"Baseline Avg ({results_base.get('time_in_range_pct', 0):.1f}%)")
ax5.set_title('Time In Range (5-Period Rolling Average)', weight='bold')
ax5.set_xlabel('Time'); ax5.set_ylabel('Time In Range (%)')
ax5.set_ylim(0, 105); ax5.legend(); ax5.tick_params(axis='x', rotation=30)

ax6 = fig.add_subplot(gs[2, 1]) 
if 'cumulative_gas_eth' in df_pred.columns and not df_pred.empty:
    sns.lineplot(data=df_pred, x='timestamp', y='cumulative_gas_eth', label='Predictive', marker='.', ax=ax6)
if 'cumulative_gas_eth' in df_base.columns and not df_base.empty:
    sns.lineplot(data=df_base, x='timestamp', y='cumulative_gas_eth', label='Baseline', marker='.', ax=ax6)
ax6.set_title('Cumulative Gas Costs (ETH)', weight='bold')
ax6.set_xlabel('Time'); ax6.set_ylabel('Total Gas Cost (ETH)')
ax6.legend(); ax6.tick_params(axis='x', rotation=30)

ax_lp_value = fig.add_subplot(gs[3, 0]) 
if 'v_lp_usd' in df_pred.columns and not df_pred.empty:
    sns.lineplot(data=df_pred, x='timestamp', y='v_lp_usd', label='Predictive LP Value (USD)', marker='.', ax=ax_lp_value, color='purple')
if 'v_lp_usd' in df_base.columns and not df_base.empty:
    sns.lineplot(data=df_base, x='timestamp', y='v_lp_usd', label='Baseline LP Value (USD)', marker='.', ax=ax_lp_value, color='brown')
ax_lp_value.set_title('LP Position Value Over Time (USD)', weight='bold')
ax_lp_value.set_xlabel('Time'); ax_lp_value.set_ylabel('LP Value (USD)')
ax_lp_value.legend(); ax_lp_value.tick_params(axis='x', rotation=30)
ax_lp_value.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

ax_il = fig.add_subplot(gs[3, 1]) 
if 'il_usd' in df_pred.columns and not df_pred.empty:
    sns.lineplot(data=df_pred, x='timestamp', y='il_usd', label='Predictive IL (USD)', marker='.', ax=ax_il, color='cyan')
if 'il_usd' in df_base.columns and not df_base.empty:
    sns.lineplot(data=df_base, x='timestamp', y='il_usd', label='Baseline IL (USD)', marker='.', ax=ax_il, color='magenta')
ax_il.set_title('Impermanent Loss Over Time (USD)', weight='bold')
ax_il.set_xlabel('Time'); ax_il.set_ylabel('Impermanent Loss (USD)')
ax_il.axhline(0, color='grey', linestyle='--')
ax_il.legend(); ax_il.tick_params(axis='x', rotation=30)
ax_il.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

ax_pnl_vs_hodl = fig.add_subplot(gs[4, 0]) 
if 'pnl_vs_hodl_usd' in df_pred.columns and not df_pred.empty:
    sns.lineplot(data=df_pred, x='timestamp', y='pnl_vs_hodl_usd', label='Predictive P&L vs HODL (USD)', marker='.', ax=ax_pnl_vs_hodl, color='lime')
if 'pnl_vs_hodl_usd' in df_base.columns and not df_base.empty:
    sns.lineplot(data=df_base, x='timestamp', y='pnl_vs_hodl_usd', label='Baseline P&L vs HODL (USD)', marker='.', ax=ax_pnl_vs_hodl, color='red')
ax_pnl_vs_hodl.set_title('Net P&L vs HODL Per Period (USD)', weight='bold')
ax_pnl_vs_hodl.set_xlabel('Time'); ax_pnl_vs_hodl.set_ylabel('P&L vs HODL / Period (USD)')
ax_pnl_vs_hodl.axhline(0, color='grey', linestyle='--')
ax_pnl_vs_hodl.legend(); ax_pnl_vs_hodl.tick_params(axis='x', rotation=30)
ax_pnl_vs_hodl.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

ax_cumulative_pnl_actual = fig.add_subplot(gs[4, 1]) 
if 'cumulative_actual_pnl_usd' in df_pred.columns and not df_pred.empty:
    sns.lineplot(data=df_pred, x='timestamp', y='cumulative_actual_pnl_usd', label='Predictive Cum. Actual P&L (USD)', marker='.', ax=ax_cumulative_pnl_actual, color='navy')
if 'cumulative_actual_pnl_usd' in df_base.columns and not df_base.empty:
    sns.lineplot(data=df_base, x='timestamp', y='cumulative_actual_pnl_usd', label='Baseline Cum. Actual P&L (USD)', marker='.', ax=ax_cumulative_pnl_actual, color='saddlebrown')
ax_cumulative_pnl_actual.set_title('Cumulative Actual P&L (Portfolio Value Change - Gas)', weight='bold')
ax_cumulative_pnl_actual.set_xlabel('Time'); ax_cumulative_pnl_actual.set_ylabel('Cumulative Actual P&L (USD)')
ax_cumulative_pnl_actual.axhline(0, color='grey', linestyle='--')
ax_cumulative_pnl_actual.legend(); ax_cumulative_pnl_actual.tick_params(axis='x', rotation=30)
ax_cumulative_pnl_actual.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

ax_cumulative_net_pnl = fig.add_subplot(gs[5, 0]) # تغییر gs[5, :] به gs[5,0] برای همترازی
if 'cumulative_net_pnl_usd' in df_pred.columns and not df_pred.empty:
    sns.lineplot(data=df_pred, x='timestamp', y='cumulative_net_pnl_usd', label='Predictive Cum. Net Holding P&L (USD)', marker='.', ax=ax_cumulative_net_pnl, color='green')
if 'cumulative_net_pnl_usd' in df_base.columns and not df_base.empty:
    sns.lineplot(data=df_base, x='timestamp', y='cumulative_net_pnl_usd', label='Baseline Cum. Net Holding P&L (USD)', marker='.', ax=ax_cumulative_net_pnl, color='darkorange')
ax_cumulative_net_pnl.set_title('Cumulative Net Holding Period P&L', weight='bold') # عنوان کوتاهتر
ax_cumulative_net_pnl.set_xlabel('Time'); ax_cumulative_net_pnl.set_ylabel('Cumulative Net Holding P&L (USD)')
ax_cumulative_net_pnl.axhline(0, color='grey', linestyle='--')
ax_cumulative_net_pnl.legend(); ax_cumulative_net_pnl.tick_params(axis='x', rotation=30)
ax_cumulative_net_pnl.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

ax_ended_in_range = fig.add_subplot(gs[5, 1]) # تغییر gs[6, :] به gs[5,1] برای همترازی
ended_in_range_data = []
if pd.notna(results_pred.get('percent_periods_ended_in_range')) and results_pred.get('percent_periods_ended_in_range') is not None :
    ended_in_range_data.append({'Strategy': 'Predictive', 'Percent': results_pred.get('percent_periods_ended_in_range')})
if pd.notna(results_base.get('percent_periods_ended_in_range')) and results_base.get('percent_periods_ended_in_range') is not None:
    ended_in_range_data.append({'Strategy': 'Baseline', 'Percent': results_base.get('percent_periods_ended_in_range')})

if ended_in_range_data:
    df_ended_in_range = pd.DataFrame(ended_in_range_data)
    sns.barplot(x='Strategy', y='Percent', data=df_ended_in_range, ax=ax_ended_in_range, palette=['skyblue', 'lightcoral'])
    ax_ended_in_range.set_title('% Holding Periods Ended In Range', weight='bold') # عنوان کوتاهتر
    ax_ended_in_range.set_xlabel('Strategy'); ax_ended_in_range.set_ylabel('Percent (%)')
    ax_ended_in_range.set_ylim(0, 105)
    for p_patch in ax_ended_in_range.patches: # تغییر نام متغیر p
        ax_ended_in_range.annotate(f"{p_patch.get_height():.1f}%", (p_patch.get_x() + p_patch.get_width() / 2., p_patch.get_height()),
                                   ha='center', va='center', xytext=(0, 5), textcoords='offset points')
else:
    ax_ended_in_range.text(0.5, 0.5, "Data for 'periods ended in range' not available\n(requires at least 2 data rows per strategy)",
                           ha='center', va='center', transform=ax_ended_in_range.transAxes)


fig.suptitle('Comprehensive Analysis of Uniswap V3 AMM Strategies', fontsize=24, y=1.005) 
plt.tight_layout(rect=[0, 0, 1, 0.97]) 
plt.savefig("thesis_full_results_english_v2.png", dpi=300, bbox_inches='tight')
print("\nMain plots saved as thesis_full_results_english_v2.png")

# --- 6. Create and Save Summary Table Image ---
fig_table, ax_table = plt.subplots(figsize=(12, 5)) 
ax_table.set_title("Results Summary Table", weight='bold', size=14, y=1.18)
ax_table.axis('tight'); ax_table.axis('off')
the_table = ax_table.table(cellText=summary_df.values,
                           colLabels=summary_df.columns,
                           colWidths=[0.26] + [0.09]*(len(summary_df.columns)-1), 
                           cellLoc='center', loc='center',
                           colColours=["#E8E8E8"]*len(summary_df.columns))
the_table.auto_set_font_size(False)
the_table.set_fontsize(7.5) 
the_table.scale(1.1, 1.8) 
for (row_idx, col_idx), cell in the_table.get_celld().items(): # تغییر نام متغیرها
    if row_idx == 0: cell.set_text_props(weight='bold')
plt.savefig("thesis_summary_table_v2.png", dpi=300, bbox_inches='tight', pad_inches=0.2)
print("Summary table saved as thesis_summary_table_v2.png")

# --- 7. Show Main Plots ---
plt.show()
