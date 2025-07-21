import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
from pathlib import Path
import sys
from decimal import Decimal, getcontext, ROUND_HALF_UP

# --- 0. Setup Directories (CORRECTED) ---
try:
    script_dir = Path(__file__).parent.resolve()
except NameError:
    script_dir = Path('.').resolve()

input_csv_dir = script_dir / "processed_results"
output_plots_dir = script_dir / "plots_results"
output_plots_dir.mkdir(parents=True, exist_ok=True)

print(f"Input CSVs will be read from: {input_csv_dir}")
print(f"Plots will be saved in: {output_plots_dir}")

# --- Constants ---
getcontext().prec = 78
TOKEN0_DECIMALS = 6
TOKEN1_DECIMALS = 18
TOKEN0_NAME = "USDC" 
TOKEN1_NAME = "WETH"

# --- 1. Data Loading ---
PREDICTIVE_FINAL_FILE_PATH = input_csv_dir / "predictive_final.csv"
BASELINE_FINAL_FILE_PATH = input_csv_dir / "baseline_final.csv"

try:
    df_pred_input = pd.read_csv(PREDICTIVE_FINAL_FILE_PATH)
    df_base_input = pd.read_csv(BASELINE_FINAL_FILE_PATH)
    print("✅ Final CSV Files loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: One or both final CSV files not found in '{input_csv_dir}'.")
    sys.exit(1)
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")
    sys.exit(1)

# --- 2. Data Re-Preprocessing ---
def re_preprocess_readable_df(df_input: pd.DataFrame) -> pd.DataFrame:
    df = df_input.copy()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    cols_to_ensure_numeric = [
        'predictedPrice_api', 'actualPrice_pool', 'currentTick_pool',
        'finalTickLower_contract', 'finalTickUpper_contract',
        'gas_cost_eth', 'liquidity_contract', 'finalLiquidity_contract', 'currentLiquidity_contract',
        'amount0_provided_to_mint', 'amount1_provided_to_mint',
        'initial_contract_balance_token0', 'initial_contract_balance_token1',
        'fees_collected_token0', 'fees_collected_token1',
        'range_width_multiplier_setting', 'sqrtPriceX96_pool'
    ]
    
    for t_name in [TOKEN0_NAME, TOKEN1_NAME]:
        for prefix in ['amount', 'initial_balance', 'fees']:
            base_name_final = ""
            if prefix == 'amount': base_name_final = f'amount_{t_name}_minted'
            elif prefix == 'initial_balance': base_name_final = f'initial_balance_{t_name}'
            elif prefix == 'fees': base_name_final = f'fees_{t_name}_collected'
            
            if base_name_final:
                cols_to_ensure_numeric.append(f'{base_name_final}_readable')
                cols_to_ensure_numeric.append(f'{base_name_final}_usd')

    for col_prefix in ['currentTick_pool', 'finalTickLower_contract', 'finalTickUpper_contract', 'predictedTick_calculated']:
        cols_to_ensure_numeric.append(f'{col_prefix}_{TOKEN1_NAME}{TOKEN0_NAME}_price')

    for col in list(set(cols_to_ensure_numeric)):
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0) 

    if 'timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.sort_values('timestamp').reset_index(drop=True)
    return df

df_pred_analyzed_input = re_preprocess_readable_df(df_pred_input.copy())
df_base_analyzed_input = re_preprocess_readable_df(df_base_input.copy())

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.max_open_warning'] = 50

# --- 3. Helper & Metric Functions ---
def calculate_errors(y_true, y_pred):
    errors = y_true - y_pred
    mae = np.mean(np.abs(errors))
    y_true_safe = np.where(y_true == 0, 1e-9, y_true)
    mape = np.mean(np.abs(errors / y_true_safe)) * 100
    return errors, mae, mape

def get_amounts_in_lp(liquidity: Decimal, current_tick: Decimal, lower_tick: Decimal, upper_tick: Decimal) -> tuple[int, int]:
    L = liquidity
    if not (pd.notna(float(current_tick)) and pd.notna(float(lower_tick)) and pd.notna(float(upper_tick))): return 0,0
    if L == 0 or lower_tick >= upper_tick: return 0,0
    try:
        sqrt_P_current = Decimal('1.0001')**(current_tick / Decimal('2'))
        sqrt_P_lower = Decimal('1.0001')**(lower_tick / Decimal('2'))
        sqrt_P_upper = Decimal('1.0001')**(upper_tick / Decimal('2'))
    except Exception: return 0,0
    amount0_in_lp, amount1_in_lp = Decimal(0), Decimal(0)
    if current_tick < lower_tick:
        amount0_in_lp = L * (Decimal(1) / sqrt_P_lower - Decimal(1) / sqrt_P_upper)
    elif current_tick >= upper_tick:
        amount1_in_lp = L * (sqrt_P_upper - sqrt_P_lower)
    else:
        amount0_in_lp = L * (Decimal(1) / sqrt_P_current - Decimal(1) / sqrt_P_upper)
        amount1_in_lp = L * (sqrt_P_current - sqrt_P_lower)
    return int(amount0_in_lp.to_integral_value(rounding=ROUND_HALF_UP)), \
           int(amount1_in_lp.to_integral_value(rounding=ROUND_HALF_UP))

def analyze_data(df_input: pd.DataFrame, contract_type: str) -> tuple[dict, pd.DataFrame]:
    df = df_input.copy() 
    results = {}

    price_range_lower_col = f'finalTickUpper_contract_{TOKEN1_NAME}{TOKEN0_NAME}_price' 
    price_range_upper_col = f'finalTickLower_contract_{TOKEN1_NAME}{TOKEN0_NAME}_price'

    if 'actualPrice_pool' in df.columns and price_range_lower_col in df.columns and price_range_upper_col in df.columns:
        price_range_lower_usd_series = df[price_range_lower_col]
        price_range_upper_usd_series = df[price_range_upper_col]
        valid_comparison = pd.notna(df['actualPrice_pool']) & pd.notna(price_range_lower_usd_series) & pd.notna(price_range_upper_usd_series)
        df['is_in_range'] = False 
        df.loc[valid_comparison, 'is_in_range'] = (df.loc[valid_comparison, 'actualPrice_pool'] >= price_range_lower_usd_series[valid_comparison]) & (df.loc[valid_comparison, 'actualPrice_pool'] <= price_range_upper_usd_series[valid_comparison])
        results['time_in_range_pct'] = df['is_in_range'].mean() * 100 if not df['is_in_range'].dropna().empty else 0.0
    else:
        df['is_in_range'] = False; results['time_in_range_pct'] = 0.0
    
    if contract_type == 'Predictive' and 'actualPrice_pool' in df.columns and 'predictedPrice_api' in df.columns and not df.empty:
        valid_actual_prices = df['actualPrice_pool'][df['actualPrice_pool'] != 0]
        if not valid_actual_prices.empty:
            errors, mae, mape = calculate_errors(df['actualPrice_pool'], df['predictedPrice_api'])
            df['prediction_error_pct'] = np.where(df['actualPrice_pool'] != 0, (errors / df['actualPrice_pool']) * 100, 0.0)
            results['mae'] = mae; results['mape'] = mape
        else: df['prediction_error_pct'] = 0.0; results['mae'] = 0.0; results['mape'] = 0.0
    else: df['prediction_error_pct'] = 0.0; results['mae'] = 'N/A'; results['mape'] = 'N/A'
    
    pnl_cols = ['amount0_in_lp', 'amount1_in_lp', 'v_lp_usd', 'v_remaining_usd','v_current_total_usd', 'v_hodl_usd', 'il_usd', 'v_fees_usd','v_gas_usd', 'pnl_vs_hodl_usd', 'period_actual_pnl_usd','value_lp_at_period_end_usd', 'pnl_lp_holding_period_usd','position_i_ended_in_range', 'net_pnl_period_usd']
    for col in pnl_cols:
        if col == 'position_i_ended_in_range': df[col] = pd.NA
        else: df[col] = 0.0

    for index, row in df.iterrows():
        try:
            l_val_str = str(row.get('liquidity_contract', '0')) if contract_type == 'Predictive' else str(row.get('finalLiquidity_contract', '0'))
            L_val = Decimal(l_val_str)
            
            eth_price_usd = Decimal(str(row.get('actualPrice_pool', 0)))
            
            amt0_lp, amt1_lp = get_amounts_in_lp(L_val, Decimal(str(row.get('currentTick_pool',0))), Decimal(str(row.get('finalTickLower_contract',0))), Decimal(str(row.get('finalTickUpper_contract',0))))
            df.loc[index, 'amount0_in_lp'] = amt0_lp
            df.loc[index, 'amount1_in_lp'] = amt1_lp
            
            v_lp_usd_calc = (Decimal(amt0_lp) / (Decimal('10')**TOKEN0_DECIMALS)) + ((Decimal(amt1_lp) / (Decimal('10')**TOKEN1_DECIMALS)) * eth_price_usd)
            df.loc[index, 'v_lp_usd'] = float(v_lp_usd_calc)
            
            bal0_raw = Decimal(str(row.get('initial_contract_balance_token0',0)))
            bal1_raw = Decimal(str(row.get('initial_contract_balance_token1',0)))
            v_hodl_usd_calc = (bal0_raw / (Decimal('10')**TOKEN0_DECIMALS)) + ((bal1_raw / (Decimal('10')**TOKEN1_DECIMALS)) * eth_price_usd)
            df.loc[index, 'v_hodl_usd'] = float(v_hodl_usd_calc)

            v_fees_usd_calc = Decimal(str(row.get(f'fees_{TOKEN0_NAME}_collected_usd', 0.0))) + Decimal(str(row.get(f'fees_{TOKEN1_NAME}_collected_usd', 0.0)))
            df.loc[index, 'v_fees_usd'] = float(v_fees_usd_calc)
            
            v_gas_usd_calc = Decimal(str(row.get('gas_cost_eth',0))) * eth_price_usd
            df.loc[index, 'v_gas_usd'] = float(v_gas_usd_calc)
            
            amt0_minted_raw = Decimal(str(row.get('amount0_provided_to_mint',0)))
            amt1_minted_raw = Decimal(str(row.get('amount1_provided_to_mint',0)))
            v_remaining_usd_calc = ((bal0_raw - amt0_minted_raw) / (Decimal('10')**TOKEN0_DECIMALS)) + (((bal1_raw - amt1_minted_raw) / (Decimal('10')**TOKEN1_DECIMALS)) * eth_price_usd)
            df.loc[index, 'v_remaining_usd'] = float(v_remaining_usd_calc)

            v_current_total_usd_calc = v_lp_usd_calc + v_remaining_usd_calc
            df.loc[index, 'v_current_total_usd'] = float(v_current_total_usd_calc)
            
            il_usd_calc = v_hodl_usd_calc - v_current_total_usd_calc
            df.loc[index, 'il_usd'] = float(il_usd_calc)
            
            pnl_vs_hodl_usd_calc = (v_current_total_usd_calc + v_fees_usd_calc) - v_hodl_usd_calc - v_gas_usd_calc
            df.loc[index, 'pnl_vs_hodl_usd'] = float(pnl_vs_hodl_usd_calc)
            
            if index == 0:
                df.loc[index, 'period_actual_pnl_usd'] = float(pnl_vs_hodl_usd_calc)
            else:
                v_current_total_usd_prev = Decimal(str(df.loc[index-1, 'v_current_total_usd']))
                df.loc[index, 'period_actual_pnl_usd'] = float(v_current_total_usd_calc - v_current_total_usd_prev - v_gas_usd_calc + v_fees_usd_calc)
        except Exception as e:
            print(f"Error in main P&L loop for row {index}, contract {contract_type}: {e}.")

    if len(df) > 1:
        for i in range(len(df) - 1):
            try:
                row_i_data = df.loc[i]
                l_i_loop_str = str(row_i_data.get('liquidity_contract', '0')) if contract_type == 'Predictive' else str(row_i_data.get('finalLiquidity_contract', '0'))
                L_i_loop = Decimal(l_i_loop_str)

                if L_i_loop == Decimal(0):
                    df.loc[i, ['value_lp_at_period_end_usd', 'pnl_lp_holding_period_usd']] = 0.0
                    df.loc[i, 'position_i_ended_in_range'] = False
                    df.loc[i, 'net_pnl_period_usd'] = float(-Decimal(str(row_i_data.get('v_gas_usd',0))))
                    continue

                row_i_plus_1_data = df.loc[i+1]
                eth_price_at_end_of_i_loop = Decimal(str(row_i_plus_1_data.get('actualPrice_pool',0)))

                amt0_lp_end_i, amt1_lp_end_i = get_amounts_in_lp(L_i_loop, Decimal(str(row_i_plus_1_data.get('currentTick_pool',0))), Decimal(str(row_i_data.get('finalTickLower_contract',0))), Decimal(str(row_i_data.get('finalTickUpper_contract',0))))
                
                v_lp_end_i_usd_calc_loop = (Decimal(amt0_lp_end_i) / (Decimal('10')**TOKEN0_DECIMALS)) + ((Decimal(amt1_lp_end_i) / (Decimal('10')**TOKEN1_DECIMALS)) * eth_price_at_end_of_i_loop)
                df.loc[i, 'value_lp_at_period_end_usd'] = float(v_lp_end_i_usd_calc_loop)

                pnl_lp_holding_calc_loop = v_lp_end_i_usd_calc_loop - Decimal(str(row_i_data.get('v_lp_usd',0)))
                df.loc[i, 'pnl_lp_holding_period_usd'] = float(pnl_lp_holding_calc_loop)
                
                df.loc[i, 'net_pnl_period_usd'] = float(pnl_lp_holding_calc_loop - Decimal(str(row_i_data.get('v_gas_usd',0))))
                df.loc[i, 'position_i_ended_in_range'] = row_i_plus_1_data.get('is_in_range', False)
            except Exception as e:
                print(f"Error in holding period P&L for row {i}, contract {contract_type}: {e}")

    if not df.empty:
        last_idx = len(df)-1
        df.loc[last_idx, 'net_pnl_period_usd'] = df.loc[last_idx, 'period_actual_pnl_usd']
        df.loc[last_idx, 'pnl_lp_holding_period_usd'] = 0.0
        df.loc[last_idx, 'value_lp_at_period_end_usd'] = df.loc[last_idx, 'v_lp_usd']
    
    df['cumulative_actual_pnl_usd'] = df['period_actual_pnl_usd'].cumsum()
    df['cumulative_net_pnl_usd'] = df['net_pnl_period_usd'].cumsum()
    df['rolling_tir'] = df['is_in_range'].rolling(window=5, min_periods=1).mean() * 100 
    
    results['total_gas_cost_eth'] = df['gas_cost_eth'].sum()
    results['avg_v_lp_usd'] = df['v_lp_usd'].mean()
    results['total_il_usd'] = df['il_usd'].sum()
    results['avg_il_usd'] = df['il_usd'].mean() if not df['il_usd'].dropna().empty else 0.0
    results['total_fees_usd'] = df['v_fees_usd'].sum()
    results['total_gas_usd'] = df['v_gas_usd'].sum()
    results['total_pnl_vs_hodl_usd'] = df['pnl_vs_hodl_usd'].sum()
    results['final_cumulative_actual_pnl_usd'] = df['cumulative_actual_pnl_usd'].iloc[-1] if not df.empty else 0
    results['final_cumulative_net_pnl_usd'] = df['cumulative_net_pnl_usd'].iloc[-1] if not df.empty else 0

    ended_in_range_series = df['position_i_ended_in_range'].iloc[:-1].dropna().astype(bool)
    if not ended_in_range_series.empty:
        results['percent_periods_ended_in_range'] = ended_in_range_series.mean() * 100
    else: results['percent_periods_ended_in_range'] = pd.NA
        
    return results, df

# --- Analyze Data ---
results_pred, df_pred_analyzed = analyze_data(df_pred_analyzed_input.copy(), 'Predictive')
results_base, df_base_analyzed = analyze_data(df_base_analyzed_input.copy(), 'Baseline')

# --- 4. Create Summary DataFrame ---
summary_data = {
    'Metric': ['Time In Range (%)', 'Total Gas Cost (ETH)', 'Mean Absolute Error (MAE)', 'Mean Absolute Percentage Error (MAPE %)', 'Average LP Value (USD)', 'Total Impermanent Loss (USD)', 'Total Fees Earned (USD)', 'Total Gas Cost (USD)', 'Net P&L vs HODL (USD)', 'Final Cumulative Actual P&L (USD)', '% Periods Ended In Range', 'Final Cumulative Net Holding P&L (USD)'],
    'Predictive Strategy': [f"{results_pred.get('time_in_range_pct', 0):.2f}", f"{results_pred.get('total_gas_cost_eth', 0):.6f}", f"{results_pred.get('mae', 'N/A'):.2f}" if isinstance(results_pred.get('mae'), (int, float)) else 'N/A', f"{results_pred.get('mape', 'N/A'):.2f}" if isinstance(results_pred.get('mape'), (int, float)) else 'N/A', f"${results_pred.get('avg_v_lp_usd', 0):,.2f}", f"${results_pred.get('total_il_usd', 0):,.2f}", f"${results_pred.get('total_fees_usd', 0):,.4f}", f"${results_pred.get('total_gas_usd', 0):,.2f}", f"${results_pred.get('total_pnl_vs_hodl_usd', 0):,.2f}", f"${results_pred.get('final_cumulative_actual_pnl_usd', 0):,.2f}", f"{results_pred.get('percent_periods_ended_in_range', 'N/A'):.2f}" if pd.notna(results_pred.get('percent_periods_ended_in_range')) else 'N/A', f"${results_pred.get('final_cumulative_net_pnl_usd', 0):,.2f}"],
    'Baseline Strategy': [f"{results_base.get('time_in_range_pct', 0):.2f}", f"{results_base.get('total_gas_cost_eth', 0):.6f}", 'N/A', 'N/A', f"${results_base.get('avg_v_lp_usd', 0):,.2f}", f"${results_base.get('total_il_usd', 0):,.2f}", f"${results_base.get('total_fees_usd', 0):,.4f}", f"${results_base.get('total_gas_usd', 0):,.2f}", f"${results_base.get('total_pnl_vs_hodl_usd', 0):,.2f}", f"${results_base.get('final_cumulative_actual_pnl_usd', 0):,.2f}", f"${results_base.get('percent_periods_ended_in_range', 'N/A'):.2f}" if pd.notna(results_base.get('percent_periods_ended_in_range')) else 'N/A', f"${results_base.get('final_cumulative_net_pnl_usd', 0):,.2f}"]
}
summary_df = pd.DataFrame(summary_data)
print("\n--- Results Summary Table (Corrected Fees) ---")
print(summary_df.to_string(index=False))

# --- 5. Plotting (All plots included) ---
plot_figsize = (12, 7)
date_formatter = mdates.DateFormatter('%Y-%m-%d %H:%M')
thousands_formatter = plt.FuncFormatter(lambda x, p: f'${int(x):,}')

def can_plot(df, required_cols):
    if df.empty or 'timestamp' not in df or not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        return False
    for col in required_cols:
        if col not in df.columns or df[col].dropna().empty:
            return False
    return True

# Plot 1 & 2: Price & Liquidity Range (Original)
for df, name, color, file_name in [(df_pred_analyzed, 'Predictive', 'lightgreen', "01_predictive_price_liquidity.png"), 
                                   (df_base_analyzed, 'Baseline', 'moccasin', "02_baseline_price_liquidity.png")]:
    if can_plot(df, ['actualPrice_pool', f'finalTickUpper_contract_{TOKEN1_NAME}{TOKEN0_NAME}_price', f'finalTickLower_contract_{TOKEN1_NAME}{TOKEN0_NAME}_price']):
        fig, ax = plt.subplots(figsize=plot_figsize)
        sns.lineplot(data=df, x='timestamp', y='actualPrice_pool', label='Actual Price', marker='.', ax=ax)
        if name == 'Predictive' and 'predictedPrice_api' in df.columns:
            sns.lineplot(data=df, x='timestamp', y='predictedPrice_api', label='Predicted Price', marker='x', linestyle='--', ax=ax)
        ax.fill_between(df['timestamp'], df[f'finalTickUpper_contract_{TOKEN1_NAME}{TOKEN0_NAME}_price'], df[f'finalTickLower_contract_{TOKEN1_NAME}{TOKEN0_NAME}_price'], alpha=0.15, color=color, label=f'{name} Range')
        ax.set_title(f'{name} Strategy: Price & Liquidity Range', weight='bold'); ax.set_xlabel('Time'); ax.set_ylabel('ETH Price (USD)'); ax.legend(); ax.tick_params(axis='x', rotation=30)
        ax.xaxis.set_major_formatter(date_formatter); fig.tight_layout()
        plt.savefig(output_plots_dir / file_name, dpi=300); plt.close(fig)
        print(f"Plot saved: {file_name}")

# Plot 3: Prediction Error Over Time (Original)
if can_plot(df_pred_analyzed, ['prediction_error_pct']):
    fig3, ax3 = plt.subplots(figsize=plot_figsize)
    sns.lineplot(data=df_pred_analyzed, x='timestamp', y='prediction_error_pct', ax=ax3, color='r', label='Instantaneous Error (%)')
    ax3.axhline(0, color='grey', linestyle='--'); mape_val = results_pred.get('mape', 'N/A')
    mape_display_str = f"{mape_val:.2f}%" if isinstance(mape_val, (float, int)) else "N/A"
    ax3.set_title(f"Prediction Error Over Time (MAPE: {mape_display_str})", weight='bold'); ax3.tick_params(axis='x', rotation=30)
    ax3.set_xlabel('Time'); ax3.set_ylabel('Prediction Error (%)'); ax3.xaxis.set_major_formatter(date_formatter); fig3.tight_layout()
    plt.savefig(output_plots_dir / "03_prediction_error.png", dpi=300); plt.close(fig3)
    print("Plot saved: 03_prediction_error.png")

# Plot 4: Actual Price vs. Predicted Price (Original)
if can_plot(df_pred_analyzed, ['actualPrice_pool', 'predictedPrice_api']):
    fig4, ax4 = plt.subplots(figsize=(8,8))
    plot_data_scatter = df_pred_analyzed[['actualPrice_pool', 'predictedPrice_api']].dropna()
    if not plot_data_scatter.empty:
        sns.scatterplot(data=plot_data_scatter, x='actualPrice_pool', y='predictedPrice_api', ax=ax4, s=100, alpha=0.7, label='Predictions')
        min_val = min(plot_data_scatter['actualPrice_pool'].min(), plot_data_scatter['predictedPrice_api'].min())
        max_val = max(plot_data_scatter['actualPrice_pool'].max(), plot_data_scatter['predictedPrice_api'].max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.75, zorder=0, label='Ideal Prediction')
        ax4.set_title('Actual Price vs. Predicted Price', weight='bold'); ax4.set_xlabel('Actual Price (USD)'); ax4.set_ylabel('Predicted Price (USD)'); ax4.legend(); ax4.set_aspect('equal', adjustable='box'); fig4.tight_layout()
        plt.savefig(output_plots_dir / "04_actual_vs_predicted_price.png", dpi=300); plt.close(fig4)
        print("Plot saved: 04_actual_vs_predicted_price.png")

# Plot 5: Rolling Time-In-Range (%) (Original)
if can_plot(df_pred_analyzed, ['rolling_tir']) or can_plot(df_base_analyzed, ['rolling_tir']):
    fig5, ax5 = plt.subplots(figsize=plot_figsize)
    if can_plot(df_pred_analyzed, ['rolling_tir']):
        sns.lineplot(data=df_pred_analyzed, x='timestamp', y='rolling_tir', label='Predictive', marker='.', ax=ax5)
    if can_plot(df_base_analyzed, ['rolling_tir']):
        sns.lineplot(data=df_base_analyzed, x='timestamp', y='rolling_tir', label='Baseline', marker='.', ax=ax5)
    ax5.set_title('Rolling Time-In-Range (%)', weight='bold'); ax5.set_xlabel('Time'); ax5.set_ylabel('TIR (%)'); ax5.set_ylim(0, 105); ax5.legend(); ax5.tick_params(axis='x', rotation=30)
    ax5.xaxis.set_major_formatter(date_formatter); fig5.tight_layout()
    plt.savefig(output_plots_dir / "05_rolling_time_in_range.png", dpi=300); plt.close(fig5)
    print("Plot saved: 05_rolling_time_in_range.png")

# Plot 6: Cumulative Gas Costs (ETH) (Original)
if can_plot(df_pred_analyzed, ['gas_cost_eth']) or can_plot(df_base_analyzed, ['gas_cost_eth']):
    df_pred_analyzed['cumulative_gas_eth'] = df_pred_analyzed['gas_cost_eth'].cumsum()
    df_base_analyzed['cumulative_gas_eth'] = df_base_analyzed['gas_cost_eth'].cumsum()
    fig6, ax6 = plt.subplots(figsize=plot_figsize)
    sns.lineplot(data=df_pred_analyzed, x='timestamp', y='cumulative_gas_eth', label='Predictive', marker='.', ax=ax6)
    sns.lineplot(data=df_base_analyzed, x='timestamp', y='cumulative_gas_eth', label='Baseline', marker='.', ax=ax6)
    ax6.set_title('Cumulative Gas Costs (ETH)', weight='bold'); ax6.set_xlabel('Time'); ax6.set_ylabel('Total Gas Cost (ETH)'); ax6.legend(); ax6.tick_params(axis='x', rotation=30)
    ax6.xaxis.set_major_formatter(date_formatter); fig6.tight_layout()
    plt.savefig(output_plots_dir / "06_cumulative_gas_costs_eth.png", dpi=300); plt.close(fig6)
    print("Plot saved: 06_cumulative_gas_costs_eth.png")

# Plot 7: LP Position Value (USD) (Original)
if can_plot(df_pred_analyzed, ['v_lp_usd']) or can_plot(df_base_analyzed, ['v_lp_usd']):
    fig7, ax7 = plt.subplots(figsize=plot_figsize)
    sns.lineplot(data=df_pred_analyzed, x='timestamp', y='v_lp_usd', label='Predictive LP Value (USD)', marker='.', ax=ax7)
    sns.lineplot(data=df_base_analyzed, x='timestamp', y='v_lp_usd', label='Baseline LP Value (USD)', marker='.', ax=ax7)
    ax7.set_title('LP Position Value Over Time (USD)', weight='bold'); ax7.set_xlabel('Time'); ax7.set_ylabel('LP Value (USD)'); ax7.legend(); ax7.tick_params(axis='x', rotation=30)
    ax7.yaxis.set_major_formatter(thousands_formatter); ax7.xaxis.set_major_formatter(date_formatter); fig7.tight_layout()
    plt.savefig(output_plots_dir / "07_lp_value_over_time.png", dpi=300); plt.close(fig7)
    print("Plot saved: 07_lp_value_over_time.png")

# Plot 8: Impermanent Loss (USD) (Original)
if can_plot(df_pred_analyzed, ['il_usd']) or can_plot(df_base_analyzed, ['il_usd']):
    fig8, ax8 = plt.subplots(figsize=plot_figsize)
    sns.lineplot(data=df_pred_analyzed, x='timestamp', y='il_usd', label='Predictive IL (USD)', marker='.', ax=ax8)
    sns.lineplot(data=df_base_analyzed, x='timestamp', y='il_usd', label='Baseline IL (USD)', marker='.', ax=ax8)
    ax8.set_title('Impermanent Loss Over Time (USD)', weight='bold'); ax8.set_xlabel('Time'); ax8.set_ylabel('Impermanent Loss (USD)'); ax8.axhline(0, color='grey', linestyle='--'); ax8.legend(); ax8.tick_params(axis='x', rotation=30)
    ax8.yaxis.set_major_formatter(thousands_formatter); ax8.xaxis.set_major_formatter(date_formatter); fig8.tight_layout()
    plt.savefig(output_plots_dir / "08_impermanent_loss_over_time.png", dpi=300); plt.close(fig8)
    print("Plot saved: 08_impermanent_loss_over_time.png")

# Plot 9: P&L vs HODL (USD) (Original)
if can_plot(df_pred_analyzed, ['pnl_vs_hodl_usd']) or can_plot(df_base_analyzed, ['pnl_vs_hodl_usd']):
    fig9, ax9 = plt.subplots(figsize=plot_figsize)
    sns.lineplot(data=df_pred_analyzed, x='timestamp', y='pnl_vs_hodl_usd', label='Predictive P&L vs HODL (USD)', marker='.', ax=ax9)
    sns.lineplot(data=df_base_analyzed, x='timestamp', y='pnl_vs_hodl_usd', label='Baseline P&L vs HODL (USD)', marker='.', ax=ax9)
    ax9.set_title('Net P&L vs HODL Per Period (USD)', weight='bold'); ax9.set_xlabel('Time'); ax9.set_ylabel('P&L vs HODL / Period (USD)'); ax9.axhline(0, color='grey', linestyle='--'); ax9.legend(); ax9.tick_params(axis='x', rotation=30)
    ax9.yaxis.set_major_formatter(thousands_formatter); ax9.xaxis.set_major_formatter(date_formatter); fig9.tight_layout()
    plt.savefig(output_plots_dir / "09_pnl_vs_hodl_per_period.png", dpi=300); plt.close(fig9)
    print("Plot saved: 09_pnl_vs_hodl_per_period.png")

# Plot 10: Cumulative Actual P&L (USD) (Original)
if can_plot(df_pred_analyzed, ['cumulative_actual_pnl_usd']) or can_plot(df_base_analyzed, ['cumulative_actual_pnl_usd']):
    fig10, ax10 = plt.subplots(figsize=plot_figsize)
    sns.lineplot(data=df_pred_analyzed, x='timestamp', y='cumulative_actual_pnl_usd', label='Predictive Cum. Actual P&L (USD)', marker='.', ax=ax10)
    sns.lineplot(data=df_base_analyzed, x='timestamp', y='cumulative_actual_pnl_usd', label='Baseline Cum. Actual P&L (USD)', marker='.', ax=ax10)
    ax10.set_title('Cumulative Actual P&L (Portfolio Value Change - Gas + Fees)', weight='bold'); ax10.set_xlabel('Time'); ax10.set_ylabel('Cumulative Actual P&L (USD)'); ax10.axhline(0, color='grey', linestyle='--'); ax10.legend(); ax10.tick_params(axis='x', rotation=30)
    ax10.yaxis.set_major_formatter(thousands_formatter); ax10.xaxis.set_major_formatter(date_formatter); fig10.tight_layout()
    plt.savefig(output_plots_dir / "10_cumulative_actual_pnl.png", dpi=300); plt.close(fig10)
    print("Plot saved: 10_cumulative_actual_pnl.png")

# Plot 11: Cumulative Net Holding P&L (USD) (Original)
if can_plot(df_pred_analyzed, ['cumulative_net_pnl_usd']) or can_plot(df_base_analyzed, ['cumulative_net_pnl_usd']):
    fig11, ax11 = plt.subplots(figsize=plot_figsize)
    sns.lineplot(data=df_pred_analyzed, x='timestamp', y='cumulative_net_pnl_usd', label='Predictive Cum. Net Holding P&L (USD)', marker='.', ax=ax11)
    sns.lineplot(data=df_base_analyzed, x='timestamp', y='cumulative_net_pnl_usd', label='Baseline Cum. Net Holding P&L (USD)', marker='.', ax=ax11)
    ax11.set_title('Cumulative Net Holding Period P&L (LP Value Change - Gas)', weight='bold'); ax11.set_xlabel('Time'); ax11.set_ylabel('Cumulative Net Holding P&L (USD)'); ax11.axhline(0, color='grey', linestyle='--'); ax11.legend(); ax11.tick_params(axis='x', rotation=30)
    ax11.yaxis.set_major_formatter(thousands_formatter); ax11.xaxis.set_major_formatter(date_formatter); fig11.tight_layout()
    plt.savefig(output_plots_dir / "11_cumulative_net_holding_pnl.png", dpi=300); plt.close(fig11)
    print("Plot saved: 11_cumulative_net_holding_pnl.png")

# Plot 12: % Holding Periods Ended In Range (Original)
ended_in_range_data = []
if pd.notna(results_pred.get('percent_periods_ended_in_range')):
    ended_in_range_data.append({'Strategy': 'Predictive', 'Percent': results_pred.get('percent_periods_ended_in_range')})
if pd.notna(results_base.get('percent_periods_ended_in_range')):
    ended_in_range_data.append({'Strategy': 'Baseline', 'Percent': results_base.get('percent_periods_ended_in_range')})
if ended_in_range_data:
    fig12, ax12 = plt.subplots(figsize=(8,6))
    df_ended_in_range = pd.DataFrame(ended_in_range_data)
    sns.barplot(x='Strategy', y='Percent', data=df_ended_in_range, ax=ax12, hue='Strategy', palette={'Predictive':'skyblue', 'Baseline':'lightcoral'}, legend=False)
    ax12.set_title('% Holding Periods Ended In Range', weight='bold'); ax12.set_xlabel('Strategy'); ax12.set_ylabel('Percent (%)'); ax12.set_ylim(0, 105)
    for p_patch in ax12.patches: ax12.annotate(f"{p_patch.get_height():.1f}%", (p_patch.get_x() + p_patch.get_width() / 2., p_patch.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    fig12.tight_layout(); plt.savefig(output_plots_dir / "12_percent_periods_ended_in_range.png", dpi=300); plt.close(fig12)
    print("Plot saved: 12_percent_periods_ended_in_range.png")

# Plot 13: Distribution of Impermanent Loss (USD) (Original)
if can_plot(df_pred_analyzed, ['il_usd']) or can_plot(df_base_analyzed, ['il_usd']):
    fig13, ax13 = plt.subplots(figsize=plot_figsize)
    if can_plot(df_pred_analyzed, ['il_usd']):
        sns.histplot(df_pred_analyzed['il_usd'].dropna(), bins=20, color='blue', alpha=0.6, label='Predictive IL', ax=ax13, kde=True)
    if can_plot(df_base_analyzed, ['il_usd']):
        sns.histplot(df_base_analyzed['il_usd'].dropna(), bins=20, color='orange', alpha=0.6, label='Baseline IL', ax=ax13, kde=True)
    ax13.set_title('Distribution of Impermanent Loss (USD)', weight='bold'); ax13.set_xlabel('Impermanent Loss (USD)'); ax13.set_ylabel('Frequency'); ax13.legend(); ax13.xaxis.set_major_formatter(thousands_formatter)
    fig13.tight_layout(); plt.savefig(output_plots_dir / "13_distribution_impermanent_loss.png", dpi=300); plt.close(fig13)
    print("Plot saved: 13_distribution_impermanent_loss.png")


# --- NEW KEY PLOTS (Replaces old confusing plots) ---
# Plot 14: Cumulative Loss vs HODL (Corrected Version)
def calculate_correct_pnl(df):
    df = df.copy()
    df['pnl_vs_hodl_usd'] = 0.0
    
    for index, row in df.iterrows():
        try:
            # Use finalLiquidity_contract for both strategies for consistency
            l_val_str = str(row.get('finalLiquidity_contract', '0'))
            L_val = Decimal(l_val_str)
            eth_price_usd = Decimal(str(row.get('actualPrice_pool', 0)))

            amt0_lp, amt1_lp = get_amounts_in_lp(L_val, 
                                                 Decimal(str(row.get('currentTick_pool',0))), 
                                                 Decimal(str(row.get('finalTickLower_contract',0))), 
                                                 Decimal(str(row.get('finalTickUpper_contract',0))))
            
            v_lp_usd = (Decimal(amt0_lp) / (10**TOKEN0_DECIMALS)) + ((Decimal(amt1_lp) / (10**TOKEN1_DECIMALS)) * eth_price_usd)
            
            bal0 = Decimal(str(row.get('initial_contract_balance_token0',0)))
            bal1 = Decimal(str(row.get('initial_contract_balance_token1',0)))
            v_hodl = (bal0 / (10**TOKEN0_DECIMALS)) + ((bal1 / (10**TOKEN1_DECIMALS)) * eth_price_usd)
            
            fees = Decimal(str(row.get(f'fees_{TOKEN0_NAME}_collected_usd', 0))) + Decimal(str(row.get(f'fees_{TOKEN1_NAME}_collected_usd', 0)))
            gas = Decimal(str(row.get('gas_cost_eth',0))) * eth_price_usd

            df.loc[index, 'pnl_vs_hodl_usd'] = float(v_lp_usd + fees - v_hodl - gas)
        except Exception as e:
            print(f"Error calculating P&L for row {index}: {str(e)}")
            df.loc[index, 'pnl_vs_hodl_usd'] = 0.0
    
    df['loss_vs_hodl_usd'] = df['pnl_vs_hodl_usd'] * -1
    df['cumulative_loss_vs_hodl'] = df['loss_vs_hodl_usd'].cumsum()
    return df

# Recalculate with correct method
df_pred_corrected = calculate_correct_pnl(df_pred_analyzed_input.copy())
df_base_corrected = calculate_correct_pnl(df_base_analyzed_input.copy())

if can_plot(df_pred_corrected, ['cumulative_loss_vs_hodl']) and can_plot(df_base_corrected, ['cumulative_loss_vs_hodl']):
    fig14, ax14 = plt.subplots(figsize=(12, 7))
    
    # Use the same styling as your working version
    sns.lineplot(data=df_pred_corrected, x='timestamp', y='cumulative_loss_vs_hodl', 
                 label='Predictive Strategy Loss', marker='o', color='blue', ax=ax14)
    sns.lineplot(data=df_base_corrected, x='timestamp', y='cumulative_loss_vs_hodl', 
                 label='Baseline Strategy Loss', marker='o', color='orange', ax=ax14)

    ax14.set_title('Comparison of Cumulative Loss vs. HODL (USD)', weight='bold', pad=20)
    ax14.set_ylabel('Amount of Loss (USD)\n[Higher = More Loss]', linespacing=1.5)
    ax14.set_xlabel('Date', labelpad=10)
    
    # Formatting
    ax14.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax14.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_plots_dir / "14_Cumulative_Loss_Comparison_CORRECTED.png", 
                dpi=300, bbox_inches='tight')
    plt.close(fig14)
    print("✅ Corrected Plot 14 saved successfully.")
# KEY PLOT 15: Fees vs Gas per Period
if can_plot(df_pred_analyzed, ['v_fees_usd', 'v_gas_usd']) and can_plot(df_base_analyzed, ['v_fees_usd', 'v_gas_usd']):
    fig15, (ax_pred, ax_base) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax_pred.bar(df_pred_analyzed['timestamp'], df_pred_analyzed['v_fees_usd'], color='g', alpha=0.7, width=0.05, label='Predictive Fees')
    ax_pred.bar(df_pred_analyzed['timestamp'], -df_pred_analyzed['v_gas_usd'], color='r', alpha=0.7, width=0.05, label='Predictive Gas Cost')
    ax_pred.set_title('Predictive Strategy: Fees Earned vs. Gas Costs per Period', weight='bold'); ax_pred.legend()
    ax_base.bar(df_base_analyzed['timestamp'], df_base_analyzed['v_fees_usd'], color='c', alpha=0.7, width=0.05, label='Baseline Fees')
    ax_base.bar(df_base_analyzed['timestamp'], -df_base_analyzed['v_gas_usd'], color='m', alpha=0.7, width=0.05, label='Baseline Gas Cost')
    ax_base.set_title('Baseline Strategy: Fees Earned vs. Gas Costs per Period', weight='bold'); ax_base.legend()
    ax_base.xaxis.set_major_formatter(date_formatter); plt.xticks(rotation=45); fig15.tight_layout()
    plt.savefig(output_plots_dir / "15_KEY_fees_vs_gas.png", dpi=300); plt.close(fig15)
    print("Plot saved: 15_KEY_fees_vs_gas.png")

# KEY PLOT 16: Total Fees Comparison
fee_comparison_data = [
    {'Strategy': 'Predictive', 'Total Fees (USD)': results_pred.get('total_fees_usd', 0)},
    {'Strategy': 'Baseline', 'Total Fees (USD)': results_base.get('total_fees_usd', 0)}
]
df_fees = pd.DataFrame(fee_comparison_data)
if not df_fees.empty:
    fig16, ax16 = plt.subplots(figsize=(8, 6))
    sns.barplot(x='Strategy', y='Total Fees (USD)', data=df_fees, ax=ax16, hue='Strategy', palette={'Predictive': 'seagreen', 'Baseline': 'coral'}, legend=False)
    ax16.set_title('Comparison of Total Fees Earned', weight='bold'); ax16.set_xlabel('Strategy'); ax16.set_ylabel('Total Fees (USD)')
    for p_patch in ax16.patches:
        ax16.annotate(f"${p_patch.get_height():,.4f}", (p_patch.get_x() + p_patch.get_width() / 2., p_patch.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    fig16.tight_layout(); plt.savefig(output_plots_dir / "16_KEY_total_fees_comparison.png", dpi=300); plt.close(fig16)
    print("Plot saved: 16_KEY_total_fees_comparison.png")
# --- PLOT 17: CUMULATIVE FEES EARNED (LINE CHART) ---
# This plot shows how fee income accumulates over time for each strategy.
# It provides a clearer story of income growth compared to a simple bar chart of the total.

# First, ensure the cumulative fees column exists or calculate it.
if 'v_fees_usd' in df_pred_analyzed.columns:
    df_pred_analyzed['cumulative_fees_usd'] = df_pred_analyzed['v_fees_usd'].cumsum()
if 'v_fees_usd' in df_base_analyzed.columns:
    df_base_analyzed['cumulative_fees_usd'] = df_base_analyzed['v_fees_usd'].cumsum()

# Check if data is available for plotting
# Check if data is available for plotting
if can_plot(df_pred_analyzed, ['cumulative_fees_usd']) and can_plot(df_base_analyzed, ['cumulative_fees_usd']):
    
    fig17, ax17 = plt.subplots(figsize=plot_figsize)
    
    # Plotting the cumulative fees for both strategies
    sns.lineplot(data=df_pred_analyzed, x='timestamp', y='cumulative_fees_usd', 
                 label='Predictive Strategy Fees', marker='.', color='darkgreen', ax=ax17)
    
    sns.lineplot(data=df_base_analyzed, x='timestamp', y='cumulative_fees_usd', 
                 label='Baseline Strategy Fees', marker='.', color='crimson', ax=ax17)

    # --- Formatting the plot for clarity ---
    ax17.set_title('KEY PLOT: Cumulative Fees Earned Over Time', weight='bold')
    ax17.set_xlabel('Time')
    ax17.set_ylabel('Total Fees Earned (USD)')
    
    # Using the formatters defined at the start of your script
    ax17.yaxis.set_major_formatter(thousands_formatter)
    ax17.xaxis.set_major_formatter(date_formatter)
    
    ax17.legend()
    ax17.tick_params(axis='x', rotation=30)
    fig17.tight_layout()
    
    # --- Saving the plot ---
    plot_filename = "17_KEY_cumulative_fees_earned.png"
    plt.savefig(output_plots_dir / plot_filename, dpi=300)
    plt.close(fig17)
    
    print(f"Plot saved: {plot_filename}")
else:
    print("Skipping Plot 17: Data for cumulative fees is not available.")

# --- PLOT 18: KEY PLOT - CUMULATIVE P&L vs HODL --- 
# This plot is much better at visualizing the performance difference because it 
# isolates the strategy's "alpha" or outperformance against the HODL baseline. 
# It starts at 0 and clearly shows the growing gap between the strategies. 

# Calculate the cumulative sum of the P&L vs HODL metric 
if 'pnl_vs_hodl_usd' in df_pred_analyzed.columns: 
    df_pred_analyzed['cumulative_pnl_vs_hodl'] = df_pred_analyzed['pnl_vs_hodl_usd'].cumsum() 
if 'pnl_vs_hodl_usd' in df_base_analyzed.columns: 
    df_base_analyzed['cumulative_pnl_vs_hodl'] = df_base_analyzed['pnl_vs_hodl_usd'].cumsum() 

# Check if data is available for plotting 
if can_plot(df_pred_analyzed, ['cumulative_pnl_vs_hodl']) and can_plot(df_base_analyzed, ['cumulative_pnl_vs_hodl']): 
    
    fig18, ax18 = plt.subplots(figsize=plot_figsize) 
    
    # Plotting the cumulative P&L vs HODL for both strategies 
    sns.lineplot(data=df_pred_analyzed, x='timestamp', y='cumulative_pnl_vs_hodl', 
                 label='Predictive Strategy (vs HODL)', marker='.', color='blue', linewidth=2.5, ax=ax18) 
    
    sns.lineplot(data=df_base_analyzed, x='timestamp', y='cumulative_pnl_vs_hodl', 
                 label='Baseline Strategy (vs HODL)', marker='.', color='orange', linewidth=2.5, ax=ax18) 

    # --- Formatting the plot --- 
    ax18.set_title('KEY PLOT: Cumulative Net Profit vs. HODL Benchmark', weight='bold') 
    ax18.set_xlabel('Time') 
    ax18.set_ylabel('Cumulative Outperformance (USD)') 
    
    # Add a zero line for reference 
    ax18.axhline(0, color='grey', linestyle='--') 
    
    ax18.yaxis.set_major_formatter(thousands_formatter) 
    ax18.xaxis.set_major_formatter(date_formatter) 
    
    ax18.legend() 
    ax18.tick_params(axis='x', rotation=30) 
    fig18.tight_layout() 
    
    # --- Saving the plot --- 
    plot_filename = "18_KEY_cumulative_pnl_vs_hodl.png" 
    plt.savefig(output_plots_dir / plot_filename, dpi=300) 
    plt.close(fig18) 
    
    print(f"Plot saved: {plot_filename}") 
else: 
    print("Skipping Plot 18: Data for cumulative P&L vs HODL is not available.")

# --- Save Summary Table Image ---
fig_table, ax_table = plt.subplots(figsize=(16, 8))
ax_table.set_title("Results Summary Table", weight='bold', size=14, y=1.08)
ax_table.axis('tight'); ax_table.axis('off')
the_table = ax_table.table(cellText=summary_df.values, colLabels=summary_df.columns,
                           colWidths=[0.35] + [0.15]*(len(summary_df.columns)-1),
                           cellLoc='left', loc='center', colColours=["#E8E8E8"]*len(summary_df.columns))
the_table.auto_set_font_size(False); the_table.set_fontsize(9); the_table.scale(1, 2)
plt.savefig(output_plots_dir / "summary_table.png", dpi=300, bbox_inches='tight')
plt.close(fig_table)
print(f"Summary table image saved as {output_plots_dir / 'summary_table.png'}")

print("\nAnalysis script finished.")