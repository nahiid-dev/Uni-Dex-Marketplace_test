import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns  
import numpy as np
from pathlib import Path
import sys # To stop the program in case of an error

# --- 1. Data Loading (Modified with Parent Directory) ---

PREDICTIVE_FILE_NAME = "position_results_predictive.csv"
BASELINE_FILE_NAME = "position_results_baseline.csv"

try:
    # Get the full path of the currently running script
    script_path = Path(__file__).resolve()
    # Get the directory where the script is located (e.g., 'analysis')
    script_dir = script_path.parent
    # !! Important !! Go to the parent directory (one level up) to find CSVs
    csv_dir = script_dir.parent

except NameError:
    # If __file__ is not defined (e.g., in Jupyter or interactive console),
    # use the current working directory as the script directory
    # and its parent as the CSV directory.
    # !! Note !! In this case, ensure you are running from the 'analysis' folder.
    print("Warning: __file__ variable not found. Using the current working directory.")
    script_dir = Path('.').resolve()
    csv_dir = script_dir.parent # Go up one level from the current dir
    print(f"Assumed script path: {script_dir}")


# Construct the full path to the CSV files using the parent directory path
predictive_path = csv_dir / PREDICTIVE_FILE_NAME
baseline_path = csv_dir / BASELINE_FILE_NAME

print(f"Script path: {script_dir}")
print(f"CSV search path: {csv_dir}")
print(f"Attempting to read file: {predictive_path}")
print(f"Attempting to read file: {baseline_path}")

# Use try-except to handle potential FileNotFoundError
try:
    # Read the CSV files directly using the full path
    df_pred = pd.read_csv(predictive_path)
    df_base = pd.read_csv(baseline_path)
    print("✅ Files loaded successfully.")

except FileNotFoundError:
    print("❌ Error: One or both CSV files were not found.")
    print(f"   Please ensure that the file '{PREDICTIVE_FILE_NAME}' exists at path '{predictive_path}'.")
    print(f"   And that the file '{BASELINE_FILE_NAME}' exists at path '{baseline_path}'.")
    sys.exit(1) # Stop execution with an error code
except Exception as e:
    print(f"❌ An unexpected error occurred while reading the files: {e}")
    sys.exit(1) # Stop execution with an error code

# --- 2. Data Preprocessing & Style ---
# ... Rest of your code continues here ...
def preprocess(df):
    """Preprocesses the DataFrame: sets datetime, converts numeric, handles NaNs."""
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    numeric_cols = ['predictedPrice_api', 'actualPrice_pool', 'currentTick_pool',
                    'finalTickLower_contract', 'finalTickUpper_contract',
                    'gas_cost_eth', 'liquidity_contract', 'finalLiquidity_contract']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'liquidity_contract' not in df.columns and 'finalLiquidity_contract' in df.columns:
        df['liquidity_contract'] = df['finalLiquidity_contract']
    df = df.fillna(0)
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

df_pred = preprocess(df_pred)
df_base = preprocess(df_base)


# --- 2. Data Preprocessing & Style ---
def preprocess(df):
    """Preprocesses the DataFrame: sets datetime, converts numeric, handles NaNs."""
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    numeric_cols = ['predictedPrice_api', 'actualPrice_pool', 'currentTick_pool',
                    'finalTickLower_contract', 'finalTickUpper_contract',
                    'gas_cost_eth', 'liquidity_contract', 'finalLiquidity_contract']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'liquidity_contract' not in df.columns and 'finalLiquidity_contract' in df.columns:
        df['liquidity_contract'] = df['finalLiquidity_contract']
    df = df.fillna(0)
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

df_pred = preprocess(df_pred)
df_base = preprocess(df_base)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# --- 3. Helper & Metric Functions ---
def tick_to_price(tick):
    """Converts a Uniswap V3 tick to price."""
    return 1.0001 ** tick

def calculate_errors(y_true, y_pred):
    """Calculates errors, MAE, and MAPE."""
    errors = y_true - y_pred
    mae = np.mean(np.abs(errors))
    mape = np.mean(np.abs(errors / y_true)) * 100
    return errors, mae, mape

def analyze_data(df, contract_type):
    """Analyzes the DataFrame to calculate key performance metrics."""
    results = {}
    if 'finalTickLower_contract' in df.columns and 'finalTickUpper_contract' in df.columns and \
       not df['finalTickLower_contract'].eq(0).all() and not df['finalTickUpper_contract'].eq(0).all():
        df['is_in_range'] = (df['currentTick_pool'] >= df['finalTickLower_contract']) & \
                           (df['currentTick_pool'] <= df['finalTickUpper_contract'])
        time_in_range_pct = df['is_in_range'].mean() * 100
    else:
        df['is_in_range'] = False
        time_in_range_pct = 0
    results['time_in_range_pct'] = time_in_range_pct

    if contract_type == 'Predictive':
        errors, mae, mape = calculate_errors(df['actualPrice_pool'], df['predictedPrice_api'])
        df['prediction_error_pct'] = (errors / df['actualPrice_pool']) * 100
        results['mae'] = mae
        results['mape'] = mape

    results['total_gas_cost_eth'] = df['gas_cost_eth'].sum()
    df['cumulative_gas_eth'] = df['gas_cost_eth'].cumsum()
    df['rolling_tir'] = df['is_in_range'].rolling(window=5, min_periods=1).mean() * 100
    return results, df

results_pred, df_pred = analyze_data(df_pred, 'Predictive')
results_base, df_base = analyze_data(df_base, 'Baseline')

# --- 4. Create Summary DataFrame ---
summary_data = {
    'Metric': [
        'Time In Range (%)',
        'Total Gas Cost (ETH)',
        'Mean Absolute Error (MAE)',
        'Mean Absolute Percentage Error (MAPE %)'
    ],
    'Predictive Strategy': [
        f"{results_pred.get('time_in_range_pct', 0):.2f}",
        f"{results_pred.get('total_gas_cost_eth', 0):.6f}",
        f"{results_pred.get('mae', 'N/A'):.2f}",
        f"{results_pred.get('mape', 'N/A'):.2f}"
    ],
    'Baseline Strategy': [
        f"{results_base.get('time_in_range_pct', 0):.2f}",
        f"{results_base.get('total_gas_cost_eth', 0):.6f}",
        'N/A',
        'N/A'
    ]
}
summary_df = pd.DataFrame(summary_data)
print("--- Results Summary Table ---")
print(summary_df.to_string(index=False))
print("\nN/A = Not Applicable")
print("Warning: Ensure you are using your full dataset for final results.")


# --- 5. Plotting for Thesis ---
fig = plt.figure(figsize=(20, 18))
gs = fig.add_gridspec(3, 2)

ax1 = fig.add_subplot(gs[0, 0])
# ... (Plotting code for ax1 to ax6 remains the same as previous version) ...
sns.lineplot(data=df_pred, x='timestamp', y='actualPrice_pool', label='Actual Price', marker='o', ax=ax1, color='b')
sns.lineplot(data=df_pred, x='timestamp', y='predictedPrice_api', label='Predicted Price', marker='x', linestyle='--', ax=ax1, color='g')
for i, row in df_pred.iterrows():
    lower_price = tick_to_price(row['finalTickLower_contract'])
    upper_price = tick_to_price(row['finalTickUpper_contract'])
    ax1.fill_between(df_pred['timestamp'], lower_price, upper_price,
                     where=(df_pred['timestamp'] >= row['timestamp']),
                     alpha=0.15, color='g', label='Predictive Range' if i == 0 else "")
ax1.set_title('Predictive Strategy: Price & Liquidity Range', weight='bold')
ax1.set_xlabel('Time'); ax1.set_ylabel('ETH Price (USD)')
ax1.legend(); ax1.tick_params(axis='x', rotation=30)

ax2 = fig.add_subplot(gs[0, 1])
sns.lineplot(data=df_base, x='timestamp', y='actualPrice_pool', label='Actual Price', marker='o', ax=ax2, color='b')
for i, row in df_base.iterrows():
    lower_price = tick_to_price(row['finalTickLower_contract'])
    upper_price = tick_to_price(row['finalTickUpper_contract'])
    ax2.fill_between(df_base['timestamp'], lower_price, upper_price,
                     where=(df_base['timestamp'] >= row['timestamp']),
                     alpha=0.2, color='orange', label='Baseline Range' if i == 0 else "")
ax2.set_title('Baseline Strategy: Price & Liquidity Range', weight='bold')
ax2.set_xlabel('Time'); ax2.set_ylabel('ETH Price (USD)')
ax2.legend(); ax2.tick_params(axis='x', rotation=30)

ax3 = fig.add_subplot(gs[1, 0])
sns.lineplot(data=df_pred, x='timestamp', y='prediction_error_pct', ax=ax3, color='r', label='Instantaneous Error (%)')
ax3.axhline(0, color='grey', linestyle='--')
ax3.set_title(f"Prediction Error Over Time (MAPE: {results_pred.get('mape', 0):.2f}%)", weight='bold')
ax3.set_xlabel('Time'); ax3.set_ylabel('Prediction Error (%)')
ax3.tick_params(axis='x', rotation=30)
ax3_hist = ax3.inset_axes([0.65, 0.65, 0.3, 0.3])
sns.histplot(df_pred['prediction_error_pct'], bins=10, kde=False, ax=ax3_hist, color='r', alpha=0.6)
ax3_hist.set_title('Error Distribution', fontsize=9)
ax3_hist.set_xlabel(''); ax3_hist.set_ylabel('')

ax4 = fig.add_subplot(gs[1, 1])
sns.scatterplot(data=df_pred, x='actualPrice_pool', y='predictedPrice_api', ax=ax4, s=100, alpha=0.7, label='Predictions')
lims = [ np.min([ax4.get_xlim(), ax4.get_ylim()]), np.max([ax4.get_xlim(), ax4.get_ylim()]), ]
ax4.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='Ideal Prediction')
ax4.set_title('Actual Price vs. Predicted Price', weight='bold')
ax4.set_xlabel('Actual Price (USD)'); ax4.set_ylabel('Predicted Price (USD)')
ax4.legend(); ax4.set_aspect('equal', adjustable='box')

ax5 = fig.add_subplot(gs[2, 0])
sns.lineplot(data=df_pred, x='timestamp', y='rolling_tir', label='Predictive', marker='.', ax=ax5)
sns.lineplot(data=df_base, x='timestamp', y='rolling_tir', label='Baseline', marker='.', ax=ax5)
ax5.axhline(results_pred.get('time_in_range_pct', 0), color='blue', linestyle=':', label=f"Predictive Avg ({results_pred.get('time_in_range_pct', 0):.1f}%)")
ax5.axhline(results_base.get('time_in_range_pct', 0), color='orange', linestyle=':', label=f"Baseline Avg ({results_base.get('time_in_range_pct', 0):.1f}%)")
ax5.set_title('Time In Range (5-Period Rolling Average)', weight='bold')
ax5.set_xlabel('Time'); ax5.set_ylabel('Time In Range (%)')
ax5.set_ylim(0, 105); ax5.legend(); ax5.tick_params(axis='x', rotation=30)

ax6 = fig.add_subplot(gs[2, 1])
sns.lineplot(data=df_pred, x='timestamp', y='cumulative_gas_eth', label='Predictive', marker='.', ax=ax6)
sns.lineplot(data=df_base, x='timestamp', y='cumulative_gas_eth', label='Baseline', marker='.', ax=ax6)
ax6.set_title('Cumulative Gas Costs', weight='bold')
ax6.set_xlabel('Time'); ax6.set_ylabel('Total Gas Cost (ETH)')
ax6.legend(); ax6.tick_params(axis='x', rotation=30)

fig.suptitle('Comparative Analysis of Uniswap V3 AMM Strategies', fontsize=24, y=1.01, weight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig("thesis_full_results_english.png", dpi=300, bbox_inches='tight')
print("\nMain plots saved as thesis_full_results_english.png")

# --- 6. Create and Save Summary Table Image ---
fig_table, ax_table = plt.subplots(figsize=(8, 2.5)) # Adjust figsize as needed
ax_table.set_title("Results Summary Table", weight='bold', size=14, y=1.15) # Add title
ax_table.axis('tight')
ax_table.axis('off')

# Create the table - set 'colWidths' for better spacing
the_table = ax_table.table(cellText=summary_df.values,
                           colLabels=summary_df.columns,
                           colWidths=[0.4, 0.3, 0.3], # Adjust these widths
                           cellLoc='center',
                           loc='center',
                           colColours=["#E8E8E8", "#E8E8E8", "#E8E8E8"]) # Header color

the_table.auto_set_font_size(False)
the_table.set_fontsize(11)
the_table.scale(1.1, 1.4) # Adjust scale (width, height)

# Make header text bold
for (row, col), cell in the_table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold')

plt.savefig("thesis_summary_table.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
print("Summary table saved as thesis_summary_table.png")

# --- 7. Show Main Plots ---
plt.show() # Display the plots (will show both figures)