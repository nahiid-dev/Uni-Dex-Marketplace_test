import csv
from decimal import Decimal, getcontext
import math

# Set higher precision for decimal calculations
getcontext().prec = 50

def tick_to_price(tick):
    """Convert a tick value to a human readable price."""
    if tick is None or tick == "" or tick == "0":
        return None
    try:
        tick = float(tick)
        # Using the proper Uniswap V3 formula: price = 1.0001^tick
        price = Decimal('1.0001') ** Decimal(str(tick))
        return float(price)
    except:
        return None

def format_decimal(val, decimals=18):
    """Convert wei/raw values to human readable decimals."""
    if val is None or val == "" or val == "0":
        return "0"
    try:
        val_decimal = Decimal(str(val))
        human_readable = val_decimal / Decimal(10 ** decimals)
        return str(human_readable)
    except:
        return str(val)

input_file = "position_results_predictive.csv"
output_file = "position_results_predictive_numeric.csv"

# Fields that should be converted to numeric (based on your CSV structure)
numeric_fields = [
    # Price fields (float)
    'predictedPrice_api', 'actualPrice_pool', 'gas_cost_eth',
    # Tick fields (int)
    'predictedTick_calculated', 'currentTick_pool',
    'targetTickLower_calculated', 'targetTickUpper_calculated',
    'finalTickLower_contract', 'finalTickUpper_contract',
    # Balance and token amount fields (Decimals - keep full precision)
    'initial_contract_balance_token0', 'initial_contract_balance_token1',
    'amount0_provided_to_mint', 'amount1_provided_to_mint',
    'fees_collected_token0', 'fees_collected_token1',
    # Other numeric fields
    'sqrtPriceX96_pool', 'liquidity_contract', 'gas_used',
    'range_width_multiplier_setting'
]

# Categorize fields by their type for proper conversion
tick_fields = {
    'predictedTick_calculated', 'currentTick_pool',
    'targetTickLower_calculated', 'targetTickUpper_calculated',
    'finalTickLower_contract', 'finalTickUpper_contract'
}

balance_fields = {
    'initial_contract_balance_token0', 'initial_contract_balance_token1',
    'amount0_provided_to_mint', 'amount1_provided_to_mint',
    'fees_collected_token0', 'fees_collected_token1'
}

def parse_numeric(val, field_name=None):
    """Convert string to appropriate numeric type based on field type."""
    if val is None or val == "":
        return None, None
    try:
        # For tick fields - convert to both integer and price
        if field_name in tick_fields:
            tick_val = int(float(val))
            return tick_val, tick_to_price(tick_val)
        
        # For balance fields - use Decimal to maintain precision
        if field_name in balance_fields:
            from decimal import Decimal
            raw_val = str(Decimal(val))
            return raw_val, format_decimal(raw_val)
            
        # For other fields - try integer first, then float
        if "." not in val and "e" not in val.lower() and "E" not in val:
            try:
                num_val = int(val)
                return num_val, str(num_val)
            except ValueError:
                pass
        num_val = float(val)
        return num_val, str(num_val)
    except ValueError:
        if val.lower() == "true":
            return 1, "1"
        if val.lower() == "false":
            return 0, "0"
        return None, None

def get_token_decimals(column_name):
    """Determine the number of decimals based on the column name."""
    column_lower = column_name.lower()
    if any(token in column_lower for token in ['usdc', 'token0']):
        return 6
    elif any(token in column_lower for token in ['eth', 'weth', 'token1']):
        return 18
    return 0  # For non-token values

def should_convert_ticks(column_name):
    """Determine if a column contains tick values that need price conversion."""
    column_lower = column_name.lower()
    return 'tick' in column_lower

def format_output_value(val):
    """Format numeric values for CSV output."""
    if val is None:
        return ""
    if isinstance(val, (int, float)):
        if val == 0:
            return "0"
        elif abs(val) < 0.000001:
            return f"{val:.12e}"
        else:
            return f"{val:.8f}".rstrip('0').rstrip('.')
    return str(val)

with open(input_file, 'r', encoding="utf-8") as infile, \
     open(output_file, 'w', newline='', encoding="utf-8") as outfile:
    reader = csv.DictReader(infile)
    
    # Create new fieldnames with human-readable columns
    original_fields = reader.fieldnames
    new_fields = []
    for field in original_fields:
        new_fields.append(field)  # Original column
        new_fields.append(f"{field}_human_readable")  # Human readable column
    
    writer = csv.DictWriter(outfile, fieldnames=new_fields)
    writer.writeheader()
    
    for row in reader:
        new_row = {}
        for field in original_fields:
            value = row[field]
            new_row[field] = value  # Keep original value
            
            # Skip non-numeric fields or empty values
            if not value or value.lower() in ('', 'null', 'none', 'nan'):
                new_row[f"{field}_human_readable"] = value
                continue
                
            # Convert to human readable format based on field type
            if should_convert_ticks(field):
                # Convert tick to price
                price = tick_to_price(value)
                new_row[f"{field}_human_readable"] = format_output_value(price)
            else:
                # Try to convert as token amount first
                decimals = get_token_decimals(field)
                if decimals > 0:
                    human_readable = format_decimal(value, decimals)
                    new_row[f"{field}_human_readable"] = format_output_value(float(human_readable))
                else:
                    # Keep original value for non-token numeric fields
                    new_row[f"{field}_human_readable"] = value
        
        writer.writerow(new_row)

print(f"Converted file saved as {output_file}")
