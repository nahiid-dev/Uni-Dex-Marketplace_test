import csv

input_file = "position_results_predictive.csv"
output_file = "position_results_predictive_numeric.csv"

# Fields that should be converted to numeric (based on your CSV structure)
numeric_fields = [
    'predictedPrice_api', 'predictedTick_calculated', 'actualPrice_pool', 'sqrtPriceX96_pool',
    'currentTick_pool', 'targetTickLower_calculated', 'targetTickUpper_calculated',
    'finalTickLower_contract', 'finalTickUpper_contract', 'liquidity_contract', 'gas_used', 'gas_cost_eth'
]

def parse_numeric(val):
    """Convert string to int/float if possible, else return None."""
    if val is None or val == "":
        return None
    try:
        # Try integer
        if "." not in val and "e" not in val.lower() and "E" not in val:
            return int(val)
        # Try float (handles scientific notation)
        return float(val)
    except ValueError:
        # Handle boolean-like
        if val.lower() == "true":
            return 1
        if val.lower() == "false":
            return 0
        return None

with open(input_file, newline='', encoding="utf-8") as fin, \
     open(output_file, "w", newline='', encoding="utf-8") as fout:
    reader = csv.DictReader(fin)
    fieldnames = reader.fieldnames[:]
    # Add new numeric columns next to each numeric field
    new_fieldnames = []
    for fn in fieldnames:
        new_fieldnames.append(fn)
        if fn in numeric_fields:
            new_fieldnames.append(fn + "_numeric")
    writer = csv.DictWriter(fout, fieldnames=new_fieldnames)
    writer.writeheader()
    for row in reader:
        new_row = {}
        for fn in fieldnames:
            new_row[fn] = row[fn]
            if fn in numeric_fields:
                new_row[fn + "_numeric"] = parse_numeric(row[fn])
        writer.writerow(new_row)

print(f"Numeric results written to {output_file}")
