import pandas as pd
import os

def combine_csv_files(input_folder, output_file):
    all_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.csv')]
    all_df = [pd.read_csv(f) for f in all_files]
    combined_df = pd.concat(all_df, ignore_index=True)
    combined_df.to_csv(output_file, index=False)
    print(f"All files combined into {output_file}.")

input_folder = 'binance_data_chunks'
output_file = 'binance_data_combined.csv'
combine_csv_files(input_folder, output_file)
