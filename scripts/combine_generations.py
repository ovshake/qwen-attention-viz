import os
import pandas as pd
import glob

# Get all CSV files in the generations directory
csv_files = glob.glob("outputs/generations/*.csv")

# Initialize list to store dataframes
dfs = []

# Process each CSV file
for file in csv_files:
    # Extract model name and prompt key from filename
    basename = os.path.basename(file)
    model_name, prompt_key = basename.rsplit('_', 1)
    prompt_key = prompt_key.replace('.csv', '')
    
    # Read the CSV
    df = pd.read_csv(file)
    
    # Add model_name and prompt_key columns
    df['model_name'] = model_name
    df['prompt_key'] = prompt_key
    
    dfs.append(df)

# Combine all dataframes
combined_df = pd.concat(dfs, ignore_index=True)

# Save combined dataframe
os.makedirs("outputs/combined", exist_ok=True)
combined_df.to_csv("outputs/combined/all_generations.tsv", index=False, sep="\t")
