import pandas as pd
import numpy as np
import os
from glob import glob

def clean_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Replace 'Missing', 'NA', and empty cells with NaN
    df = df.replace(['Missing', 'NA', ''], np.nan)
    
    # Convert 'speed' column to numeric, coercing errors to NaN
    df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
    
    # Save the cleaned data
    cleaned_file_path = file_path.replace('.csv', '_cleaned.csv')
    df.to_csv(cleaned_file_path, index=False)
    
    return df

def process_directory(directory):
    total_non_empty = 0
    total_empty = 0
    
    # Process all CSV files in the directory
    for file_path in glob(os.path.join(directory, '*.csv')):
        if '_cleaned.csv' in file_path:
            continue  # Skip already cleaned files
        
        print(f"Processing {file_path}...")
        df = clean_data(file_path)
        
        # Count non-empty and empty cells
        non_empty = df.notna().sum().sum()
        empty = df.isna().sum().sum()
        
        total_non_empty += non_empty
        total_empty += empty
        
        print(f"File: {os.path.basename(file_path)}")
        print(f"Non-empty cells: {non_empty}")
        print(f"Empty cells: {empty}")
        print("--------------------")
    
    print("\nOverall Summary:")
    print(f"Total non-empty cells: {total_non_empty}")
    print(f"Total empty cells: {total_empty}")
    print(f"Total cells: {total_non_empty + total_empty}")
    print(f"Percentage of empty cells: {(total_empty / (total_non_empty + total_empty)) * 100:.2f}%")

def main():
    directory = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/preprocessed_data'
    process_directory(directory)

if __name__ == "__main__":
    main()