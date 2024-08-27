import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, time as datetime_time

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds to run.")
        return result
    return wrapper

@timer
def preprocess_and_split_data(input_path, output_dir):
    print("Loading and preprocessing data...")
    
    columns_to_read = ['user', 'date', 'time', 'speed', 'indoors', 'isapp']
    
    try:
        df = pd.read_excel(input_path, sheet_name='gpsappS_8', usecols=columns_to_read)
    except Exception as e:
        print(f"Error reading Excel file: {str(e)}")
        return
    
    print("Initial data structure:")
    print(df.dtypes)
    print(df.head())
    print(f"Initial shape: {df.shape}")
    
    # Check data types and formats
    print("\nChecking data types and formats:")
    for column in df.columns:
        print(f"{column}:")
        print(f"  Data type: {df[column].dtype}")
        print(f"  Sample values: {df[column].head().tolist()}")
        if column in ['date', 'time']:
            print(f"  Unique formats: {df[column].astype(str).unique()[:5]}")  # Show first 5 unique formats
    
    # Convert 'date' to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Function to combine date and time
    def combine_date_time(row):
        if pd.isna(row['time']):
            return pd.NaT
        try:
            if isinstance(row['time'], str):
                time_obj = pd.to_datetime(row['time']).time()
            elif isinstance(row['time'], datetime):
                time_obj = row['time'].time()
            elif isinstance(row['time'], datetime_time):
                time_obj = row['time']
            else:
                raise ValueError(f"Unexpected time format: {type(row['time'])}")
            return pd.Timestamp.combine(row['date'].date(), time_obj)
        except Exception as e:
            print(f"Error combining date {row['date']} and time {row['time']}: {str(e)}")
            return pd.NaT
    
    # Combine date and time to create Timestamp
    df['Timestamp'] = df.apply(combine_date_time, axis=1)
    print(f"\nShape after creating Timestamp: {df.shape}")
    print("Sample of Timestamps:")
    print(df['Timestamp'].head())
    
    # Drop rows with NaT timestamps
    df_cleaned = df.dropna(subset=['Timestamp'])
    print(f"Shape after dropping NaT Timestamps: {df_cleaned.shape}")
    
    if df_cleaned.empty:
        print("Warning: All rows were dropped. Check the Timestamp creation process.")
        return
    
    preprocessed_dir = os.path.join(output_dir, 'preprocessed_data')
    os.makedirs(preprocessed_dir, exist_ok=True)
    
    for (date, user), group in df_cleaned.groupby([df_cleaned['date'].dt.date, 'user']):
        filename = f"{date}_{user}.csv"
        filepath = os.path.join(preprocessed_dir, filename)
        group.to_csv(filepath, index=False)
        
        # Print head of each output file
        print(f"\nHead of {filename}:")
        print(group.head())
    
    print("\nData preprocessing and splitting completed.")
    print(f"Final processed data shape: {df_cleaned.shape}")

def main():
    input_path = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/gpsappS_9.1_excel.xlsx'
    output_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/'
    
    preprocess_and_split_data(input_path, output_dir)

if __name__ == "__main__":
    main()