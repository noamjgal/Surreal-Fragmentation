import pandas as pd
import numpy as np
import os
import time

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
    
    # Convert 'date' to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Convert 'time' to time, handling potential errors
    def parse_time(t):
        if pd.isna(t):
            return pd.NaT
        try:
            # Parse time and truncate to seconds
            return pd.to_datetime(t).floor('S').time()
        except:
            return pd.NaT

    df['time'] = df['time'].apply(parse_time)
    
    # Combine date and time, handling NaT values
    df['Timestamp'] = df.apply(lambda row: 
        pd.Timestamp.combine(row['date'], row['time']) if pd.notna(row['time']) else pd.NaT, 
        axis=1
    )
    
    # Drop rows with NaT timestamps
    df = df.dropna(subset=['Timestamp'])
    
    preprocessed_dir = os.path.join(output_dir, 'preprocessed_data')
    os.makedirs(preprocessed_dir, exist_ok=True)
    
    for (date, user), group in df.groupby([df['date'].dt.date, 'user']):
        filename = f"{date}_{user}.csv"
        filepath = os.path.join(preprocessed_dir, filename)
        group.to_csv(filepath, index=False)
        
        # Print head of each output file
        print(f"\nHead of {filename}:")
        print(group.head())
    
    print("\nData preprocessing and splitting completed.")
    print(f"Processed data shape: {df.shape}")

def main():
    input_path = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/gpsappS_9.1_excel.xlsx'
    output_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/'
    
    preprocess_and_split_data(input_path, output_dir)

if __name__ == "__main__":
    main()