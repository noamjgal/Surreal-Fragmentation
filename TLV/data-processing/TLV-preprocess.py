import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, time as datetime_time, timedelta
from tqdm import tqdm

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds to run.")
        return result
    return wrapper

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
        return pd.Timestamp.combine(row['date'].date(), time_obj).floor('S')  # Drop milliseconds
    except Exception as e:
        print(f"Error combining date {row['date']} and time {row['time']}: {str(e)}")
        return pd.NaT

@timer
def preprocess_and_clean_data(input_path, output_dir):
    print("Loading and preprocessing data...")
    columns_to_read = ['user', 'date', 'time', 'speed', 'indoors', 'isapp', 'type', 'Travel_mode', 'long', 'lat', 'school_n', 'sex', 'LU_na']
    
    try:
        df = pd.read_excel(input_path, sheet_name='gpsappS_8', usecols=columns_to_read)
    except Exception as e:
        print(f"Error reading Excel file: {str(e)}")
        return None

    print("Initial data structure:")
    print(df.dtypes)
    print(df.head())
    print(f"Initial shape: {df.shape}")

    # Convert 'date' to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    # Combine date and time to create Timestamp
    df['Timestamp'] = df.apply(combine_date_time, axis=1)

    # Clean data
    df = df.replace(['Missing', 'NA', ''], np.nan)
    df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
    df['indoors'] = df['indoors'].map({'True': True, 'False': False, True: True, False: False})
    
    # Create new boolean columns
    df['is_digital'] = df['type'].isin(['Social', 'Productive', 'Process', 'Settings', 'School', 'Spatial', 'Screen on/off/lock'])
    df['is_moving'] = df['Travel_mode'].isin(['AT', 'PT', 'Walking'])
    df['is_home'] = df['LU_na'].isin(['Home', '~Home'])

    # Drop rows with NaT timestamps
    df_cleaned = df.dropna(subset=['Timestamp'])
    print(f"Shape after dropping NaT Timestamps: {df_cleaned.shape}")

    if df_cleaned.empty:
        print("Warning: All rows were dropped. Check the Timestamp creation process.")
        return None

    # Create output directories
    preprocessed_dir = os.path.join(output_dir, 'preprocessed_data')
    os.makedirs(preprocessed_dir, exist_ok=True)

    # Create participant info CSV
    participant_info = df_cleaned[['user', 'school_n', 'sex']].drop_duplicates()
    participant_info_path = os.path.join(output_dir, 'participant_info.csv')
    participant_info.to_csv(participant_info_path, index=False)
    print(f"\nParticipant info saved to {participant_info_path}")
    print("Sample of participant info:")
    print(participant_info.head())

    # Split data by date and user
    columns_to_save = ['user', 'Timestamp', 'speed', 'indoors', 'is_digital', 'is_moving', 'is_home', 'long', 'lat']
    problematic_days = []
    for (date, user), group in df_cleaned.groupby([df_cleaned['Timestamp'].dt.date, 'user']):
        filename = f"{date}_{user}.csv"
        filepath = os.path.join(preprocessed_dir, filename)
        
        # Sort by Timestamp
        group_sorted = group.sort_values('Timestamp')
        
        # Check for day length
        day_length = group_sorted['Timestamp'].max() - group_sorted['Timestamp'].min()
        if day_length > timedelta(hours=24):
            problematic_days.append((date, user, day_length))
            print(f"Warning: Day length exceeds 24 hours for user {user} on {date}. Length: {day_length}")
        
        # Select and reorder columns
        group_sorted[columns_to_save].to_csv(filepath, index=False)
        
    print("\nData preprocessing and cleaning completed.")
    print(f"Final processed data shape: {df_cleaned.shape}")

    # Print summary of missing values
    missing_summary = df_cleaned[columns_to_save].isnull().sum()
    print("\nMissing values summary:")
    print(missing_summary)
    
    # Print summary of problematic days
    if problematic_days:
        print("\nProblematic days (exceeding 24 hours):")
        for date, user, length in problematic_days:
            print(f"User {user}, Date {date}: {length}")
    else:
        print("\nNo days exceeding 24 hours found.")
    
    return preprocessed_dir

def main():
    input_path = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/Survey/gpsappS_9.1_excel.xlsx'
    output_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/'
    preprocessed_dir = preprocess_and_clean_data(input_path, output_dir)
    
    # Additional step to check cleaned files
    if preprocessed_dir:
        print("\nChecking cleaned files...")
        total_non_empty = 0
        total_empty = 0
        for file_path in tqdm(os.listdir(preprocessed_dir)):
            if file_path.endswith('.csv'):
                df = pd.read_csv(os.path.join(preprocessed_dir, file_path))
                non_empty = df.notna().sum().sum()
                empty = df.isna().sum().sum()
                total_non_empty += non_empty
                total_empty += empty
        
        print("\nOverall Summary:")
        print(f"Total non-empty cells: {total_non_empty}")
        print(f"Total empty cells: {total_empty}")
        print(f"Total cells: {total_non_empty + total_empty}")
        print(f"Percentage of empty cells: {(total_empty / (total_non_empty + total_empty)) * 100:.2f}%")

if __name__ == "__main__":
    main()
