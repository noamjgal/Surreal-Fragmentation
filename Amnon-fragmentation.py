import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from ruptures import Pelt
from scipy import stats
import time
import concurrent.futures
import multiprocessing
import traceback
import sys
from datetime import datetime, time as dt_time

MAX_ERRORS = 5
error_count = 0

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds to run.")
        return result
    return wrapper

def parse_time(t):
    if pd.isna(t):
        return pd.NaT
    try:
        # First, try to parse as a string
        return pd.to_datetime(t).time()
    except:
        try:
            # If that fails, check if it's already a time object
            if isinstance(t, dt_time):
                return t
            # If it's neither, print detailed error info
            print(f"Failed to parse time: {t}, Type: {type(t)}")
            return pd.NaT
        except:
            print(f"Unexpected error parsing time: {t}, Type: {type(t)}")
            return pd.NaT

@timer
def preprocess_and_split_data(input_path, output_dir):
    print("Loading and preprocessing data...")
    
    columns_to_read = ['user', 'date', 'time', 'speed', 'indoors', 'isapp']
    
    try:
        df = pd.read_excel(input_path, sheet_name='gpsappS_8', usecols=columns_to_read)
    except Exception as e:
        print(f"Error reading Excel file: {str(e)}")
        return None

    print("Initial data structure:")
    print(df.dtypes)
    print(df.head())
    print(f"Initial shape: {df.shape}")

    df['date'] = pd.to_datetime(df['date'])
    df['time'] = df['time'].apply(parse_time)
    
    # Print sample of parsed times
    print("Sample of parsed times:")
    print(df[['time']].head(10))
    
    # Combine date and time
    df['Timestamp'] = df.apply(lambda row: 
        pd.Timestamp.combine(row['date'], row['time']) if pd.notna(row['time']) else pd.NaT, 
        axis=1
    )
    
    # Print sample of timestamps
    print("Sample of timestamps:")
    print(df[['date', 'time', 'Timestamp']].head(10))
    
    # Drop rows with NaT timestamps
    df_before_drop = df.copy()
    df = df.dropna(subset=['Timestamp'])
    rows_dropped = len(df_before_drop) - len(df)
    print(f"Rows dropped due to invalid timestamps: {rows_dropped}")
    
    preprocessed_dir = os.path.join(output_dir, 'preprocessed_data')
    os.makedirs(preprocessed_dir, exist_ok=True)
    
    for (date, user), group in df.groupby([df['date'].dt.date, 'user']):
        filename = f"{date}_{user}.csv"
        filepath = os.path.join(preprocessed_dir, filename)
        group.to_csv(filepath, index=False)
    
    print("Data preprocessing and splitting completed.")
    print(f"Final processed data shape: {df.shape}")
    
    return df


def classify_movement(speed):
    if pd.isna(speed):
        return 'Unknown'
    elif speed < 1.5:
        return 'Stationary'
    elif speed < 7:
        return 'Active Transport'
    else:
        return 'Mechanized Transport'

def detect_changepoints(data, column, min_size=5, jump=5, pen=1):
    model = Pelt(model="rbf", jump=jump, min_size=min_size).fit(data[column].values.reshape(-1, 1))
    change_points = model.predict(pen=pen)
    return change_points

def create_episodes(df, change_points):
    episodes = []
    for i in range(len(change_points) - 1):
        start_idx = change_points[i]
        end_idx = change_points[i + 1]
        
        episode = df.iloc[start_idx:end_idx]
        
        episode_summary = {
            'start_time': episode['Timestamp'].iloc[0],
            'end_time': episode['Timestamp'].iloc[-1],
            'duration': (episode['Timestamp'].iloc[-1] - episode['Timestamp'].iloc[0]).total_seconds() / 60,
            'movement_type': stats.mode(episode['movement_type'])[0][0],
            'indoor_outdoor': stats.mode(episode['indoors'])[0][0],
            'digital_use': 'Yes' if (episode['isapp'] == 1).any() else 'No',
            'avg_speed': episode['speed'].mean()
        }
        
        episodes.append(episode_summary)
    
    return pd.DataFrame(episodes)

def calculate_fragmentation_index(episodes_df, column):
    total_duration = episodes_df['duration'].sum()
    episode_counts = episodes_df[column].value_counts()
    fragmentation_indices = {}
    
    for category in episode_counts.index:
        category_episodes = episodes_df[episodes_df[column] == category]
        S = len(category_episodes)
        T = category_episodes['duration'].sum()
        
        if S > 1:
            index = (1 - sum((category_episodes['duration'] / T) ** 2)) / (1 - (1 / S))
        else:
            index = 0
        
        fragmentation_indices[f"{category}_index"] = index
    
    return fragmentation_indices

def preprocess_data(df):
    df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
    df['movement_type'] = df['speed'].apply(classify_movement)
    df['isapp'] = df['isapp'].astype(int)
    return df

def analyze_participant_day(participant_df):
    global error_count
    try:
        if participant_df.empty:
            return None, None

        participant_df = preprocess_data(participant_df)
        
        if 'speed' not in participant_df.columns or participant_df.empty:
            return None, None

        change_points = detect_changepoints(participant_df, 'speed')
        if not change_points:
            return None, None

        episodes_df = create_episodes(participant_df, change_points)

        fragmentation_indices_movement = calculate_fragmentation_index(episodes_df, 'movement_type')
        fragmentation_indices_io = calculate_fragmentation_index(episodes_df, 'indoor_outdoor')
        fragmentation_indices_digital = calculate_fragmentation_index(episodes_df, 'digital_use')

        all_indices = {**fragmentation_indices_movement, **fragmentation_indices_io, **fragmentation_indices_digital}

        modes = {
            'movement_mode': stats.mode(participant_df['movement_type'])[0][0],
            'indoor_outdoor_mode': stats.mode(participant_df['indoors'])[0][0],
            'digital_use_mode': 'Yes' if stats.mode(participant_df['isapp'])[0][0] == 1 else 'No'
        }

        result = {
            'participant_id': participant_df['user'].iloc[0],
            'date': participant_df['date'].iloc[0],
            **all_indices,
            **modes
        }

        return result, episodes_df

    except Exception as e:
        error_count += 1
        if error_count <= MAX_ERRORS:
            print(f"Error processing participant {participant_df['user'].iloc[0]}: {str(e)}")
        if error_count == MAX_ERRORS:
            print("Maximum number of errors reached. Some participants may not be processed.")
        return None, None

def main():
    input_path = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/gpsappS_9.1_excel.xlsx'
    output_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/'
    episode_dir = os.path.join(output_dir, 'fragment-episodes')
    os.makedirs(episode_dir, exist_ok=True)

    df = preprocess_and_split_data(input_path, output_dir)
    if df is None or df.empty:
        print("Preprocessing failed or resulted in empty dataset. Exiting.")
        return

    all_results = []
    
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {num_workers} workers for parallel processing")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(analyze_participant_day, group) for _, group in df.groupby('user')]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    result, episodes_df = result
                    if result is not None and episodes_df is not None and not episodes_df.empty:
                        all_results.append(result)
                        episodes_df.to_csv(os.path.join(episode_dir, f"participant_{result['participant_id']}_{result['date']}_episodes.csv"), index=False)
            except Exception as e:
                print(f"Error processing future: {str(e)}")

    summary_df = pd.DataFrame(all_results)

    if not summary_df.empty:
        summary_df.to_csv(os.path.join(output_dir, 'fragmentation_summary.csv'), index=False)
        print("Summary saved to CSV.")

        print("\nGenerating descriptive statistics...")
        print(summary_df.describe())

        print("\nCreating visualizations...")
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=summary_df[[col for col in summary_df.columns if col.endswith('_index')]])
        plt.title('Distribution of Fragmentation Indices')
        plt.ylabel('Fragmentation Index')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fragmentation_indices_distribution.png'))
        plt.close()

        print("Visualization saved as 'fragmentation_indices_distribution.png' in the output directory.")
    else:
        print("No valid results were generated. Please check your data and error messages.")

if __name__ == "__main__":
    main()