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

def classify_movement(speed):
    if pd.isna(speed):
        return 'Unknown'
    elif speed < 1.5:
        return 'Stationary'
    else:
        return 'Mobile'

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
            'movement_type': stats.mode(episode['movement_type'], keepdims=False)[0],
            'indoor_outdoor': stats.mode(episode['indoors'], keepdims=False)[0],
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

def calculate_average_interepisode_duration(episodes_df, column):
    episodes_df = episodes_df.sort_values('start_time')
    interepisode_durations = {}
    
    for category in episodes_df[column].unique():
        category_episodes = episodes_df[episodes_df[column] == category]
        if len(category_episodes) > 1:
            durations = category_episodes['start_time'].diff().dropna().dt.total_seconds() / 60
            interepisode_durations[f"{category}_AID"] = durations.mean()
        else:
            interepisode_durations[f"{category}_AID"] = np.nan
    
    return interepisode_durations

def parse_time(t):
    if pd.isna(t):
        return pd.NaT
    try:
        return pd.to_datetime(t).floor('s').time()
    except:
        return pd.NaT

def preprocess_data(df):
    df['time'] = df['time'].apply(parse_time)
    df = df.dropna(subset=['time'])
    df['Timestamp'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
    df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
    df['movement_type'] = df['speed'].apply(classify_movement)
    df['isapp'] = df['isapp'].astype(int)
    return df

def analyze_participant_day(file_path):
    global error_count
    try:
        participant_df = pd.read_csv(file_path)
        if participant_df.empty:
            return None

        participant_df = preprocess_data(participant_df)
        
        if 'speed' not in participant_df.columns or participant_df.empty:
            return None

        change_points = detect_changepoints(participant_df, 'speed')
        if not change_points:
            return None

        episodes_df = create_episodes(participant_df, change_points)

        fragmentation_indices_movement = calculate_fragmentation_index(episodes_df, 'movement_type')
        fragmentation_indices_io = calculate_fragmentation_index(episodes_df, 'indoor_outdoor')
        fragmentation_indices_digital = calculate_fragmentation_index(episodes_df, 'digital_use')

        average_interepisode_durations = calculate_average_interepisode_duration(episodes_df, 'movement_type')

        result = {
            'participant_id': participant_df['user'].iloc[0],
            'date': participant_df['date'].iloc[0],
            'total_episodes': len(episodes_df),
            'stationary_episodes': len(episodes_df[episodes_df['movement_type'] == 'Stationary']),
            'mobile_episodes': len(episodes_df[episodes_df['movement_type'] == 'Mobile']),
            'total_duration': episodes_df['duration'].sum(),
            'stationary_duration': episodes_df[episodes_df['movement_type'] == 'Stationary']['duration'].sum(),
            'mobile_duration': episodes_df[episodes_df['movement_type'] == 'Mobile']['duration'].sum(),
            **fragmentation_indices_movement,
            **fragmentation_indices_io,
            **fragmentation_indices_digital,
            **average_interepisode_durations
        }

        return result

    except Exception as e:
        error_count += 1
        if error_count <= MAX_ERRORS:
            print(f"Error processing file {os.path.basename(file_path)}: {str(e)}")
        if error_count == MAX_ERRORS:
            print("Maximum number of errors reached. Some files may not be processed.")
        return None

@timer
def main():
    input_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/preprocessed_data'
    output_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/'

    all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv')]

    all_results = []
    
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {num_workers} workers for parallel processing")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(analyze_participant_day, file) for file in all_files]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    all_results.append(result)
            except Exception as e:
                print(f"Error processing future: {str(e)}")

    print(f"Number of results: {len(all_results)}")
    print("Sample of first result:")
    print(all_results[0] if all_results else "No results")

    summary_df = pd.DataFrame(all_results)

    print("\nSummary DataFrame info:")
    print(summary_df.info())

    print("\nSummary DataFrame head:")
    print(summary_df.head())

    if not summary_df.empty:
        summary_df.to_csv(os.path.join(output_dir, 'fragmentation_summary.csv'), index=False)
        print("Summary saved to CSV.")

        print("\nGenerating descriptive statistics...")
        print(summary_df.describe())

        # Check if the expected columns exist
        expected_columns = ['Stationary_index', 'Mobile_index', 'Stationary_AID', 'Mobile_AID']
        missing_columns = [col for col in expected_columns if col not in summary_df.columns]
        
        if missing_columns:
            print(f"Warning: The following expected columns are missing: {missing_columns}")
            print("Available columns:")
            print(summary_df.columns)
        else:
            print("\nSummary of fragmentation indices:")
            fragmentation_columns = [col for col in summary_df.columns if col.endswith('_index')]
            print(summary_df[fragmentation_columns].describe())

            print("\nSummary of average interepisode durations:")
            aid_columns = [col for col in summary_df.columns if col.endswith('_AID')]
            print(summary_df[aid_columns].describe())

            print("\nCreating visualizations...")
            create_boxplot(summary_df, fragmentation_columns, 'Fragmentation Indices', 'Fragmentation Index', output_dir)
            create_boxplot(summary_df, aid_columns, 'Average Interepisode Durations', 'Duration (minutes)', output_dir)

            print("\nParticipant summary:")
            create_participant_summary(summary_df, output_dir)
    else:
        print("No valid results were generated. Please check your data and error messages.")

def create_boxplot(df, columns, title, ylabel, output_dir):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[columns])
    plt.title(f'Distribution of {title}')
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{title.lower().replace(" ", "_")}_distribution.png'))
    plt.close()

def create_participant_summary(df, output_dir):
    participant_summary = df.groupby('participant_id').agg({
        'date': 'count',
        'total_episodes': 'mean',
        'stationary_episodes': 'mean',
        'mobile_episodes': 'mean',
        'Stationary_index': 'mean',
        'Mobile_index': 'mean',
        'Stationary_AID': 'mean',
        'Mobile_AID': 'mean'
    }).reset_index()
    participant_summary = participant_summary.rename(columns={'date': 'days_with_data'})
    print(participant_summary)
    participant_summary.to_csv(os.path.join(output_dir, 'participant_summary.csv'), index=False)

if __name__ == "__main__":
    main()