import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import medfilt
import time
import concurrent.futures
import multiprocessing
from tqdm import tqdm
from functools import wraps

MAX_ERRORS = 5
error_count = 0

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

def smooth_data(data, window_size=5):
    return medfilt(data, kernel_size=window_size)

def classify_movement(speed, threshold=1.5):
    if pd.isna(speed):
        return 'Unknown'
    elif speed < threshold:
        return 'Stationary'
    else:
        return 'Mobile'

@timeit
def detect_episodes(df, movement_min_duration=5, digital_min_duration=1):
    df['smoothed_speed'] = smooth_data(df['speed'].values)
    df['movement_type'] = df['smoothed_speed'].apply(classify_movement)
    
    episodes = []
    current_episode = {
        'start_time': df['Timestamp'].iloc[0],
        'movement_type': df['movement_type'].iloc[0],
        'indoor_outdoor': df['indoors'].iloc[0],
        'digital_use': 'Yes' if df['isapp'].iloc[0] == 1 else 'No'
    }
    
    for i in range(1, len(df)):
        if (df['movement_type'].iloc[i] != current_episode['movement_type'] or
            df['indoors'].iloc[i] != current_episode['indoor_outdoor'] or
            (df['isapp'].iloc[i] == 1) != (current_episode['digital_use'] == 'Yes')):
            
            current_episode['end_time'] = df['Timestamp'].iloc[i-1]
            current_episode['duration'] = (current_episode['end_time'] - current_episode['start_time']).total_seconds() / 60
            current_episode['avg_speed'] = df.loc[df['Timestamp'].between(current_episode['start_time'], current_episode['end_time']), 'speed'].mean()
            
            if ((current_episode['movement_type'] != 'Unknown' and current_episode['duration'] >= movement_min_duration) or
                (current_episode['digital_use'] == 'Yes' and current_episode['duration'] >= digital_min_duration)):
                episodes.append(current_episode)
            
            current_episode = {
                'start_time': df['Timestamp'].iloc[i],
                'movement_type': df['movement_type'].iloc[i],
                'indoor_outdoor': df['indoors'].iloc[i],
                'digital_use': 'Yes' if df['isapp'].iloc[i] == 1 else 'No'
            }
    
    # Add the last episode
    current_episode['end_time'] = df['Timestamp'].iloc[-1]
    current_episode['duration'] = (current_episode['end_time'] - current_episode['start_time']).total_seconds() / 60
    current_episode['avg_speed'] = df.loc[df['Timestamp'].between(current_episode['start_time'], current_episode['end_time']), 'speed'].mean()
    
    if ((current_episode['movement_type'] != 'Unknown' and current_episode['duration'] >= movement_min_duration) or
        (current_episode['digital_use'] == 'Yes' and current_episode['duration'] >= digital_min_duration)):
        episodes.append(current_episode)
    
    return pd.DataFrame(episodes)

@timeit
def calculate_fragmentation_index(episodes_df, column):
    fragmentation_indices = {}
    
    for category in episodes_df[column].unique():
        category_episodes = episodes_df[episodes_df[column] == category]
        S = len(category_episodes)
        T = category_episodes['duration'].sum()
        
        if S > 1 and T > 0:
            normalized_durations = category_episodes['duration'] / T
            sum_squared = sum(normalized_durations ** 2)
            index = (1 - sum_squared) / (1 - (1 / S))
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
        return pd.to_datetime(t).floor('s')
    except:
        return pd.NaT

def preprocess_data(df):
    df = df.copy()
    df['Timestamp'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
    df['Timestamp'] = df['Timestamp'].apply(parse_time)
    df = df.dropna(subset=['Timestamp'])
    df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
    df['isapp'] = df['isapp'].astype(int)
    df['indoors'] = df['indoors'].astype(str)
    return df.sort_values('Timestamp')

def visualize_data(df, episodes_df, output_path):
    plt.figure(figsize=(15, 10))
    plt.plot(df['Timestamp'], df['speed'], label='Original Speed', alpha=0.5)
    plt.plot(df['Timestamp'], df['smoothed_speed'], label='Smoothed Speed')
    plt.axhline(y=1.5, color='r', linestyle='--', label='Speed Threshold')
    
    for _, episode in episodes_df.iterrows():
        color = 'green' if episode['movement_type'] == 'Stationary' else 'blue'
        plt.axvspan(episode['start_time'], episode['end_time'], alpha=0.2, color=color)
    
    plt.title('Speed Over Time with Detected Episodes')
    plt.xlabel('Time')
    plt.ylabel('Speed')
    plt.legend()
    plt.savefig(output_path)
    plt.close()

@timeit
def analyze_participant_day(file_path, output_dir):
    global error_count
    try:
        participant_df = pd.read_csv(file_path)
        if participant_df.empty:
            return None

        participant_df = preprocess_data(participant_df)
        
        if 'speed' not in participant_df.columns or participant_df.empty:
            return None

        episodes_df = detect_episodes(participant_df)

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
            'unknown_episodes': len(episodes_df[episodes_df['movement_type'] == 'Unknown']),
            'total_duration': episodes_df['duration'].sum(),
            'stationary_duration': episodes_df[episodes_df['movement_type'] == 'Stationary']['duration'].sum(),
            'mobile_duration': episodes_df[episodes_df['movement_type'] == 'Mobile']['duration'].sum(),
            'unknown_duration': episodes_df[episodes_df['movement_type'] == 'Unknown']['duration'].sum(),
            **fragmentation_indices_movement,
            **fragmentation_indices_io,
            **fragmentation_indices_digital,
            **average_interepisode_durations
        }

        # Visualize data for the first 5 participants
        if result['participant_id'] <= 5:
            vis_output_path = os.path.join(output_dir, f"participant_{result['participant_id']}_{result['date']}_speed_visualization.png")
            visualize_data(participant_df, episodes_df, vis_output_path)

        return result

    except Exception as e:
        error_count += 1
        if error_count <= MAX_ERRORS:
            print(f"Error processing file {os.path.basename(file_path)}: {str(e)}")
        if error_count == MAX_ERRORS:
            print("Maximum number of errors reached. Some files may not be processed.")
        return None

@timeit
def main(test_mode=True, num_test_files=10):
    input_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/preprocessed_data'
    output_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/'

    all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv')]

    if test_mode:
        print(f"Running in test mode with {num_test_files} files")
        all_files = all_files[:num_test_files]
    else:
        print(f"Running full analysis on {len(all_files)} files")

    all_results = []
    
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {num_workers} workers for parallel processing")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(analyze_participant_day, file, output_dir) for file in all_files]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(all_files), desc="Processing files"):
            try:
                result = future.result()
                if result is not None:
                    all_results.append(result)
            except Exception as e:
                print(f"Error processing future: {str(e)}")

    print(f"Processed {len(all_results)} files successfully.")

    summary_df = pd.DataFrame(all_results)

    if not summary_df.empty:
        summary_df.to_csv(os.path.join(output_dir, 'fragmentation_daily_summary.csv'), index=False)
        print("Daily summary saved to CSV.")

        print("\nGenerating descriptive statistics...")
        print(summary_df.describe())

        print("\nSummary of fragmentation indices:")
        fragmentation_columns = [col for col in summary_df.columns if col.endswith('_index')]
        print(summary_df[fragmentation_columns].describe())

        print("\nSummary of average interepisode durations:")
        aid_columns = [col for col in summary_df.columns if col.endswith('_AID')]
        print(summary_df[aid_columns].describe())

        print("\nCreating visualizations...")
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=summary_df[fragmentation_columns])
        plt.title('Distribution of Fragmentation Indices')
        plt.ylabel('Fragmentation Index')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fragmentation_indices_distribution.png'))
        plt.close()

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=summary_df[aid_columns])
        plt.title('Distribution of Average Interepisode Durations')
        plt.ylabel('Duration (minutes)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'average_interepisode_durations_distribution.png'))
        plt.close()

        print("Visualizations saved in the output directory.")

        print("\nGenerating participant summary...")
        existing_columns = summary_df.columns
        fragmentation_columns = [col for col in existing_columns if col.endswith('_index')]
        aid_columns = [col for col in existing_columns if col.endswith('_AID')]
        episode_columns = [col for col in existing_columns if col.endswith('_episodes')]
        
        agg_dict = {
            'date': 'count',
            'total_episodes': 'mean',
            **{col: 'mean' for col in episode_columns},
            **{col: 'mean' for col in fragmentation_columns},
            **{col: 'mean' for col in aid_columns}
        }
        
        participant_summary = summary_df.groupby('participant_id').agg(agg_dict).reset_index()
        participant_summary = participant_summary.rename(columns={'date': 'days_with_data'})
        print(participant_summary)
        participant_summary.to_csv(os.path.join(output_dir, 'participant_summary.csv'), index=False)
        print("Participant summary saved to CSV.")
    else:
        print("No valid results were generated. Please check your data and error messages.")

if __name__ == "__main__":
    main(test_mode=True, num_test_files=10)