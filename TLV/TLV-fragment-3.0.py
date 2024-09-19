import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

def calculate_fragmentation_index(episodes_df, column, min_episodes=5):
    S = len(episodes_df)
    T = episodes_df['duration'].sum()
    if S >= min_episodes and T > 0:
        normalized_durations = episodes_df['duration'] / T
        sum_squared = sum(normalized_durations ** 2)
        index = (1 - sum_squared) / (1 - (1 / S))
        if index > 0.9999:
            print(f"High fragmentation detected: {index}")
            print(f"Number of episodes: {S}")
            print(f"Total duration: {T}")
            print(f"Normalized durations: {normalized_durations.tolist()}")
            print(f"Sum of squared normalized durations: {sum_squared}")
        return index
    else:
        print(f"Insufficient data for fragmentation index: {S} episodes, {T} total duration")
        return np.nan

def calculate_aid(episodes_df):
    if len(episodes_df) > 1:
        inter_episode_durations = np.abs((episodes_df['start_time'].iloc[1:] - episodes_df['end_time'].iloc[:-1]).dt.total_seconds() / 60)
        
        if len(inter_episode_durations) > 0:
            aid_mean = np.mean(inter_episode_durations)
            aid_median = np.median(inter_episode_durations)
            aid_std = np.std(inter_episode_durations)
            aid_min = np.min(inter_episode_durations)
            aid_max = np.max(inter_episode_durations)
            aid_counts = {
                '1-5min': sum((inter_episode_durations >= 1) & (inter_episode_durations < 5)),
                '5-15min': sum((inter_episode_durations >= 5) & (inter_episode_durations < 15)),
                '15-60min': sum((inter_episode_durations >= 15) & (inter_episode_durations < 60)),
                '60+min': sum(inter_episode_durations >= 60)
            }
        else:
            aid_mean = aid_median = aid_std = aid_min = aid_max = np.nan
            aid_counts = {k: 0 for k in ['1-5min', '5-15min', '15-60min', '60+min']}
    else:
        aid_mean = aid_median = aid_std = aid_min = aid_max = np.nan
        aid_counts = {k: 0 for k in ['1-5min', '5-15min', '15-60min', '60+min']}
    
    return aid_mean, aid_median, aid_std, aid_min, aid_max, aid_counts

def extract_info_from_filename(filename):
    parts = filename.split('_')
    date_str = parts[-1].split('.')[0]  # Remove the .csv extension
    participant_id = parts[-2]
    date = datetime.strptime(date_str, "%Y-%m-%d").date()
    return participant_id, date

def process_episode_summary(file_path, episode_type, print_sample=False):
    try:
        df = pd.read_csv(file_path)
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])

        if print_sample:
            print(f"\nSample data for {os.path.basename(file_path)}:")
            print(df.head())
            print(f"\nTotal episodes: {len(df)}")

        participant_id, date = extract_info_from_filename(os.path.basename(file_path))

        result = {
            'participant_id': participant_id,
            'date': date,
            'total_episodes': len(df),
            'total_duration': df['duration'].sum(),
            'avg_episode_length': df['duration'].mean(),
        }

        result['fragmentation_index'] = calculate_fragmentation_index(df, 'duration')
        
        # Add debugging for extreme fragmentation values
        if result['fragmentation_index'] > 0.9999 or result['fragmentation_index'] < 0.0001:
            print(f"Extreme fragmentation index detected: {result['fragmentation_index']}")
            print(f"File: {file_path}")
            print(f"Number of episodes: {len(df)}")
            print(f"Episode durations: {df['duration'].tolist()}")

        result['fragmentation_index'] = calculate_fragmentation_index(df, 'duration')
        aid_mean, aid_median, aid_std, aid_min, aid_max, aid_counts = calculate_aid(df)
        result['AID_mean'] = aid_mean
        result['AID_median'] = aid_median
        result['AID_std'] = aid_std
        result['AID_min'] = aid_min
        result['AID_max'] = aid_max
        for k, v in aid_counts.items():
            result[f'AID_{k}'] = v

        return pd.DataFrame([result])
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def print_summary_statistics(df, episode_type):
    print(f"\nSummary Statistics for {episode_type}:")
    print(f"Total participants: {df['participant_id'].nunique()}")
    print(f"Total days: {len(df)}")
    print(f"Average episodes per day: {df['total_episodes'].mean():.2f}")
    print(f"Average episode length: {df['avg_episode_length'].mean():.2f} minutes")
    print(f"Average total duration per day: {df['total_duration'].mean():.2f} minutes")
    print(f"Average Fragmentation Index: {df['fragmentation_index'].mean():.4f}")
    print(f"Maximum Fragmentation Index: {df['fragmentation_index'].max():.4f}")
    print(f"Minimum Fragmentation Index: {df['fragmentation_index'].min():.4f}")
    print(f"Average AID (mean): {df['AID_mean'].mean():.2f} minutes")
    print(f"Average AID (median): {df['AID_median'].mean():.2f} minutes")
    print(f"Average AID (std): {df['AID_std'].mean():.2f} minutes")
    print(f"Average AID (min): {df['AID_min'].mean():.2f} minutes")
    print(f"Average AID (max): {df['AID_max'].mean():.2f} minutes")

    print("\nAID Interval Counts:")
    for interval in ['1-5min', '5-15min', '15-60min', '60+min']:
        print(f"  {interval}: {df[f'AID_{interval}'].mean():.2f}")

    # Create histogram of Fragmentation Index
    plt.figure(figsize=(10, 6))
    plt.hist(df['fragmentation_index'].dropna(), bins=50)
    plt.title(f'Histogram of {episode_type} Fragmentation Index')
    plt.xlabel('Fragmentation Index')
    plt.ylabel('Frequency')
    plt.savefig(f'{episode_type}_fragmentation_histogram.png')
    plt.close()

    # Create histogram of AID
    plt.figure(figsize=(10, 6))
    plt.hist(df['AID_mean'].dropna(), bins=50)
    plt.title(f'Histogram of {episode_type} AID (mean)')
    plt.xlabel('AID (minutes)')
    plt.ylabel('Frequency')
    plt.savefig(f'{episode_type}_AID_histogram.png')
    plt.close()

def main(input_dir, output_dir):
    episode_types = ['digital', 'moving']
    
    for episode_type in episode_types:
        input_files = [f for f in os.listdir(input_dir) if f.startswith(f'{episode_type}_episodes_') and f.endswith('.csv')]
        
        if input_files:
            all_results = []
            for i, input_file in enumerate(tqdm(input_files, desc=f"Processing {episode_type} episodes")):
                file_path = os.path.join(input_dir, input_file)
                # Print sample data for the first file of each episode type
                results = process_episode_summary(file_path, episode_type, print_sample=(i==0))
                if results is not None:
                    all_results.append(results)
            
            combined_results = pd.concat(all_results, ignore_index=True)
            output_file = os.path.join(output_dir, f'{episode_type}_fragmentation_summary.csv')
            combined_results.to_csv(output_file, index=False)
            print(f"Saved fragmentation summary for {episode_type} episodes to {output_file}")
            print_summary_statistics(combined_results, episode_type)
        else:
            print(f"Warning: No {episode_type} episode files found in {input_dir}. Skipping {episode_type} analysis.")

if __name__ == "__main__":
    input_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/episodes'
    output_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/fragmentation'
    main(input_dir, output_dir)