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

def calculate_mobility_metrics(df):
    total_duration = df['duration'].sum()
    avg_duration = df['duration'].mean()
    episode_count = len(df)
    return total_duration, avg_duration, episode_count

def calculate_digital_frag_during_mobility(digital_df, moving_df):
    digital_episodes_during_mobility = []
    for _, mobility_episode in moving_df.iterrows():
        mobility_start = mobility_episode['start_time']
        mobility_end = mobility_episode['end_time']
        overlapping_digital = digital_df[
            (digital_df['start_time'] < mobility_end) & 
            (digital_df['end_time'] > mobility_start)
        ]
        if not overlapping_digital.empty:
            digital_episodes_during_mobility.append(overlapping_digital)
    
    if digital_episodes_during_mobility:
        combined_digital = pd.concat(digital_episodes_during_mobility)
        return calculate_fragmentation_index(combined_digital, 'duration')
    else:
        return np.nan

def process_episode_summary(digital_file_path, moving_file_path, print_sample=False):
    try:
        digital_df = pd.read_csv(digital_file_path)
        moving_df = pd.read_csv(moving_file_path)
        
        for df in [digital_df, moving_df]:
            df['start_time'] = pd.to_datetime(df['start_time'])
            df['end_time'] = pd.to_datetime(df['end_time'])

        if print_sample:
            print(f"\nSample data for {os.path.basename(digital_file_path)}:")
            print(digital_df.head())
            print(f"\nSample data for {os.path.basename(moving_file_path)}:")
            print(moving_df.head())

        participant_id, date = extract_info_from_filename(os.path.basename(digital_file_path))

        digital_frag_index = calculate_fragmentation_index(digital_df, 'duration')
        moving_frag_index = calculate_fragmentation_index(moving_df, 'duration')
        digital_frag_during_mobility = calculate_digital_frag_during_mobility(digital_df, moving_df)

        total_duration_mobility, avg_duration_mobility, count_mobility = calculate_mobility_metrics(moving_df)

        result = {
            'participant_id': participant_id,
            'date': date,
            'digital_fragmentation_index': digital_frag_index,
            'moving_fragmentation_index': moving_frag_index,
            'digital_frag_during_mobility': digital_frag_during_mobility,
            'total_duration_mobility': total_duration_mobility,
            'avg_duration_mobility': avg_duration_mobility,
            'count_mobility': count_mobility
        }

        # Calculate AID for both digital and moving episodes
        for episode_type, df in [('digital', digital_df), ('moving', moving_df)]:
            aid_mean, aid_median, aid_std, aid_min, aid_max, aid_counts = calculate_aid(df)
            result[f'{episode_type}_AID_mean'] = aid_mean
            result[f'{episode_type}_AID_median'] = aid_median
            result[f'{episode_type}_AID_std'] = aid_std
            result[f'{episode_type}_AID_min'] = aid_min
            result[f'{episode_type}_AID_max'] = aid_max
            for k, v in aid_counts.items():
                result[f'{episode_type}_AID_{k}'] = v

        return pd.DataFrame([result])
    except Exception as e:
        print(f"Error processing files {digital_file_path} and {moving_file_path}: {str(e)}")
        return None

def print_summary_statistics(df):
    print("\nSummary Statistics:")
    print(f"Total participants: {df['participant_id'].nunique()}")
    print(f"Total days: {len(df)}")
    
    for col in df.columns:
        if col not in ['participant_id', 'date']:
            print(f"Average {col}: {df[col].mean():.4f}")

    # Create histograms
    for col in ['digital_fragmentation_index', 'moving_fragmentation_index', 'digital_frag_during_mobility']:
        plt.figure(figsize=(10, 6))
        plt.hist(df[col].dropna(), bins=50)
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.savefig(f'{col}_histogram.png')
        plt.close()

def main(input_dir, output_dir):
    digital_files = sorted([f for f in os.listdir(input_dir) if f.startswith('digital_episodes_') and f.endswith('.csv')])
    moving_files = sorted([f for f in os.listdir(input_dir) if f.startswith('moving_episodes_') and f.endswith('.csv')])
    
    if len(digital_files) != len(moving_files):
        print("Warning: Mismatch in the number of digital and moving episode files.")
    
    all_results = []
    for i, (digital_file, moving_file) in enumerate(tqdm(zip(digital_files, moving_files), desc="Processing episodes")):
        digital_path = os.path.join(input_dir, digital_file)
        moving_path = os.path.join(input_dir, moving_file)
        results = process_episode_summary(digital_path, moving_path, print_sample=(i==0))
        if results is not None:
            all_results.append(results)
    
    combined_results = pd.concat(all_results, ignore_index=True)
    output_file = os.path.join(output_dir, 'fragmentation_summary.csv')
    combined_results.to_csv(output_file, index=False)
    print(f"Saved fragmentation summary to {output_file}")
    print_summary_statistics(combined_results)

if __name__ == "__main__":
    input_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/episodes'
    output_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/fragmentation'
    main(input_dir, output_dir)