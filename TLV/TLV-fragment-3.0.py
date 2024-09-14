import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def merge_short_episodes(df, column, threshold=60):
    merged = []
    current_episode = None
    for _, row in df.iterrows():
        if current_episode is None:
            current_episode = row.to_dict()
        else:
            time_diff = (row['start_time'] - current_episode['end_time']).total_seconds()
            if time_diff < threshold and row[column] == current_episode[column]:
                current_episode['end_time'] = row['end_time']
                current_episode['duration'] += row['duration']
            else:
                merged.append(current_episode)
                current_episode = row.to_dict()
    if current_episode is not None:
        merged.append(current_episode)
    return pd.DataFrame(merged)

def calculate_fragmentation_index(episodes_df, column, positive_category):
    category_episodes = episodes_df[episodes_df[column] == positive_category]
    S = len(category_episodes)
    T = category_episodes['duration'].sum()
    if S > 1 and T > 0:
        normalized_durations = category_episodes['duration'] / T
        sum_squared = sum(normalized_durations ** 2)
        index = (1 - sum_squared) / (1 - (1 / S))
    else:
        index = np.nan
    return index

def calculate_aid(episodes_df, column, positive_category):
    category_episodes = episodes_df[episodes_df[column] == positive_category].sort_values('start_time')
    if len(category_episodes) > 1:
        inter_episode_durations = np.abs((category_episodes['start_time'].iloc[1:] - category_episodes['end_time'].iloc[:-1]).dt.total_seconds() / 60)
        
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

def extract_info_from_file(df, filename):
    participant_id = filename.split('_')[1]
    date = pd.to_datetime(df['start_time'].iloc[0]).date()
    return participant_id, date

def process_episode_summary(file_path, episode_type, print_sample=False):
    try:
        df = pd.read_csv(file_path)
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])

        if episode_type == 'mobility_episodes':
            column = 'mobility'
            positive_category = 'Moving'
        elif episode_type == 'digital_episodes':
            column = 'digital_use'
            positive_category = 'Digital'
        else:
            print(f"Error: Unknown episode type '{episode_type}'")
            return None

        if column not in df.columns:
            print(f"Error: Column '{column}' not found in the DataFrame.")
            return None

        # Merge short episodes
        df_merged = merge_short_episodes(df, column)
        
        if print_sample:
            print(f"\nSample data for {os.path.basename(file_path)} (after merging):")
            print(df_merged.head())
            print(f"\nTotal episodes after merging: {len(df_merged)}")

        participant_id, date = extract_info_from_file(df, os.path.basename(file_path))

        result = {
            'participant_id': participant_id,
            'date': date,
            'total_episodes': len(df_merged),
            'total_duration': df_merged['duration'].sum(),
            'avg_episode_length': df_merged['duration'].mean(),
        }

        positive_episodes = df_merged[df_merged[column] == positive_category]
        result[f'{positive_category}_episodes'] = len(positive_episodes)
        result[f'{positive_category}_duration'] = positive_episodes['duration'].sum()
        result[f'{positive_category}_avg_length'] = positive_episodes['duration'].mean() if len(positive_episodes) > 0 else np.nan

        result[f'{positive_category}_fragmentation_index'] = calculate_fragmentation_index(df_merged, column, positive_category)
        aid_mean, aid_median, aid_std, aid_min, aid_max, aid_counts = calculate_aid(df_merged, column, positive_category)
        result[f'{positive_category}_AID_mean'] = aid_mean
        result[f'{positive_category}_AID_median'] = aid_median
        result[f'{positive_category}_AID_std'] = aid_std
        result[f'{positive_category}_AID_min'] = aid_min
        result[f'{positive_category}_AID_max'] = aid_max
        for k, v in aid_counts.items():
            result[f'{positive_category}_AID_{k}'] = v

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

    positive_category = 'Moving' if 'mobility' in episode_type else 'Digital'

    print(f"\n{positive_category} Statistics:")
    print(f"Average {positive_category} episodes per day: {df[f'{positive_category}_episodes'].mean():.2f}")
    print(f"Average {positive_category} duration per day: {df[f'{positive_category}_duration'].mean():.2f} minutes")
    print(f"Average {positive_category} episode length: {df[f'{positive_category}_avg_length'].mean():.2f} minutes")
    print(f"Average {positive_category} Fragmentation Index: {df[f'{positive_category}_fragmentation_index'].mean():.4f}")
    print(f"Average {positive_category} AID (mean): {df[f'{positive_category}_AID_mean'].mean():.2f} minutes")
    print(f"Average {positive_category} AID (median): {df[f'{positive_category}_AID_median'].mean():.2f} minutes")
    print(f"Average {positive_category} AID (std): {df[f'{positive_category}_AID_std'].mean():.2f} minutes")
    print(f"Average {positive_category} AID (min): {df[f'{positive_category}_AID_min'].mean():.2f} minutes")
    print(f"Average {positive_category} AID (max): {df[f'{positive_category}_AID_max'].mean():.2f} minutes")

    print("\nAID Interval Counts:")
    for interval in ['1-5min', '5-15min', '15-60min', '60+min']:
        print(f"  {interval}: {df[f'{positive_category}_AID_{interval}'].mean():.2f}")

    # Create histogram of AID
    plt.figure(figsize=(10, 6))
    plt.hist(df[f'{positive_category}_AID_mean'].dropna(), bins=50)
    plt.title(f'Histogram of {positive_category} AID (mean)')
    plt.xlabel('AID (minutes)')
    plt.ylabel('Frequency')
    plt.savefig(f'{episode_type}_AID_histogram.png')
    plt.close()

def main(input_dir, output_dir):
    episode_types = ['mobility_episodes', 'digital_episodes']
    
    for episode_type in episode_types:
        input_files = [f for f in os.listdir(input_dir) if f.startswith(f'{episode_type}_') and f.endswith('.csv')]
        
        if input_files:
            all_results = []
            for i, input_file in enumerate(tqdm(input_files, desc=f"Processing {episode_type}")):
                file_path = os.path.join(input_dir, input_file)
                # Print sample data for the first file of each episode type
                results = process_episode_summary(file_path, episode_type, print_sample=(i==0))
                all_results.append(results)
            
            combined_results = pd.concat(all_results, ignore_index=True)
            output_file = os.path.join(output_dir, f'{episode_type}_fragmentation_summary.csv')
            combined_results.to_csv(output_file, index=False)
            print(f"Saved fragmentation summary for {episode_type} to {output_file}")
            print_summary_statistics(combined_results, episode_type)
        else:
            print(f"Warning: No {episode_type} files found in {input_dir}. Skipping {episode_type} analysis.")

if __name__ == "__main__":
    input_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/episodes'
    output_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/fragmentation'
    main(input_dir, output_dir)