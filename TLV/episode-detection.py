import pandas as pd
import numpy as np
import os
from datetime import timedelta
from tqdm import tqdm
from collections import defaultdict

def detect_episodes(df, column, min_duration, merge_gap):
    episodes = []
    start_time = None
    prev_time = None
    in_episode = False
    merged_count = 0
    changed_values = 0

    for idx, row in df.iterrows():
        current_time = row['Timestamp']
        current_state = row[column]

        if not in_episode and current_state:
            start_time = current_time
            in_episode = True
            changed_values += 1
        elif in_episode:
            if not current_state or (idx == len(df) - 1):
                end_time = current_time if current_state else prev_time
                duration = end_time - start_time
                if duration >= min_duration:
                    episodes.append((start_time, end_time))
                in_episode = False
                start_time = None
                changed_values += 1

        prev_time = current_time

    # Merge episodes
    merged_episodes = []
    for episode in episodes:
        if not merged_episodes or (episode[0] - merged_episodes[-1][1]) > merge_gap:
            merged_episodes.append(episode)
        else:
            merged_episodes[-1] = (merged_episodes[-1][0], episode[1])
            merged_count += 1

    return merged_episodes, merged_count, changed_values

def process_user_day(file_path, verbose=False):
    if verbose:
        print(f"Processing file: {file_path}")
    df = pd.read_csv(file_path, parse_dates=['Timestamp'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.floor('s')  # Drop milliseconds, using 's' instead of 'S'  # Drop milliseconds
    
    if verbose:
        print("Columns in the DataFrame:")
        print(df.columns)
        print("\nFirst few rows of the DataFrame:")
        print(df.head())
    
    if 'is_digital' not in df.columns or 'is_moving' not in df.columns:
        if verbose:
            print("Warning: 'is_digital' or 'is_moving' column not found. Available columns are:")
            print(df.columns)
        return None

    digital_episodes, digital_merged, digital_changed = detect_episodes(df, 'is_digital', 
                                                                        min_duration=timedelta(minutes=1), 
                                                                        merge_gap=timedelta(seconds=40))
    moving_episodes, moving_merged, moving_changed = detect_episodes(df, 'is_moving', 
                                                                     min_duration=timedelta(minutes=2), 
                                                                     merge_gap=timedelta(seconds=80))
    
    return {
        'digital_episodes': digital_episodes,
        'moving_episodes': moving_episodes,
        'date': df['Timestamp'].dt.date.iloc[0],
        'user': df['user'].iloc[0],
        'digital_merged': digital_merged,
        'moving_merged': moving_merged,
        'digital_changed': digital_changed,
        'moving_changed': moving_changed
    }

def calculate_statistics(all_episodes):
    stats = defaultdict(lambda: defaultdict(list))
    
    for day_data in all_episodes:
        for episode_type in ['digital', 'moving']:
            episodes = day_data[f'{episode_type}_episodes']
            durations = [(end - start).total_seconds() / 60 for start, end in episodes]
            
            stats[episode_type]['total_episodes'].append(len(episodes))
            stats[episode_type]['total_duration'].append(sum(durations))
            if durations:
                stats[episode_type]['min_duration'].append(min(durations))
                stats[episode_type]['max_duration'].append(max(durations))
            stats[episode_type]['merged_episodes'].append(day_data[f'{episode_type}_merged'])
            stats[episode_type]['changed_values'].append(day_data[f'{episode_type}_changed'])
    
    for episode_type in ['digital', 'moving']:
        stats[episode_type]['avg_episodes_per_day'] = np.mean(stats[episode_type]['total_episodes'])
        stats[episode_type]['avg_total_duration_per_day'] = np.mean(stats[episode_type]['total_duration'])
        stats[episode_type]['avg_duration_per_episode'] = (
            np.sum(stats[episode_type]['total_duration']) / np.sum(stats[episode_type]['total_episodes'])
            if np.sum(stats[episode_type]['total_episodes']) > 0 else 0
        )
        stats[episode_type]['min_duration'] = min(stats[episode_type]['min_duration']) if stats[episode_type]['min_duration'] else 0
        stats[episode_type]['max_duration'] = max(stats[episode_type]['max_duration']) if stats[episode_type]['max_duration'] else 0
        stats[episode_type]['total_merged_episodes'] = np.sum(stats[episode_type]['merged_episodes'])
        stats[episode_type]['total_changed_values'] = np.sum(stats[episode_type]['changed_values'])
    
    return stats

def main():
    preprocessed_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/preprocessed_data'
    output_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/episodes'
    os.makedirs(output_dir, exist_ok=True)
    all_episodes = []
    
    for i, file_name in enumerate(tqdm(os.listdir(preprocessed_dir))):
        if file_name.endswith('.csv'):
            file_path = os.path.join(preprocessed_dir, file_name)
            day_episodes = process_user_day(file_path, verbose=(i < 3))  # Only print details for first 3 files
            if day_episodes is not None:
                all_episodes.append(day_episodes)
                
                # Save episodes to separate CSV files
                for episode_type in ['digital', 'moving']:
                    episodes_df = pd.DataFrame(day_episodes[f'{episode_type}_episodes'], 
                                               columns=['start_time', 'end_time'])
                    episodes_df['start_time'] = pd.to_datetime(episodes_df['start_time'])
                    episodes_df['end_time'] = pd.to_datetime(episodes_df['end_time'])
                    episodes_df['duration'] = (episodes_df['end_time'] - episodes_df['start_time']).dt.total_seconds() / 60
                    episodes_df['user'] = day_episodes['user']
                    episodes_df['date'] = day_episodes['date']
                    episodes_df.to_csv(os.path.join(output_dir, f"{episode_type}_episodes_{day_episodes['user']}_{day_episodes['date']}.csv"), 
                                       index=False)
    
    if not all_episodes:
        print("No episodes were processed successfully. Check the data and column names.")
        return

    stats = calculate_statistics(all_episodes)
    
    print("\nDescriptive Statistics:")
    for episode_type in ['digital', 'moving']:
        print(f"\n{episode_type.capitalize()} Episodes:")
        print(f"Total number of episodes: {int(np.sum(stats[episode_type]['total_episodes']))}")
        print(f"Total duration (minutes): {np.sum(stats[episode_type]['total_duration']):.2f}")
        print(f"Minimum episode duration (minutes): {stats[episode_type]['min_duration']:.2f}")
        print(f"Maximum episode duration (minutes): {stats[episode_type]['max_duration']:.2f}")
        print(f"Average duration per episode (minutes): {stats[episode_type]['avg_duration_per_episode']:.2f}")
        print(f"Average total duration per day (minutes): {stats[episode_type]['avg_total_duration_per_day']:.2f}")
        print(f"Average number of episodes per day: {stats[episode_type]['avg_episodes_per_day']:.2f}")
        print(f"Total merged episodes: {stats[episode_type]['total_merged_episodes']}")
        print(f"Total changed values: {stats[episode_type]['total_changed_values']}")

if __name__ == "__main__":
    main()