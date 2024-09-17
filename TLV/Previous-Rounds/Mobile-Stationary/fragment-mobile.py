#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 22:04:45 2024

@author: noamgal
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from scipy.signal import medfilt

def smooth_data(data, window_size=5):
    return medfilt(data, kernel_size=window_size)

def classify_movement(speed, stationary_threshold=0.5, mobile_threshold=2.0):
    if pd.isna(speed):
        return 'Unknown'
    elif speed < stationary_threshold:
        return 'Stationary'
    elif speed >= mobile_threshold:
        return 'Mobile'
    else:
        return 'Transition'

def detect_episodes(df, movement_min_duration=10, digital_min_duration=5, min_speed_change=1.0):
    df['smoothed_speed'] = smooth_data(df['speed'].fillna(0))
    df['movement_type'] = df['smoothed_speed'].apply(classify_movement)

    episodes = []
    current_episode = {
        'start_time': df['Timestamp'].iloc[0],
        'movement_type': df['movement_type'].iloc[0],
        'indoor_outdoor': df['indoors'].iloc[0],
        'digital_use': 'Yes' if df['isapp'].iloc[0] else 'No'
    }

    for i in range(1, len(df)):
        if (df['movement_type'].iloc[i] != current_episode['movement_type'] or
            df['indoors'].iloc[i] != current_episode['indoor_outdoor'] or
            df['isapp'].iloc[i] != (current_episode['digital_use'] == 'Yes')):

            if abs(df['smoothed_speed'].iloc[i] - df['smoothed_speed'].iloc[i-1]) >= min_speed_change:
                current_episode['end_time'] = df['Timestamp'].iloc[i-1]
                current_episode['duration'] = (current_episode['end_time'] - current_episode['start_time']).total_seconds() / 60

                if ((current_episode['movement_type'] != 'Unknown' and current_episode['duration'] >= movement_min_duration) or
                    (current_episode['digital_use'] == 'Yes' and current_episode['duration'] >= digital_min_duration)):
                    episodes.append(current_episode)

                current_episode = {
                    'start_time': df['Timestamp'].iloc[i],
                    'movement_type': df['movement_type'].iloc[i],
                    'indoor_outdoor': df['indoors'].iloc[i],
                    'digital_use': 'Yes' if df['isapp'].iloc[i] else 'No'
                }

    # Add the last episode
    current_episode['end_time'] = df['Timestamp'].iloc[-1]
    current_episode['duration'] = (current_episode['end_time'] - current_episode['start_time']).total_seconds() / 60

    if ((current_episode['movement_type'] != 'Unknown' and current_episode['duration'] >= movement_min_duration) or
        (current_episode['digital_use'] == 'Yes' and current_episode['duration'] >= digital_min_duration)):
        episodes.append(current_episode)

    return pd.DataFrame(episodes)

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
        elif S == 1:
            index = 0  # No fragmentation for a single episode
        else:
            index = np.nan  # Not enough data to calculate fragmentation

        fragmentation_indices[f"{category}_index"] = index

    return fragmentation_indices

def calculate_aid(episodes_df, column):
    aid_values = {}

    for category in episodes_df[column].unique():
        category_episodes = episodes_df[episodes_df[column] == category].sort_values('start_time')
        if len(category_episodes) > 1:
            inter_episode_durations = (category_episodes['start_time'].iloc[1:] - category_episodes['end_time'].iloc[:-1]).dt.total_seconds() / 60
            aid = inter_episode_durations.mean()
        else:
            aid = np.nan

        aid_values[f"{category}_AID"] = aid

    return aid_values

def check_data_quality(result):
    issues = []
    if result['total_episodes'] < 2:
        issues.append("Too few episodes")
    if result['total_duration'] < 360:  # Less than 6 hours
        issues.append("Short duration")
    if result['stationary_duration'] / result['total_duration'] > 0.99 or result['mobile_duration'] / result['total_duration'] > 0.99:
        issues.append("Extreme movement imbalance")
    return issues

def analyze_participant_day(file_path):
    try:
        participant_df = pd.read_csv(file_path)
        participant_df['Timestamp'] = pd.to_datetime(participant_df['Timestamp'])
        participant_df['speed'] = pd.to_numeric(participant_df['speed'], errors='coerce')
        participant_df['isapp'] = participant_df['isapp'].astype(bool)
        participant_df['indoors'] = participant_df['indoors'].astype(str)
        participant_df = participant_df.sort_values('Timestamp').reset_index(drop=True)

        if participant_df.empty or 'speed' not in participant_df.columns:
            print(f"Empty DataFrame or missing 'speed' column in {file_path}")
            return None

        episodes_df = detect_episodes(participant_df)

        if episodes_df.empty:
            print(f"No episodes detected in {file_path}")
            return None

        fragmentation_indices = calculate_fragmentation_index(episodes_df, 'movement_type')
        indoor_outdoor_indices = calculate_fragmentation_index(episodes_df, 'indoor_outdoor')
        digital_use_indices = calculate_fragmentation_index(episodes_df, 'digital_use')
        
        aid_values = calculate_aid(episodes_df, 'movement_type')

        result = {
            'participant_id': participant_df['user'].iloc[0],
            'date': participant_df['Timestamp'].dt.date.iloc[0],
            'total_episodes': len(episodes_df),
            'stationary_episodes': len(episodes_df[episodes_df['movement_type'] == 'Stationary']),
            'mobile_episodes': len(episodes_df[episodes_df['movement_type'] == 'Mobile']),
            'total_duration': episodes_df['duration'].sum(),
            'stationary_duration': episodes_df[episodes_df['movement_type'] == 'Stationary']['duration'].sum(),
            'mobile_duration': episodes_df[episodes_df['movement_type'] == 'Mobile']['duration'].sum(),
            **fragmentation_indices,
            **indoor_outdoor_indices,
            **digital_use_indices,
            **aid_values
        }

        result['data_quality_issues'] = check_data_quality(result)
        return result

    except Exception as e:
        print(f"Error processing file {os.path.basename(file_path)}: {str(e)}")
        return None

def main(input_dir, output_dir):
    all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv')]
    print(f"Found {len(all_files)} CSV files in the input directory")

    all_results = []
    problematic_days = []

    for file in tqdm(all_files, desc="Processing files"):
        result = analyze_participant_day(file)
        if result is not None:
            if result['data_quality_issues']:
                problematic_days.append(result)
            else:
                all_results.append(result)

    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv(os.path.join(output_dir, 'fragmentation_daily_summary.csv'), index=False)
        print(f"Daily summary saved to CSV. Generated {len(all_results)} valid results out of {len(all_files)} files.")

        print("\nDays with potential issues (discarded from analysis):")
        for day in problematic_days:
            print(f"Participant {day['participant_id']} on {day['date']}: {', '.join(day['data_quality_issues'])}")

        problematic_percentage = (len(problematic_days) / len(all_files)) * 100
        print(f"\n{problematic_percentage:.2f}% of days have potential data quality issues and were discarded.")
    else:
        print("No valid results were generated. Please check the detailed error messages above.")

    print("\nSummary of all processed data:")
    print(f"Total files processed: {len(all_files)}")
    print(f"Valid results: {len(all_results)}")
    print(f"Problematic days: {len(problematic_days)}")

    if problematic_days:
        issue_counts = {}
        for day in problematic_days:
            for issue in day['data_quality_issues']:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        print("\nBreakdown of data quality issues:")
        for issue, count in issue_counts.items():
            print(f"{issue}: {count}")

if __name__ == "__main__":
    input_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/preprocessed_data'
    output_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/'
    main(input_dir, output_dir)