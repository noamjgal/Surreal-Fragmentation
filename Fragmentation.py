#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import MultiPoint
from ruptures import Pelt
from datetime import datetime, timedelta
import traceback

# Set the base directory
base_dir = "/Users/noamgal/Downloads/Research-Projects/SURREAL/HUJI_data-main/"
processed_dir = os.path.join(base_dir, "Processed", "fragment-processed")
output_dir = os.path.join(processed_dir, "fragmentation-outputs")
os.makedirs(output_dir, exist_ok=True)

def load_preprocessed_data(participant_id):
    qstarz_file = os.path.join(processed_dir, f'{participant_id}_qstarz_preprocessed.csv')
    qstarz_df = pd.read_csv(qstarz_file, parse_dates=['UTC DATE TIME'])
    return qstarz_df

def classify_movement(speed):
    if speed < 1.5:
        return 'Stationary'
    elif speed < 7:
        return 'Active Transport'
    else:
        return 'Mechanized Transport'

def classify_io_status(nsat_used, window_size=5, threshold=7):
    """
    Classify indoor/outdoor status using a rolling window approach.
    """
    rolling_mean = pd.Series(nsat_used).rolling(window=window_size, center=True).mean()
    initial_classification = (rolling_mean >= threshold).astype(int)
    
    # Smooth out brief changes
    smoothed = initial_classification.rolling(window=3, center=True).median().fillna(method='ffill').fillna(method='bfill')
    
    return np.where(smoothed == 1, 'Outdoor', 'Indoor')

def create_and_classify_episodes(qstarz_df, min_size=5, jump=5, pen=1):
    speed_data = qstarz_df['SPEED_MS'].values
    nsat_data = qstarz_df['NSAT_USED'].values
    timestamps = qstarz_df['UTC DATE TIME'].values

    if len(speed_data) < min_size:
        print(f"  Warning: Not enough data points ({len(speed_data)}) to create episodes. Minimum required: {min_size}")
        return pd.DataFrame()

    model = Pelt(model="rbf", jump=jump, min_size=min_size).fit(speed_data.reshape(-1, 1))
    change_points = model.predict(pen=pen)

    io_status = classify_io_status(nsat_data)

    episodes = []
    for i in range(len(change_points) - 1):
        start_idx = change_points[i]
        end_idx = change_points[i + 1]
        
        start_time = pd.Timestamp(timestamps[start_idx]).to_pydatetime()
        end_time = pd.Timestamp(timestamps[end_idx - 1]).to_pydatetime()
        
        avg_speed = np.mean(speed_data[start_idx:end_idx])
        avg_nsat = np.mean(nsat_data[start_idx:end_idx])
        
        movement_type = classify_movement(avg_speed)
        io_status_mode = pd.Series(io_status[start_idx:end_idx]).mode().iloc[0]
        
        episode = {
            'start_time': start_time,
            'end_time': end_time,
            'avg_speed': avg_speed,
            'MOVEMENT_TYPE': movement_type,
            'IO_STATUS': io_status_mode,
            'points': list(zip(qstarz_df['LATITUDE'].iloc[start_idx:end_idx], qstarz_df['LONGITUDE'].iloc[start_idx:end_idx])),
            'avg_nsat_used': avg_nsat
        }
        
        episode['duration'] = (episode['end_time'] - episode['start_time']).total_seconds() / 60
        
        if len(episode['points']) > 1:
            coords = MultiPoint(episode['points'])
            episode['centroid'] = coords.centroid.coords[0]
        else:
            episode['centroid'] = episode['points'][0]
        
        episodes.append(episode)
    
    # Merge consecutive episodes with the same movement type and IO status
    merged_episodes = []
    for episode in episodes:
        if not merged_episodes or \
           episode['MOVEMENT_TYPE'] != merged_episodes[-1]['MOVEMENT_TYPE'] or \
           episode['IO_STATUS'] != merged_episodes[-1]['IO_STATUS']:
            merged_episodes.append(episode)
        else:
            merged_episodes[-1]['end_time'] = episode['end_time']
            merged_episodes[-1]['duration'] += episode['duration']
            merged_episodes[-1]['points'].extend(episode['points'])
            merged_episodes[-1]['avg_speed'] = np.mean([merged_episodes[-1]['avg_speed'], episode['avg_speed']])
            merged_episodes[-1]['avg_nsat_used'] = np.mean([merged_episodes[-1]['avg_nsat_used'], episode['avg_nsat_used']])
    
    return pd.DataFrame(merged_episodes)

def calculate_fragmentation_index(episodes_df, column):
    episodes_df['duration_minutes'] = episodes_df['duration']
    total_duration = episodes_df['duration_minutes'].sum()
    
    episode_counts = episodes_df[column].value_counts()
    fragmentation_indices = {}
    
    for category in episode_counts.index:
        category_episodes = episodes_df[episodes_df[column] == category]
        S = len(category_episodes)
        T = category_episodes['duration_minutes'].sum()
        
        if S > 1:
            index = (1 - sum((category_episodes['duration_minutes'] / T) ** 2)) / (1 - (1 / S))
        else:
            index = 0
        
        fragmentation_indices[f"{category}_index"] = index
    
    return fragmentation_indices

def calculate_inter_episode_duration(episodes_df):
    episodes_df = episodes_df.sort_values('start_time')
    inter_episode_durations = {}
    
    for category in ['MOVEMENT_TYPE', 'IO_STATUS']:
        durations = []
        for type in episodes_df[category].unique():
            type_episodes = episodes_df[episodes_df[category] == type]
            if len(type_episodes) > 1:
                diff = type_episodes['start_time'].diff().dropna()
                durations.extend(diff.dt.total_seconds() / 60)
        
        if durations:
            inter_episode_durations[f'{category}_inter_episode_duration'] = np.mean(durations)
        else:
            inter_episode_durations[f'{category}_inter_episode_duration'] = np.nan
    
    return inter_episode_durations

def analyze_participant_daily(participant_id):
    print(f"\nAnalyzing participant {participant_id}...")
    try:
        qstarz_df = load_preprocessed_data(participant_id)
    except Exception as e:
        print(f"Error loading data for participant {participant_id}: {str(e)}")
        return []

    # Create participant subfolder
    participant_dir = os.path.join(output_dir, participant_id)
    os.makedirs(participant_dir, exist_ok=True)
    
    # Group data by date
    qstarz_df['date'] = qstarz_df['UTC DATE TIME'].dt.date
    grouped = qstarz_df.groupby('date')
    
    daily_results = []
    
    for date, day_data in grouped:
        print(f"Processing date: {date}")
        print(f"  Data range: {day_data['UTC DATE TIME'].min()} to {day_data['UTC DATE TIME'].max()}")
        print(f"  Total data points: {len(day_data)}")
        
        try:
            if day_data.empty:
                print(f"  Warning: No data for date {date}")
                continue

            episodes_df = create_and_classify_episodes(day_data, pen=0.5)
            
            if len(episodes_df) == 0:
                print(f"  Warning: No episodes detected for date {date}")
                continue
            
            print(f"  Total episodes detected: {len(episodes_df)}")
            print(f"  Episode types:")
            print(episodes_df['MOVEMENT_TYPE'].value_counts())
            print(f"  IO Status:")
            print(episodes_df['IO_STATUS'].value_counts())
            
            fragmentation_indices_movement = calculate_fragmentation_index(episodes_df, 'MOVEMENT_TYPE')
            fragmentation_indices_io = calculate_fragmentation_index(episodes_df, 'IO_STATUS')
            
            inter_episode_durations = calculate_inter_episode_duration(episodes_df)
            
            daily_summary = {
                'date': date,
                'total_episodes': len(episodes_df),
                'avg_episode_duration': episodes_df['duration'].mean(),
                'stationary_episodes': len(episodes_df[episodes_df['MOVEMENT_TYPE'] == 'Stationary']),
                'active_transport_episodes': len(episodes_df[episodes_df['MOVEMENT_TYPE'] == 'Active Transport']),
                'mechanized_transport_episodes': len(episodes_df[episodes_df['MOVEMENT_TYPE'] == 'Mechanized Transport']),
                'indoor_episodes': len(episodes_df[episodes_df['IO_STATUS'] == 'Indoor']),
                'outdoor_episodes': len(episodes_df[episodes_df['IO_STATUS'] == 'Outdoor']),
                'total_time_hours': (day_data['UTC DATE TIME'].max() - day_data['UTC DATE TIME'].min()).total_seconds() / 3600,
                'time_stationary_hours': episodes_df[episodes_df['MOVEMENT_TYPE'] == 'Stationary']['duration'].sum() / 60,
                'time_active_transport_hours': episodes_df[episodes_df['MOVEMENT_TYPE'] == 'Active Transport']['duration'].sum() / 60,
                'time_mechanized_transport_hours': episodes_df[episodes_df['MOVEMENT_TYPE'] == 'Mechanized Transport']['duration'].sum() / 60,
                'time_indoor_hours': episodes_df[episodes_df['IO_STATUS'] == 'Indoor']['duration'].sum() / 60,
                'time_outdoor_hours': episodes_df[episodes_df['IO_STATUS'] == 'Outdoor']['duration'].sum() / 60,
            }
            daily_summary.update(fragmentation_indices_movement)
            daily_summary.update(fragmentation_indices_io)
            daily_summary.update(inter_episode_durations)
            
            daily_results.append(daily_summary)
            
            # Create detailed CSV output for each day
            detailed_df = pd.DataFrame()
            for _, episode in episodes_df.iterrows():
                episode_data = day_data[(day_data['UTC DATE TIME'] >= episode['start_time']) & 
                                        (day_data['UTC DATE TIME'] <= episode['end_time'])].copy()
                episode_data['MOVEMENT_TYPE'] = episode['MOVEMENT_TYPE']
                episode_data['Movement_duration'] = episode['duration']
                episode_data['IO_STATUS'] = episode['IO_STATUS']
                episode_data['io_duration'] = episode['duration']
                detailed_df = pd.concat([detailed_df, episode_data])
            
            detailed_df = detailed_df.sort_values('UTC DATE TIME')
            detailed_df['event_type'] = np.where(detailed_df['MOVEMENT_TYPE'].ne(detailed_df['MOVEMENT_TYPE'].shift()), 'Movement Change', 
                                                 np.where(detailed_df['IO_STATUS'].ne(detailed_df['IO_STATUS'].shift()), 'IO Change', 'No Change'))
            detailed_df['event_details'] = detailed_df['MOVEMENT_TYPE'] + ' - ' + detailed_df['IO_STATUS']
            
            output_columns = ['UTC DATE TIME', 'event_type', 'event_details', 'MOVEMENT_TYPE', 'Movement_duration', 
                              'IO_STATUS', 'io_duration', 'SPEED_MS', 'LATITUDE', 'LONGITUDE', 'NSAT_USED']
            detailed_df[output_columns].to_csv(os.path.join(participant_dir, f'{date}_fragmentation_output.csv'), index=False)
            
            print(f"  Completed processing for date {date}")
            print(f"  Summary:")
            print(f"    Total time: {daily_summary['total_time_hours']:.2f} hours")
            print(f"    Time stationary: {daily_summary['time_stationary_hours']:.2f} hours")
            print(f"    Time active transport: {daily_summary['time_active_transport_hours']:.2f} hours")
            print(f"    Time mechanized transport: {daily_summary['time_mechanized_transport_hours']:.2f} hours")
            print(f"    Time indoor: {daily_summary['time_indoor_hours']:.2f} hours")
            print(f"    Time outdoor: {daily_summary['time_outdoor_hours']:.2f} hours")
            print(f"    Avg inter-episode duration (Movement): {daily_summary['MOVEMENT_TYPE_inter_episode_duration']:.2f} minutes")
            print(f"    Avg inter-episode duration (IO): {daily_summary['IO_STATUS_inter_episode_duration']:.2f} minutes")
        except Exception as e:
            print(f"  Error processing date {date}:")
            print(traceback.format_exc())
            continue
    
    # Create summary CSV for all days
    if daily_results:
        summary_df = pd.DataFrame(daily_results)
        summary_df.to_csv(os.path.join(participant_dir, f'{participant_id}_daily_summary.csv'), index=False)
        
        # Print summary statistics
        print(f"\nSummary for participant {participant_id}:")
        print(summary_df.describe())
        
        # Print average fragmentation indices
        fragmentation_columns = [col for col in summary_df.columns if col.endswith('_index')]
        print("\nAverage Fragmentation Indices:")
        print(summary_df[fragmentation_columns].mean())
    else:
        print(f"Warning: No valid daily results for participant {participant_id}")
    
    return daily_results



# Analyze all participants
all_participants = [os.path.basename(f).split('_')[0] for f in glob.glob(os.path.join(processed_dir, '*_qstarz_preprocessed.csv'))]

all_results = {}
for i, participant_id in enumerate(all_participants, 1):
    print(f"\nProcessing participant {i} of {len(all_participants)}: {participant_id}")
    daily_results = analyze_participant_daily(participant_id)
    all_results[participant_id] = daily_results
    print(f"Completed analysis for participant {participant_id}")

# Overall summary
print("\nOverall Summary:")
all_days = pd.concat([pd.DataFrame(result) for result in all_results.values()])

print("\nAverage daily statistics across all participants:")
numeric_columns = all_days.select_dtypes(include=[np.number]).columns
print(all_days[numeric_columns].mean())

# Visualizations
os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)

# 1. Distribution of daily episode counts by mobility type
plt.figure(figsize=(12, 6))
sns.boxplot(data=all_days[['stationary_episodes', 'active_transport_episodes', 'mechanized_transport_episodes']])
plt.title('Distribution of Daily Episode Counts by Mobility Type')
plt.ylabel('Number of Episodes')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "visualizations", 'daily_episode_counts_distribution.png'))
plt.close()

# 2. Distribution of daily fragmentation indices
plt.figure(figsize=(12, 6))
sns.boxplot(data=all_days[['Stationary_index', 'Active Transport_index', 'Mechanized Transport_index', 'Indoor_index', 'Outdoor_index']])
plt.title('Distribution of Daily Fragmentation Indices')
plt.ylabel('Fragmentation Index')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "visualizations", 'daily_fragmentation_indices_distribution.png'))
plt.close()

print("\nFragmentation analysis completed. Visualizations saved in the 'fragmentation-outputs/visualizations' folder.")
print(f"Daily CSV outputs for each participant saved in their respective subfolders within the '{output_dir}' folder.")