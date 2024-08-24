#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 12:07:50 2024

@author: noamgal
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from ruptures import Pelt
from scipy import stats

# Load the data
tlv_path = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/gpsappS_9.1_excel.xlsx'
tlv_df = pd.read_excel(tlv_path, sheet_name='gpsappS_8')
print("Data loaded successfully.")
print(f"Shape of the DataFrame: {tlv_df.shape}")
print("\nColumn names:")
print(tlv_df.columns)

# Create output directories
output_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/'
episode_dir = os.path.join(output_dir, 'fragment-episodes')
os.makedirs(episode_dir, exist_ok=True)

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

def analyze_participant(participant_df):
    try:
        # Preprocess data
        participant_df['Timestamp'] = pd.to_datetime(participant_df['date'] + ' ' + participant_df['time'])
        participant_df['movement_type'] = participant_df['speed'].apply(classify_movement)

        # Detect changepoints
        change_points = detect_changepoints(participant_df, 'speed')

        # Create episodes
        episodes_df = create_episodes(participant_df, change_points)

        # Calculate fragmentation indices
        fragmentation_indices_movement = calculate_fragmentation_index(episodes_df, 'movement_type')
        fragmentation_indices_io = calculate_fragmentation_index(episodes_df, 'indoor_outdoor')
        fragmentation_indices_digital = calculate_fragmentation_index(episodes_df, 'digital_use')

        # Combine all fragmentation indices
        all_indices = {**fragmentation_indices_movement, **fragmentation_indices_io, **fragmentation_indices_digital}

        # Calculate modes
        modes = {
            'movement_mode': stats.mode(participant_df['movement_type'])[0][0],
            'indoor_outdoor_mode': stats.mode(participant_df['indoors'])[0][0],
            'digital_use_mode': 'Yes' if stats.mode(participant_df['isapp'])[0][0] == 1 else 'No'
        }

        return episodes_df, all_indices, modes

    except Exception as e:
        print(f"Error processing participant: {str(e)}")
        return None, None, None

# Analyze all participants
all_participants = tlv_df['user'].unique()
all_results = []

for participant_id in all_participants:
    print(f"Processing participant {participant_id}")
    participant_df = tlv_df[tlv_df['user'] == participant_id]
    
    episodes_df, fragmentation_indices, modes = analyze_participant(participant_df)
    
    if episodes_df is not None and fragmentation_indices is not None and modes is not None:
        result = {'participant_id': participant_id, **fragmentation_indices, **modes}
        all_results.append(result)
        
        # Save episode details
        episodes_df.to_csv(os.path.join(episode_dir, f'participant_{participant_id}_episodes.csv'), index=False)

# Create summary DataFrame
summary_df = pd.DataFrame(all_results)
summary_df.to_csv(os.path.join(output_dir, 'fragmentation_summary.csv'), index=False)

print("\nAnalysis completed. Results saved in the specified directories.")

# Print descriptive statistics
print("\nDescriptive Statistics:")
print(summary_df.describe())

# Visualizations
plt.figure(figsize=(12, 6))
sns.boxplot(data=summary_df[[col for col in summary_df.columns if col.endswith('_index')]])
plt.title('Distribution of Fragmentation Indices')
plt.ylabel('Fragmentation Index')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fragmentation_indices_distribution.png'))
plt.close()

print("\nVisualization saved as 'fragmentation_indices_distribution.png' in the output directory.")