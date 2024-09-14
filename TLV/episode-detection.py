import pandas as pd
import numpy as np
import os
from datetime import timedelta

def detect_mobility_episodes(df, min_episode_duration=5):
    df['mobility'] = np.where(df['Travel_mode'].isin(['Staying', 'nan', 'Missing']), 'Stationary', 'Mobile')
    df = df.dropna(subset=['mobility'])
    
    episodes = []
    current_episode = {'start_time': df['Timestamp'].iloc[0], 'mobility': df['mobility'].iloc[0]}

    for i in range(1, len(df)):
        if df['mobility'].iloc[i] != df['mobility'].iloc[i-1]:
            current_episode['end_time'] = df['Timestamp'].iloc[i-1]
            current_episode['duration'] = (current_episode['end_time'] - current_episode['start_time']).total_seconds() / 60
            
            if current_episode['duration'] >= min_episode_duration:
                episodes.append(current_episode)
            
            current_episode = {'start_time': df['Timestamp'].iloc[i], 'mobility': df['mobility'].iloc[i]}

    # Add the last episode
    current_episode['end_time'] = df['Timestamp'].iloc[-1]
    current_episode['duration'] = (current_episode['end_time'] - current_episode['start_time']).total_seconds() / 60
    if current_episode['duration'] >= min_episode_duration:
        episodes.append(current_episode)

    return pd.DataFrame(episodes)

def detect_indoor_outdoor_episodes(df, min_episode_duration=5):
    df['indoor_outdoor'] = np.where(df['indoors'] == 'True', 'Indoor', 'Outdoor')
    df = df.dropna(subset=['indoor_outdoor'])
    
    episodes = []
    current_episode = {'start_time': df['Timestamp'].iloc[0], 'indoor_outdoor': df['indoor_outdoor'].iloc[0]}

    for i in range(1, len(df)):
        if df['indoor_outdoor'].iloc[i] != df['indoor_outdoor'].iloc[i-1]:
            current_episode['end_time'] = df['Timestamp'].iloc[i-1]
            current_episode['duration'] = (current_episode['end_time'] - current_episode['start_time']).total_seconds() / 60
            
            if current_episode['duration'] >= min_episode_duration:
                episodes.append(current_episode)
            
            current_episode = {'start_time': df['Timestamp'].iloc[i], 'indoor_outdoor': df['indoor_outdoor'].iloc[i]}

    # Add the last episode
    current_episode['end_time'] = df['Timestamp'].iloc[-1]
    current_episode['duration'] = (current_episode['end_time'] - current_episode['start_time']).total_seconds() / 60
    if current_episode['duration'] >= min_episode_duration:
        episodes.append(current_episode)

    return pd.DataFrame(episodes)

def detect_digital_episodes(df, min_episode_duration=0.5):
    df['digital_use'] = np.where(df['type'] == 'No use', 'Non-Digital', 'Digital')
    df = df.dropna(subset=['digital_use'])
    
    episodes = []
    current_episode = {'start_time': df['Timestamp'].iloc[0], 'digital_use': df['digital_use'].iloc[0]}

    for i in range(1, len(df)):
        if df['digital_use'].iloc[i] != df['digital_use'].iloc[i-1]:
            current_episode['end_time'] = df['Timestamp'].iloc[i-1]
            current_episode['duration'] = (current_episode['end_time'] - current_episode['start_time']).total_seconds() / 60
            
            if current_episode['duration'] >= min_episode_duration:
                episodes.append(current_episode)
            
            current_episode = {'start_time': df['Timestamp'].iloc[i], 'digital_use': df['digital_use'].iloc[i]}

    # Add the last episode
    current_episode['end_time'] = df['Timestamp'].iloc[-1]
    current_episode['duration'] = (current_episode['end_time'] - current_episode['start_time']).total_seconds() / 60
    if current_episode['duration'] >= min_episode_duration:
        episodes.append(current_episode)

    return pd.DataFrame(episodes)

def process_participant_day(file_path, output_dir):
    df = pd.read_csv(file_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    mobility_episodes = detect_mobility_episodes(df)
    indoor_outdoor_episodes = detect_indoor_outdoor_episodes(df)
    digital_episodes = detect_digital_episodes(df)
    
    participant_id = df['user'].iloc[0]
    date = df['Timestamp'].dt.date.iloc[0]
    
    mobility_episodes.to_csv(os.path.join(output_dir, f'mobility_episodes_{participant_id}_{date}.csv'), index=False)
    indoor_outdoor_episodes.to_csv(os.path.join(output_dir, f'indoor_outdoor_episodes_{participant_id}_{date}.csv'), index=False)
    digital_episodes.to_csv(os.path.join(output_dir, f'digital_episodes_{participant_id}_{date}.csv'), index=False)

def main(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_dir, filename)
            process_participant_day(file_path, output_dir)

if __name__ == "__main__":
    input_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/preprocessed_data'
    output_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/episodes'
    main(input_dir, output_dir)