import pandas as pd
import numpy as np
import os
from datetime import timedelta
from collections import defaultdict

def accumulate_value_counts(df, column, accumulator):
    value_counts = df[column].value_counts(dropna=False)
    for value, count in value_counts.items():
        accumulator[column][value] += count

def detect_mobility_episodes(df, min_episode_duration=5, accumulator=None):
    df['mobility'] = np.where(df['Travel_mode'].isin(['AT', 'PT', 'Walking']), 'Moving', 
                              np.where(df['Travel_mode'] == 'Staying', 'Stationary', np.nan))
    
    dropped_values = df[df['mobility'].isna()]['Travel_mode'].value_counts()
    df_mobility = df.dropna(subset=['mobility'])
    
    if accumulator is not None:
        accumulate_value_counts(df, 'Travel_mode', accumulator['before'])
        accumulate_value_counts(df_mobility, 'mobility', accumulator['after'])
    
    episodes = []
    current_episode = {'start_time': df_mobility['Timestamp'].iloc[0], 'mobility': df_mobility['mobility'].iloc[0]}

    for i in range(1, len(df_mobility)):
        if df_mobility['mobility'].iloc[i] != df_mobility['mobility'].iloc[i-1]:
            current_episode['end_time'] = df_mobility['Timestamp'].iloc[i-1]
            current_episode['duration'] = (current_episode['end_time'] - current_episode['start_time']).total_seconds() / 60
            
            if current_episode['duration'] >= min_episode_duration:
                episodes.append(current_episode)
            
            current_episode = {'start_time': df_mobility['Timestamp'].iloc[i], 'mobility': df_mobility['mobility'].iloc[i]}

    # Add the last episode
    current_episode['end_time'] = df_mobility['Timestamp'].iloc[-1]
    current_episode['duration'] = (current_episode['end_time'] - current_episode['start_time']).total_seconds() / 60
    if current_episode['duration'] >= min_episode_duration:
        episodes.append(current_episode)

    return pd.DataFrame(episodes), dropped_values

def detect_indoor_outdoor_episodes(df, min_episode_duration=5, accumulator=None):
    # Convert all variations of True/False to boolean
    df['indoors'] = df['indoors'].map({'True': True, 'False': False, True: True, False: False})
    
    df['indoor_outdoor'] = np.where(df['indoors'] == True, 'Indoor', 
                                    np.where(df['indoors'] == False, 'Outdoor', np.nan))
    
    dropped_values = df[df['indoor_outdoor'].isna()]['indoors'].value_counts()
    df_io = df.dropna(subset=['indoor_outdoor'])
    
    if accumulator is not None:
        accumulate_value_counts(df, 'indoors', accumulator['before'])
        accumulate_value_counts(df_io, 'indoor_outdoor', accumulator['after'])
    
    episodes = []
    current_episode = {'start_time': df_io['Timestamp'].iloc[0], 'indoor_outdoor': df_io['indoor_outdoor'].iloc[0]}

    for i in range(1, len(df_io)):
        if df_io['indoor_outdoor'].iloc[i] != df_io['indoor_outdoor'].iloc[i-1]:
            current_episode['end_time'] = df_io['Timestamp'].iloc[i-1]
            current_episode['duration'] = (current_episode['end_time'] - current_episode['start_time']).total_seconds() / 60
            
            if current_episode['duration'] >= min_episode_duration:
                episodes.append(current_episode)
            
            current_episode = {'start_time': df_io['Timestamp'].iloc[i], 'indoor_outdoor': df_io['indoor_outdoor'].iloc[i]}

    # Add the last episode
    current_episode['end_time'] = df_io['Timestamp'].iloc[-1]
    current_episode['duration'] = (current_episode['end_time'] - current_episode['start_time']).total_seconds() / 60
    if current_episode['duration'] >= min_episode_duration:
        episodes.append(current_episode)

    return pd.DataFrame(episodes), dropped_values

def detect_digital_episodes(df, min_episode_duration=0.5, accumulator=None):
    digital_categories = ['Social', 'Productive', 'Process', 'Settings', 'School', 'Spatial', 'Screen on/off/lock']
    
    df['digital_use'] = np.where(df['type'].isin(digital_categories), 'Digital',
                                 np.where(df['type'] == 'No use', 'Non-Digital', np.nan))
    
    dropped_values = df[df['digital_use'].isna()]['type'].value_counts()
    df_digital = df.dropna(subset=['digital_use'])
    
    if accumulator is not None:
        accumulate_value_counts(df, 'type', accumulator['before'])
        accumulate_value_counts(df_digital, 'digital_use', accumulator['after'])
    
    episodes = []
    current_episode = {'start_time': df_digital['Timestamp'].iloc[0], 'digital_use': df_digital['digital_use'].iloc[0]}

    for i in range(1, len(df_digital)):
        if df_digital['digital_use'].iloc[i] != df_digital['digital_use'].iloc[i-1]:
            current_episode['end_time'] = df_digital['Timestamp'].iloc[i-1]
            current_episode['duration'] = (current_episode['end_time'] - current_episode['start_time']).total_seconds() / 60
            
            if current_episode['duration'] >= min_episode_duration:
                episodes.append(current_episode)
            
            current_episode = {'start_time': df_digital['Timestamp'].iloc[i], 'digital_use': df_digital['digital_use'].iloc[i]}

    # Add the last episode
    current_episode['end_time'] = df_digital['Timestamp'].iloc[-1]
    current_episode['duration'] = (current_episode['end_time'] - current_episode['start_time']).total_seconds() / 60
    if current_episode['duration'] >= min_episode_duration:
        episodes.append(current_episode)

    return pd.DataFrame(episodes), dropped_values

def process_participant_day(file_path, output_dir, accumulators):
    df = pd.read_csv(file_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    mobility_episodes, mobility_dropped = detect_mobility_episodes(df, accumulator=accumulators['mobility'])
    indoor_outdoor_episodes, indoor_outdoor_dropped = detect_indoor_outdoor_episodes(df, accumulator=accumulators['indoor_outdoor'])
    digital_episodes, digital_dropped = detect_digital_episodes(df, accumulator=accumulators['digital'])
    
    participant_id = df['user'].iloc[0]
    date = df['Timestamp'].dt.date.iloc[0]
    
    mobility_episodes.to_csv(os.path.join(output_dir, f'mobility_episodes_{participant_id}_{date}.csv'), index=False)
    indoor_outdoor_episodes.to_csv(os.path.join(output_dir, f'indoor_outdoor_episodes_{participant_id}_{date}.csv'), index=False)
    digital_episodes.to_csv(os.path.join(output_dir, f'digital_episodes_{participant_id}_{date}.csv'), index=False)

    # Calculate total duration for each type of episode
    day_start = df['Timestamp'].min()
    day_end = df['Timestamp'].max()
    total_duration = (day_end - day_start).total_seconds() / 3600  # in hours

    return {
        'participant_id': participant_id,
        'date': date,
        'mobility_dropped': mobility_dropped,
        'indoor_outdoor_dropped': indoor_outdoor_dropped,
        'digital_dropped': digital_dropped,
        'mobility_episodes': len(mobility_episodes),
        'indoor_outdoor_episodes': len(indoor_outdoor_episodes),
        'digital_episodes': len(digital_episodes),
        'total_duration': total_duration,
        'indoor_outdoor_data': indoor_outdoor_episodes
    }

def main(input_dir, output_dir):
    all_dropped_values = defaultdict(lambda: defaultdict(int))
    single_episode_days = defaultdict(list)
    
    accumulators = {
        'mobility': {'before': defaultdict(lambda: defaultdict(int)), 'after': defaultdict(lambda: defaultdict(int))},
        'indoor_outdoor': {'before': defaultdict(lambda: defaultdict(int)), 'after': defaultdict(lambda: defaultdict(int))},
        'digital': {'before': defaultdict(lambda: defaultdict(int)), 'after': defaultdict(lambda: defaultdict(int))}
    }
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_dir, filename)
            result = process_participant_day(file_path, output_dir, accumulators)
            
            # Accumulate dropped values
            for feature, dropped in result['mobility_dropped'].items():
                all_dropped_values['mobility'][feature] += dropped
            for feature, dropped in result['indoor_outdoor_dropped'].items():
                all_dropped_values['indoor_outdoor'][feature] += dropped
            for feature, dropped in result['digital_dropped'].items():
                all_dropped_values['digital'][feature] += dropped
            
            # Check for single-episode days
            if result['mobility_episodes'] == 1:
                single_episode_days['mobility'].append((result['participant_id'], result['date'], result['total_duration']))
            if result['indoor_outdoor_episodes'] == 1:
                single_episode_days['indoor_outdoor'].append((result['participant_id'], result['date'], result['total_duration'], result['indoor_outdoor_data']))
            if result['digital_episodes'] == 1:
                single_episode_days['digital'].append((result['participant_id'], result['date'], result['total_duration']))

    # Print accumulated value counts
    for category in ['mobility', 'indoor_outdoor', 'digital']:
        print(f"\n{category.capitalize()} data:")
        print("Before processing:")
        for column, counts in accumulators[category]['before'].items():
            print(f"\n{column}:")
            for value, count in counts.items():
                print(f"  {value}: {count}")
        print("\nAfter processing:")
        for column, counts in accumulators[category]['after'].items():
            print(f"\n{column}:")
            for value, count in counts.items():
                print(f"  {value}: {count}")

    # Report dropped values
    print("\nDropped values:")
    for category, values in all_dropped_values.items():
        print(f"\n{category.capitalize()}:")
        for feature, count in values.items():
            print(f"  {feature}: {count}")

    # Report single-episode days with duration
    print("\nSingle-episode days:")
    for category, days in single_episode_days.items():
        print(f"\n{category.capitalize()}: {len(days)}")
        if category == 'indoor_outdoor':
            for day in days:
                participant_id, date, duration, episode_data = day
                episode_type = episode_data['indoor_outdoor'].iloc[0] if not episode_data.empty else "Unknown"
                print(f"  Participant: {participant_id}, Date: {date}, Duration: {duration:.2f} hours, Type: {episode_type}")
        else:
            for day in days:
                participant_id, date, duration = day
                print(f"  Participant: {participant_id}, Date: {date}, Duration: {duration:.2f} hours")

if __name__ == "__main__":
    input_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/preprocessed_data'
    output_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/episodes'
    main(input_dir, output_dir)