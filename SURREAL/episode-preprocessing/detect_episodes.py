#!/usr/bin/env python3
"""
Simplified episode detection focusing on movement/digital use states
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime
import traceback

# Configuration
MOVEMENT_CUTOFF = 1.5  # m/s (stationary vs moving)
MIN_EPISODE_DURATION = '30s'  # Pandas-compatible duration string
DIGITAL_USE_COL = 'action'  # Changed from 'foreground'
MAX_SCREEN_GAP = '5min'  # Consider gaps longer than this as screen off

def load_participant_data(participant_id, processed_dir):
    """Load and merge preprocessed GPS + app data"""
    gps_path = os.path.join(processed_dir, f'{participant_id}_qstarz_prep.csv')
    app_path = os.path.join(processed_dir, f'{participant_id}_app_prep.csv')
    
    # Load GPS data with proper datetime parsing
    gps_df = pd.read_csv(gps_path, parse_dates=['UTC DATE TIME'])
    
    # Load app data with timestamp handling
    app_df = pd.read_csv(app_path)
    
    # Handle timestamp column in app data
    time_col = next((col for col in ['Timestamp', 'UTC DATE TIME', 'datetime', 'time'] 
                    if col in app_df.columns), None)
    if time_col:
        app_df = app_df.rename(columns={time_col: 'timestamp'})
        app_df['timestamp'] = pd.to_datetime(app_df['timestamp'])
    else:
        raise ValueError(f"App data for {participant_id} missing timestamp column")

    # Process digital episodes first
    digital_episodes = process_digital_episodes(app_df)
    gps_df.attrs['digital_episodes'] = digital_episodes
    
    # Merge datasets
    merged = pd.merge_asof(
        gps_df.sort_values('UTC DATE TIME'),
        app_df.sort_values('timestamp'),
        left_on='UTC DATE TIME',
        right_on='timestamp',
        direction='nearest',
        tolerance=pd.Timedelta('30s')
    )
    
    return merged

def clean_time_components(df):
    """Clean and standardize time formats before parsing"""
    # Pad single-digit time components
    df['time'] = df['time'].str.replace(
        r'^(\d+):(\d+):(\d+)$',
        lambda m: f"{m[1].zfill(2)}:{m[2].zfill(2)}:{m[3].zfill(2)}",
        regex=True
    )
    return df

def process_digital_episodes(app_df):
    """Convert raw app events to continuous screen on/off episodes"""
    # Filter relevant events
    screen_events = app_df[app_df[DIGITAL_USE_COL].isin(['SCREEN ON', 'SCREEN OFF'])].copy()
    
    # Clean and parse timestamps
    screen_events = clean_time_components(screen_events)
    
    try:
        # Handle European date format (day/month/year)
        screen_events['timestamp'] = pd.to_datetime(
            screen_events['date'] + ' ' + screen_events['time'],
            dayfirst=True,
            format='mixed',
            errors='coerce'
        )
        
        # Drop rows with invalid timestamps
        invalid_count = screen_events['timestamp'].isna().sum()
        if invalid_count > 0:
            print(f"Warning: Dropped {invalid_count} rows with invalid timestamps")
            screen_events = screen_events.dropna(subset=['timestamp'])
            
    except Exception as e:
        print("Critical timestamp parsing error:")
        print(f"Sample dates: {screen_events['date'].unique()[:5]}")
        print(f"Sample times: {screen_events['time'].unique()[:5]}")
        raise

    # Rest of the function remains unchanged...
    screen_events = screen_events.sort_values('timestamp')
    screen_events['time_diff'] = screen_events['timestamp'].diff()
    
    # Insert implicit screen off events for long gaps
    gap_mask = screen_events['time_diff'] > pd.Timedelta(MAX_SCREEN_GAP)
    implicit_offs = screen_events[gap_mask].copy()
    implicit_offs[DIGITAL_USE_COL] = 'SCREEN OFF'
    implicit_offs['timestamp'] = screen_events['timestamp'] + pd.Timedelta(MAX_SCREEN_GAP)
    
    # Combine explicit and implicit events
    all_events = pd.concat([screen_events, implicit_offs]).sort_values('timestamp')
    
    # Create digital episodes
    digital_episodes = []
    current_on = None
    for _, row in all_events.iterrows():
        if row[DIGITAL_USE_COL] == 'SCREEN ON' and not current_on:
            current_on = row['timestamp']
        elif row[DIGITAL_USE_COL] == 'SCREEN OFF' and current_on:
            digital_episodes.append({
                'start': current_on,
                'end': row['timestamp'],
                'digital_use': 'digital'
            })
            current_on = None
    
    return pd.DataFrame(digital_episodes)

def classify_states(df):
    """Classify movement and digital use states"""
    # Movement classification
    df['movement'] = np.where(df['SPEED_MS'] > MOVEMENT_CUTOFF, 'moving', 'stationary')
    
    # Digital use classification
    df['digital_use'] = 'no_digital'
    if 'digital_episodes' in df.attrs:
        for _, episode in df.attrs['digital_episodes'].iterrows():
            mask = (df['UTC DATE TIME'] >= episode['start']) & (df['UTC DATE TIME'] <= episode['end'])
            df.loc[mask, 'digital_use'] = 'digital'
    
    return df

def detect_episodes(df):
    """Identify state change points and create episodes"""
    # Detect changes in either state dimension
    state_change = (
        df['movement'].ne(df['movement'].shift()) |
        df['digital_use'].ne(df['digital_use'].shift())
    )
    
    # Create episode groups
    df['episode_id'] = state_change.cumsum()
    
    # Aggregate episode properties
    episodes = df.groupby('episode_id').agg(
        start_time=('UTC DATE TIME', 'min'),
        end_time=('UTC DATE TIME', 'max'),
        movement=('movement', 'first'),
        digital_use=('digital_use', 'first'),
        latitude=('LATITUDE', 'mean'),
        longitude=('LONGITUDE', 'mean'),
        n_points=('episode_id', 'count')
    )
    
    # Calculate duration and filter short episodes
    episodes['duration'] = episodes['end_time'] - episodes['start_time']
    return episodes[episodes['duration'] >= pd.Timedelta(MIN_EPISODE_DURATION)]

def process_participant(participant_id, processed_dir, output_dir):
    """Full processing pipeline for single participant"""
    print(f"Processing {participant_id}")
    try:
        # Load and prepare data
        df = load_participant_data(participant_id, processed_dir)
        df = classify_states(df)
        
        # Detect and clean episodes
        episodes = detect_episodes(df)
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{participant_id}_episodes.csv')
        episodes.to_csv(output_path, index=False)
        
        print(f"Detected {len(episodes)} episodes for {participant_id}")
        return True
        
    except Exception as e:
        print(f"Error processing {participant_id}: {str(e)}")
        print(traceback.format_exc())
        return False

def validate_data_structure(processed_dir):
    """Check column consistency across files"""
    sample_gps = pd.read_csv(os.path.join(processed_dir, '005_qstarz_preprocessed.csv'), nrows=1)
    sample_app = pd.read_csv(os.path.join(processed_dir, '005_app_preprocessed.csv'), nrows=1)
    
    print("GPS Data Columns:", sample_gps.columns.tolist())
    print("App Data Columns:", sample_app.columns.tolist())
    
    # Check GPS time parsing
    try:
        gps_time = pd.to_datetime(sample_gps['UTC DATE TIME'].iloc[0])
        print("GPS Time parsed successfully:", gps_time)
    except Exception as e:
        print("GPS Time parsing error:", str(e))
    
    # Check app timestamp handling
    time_col = next((col for col in ['Timestamp', 'UTC DATE TIME', 'datetime', 'time'] 
                    if col in sample_app.columns), None)
    if time_col:
        print(f"Found app time column: '{time_col}'")
        sample_app['timestamp'] = pd.to_datetime(sample_app[time_col])
        print("App Time example:", sample_app['timestamp'].iloc[0])
    else:
        print("No valid timestamp column found in app data")

# Main execution
if __name__ == "__main__":
    # Config paths
    base_dir = "/Users/noamgal/Downloads/Research-Projects/SURREAL/HUJI_data-main/"
    processed_dir = os.path.join(base_dir, "Processed", "fragment-processed")
    output_dir = os.path.join(base_dir, "Processed", "episode-outputs")
    
    # Get all participants with both data files
    qstarz_files = set(f.split('_')[0] for f in os.listdir(processed_dir) if '_qstarz_' in f)
    app_files = set(f.split('_')[0] for f in os.listdir(processed_dir) if '_app_' in f)
    participants = list(qstarz_files & app_files)
    
    # Validate data structure
    validate_data_structure(processed_dir) 
    # Process all valid participants
    for pid in participants:
        process_participant(pid, processed_dir, output_dir)
    
    print(f"\nProcessing complete. Outputs saved to: {output_dir}")
