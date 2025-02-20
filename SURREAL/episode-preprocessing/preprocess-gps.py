#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 11:48:11 2024

@author: noamgal
"""

import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from pathlib import Path
import sys
import logging
from datetime import datetime

# Add parent directory to path to find config
sys.path.append(str(Path(__file__).parent.parent))
from config.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR, MAP_OUTPUT_DIR, GPS_PREP_DIR

# Set the base directory
base_dir = RAW_DATA_DIR
processed_dir = PROCESSED_DATA_DIR
participants_dir = RAW_DATA_DIR / "Participants"

# Initialize quality report metrics after imports
quality_report = {
    'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'participants_processed': 0,
    'missing_files': [],
    'encoding_issues': [],
    'date_errors': [],
    'failed_merges': [],
    'successful_participants': []
}

def load_and_preprocess_qstarz(file_path):
    df = pd.read_csv(file_path)
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Check for different possible column names
    datetime_col = next((col for col in df.columns if 'DATE TIME' in col), None)
    speed_col = next((col for col in df.columns if 'SPEED' in col), None)
    nsat_col = next((col for col in df.columns if 'NSAT' in col), None)
    
    if datetime_col is None or speed_col is None or nsat_col is None:
        raise ValueError(f"Required columns not found in {file_path}")
    
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df['SPEED_MS'] = df[speed_col].apply(lambda x: float(x.split()[0]) if isinstance(x, str) else x) * 1000 / 3600
    df['NSAT_USED'] = df[nsat_col].apply(lambda x: int(x.split('/')[0]))
    
    return df.sort_values(datetime_col)

def process_app_and_gps_data(app_file, gps_file):
    try:
        # Modified CSV reading with delimiter detection
        encodings = ['utf-8', 'latin-1', 'utf-16']
        for encoding in encodings:
            try:
                # First try with semicolon delimiter
                app_df = pd.read_csv(app_file, encoding=encoding, delimiter=';')
                gps_df = pd.read_csv(gps_file, encoding=encoding, delimiter=';')
                
                # Fallback to comma if only 1 column found
                if len(app_df.columns) == 1:
                    app_df = pd.read_csv(app_file, encoding=encoding, delimiter=',')
                if len(gps_df.columns) == 1:
                    gps_df = pd.read_csv(gps_file, encoding=encoding, delimiter=',')
                break
            except UnicodeDecodeError as e:
                if encoding == encodings[-1]:  # Last encoding failed
                    quality_report['encoding_issues'].append({
                        'participant': os.path.basename(app_file).split('-')[0],
                        'error': str(e)
                    })
                    raise
                continue
                
        # Add column cleanup for semicolon-separated headers
        app_df.columns = app_df.columns.str.replace(';', '').str.strip().str.lower()
        gps_df.columns = gps_df.columns.str.replace(';', '').str.strip().str.lower()

        # Add debug logging for date columns
        print(f"\nColumns in app data: {app_df.columns.tolist()}")
        print(f"Sample app data:\n{app_df.head(1)}")
        print(f"\nColumns in GPS data: {gps_df.columns.tolist()}")
        print(f"Sample GPS data:\n{gps_df.head(1)}")

        # Enhanced date column detection
        def find_datetime_columns(df):
            date_candidates = ['date', 'timestamp', 'datetime']
            time_candidates = ['time', 'timestamp', 'datetime']
            
            date_col = next((col for col in df.columns if col in date_candidates), 'date')
            time_col = next((col for col in df.columns if col in time_candidates), 'time')
            return date_col, time_col

        # Modified date parsing with column detection
        def parse_datetime_with_debug(df):
            date_col, time_col = find_datetime_columns(df)
            
            # Validate columns exist
            if date_col not in df.columns:
                raise KeyError(f"Missing date column. Found columns: {df.columns.tolist()}")
            if time_col not in df.columns:
                raise KeyError(f"Missing time column. Found columns: {df.columns.tolist()}")

            # Check for null values
            null_dates = df[date_col].isna().sum()
            null_times = df[time_col].isna().sum()
            if null_dates > 0 or null_times > 0:
                print(f"Warning: Found {null_dates} null dates and {null_times} null times")

            # Convert to string if necessary
            df[date_col] = df[date_col].astype(str)
            df[time_col] = df[time_col].astype(str)
            
            # Log sample date-time strings for debugging
            sample_combos = df[[date_col, time_col]].head(3).apply(lambda x: f"{x[0]} {x[1]}", axis=1).tolist()
            print(f"Sample date-time strings: {sample_combos}")
            
            # Store sample_combos in dataframe for error reporting
            df.attrs['debug_samples'] = sample_combos[:3]
            return pd.to_datetime(df[date_col] + ' ' + df[time_col], errors='coerce')

        # Parse timestamps with enhanced detection
        app_df['Timestamp'] = parse_datetime_with_debug(app_df)
        gps_df['Timestamp'] = parse_datetime_with_debug(gps_df)

        # Log parsing success rates
        app_na = app_df['Timestamp'].isna().sum()
        gps_na = gps_df['Timestamp'].isna().sum()
        print(f"Failed to parse {app_na}/{len(app_df)} app timestamps")
        print(f"Failed to parse {gps_na}/{len(gps_df)} GPS timestamps")

        # Drop rows with NaT values
        app_df = app_df.dropna(subset=['Timestamp'])
        gps_df = gps_df.dropna(subset=['Timestamp'])
        
        # Merge data
        merged_df = pd.merge_asof(app_df.sort_values('Timestamp'), 
                                  gps_df[['Timestamp', 'long', 'lat', 'accuracy', 'provider']].sort_values('Timestamp'), 
                                  on='Timestamp', 
                                  direction='nearest', 
                                  tolerance=pd.Timedelta('1min'))
        
        return merged_df

    except Exception as e:
        if 'date' in str(e):
            quality_report['date_errors'].append({
                'file': app_file,
                'error': str(e),
                'sample_dates': app_df.attrs.get('debug_samples', [])  # Get from dataframe attributes
            })
        raise

def save_preprocessed_data(participant_id, qstarz_df, app_df, processed_dir):
    # Use configured GPS prep directory
    qstarz_file = GPS_PREP_DIR / f'{participant_id}_qstarz_prep.csv'
    app_file = GPS_PREP_DIR / f'{participant_id}_app_prep.csv'
    
    qstarz_df.to_csv(qstarz_file, index=False)
    app_df.to_csv(app_file, index=False)
    print(f"Preprocessed data saved in {GPS_PREP_DIR}")

# Process Qstarz data
all_qstarz_data = {}
for qstarz_file in glob.glob(os.path.join(processed_dir, "*_1_Qstarz_processed.csv")):
    participant_id = os.path.basename(qstarz_file).split('_')[0]
    print(f"\nProcessing Qstarz data for Participant {participant_id}")
    try:
        qstarz_df = load_and_preprocess_qstarz(qstarz_file)
        all_qstarz_data[participant_id] = qstarz_df
        print(f"Successfully processed Qstarz data for Participant {participant_id}")
        quality_report['successful_participants'].append(participant_id)
    except Exception as e:
        print(f"Error processing Qstarz data for Participant {participant_id}: {str(e)}")
        quality_report['failed_processing'].append(participant_id)

# Process digital environment data
all_app_data = {}
for participant_folder in glob.glob(os.path.join(participants_dir, 'P*')):
    folder_name = os.path.basename(participant_folder)
    participant_id = folder_name.split('_')[-1]  # Get the last part after underscore
    app_folder = os.path.join(participant_folder, '9 - Smartphone Tracking App')
    
    print(f"Looking for app files in: {app_folder}")
    print(f"Files in directory: {os.listdir(app_folder)}")
    app_files = glob.glob(os.path.join(app_folder, '*-apps.csv'))
    print(f"Found app files: {app_files}")
    
    gps_files = glob.glob(os.path.join(app_folder, '*-gps.csv'))
    
    if app_files and gps_files:
        app_file = app_files[0]
        gps_file = gps_files[0]
        
        print(f"\nProcessing digital environment data for Participant {participant_id}")
        try:
            participant_data = process_app_and_gps_data(app_file, gps_file)
            if not participant_data.empty:
                all_app_data[participant_id] = participant_data
                print(f"Successfully processed digital environment data for Participant {participant_id}")
            else:
                print(f"No valid merged data for Participant {participant_id}")
        except Exception as e:
            print(f"Error processing digital environment data for Participant {participant_id}: {str(e)}")
    else:
        print(f"Missing digital environment data files for Participant {participant_id}")
        quality_report['missing_files'].append({
            'participant': participant_id,
            'missing_files': {'apps': not bool(app_files), 'gps': not bool(gps_files)}
        })

# Find common participants
common_participants = set(all_qstarz_data.keys()) & set(all_app_data.keys())

print(f"\nNumber of participants with both Qstarz and digital environment data: {len(common_participants)}")
print("Participant IDs:", common_participants)

def create_map(data, filename, title, fallback_data=None):
    # Determine column names for latitude and longitude
    lat_col = 'lat' if 'lat' in data.columns else 'LATITUDE'
    lon_col = 'long' if 'long' in data.columns else 'LONGITUDE'

    # Filter out NaN values
    valid_data = data.dropna(subset=[lat_col, lon_col])

    if valid_data.empty and fallback_data is not None:
        print(f"Warning: No valid GPS data found. Falling back to Qstarz data for {title}")
        fallback_lat_col = 'LATITUDE' if 'LATITUDE' in fallback_data.columns else 'lat'
        fallback_lon_col = 'LONGITUDE' if 'LONGITUDE' in fallback_data.columns else 'long'
        valid_data = fallback_data.dropna(subset=[fallback_lat_col, fallback_lon_col])
        lat_col, lon_col = fallback_lat_col, fallback_lon_col

    if valid_data.empty:
        print(f"Error: No valid location data found for {title}. Skipping map creation.")
        return

    # Create a map centered on the mean latitude and longitude
    center_lat = valid_data[lat_col].mean()
    center_lon = valid_data[lon_col].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    # Add a heatmap layer
    heat_data = [[row[lat_col], row[lon_col]] for _, row in valid_data.iterrows()]
    HeatMap(heat_data).add_to(m)

    # Add a title to the map
    title_html = f'<h3 align="center" style="font-size:16px"><b>{title}</b></h3>'
    m.get_root().html.add_child(folium.Element(title_html))

    # Save the map
    m.save(filename)
    print(f"Map created: {filename}")

# Combine data for common participants and save preprocessed data
combined_data = {}
for participant_id in common_participants:
    qstarz_data = all_qstarz_data[participant_id]
    app_data = all_app_data[participant_id]
    
    # Add participant ID to both dataframes
    qstarz_data['Participant ID'] = participant_id
    app_data['Participant ID'] = participant_id
    
    # Save preprocessed data
    save_preprocessed_data(participant_id, qstarz_data, app_data, processed_dir)

    # Create maps using configured output directory
    create_map(qstarz_data, 
               MAP_OUTPUT_DIR / f'{participant_id}_qstarz_map.html',
               f'Participant {participant_id} - Qstarz Data')

    create_map(app_data, 
               MAP_OUTPUT_DIR / f'{participant_id}_app_gps_map.html',
               f'Participant {participant_id} - App GPS Data',
               fallback_data=qstarz_data)

    print(f"Maps created for Participant {participant_id}")

print("\nPreprocessing completed. Preprocessed data and maps saved in the 'fragment-processed' and 'maps' folders.")

# Add this function before the final print statements
def generate_quality_report(processed_dir):
    report_path = processed_dir / "preprocessing_quality_report.txt"
    with open(report_path, 'w') as f:
        f.write(f"Preprocessing Quality Report ({quality_report['start_time']})\n")
        f.write("="*50 + "\n")
        f.write(f"Participants successfully processed: {len(quality_report['successful_participants'])}\n")
        f.write(f"Participants with errors: {len(quality_report['encoding_issues'] + quality_report['date_errors'])}\n")
        f.write(f"Missing data files: {len(quality_report['missing_files'])}\n")
        
        f.write("\nEncoding Issues:\n")
        for issue in quality_report['encoding_issues']:
            f.write(f"- Participant {issue['participant']}: {issue['error']}\n")
            
        f.write("\nDate Parsing Errors:\n")
        for error in quality_report['date_errors']:
            f.write(f"- File: {error['file']}\n  Error: {error['error']}\n")
            
        f.write("\nMissing Files:\n")
        for files in quality_report['missing_files']:
            f.write(f"- {files}\n")
            
        f.write("\nSuccessful Participants:\n")
        f.write(", ".join(quality_report['successful_participants']))
    
    print(f"\nQuality report saved to {report_path}")

# Update processing loops to track metrics
# In Qstarz processing loop:
for qstarz_file in glob.glob(os.path.join(processed_dir, "*_1_Qstarz_processed.csv")):
    participant_id = os.path.basename(qstarz_file).split('_')[0]
    print(f"\nProcessing Qstarz data for Participant {participant_id}")
    try:
        qstarz_df = load_and_preprocess_qstarz(qstarz_file)
        all_qstarz_data[participant_id] = qstarz_df
        print(f"Successfully processed Qstarz data for Participant {participant_id}")
        quality_report['successful_participants'].append(participant_id)
    except Exception as e:
        print(f"Error processing Qstarz data for Participant {participant_id}: {str(e)}")
        quality_report['failed_processing'].append(participant_id)

# In digital environment processing:
for participant_folder in glob.glob(os.path.join(participants_dir, 'P*')):
    folder_name = os.path.basename(participant_folder)
    participant_id = folder_name.split('_')[-1]  # Get the last part after underscore
    app_folder = os.path.join(participant_folder, '9 - Smartphone Tracking App')
    
    print(f"Looking for app files in: {app_folder}")
    print(f"Files in directory: {os.listdir(app_folder)}")
    app_files = glob.glob(os.path.join(app_folder, '*-apps.csv'))
    print(f"Found app files: {app_files}")
    
    gps_files = glob.glob(os.path.join(app_folder, '*-gps.csv'))
    
    if app_files and gps_files:
        app_file = app_files[0]
        gps_file = gps_files[0]
        
        print(f"\nProcessing digital environment data for Participant {participant_id}")
        try:
            participant_data = process_app_and_gps_data(app_file, gps_file)
            if not participant_data.empty:
                all_app_data[participant_id] = participant_data
                print(f"Successfully processed digital environment data for Participant {participant_id}")
            else:
                print(f"No valid merged data for Participant {participant_id}")
        except Exception as e:
            print(f"Error processing digital environment data for Participant {participant_id}: {str(e)}")
    else:
        print(f"Missing digital environment data files for Participant {participant_id}")
        quality_report['missing_files'].append({
            'participant': participant_id,
            'missing_files': {'apps': not bool(app_files), 'gps': not bool(gps_files)}
        })

# At end of script before final print:
generate_quality_report(processed_dir)