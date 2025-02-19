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

# Add parent directory to path to find config
sys.path.append(str(Path(__file__).parent.parent))
from config.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR, MAP_OUTPUT_DIR, GPS_PREP_DIR

# Set the base directory
base_dir = RAW_DATA_DIR
processed_dir = PROCESSED_DATA_DIR
participants_dir = RAW_DATA_DIR / "Participants"

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
    app_df = pd.read_csv(app_file)
    gps_df = pd.read_csv(gps_file)
    
    # Parse timestamps for app data
    app_df['Timestamp'] = pd.to_datetime(app_df['date'] + ' ' + app_df['time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    
    # Parse timestamps for GPS data
    gps_df['Timestamp'] = pd.to_datetime(gps_df['date'] + ' ' + gps_df['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    
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
    except Exception as e:
        print(f"Error processing Qstarz data for Participant {participant_id}: {str(e)}")

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
