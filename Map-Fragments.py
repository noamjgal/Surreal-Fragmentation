#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:48:30 2024

@author: noamgal
"""

import folium
from folium import plugins
import pandas as pd
import os
import glob
from datetime import datetime, timedelta

# Set the base directory
base_dir = "/Users/noamgal/Downloads/Research-Projects/SURREAL/HUJI_data-main/"
processed_dir = os.path.join(base_dir, "Processed", "fragment-processed")
fragmentation_output_dir = os.path.join(processed_dir, "fragmentation-outputs")
map_output_dir = os.path.join(fragmentation_output_dir, "maps")
os.makedirs(map_output_dir, exist_ok=True)

def load_data(participant_id, date):
    qstarz_file = os.path.join(processed_dir, f'{participant_id}_qstarz_preprocessed.csv')
    qstarz_df = pd.read_csv(qstarz_file, parse_dates=['UTC DATE TIME'])
    qstarz_df = qstarz_df[qstarz_df['UTC DATE TIME'].dt.date == date]
    
    fragmentation_file = os.path.join(fragmentation_output_dir, participant_id, f'{date}_fragmentation_output.csv')
    frag_df = pd.read_csv(fragmentation_file, parse_dates=['UTC DATE TIME'])
    
    return qstarz_df, frag_df

def create_map(qstarz_df, frag_df, participant_id, date):
    # Create a map centered on the mean coordinates
    center_lat = qstarz_df['LATITUDE'].mean()
    center_lon = qstarz_df['LONGITUDE'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    # Add markers for all points
    for _, row in frag_df.iterrows():
        popup_text = f"""
        Time: {row['UTC DATE TIME']}
        Movement: {row['MOVEMENT_TYPE']}
        IO Status: {row['IO_STATUS']}
        Speed: {row['SPEED_MS']:.2f} m/s
        """
        folium.CircleMarker(
            location=[row['LATITUDE'], row['LONGITUDE']],
            radius=2,
            popup=popup_text,
            color='blue',
            fill=True
        ).add_to(m)

    # Add special markers for start/end of mobility and indoor/outdoor transitions
    for i, row in frag_df.iterrows():
        if i == 0 or row['event_type'] != 'No Change':
            icon = 'play' if 'Movement Change' in row['event_type'] else 'exchange'
            color = 'green' if 'Movement Change' in row['event_type'] else 'orange'
            folium.Marker(
                location=[row['LATITUDE'], row['LONGITUDE']],
                popup=f"Start: {row['event_details']} at {row['UTC DATE TIME']}",
                icon=folium.Icon(color=color, icon=icon)
            ).add_to(m)
        
        if i == len(frag_df) - 1 or frag_df.iloc[i+1]['event_type'] != 'No Change':
            icon = 'stop' if 'Movement Change' in row['event_type'] else 'exchange'
            color = 'red' if 'Movement Change' in row['event_type'] else 'orange'
            folium.Marker(
                location=[row['LATITUDE'], row['LONGITUDE']],
                popup=f"End: {row['event_details']} at {row['UTC DATE TIME']}",
                icon=folium.Icon(color=color, icon=icon)
            ).add_to(m)

    # Add lines for mobility activities
    mobility_episodes = frag_df[frag_df['MOVEMENT_TYPE'] != 'Stationary']
    folium.PolyLine(
        locations=mobility_episodes[['LATITUDE', 'LONGITUDE']].values,
        color="red",
        weight=2,
        opacity=0.8
    ).add_to(m)

    # Calculate summary statistics
    total_time = (frag_df['UTC DATE TIME'].max() - frag_df['UTC DATE TIME'].min()).total_seconds() / 3600
    time_outside = frag_df[frag_df['IO_STATUS'] == 'Outdoor']['Movement_duration'].sum() / 60
    time_inside = frag_df[frag_df['IO_STATUS'] == 'Indoor']['Movement_duration'].sum() / 60
    time_traveling = frag_df[frag_df['MOVEMENT_TYPE'] != 'Stationary']['Movement_duration'].sum() / 60
    time_stationary = frag_df[frag_df['MOVEMENT_TYPE'] == 'Stationary']['Movement_duration'].sum() / 60
    avg_episode_length = frag_df.groupby('event_details')['Movement_duration'].mean().mean()
    avg_interepisode_duration = frag_df['inter_episode_duration'].mean()

    # Load daily summary for fragmentation indices
    summary_file = os.path.join(fragmentation_output_dir, participant_id, f'{participant_id}_daily_summary.csv')
    summary_df = pd.read_csv(summary_file)
    day_summary = summary_df[summary_df['date'] == str(date)].iloc[0]

    # Create a scrollable legend
    legend_html = f"""
    <div id="legend" style="position: fixed; bottom: 50px; right: 50px; width: 250px; height: 400px; 
    overflow-y: scroll; background-color: white; padding: 10px; border: 1px solid grey;">
    <h4>Legend and Summary</h4>
    <p><strong>Date:</strong> {date}</p>
    <p><strong>Total Time:</strong> {total_time:.2f} hours</p>
    <p><strong>Time Outside:</strong> {time_outside:.2f} hours</p>
    <p><strong>Time Inside:</strong> {time_inside:.2f} hours</p>
    <p><strong>Time Traveling:</strong> {time_traveling:.2f} hours</p>
    <p><strong>Time Stationary:</strong> {time_stationary:.2f} hours</p>
    <p><strong>Avg Episode Length:</strong> {avg_episode_length:.2f} minutes</p>
    <p><strong>Avg Interepisode Duration:</strong> {avg_interepisode_duration:.2f} minutes</p>
    <h5>Fragmentation Indices:</h5>
    <p>Stationary: {day_summary['Stationary_index']:.4f}</p>
    <p>Active Transport: {day_summary['Active Transport_index']:.4f}</p>
    <p>Mechanized Transport: {day_summary['Mechanized Transport_index']:.4f}</p>
    <p>Indoor: {day_summary['Indoor_index']:.4f}</p>
    <p>Outdoor: {day_summary['Outdoor_index']:.4f}</p>
    <h5>Marker Types:</h5>
    <p>🟢 Green Play: Start of Movement</p>
    <p>🔴 Red Stop: End of Movement</p>
    <p>🟠 Orange Exchange: Indoor/Outdoor Transition</p>
    <p>🔵 Blue Circle: Regular GPS Point</p>
    <p>🔴 Red Line: Movement Path</p>
    </div>
    """

    m.get_root().html.add_child(folium.Element(legend_html))

    return m

def process_participant(participant_id):
    print(f"Processing participant: {participant_id}")
    fragmentation_files = glob.glob(os.path.join(fragmentation_output_dir, participant_id, '*_fragmentation_output.csv'))
    
    for frag_file in fragmentation_files:
        date_str = os.path.basename(frag_file).split('_')[0]
        date = datetime.strptime(date_str, '%Y-%m-%d').date()
        
        print(f"  Creating map for date: {date}")
        qstarz_df, frag_df = load_data(participant_id, date)
        m = create_map(qstarz_df, frag_df, participant_id, date)
        
        output_file = os.path.join(map_output_dir, f'{participant_id}_{date}_map.html')
        m.save(output_file)
        print(f"  Map saved to: {output_file}")

# Process all participants
all_participants = [os.path.basename(f).split('_')[0] for f in glob.glob(os.path.join(processed_dir, '*_qstarz_preprocessed.csv'))]

for participant_id in all_participants:
    process_participant(participant_id)

print("All maps have been generated.")