#!/usr/bin/env python3
"""
Visual validation of detected episodes
"""
import os
import glob
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path to find config
sys.path.append(str(Path(__file__).parent.parent))
from config.paths import PROCESSED_DATA_DIR, EPISODE_OUTPUT_DIR, MAP_OUTPUT_DIR, GPS_PREP_DIR

# Rest of imports
import folium
import pandas as pd

# Configuration - REPLACED with centralized config
BASE_DIR = PROCESSED_DATA_DIR.parent  # Gets SURREAL/data parent directory
EPISODE_DIR = EPISODE_OUTPUT_DIR
MAP_DIR = MAP_OUTPUT_DIR
GPS_DIR = GPS_PREP_DIR

def load_episode_data(participant_id):
    """Load detected episodes and raw GPS data"""
    episode_path = EPISODE_OUTPUT_DIR / f'{participant_id}_episodes.csv'
    qstarz_path = GPS_PREP_DIR / f'{participant_id}_qstarz_prep.csv'
    
    # Validate files exist
    if not episode_path.exists():
        raise FileNotFoundError(f"Missing episode file: {episode_path}")
    if not qstarz_path.exists():
        raise FileNotFoundError(f"Missing GPS data: {qstarz_path}")
    
    episodes = pd.read_csv(episode_path, parse_dates=['start_time', 'end_time'])
    gps_df = pd.read_csv(qstarz_path, parse_dates=['UTC DATE TIME'])
    
    return episodes, gps_df

def create_episode_map(episodes, gps_df, participant_id):
    """Create interactive map with episodes and raw GPS points"""
    # Base map with improved styling
    m = folium.Map(
        location=[gps_df['LATITUDE'].median(), gps_df['LONGITUDE'].median()],
        zoom_start=12,
        tiles='CartoDB positron',  # Modern basemap
        control_scale=True
    )
    
    # Enhanced GPS point visualization
    gps_layer = folium.FeatureGroup(name='Raw GPS', show=False)  # Start hidden
    for _, row in gps_df.iterrows():
        gps_layer.add_child(
            folium.CircleMarker(
                location=[row['LATITUDE'], row['LONGITUDE']],
                radius=3,
                color='#4a4a4a',
                fill=True,
                fill_opacity=0.7,
                popup=folium.Popup(
                    f"<b>{row['UTC DATE TIME'].strftime('%Y-%m-%d %H:%M')}</b><br>"
                    f"Speed: {row['SPEED_MS']:.1f} m/s<br>"
                    f"Sats: {row['NSAT_USED']}",
                    max_width=250
                )
            )
        )
    m.add_child(gps_layer)

    # Add episodes with color coding
    episode_layer = folium.FeatureGroup(name='Detected Episodes')
    for _, episode in episodes.iterrows():
        # Get GPS points within episode timeframe
        episode_points = gps_df[
            (gps_df['UTC DATE TIME'] >= episode['start_time']) &
            (gps_df['UTC DATE TIME'] <= episode['end_time'])
        ]
        
        if not episode_points.empty:
            # Create polyline for movement episodes
            if episode['movement'] == 'moving':
                color = '#ff0000'  # Red for movement
                weight = 3
            else:
                color = '#666666'  # Gray for stationary
                weight = 2
                
            episode_layer.add_child(
                folium.PolyLine(
                    locations=episode_points[['LATITUDE', 'LONGITUDE']].values,
                    color=color,
                    weight=weight,
                    opacity=0.7,
                    popup=(
                        f"Episode {episode.name}<br>"
                        f"Duration: {episode['duration']}<br>"
                        f"Movement: {episode['movement']}<br>"
                        f"Digital Use: {episode['digital_use']}"
                    )
                )
            )

    m.add_child(episode_layer)
    folium.LayerControl().add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 150px; 
                background: white; padding: 10px; border: 1px solid grey;">
      <h4>Legend</h4>
      <p><i style="background:#ff0000; width:10px; height:10px; display:inline-block;"></i> Moving</p>
      <p><i style="background:#666666; width:10px; height:10px; display:inline-block;"></i> Stationary</p>
      <p><i style="background:#555555; width:10px; height:10px; display:inline-block;"></i> Raw GPS</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    return m

def process_participant(participant_id):
    """Generate map for a single participant"""
    print(f"Processing {participant_id}")
    try:
        episodes, gps_df = load_episode_data(participant_id)
        m = create_episode_map(episodes, gps_df, participant_id)
        
        MAP_DIR.mkdir(parents=True, exist_ok=True)  # Path object directory creation
        map_path = MAP_DIR / f"{participant_id}_episodes.html"
        m.save(map_path.as_posix())
        print(f"Map saved to {map_path}")
        
    except Exception as e:
        print(f"Error mapping {participant_id}: {str(e)}")

if __name__ == "__main__":
    # Get all processed participants
    episode_files = glob.glob(os.path.join(EPISODE_DIR, "*_episodes.csv"))
    participants = [os.path.basename(f).split("_")[0] for f in episode_files]
    
    print(f"Generating maps for {len(participants)} participants")
    for pid in participants:
        process_participant(pid)
    
    print(f"\nMaps saved to: {MAP_DIR}")