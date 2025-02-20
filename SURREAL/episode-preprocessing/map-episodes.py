#!/usr/bin/env python3
"""
Enhanced episode visualization with improved maps and statistical plots
"""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Tuple

# Add parent directory to path to find config
sys.path.append(str(Path(__file__).parent.parent))
from config.paths import PROCESSED_DATA_DIR, EPISODE_OUTPUT_DIR, MAP_OUTPUT_DIR, GPS_PREP_DIR

import folium
from folium.plugins import FastMarkerCluster
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class EpisodeVisualizer:
    def __init__(self, participant_id: str):
        self.participant_id = participant_id
        self.episode_dir = EPISODE_OUTPUT_DIR / participant_id
        self.gps_file = GPS_PREP_DIR / f'{participant_id}_qstarz_prep.csv'
        self.map_dir = MAP_OUTPUT_DIR
        
        # Set up enhanced logging
        self.setup_logging()
        
        # Enhanced color scheme for better visibility
        self.colors = {
            'digital': '#1f77b4',    # Darker blue
            'movement': '#d62728',    # Brighter red
            'overlap': '#9467bd',     # Rich purple
            'stationary': '#2f2f2f'   # Dark grey
        }
    
    def setup_logging(self):
        """Configure detailed logging"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'episode_visualizer_{self.participant_id}.log')
            ]
        )
        self.logger = logging.getLogger(f"EpisodeVisualizer_{self.participant_id}")
        
    def load_data(self):
        """Load GPS and episode data with enhanced error handling"""
        try:
            self.logger.info(f"Loading data for participant {self.participant_id}")
            
            # Load GPS data with optimized dtypes
            dtypes = {
                'LATITUDE': np.float32,
                'LONGITUDE': np.float32,
                'SPEED_MS': np.float32,
                'NSAT_USED': np.int8
            }
            self.gps_df = pd.read_csv(self.gps_file, dtype=dtypes, 
                                    parse_dates=['UTC DATE TIME'])
            self.logger.info(f"Successfully loaded {len(self.gps_df):,} GPS points")
            
            # Load daily timeline data
            timeline_files = list(self.episode_dir.glob('*_daily_timeline.csv'))
            if not timeline_files:
                self.logger.error("No timeline files found")
                return False
                
            self.timeline_df = pd.concat(
                [pd.read_csv(f, parse_dates=['start_time', 'end_time']) 
                 for f in timeline_files]
            )
            self.timeline_df['duration'] = pd.to_timedelta(self.timeline_df['duration'])
            self.timeline_df['date'] = self.timeline_df['start_time'].dt.date
            
            self.logger.info(f"Successfully loaded {len(self.timeline_df):,} timeline entries")
            self.logger.info(f"Date range: {self.timeline_df['date'].min()} to {self.timeline_df['date'].max()}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}", exc_info=True)
            return False

    def create_enhanced_map(self, date, episode_data, gps_data):
        """Create enhanced interactive map with improved styling"""
        self.logger.info(f"Creating map for date {date}")
        
        # Initialize map with improved base layer
        center_lat = float(gps_data['LATITUDE'].median())
        center_lon = float(gps_data['LONGITUDE'].median())
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=13,
            tiles='CartoDB positron',
            control_scale=True,
            prefer_canvas=True
        )
        
        # Add improved GPS point clusters
        callback = """
            function (row) {
                var marker = L.circleMarker(new L.LatLng(row[0], row[1]), {
                    radius: 2,
                    color: '#696969',
                    weight: 1,
                    opacity: 0.6,
                    fillOpacity: 0.4
                });
                return marker;
            };
        """
        
        gps_points = gps_data[['LATITUDE', 'LONGITUDE']].values.tolist()
        FastMarkerCluster(data=gps_points, callback=callback, 
                         name='GPS Points').add_to(m)
        
        # Create separate feature groups for episodes
        layers = {
            'digital': folium.FeatureGroup(name='Digital Episodes'),
            'movement': folium.FeatureGroup(name='Movement Episodes'),
            'overlap': folium.FeatureGroup(name='Overlap Episodes')
        }
        
        # Add episodes with enhanced styling
        for idx, episode in episode_data.iterrows():
            episode_points = self.gps_df[
                (self.gps_df['UTC DATE TIME'] >= episode['start_time']) &
                (self.gps_df['UTC DATE TIME'] <= episode['end_time'])
            ]
            
            if not episode_points.empty:
                # Subsample large point sets
                if len(episode_points) > 1000:
                    episode_points = episode_points.iloc[::len(episode_points)//1000]
                
                # Enhanced color selection
                color = self.colors[episode['episode_type']]
                if episode['episode_type'] == 'movement' and \
                   episode.get('movement_state') == 'stationary':
                    color = self.colors['stationary']
                
                # Create enhanced polyline
                folium.PolyLine(
                    locations=episode_points[['LATITUDE', 'LONGITUDE']].values,
                    color=color,
                    weight=4,
                    opacity=0.8,
                    popup=folium.Popup(
                        f"<div style='font-family: Arial, sans-serif;'>"
                        f"<h4 style='margin: 0;'>Episode {idx + 1}</h4>"
                        f"<p><b>Type:</b> {episode['episode_type'].title()}</p>"
                        f"<p><b>Start:</b> {episode['start_time'].strftime('%H:%M:%S')}</p>"
                        f"<p><b>End:</b> {episode['end_time'].strftime('%H:%M:%S')}</p>"
                        f"<p><b>Duration:</b> {episode['duration'].total_seconds()/60:.1f} min</p>"
                        + (f"<p><b>Movement:</b> {episode.get('movement_state', 'N/A')}</p>"
                           if 'movement_state' in episode else "")
                        + "</div>",
                        max_width=300
                    )
                ).add_to(layers[episode['episode_type']])
        
        # Add all layers to map
        for layer in layers.values():
            m.add_child(layer)
        
        # Add enhanced layer control
        folium.LayerControl(collapsed=False).add_to(m)
        
        # Add enhanced legend
        legend_html = f'''
        <div style="position: fixed; bottom: 50px; right: 50px; 
                    background: white; padding: 20px; 
                    border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    font-family: Arial, sans-serif; max-width: 250px;">
            <h4 style="margin-top: 0; border-bottom: 2px solid #eee; 
                      padding-bottom: 10px;">Episode Types</h4>
            <div style="display: flex; flex-direction: column; gap: 10px;">
                <div>
                    <span style="display: inline-block; width: 20px; height: 4px; 
                           background: {self.colors['digital']}; margin-right: 8px;"></span>
                    Digital Activity
                </div>
                <div>
                    <span style="display: inline-block; width: 20px; height: 4px; 
                           background: {self.colors['movement']}; margin-right: 8px;"></span>
                    Moving
                </div>
                <div>
                    <span style="display: inline-block; width: 20px; height: 4px; 
                           background: {self.colors['stationary']}; margin-right: 8px;"></span>
                    Stationary
                </div>
                <div>
                    <span style="display: inline-block; width: 20px; height: 4px; 
                           background: {self.colors['overlap']}; margin-right: 8px;"></span>
                    Overlap
                </div>
                <div>
                    <span style="display: inline-block; width: 6px; height: 6px; 
                           background: #696969; border-radius: 50%; margin-right: 8px;"></span>
                    GPS Points
                </div>
            </div>
        </div>
        '''
        
        m.get_root().html.add_child(folium.Element(legend_html))
        
        self.logger.info(f"Map created successfully with {len(episode_data)} episodes")
        return m

    def process_all_days(self):
        """Process all available days with enhanced error handling"""
        if not self.load_data():
            return
        
        self.map_dir.mkdir(parents=True, exist_ok=True)
        
        for date in self.timeline_df['date'].unique():
            self.logger.info(f"Processing visualizations for {date}")
            
            try:
                # Filter data for this day
                date_data = self.timeline_df[self.timeline_df['date'] == date]
                date_gps = self.gps_df[
                    (self.gps_df['UTC DATE TIME'].dt.date == date)
                ]
                
                if len(date_data) == 0 or len(date_gps) == 0:
                    self.logger.warning(f"No data available for {date}")
                    continue
                
                self.logger.info(f"Creating visualizations for {len(date_data)} episodes and {len(date_gps)} GPS points")
                
                # Create and save enhanced map
                m = self.create_enhanced_map(date, date_data, date_gps)
                map_path = self.map_dir / f"{self.participant_id}_{date}_map.html"
                m.save(str(map_path))
                self.logger.info(f"Saved interactive map to {map_path}")

                # Removed statistics plot creation and saving
                
            except Exception as e:
                self.logger.error(f"Error processing {date}: {str(e)}", exc_info=True)
                continue
        
        self.logger.info(f"Completed processing all days for participant {self.participant_id}")

def main():
    """Main execution function with enhanced logging"""
    # Set up root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('episode_visualization.log')
        ]
    )
    logger = logging.getLogger("EpisodeVisualization")
    
    try:
        # Get list of participants
        participant_dirs = [d for d in EPISODE_OUTPUT_DIR.iterdir() if d.is_dir()]
        logger.info(f"Found {len(participant_dirs)} participant directories")
        
        for participant_dir in participant_dirs:
            participant_id = participant_dir.name
            logger.info(f"Starting visualization process for participant {participant_id}")
            
            try:
                visualizer = EpisodeVisualizer(participant_id)
                visualizer.process_all_days()
                logger.info(f"Successfully completed visualization for participant {participant_id}")
                
            except Exception as e:
                logger.error(f"Error processing participant {participant_id}: {str(e)}", 
                           exc_info=True)
                continue
        
        logger.info("Completed visualization process for all participants")
        
    except Exception as e:
        logger.error(f"Critical error in main execution: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()