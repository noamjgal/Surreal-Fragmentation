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

# Create logs directory
LOG_DIR = Path(__file__).parent.parent / "logs"/"maps"
LOG_DIR.mkdir(parents=True, exist_ok=True)

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
                logging.FileHandler(LOG_DIR / f'episode_visualizer_{self.participant_id}.log')
            ]
        )
        self.logger = logging.getLogger(f"EpisodeVisualizer_{self.participant_id}")
        
    def load_data(self):
        """Load GPS and episode data with enhanced error handling"""
        try:
            self.logger.info(f"Loading data for participant {self.participant_id}")
            
            # Try to read GPS data using different encodings
            try:
                # Check the headers first
                encodings = ['utf-8', 'latin1', 'cp1252', 'ISO-8859-1']
                gps_headers = None
                
                for encoding in encodings:
                    try:
                        gps_headers = pd.read_csv(self.gps_file, nrows=0, encoding=encoding).columns.tolist()
                        self.logger.info(f"GPS file columns: {gps_headers}")
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        self.logger.warning(f"Error reading GPS headers with encoding {encoding}: {str(e)}")
                        continue
                
                if not gps_headers:
                    self.logger.error("Could not read GPS file with any available encodings")
                    return False
                
                # Determine datetime column
                if 'tracked_at' in gps_headers:
                    datetime_col = 'tracked_at'
                elif 'UTC DATE TIME' in gps_headers:
                    datetime_col = 'UTC DATE TIME'
                else:
                    # Look for any time-related column
                    time_cols = [col for col in gps_headers if 'time' in col.lower() or 'date' in col.lower()]
                    if time_cols:
                        datetime_col = time_cols[0]
                    else:
                        self.logger.warning(f"Could not find datetime column. Using first column as index.")
                        datetime_col = None
                
                # Determine lat/lon columns
                if 'latitude' in gps_headers and 'longitude' in gps_headers:
                    lat_col, lon_col = 'latitude', 'longitude'
                elif 'LATITUDE' in gps_headers and 'LONGITUDE' in gps_headers:
                    lat_col, lon_col = 'LATITUDE', 'LONGITUDE'
                else:
                    # Try to find columns with lat/lon in their names
                    lat_candidates = [col for col in gps_headers if 'lat' in col.lower()]
                    lon_candidates = [col for col in gps_headers if 'lon' in col.lower()]
                    
                    if lat_candidates and lon_candidates:
                        lat_col, lon_col = lat_candidates[0], lon_candidates[0]
                    else:
                        self.logger.warning(f"Could not find lat/lon columns. Using default names.")
                        lat_col, lon_col = 'latitude', 'longitude'
                
                # Define dtypes for efficient loading
                dtypes = {
                    lat_col: np.float32,
                    lon_col: np.float32
                }
                
                # Add parse_dates parameter only if datetime column exists
                parse_dates = [datetime_col] if datetime_col else None
                
                # Now load the full data with the same encoding that worked for headers
                self.gps_df = pd.read_csv(self.gps_file, dtype=dtypes, 
                                        parse_dates=parse_dates, encoding=encoding)
                
                # Standardize column names
                col_mapping = {}
                if datetime_col:
                    col_mapping[datetime_col] = 'timestamp'
                col_mapping[lat_col] = 'LATITUDE'
                col_mapping[lon_col] = 'LONGITUDE'
                self.gps_df = self.gps_df.rename(columns=col_mapping)
                
                self.logger.info(f"Successfully loaded {len(self.gps_df):,} GPS points")
                
            except Exception as e:
                self.logger.error(f"Error loading GPS data: {str(e)}")
                return False
            
            # Load daily timeline data with encoding handling
            try:
                timeline_files = list(self.episode_dir.glob('*_daily_timeline.csv'))
                if not timeline_files:
                    self.logger.error("No timeline files found")
                    return False
                
                # Filter out macOS hidden files
                timeline_files = [f for f in timeline_files if not f.name.startswith('._')]
                
                if not timeline_files:
                    self.logger.error("No valid timeline files found after filtering hidden files")
                    return False
                
                # Try multiple encodings
                encodings = ['utf-8', 'latin1', 'cp1252', 'ISO-8859-1']
                all_timelines = []
                
                for file in timeline_files:
                    file_loaded = False
                    for encoding in encodings:
                        try:
                            df = pd.read_csv(file, encoding=encoding, parse_dates=['start_time', 'end_time'])
                            all_timelines.append(df)
                            file_loaded = True
                            self.logger.debug(f"Successfully read {file} with encoding {encoding}")
                            break
                        except UnicodeDecodeError:
                            continue
                        except Exception as e:
                            self.logger.warning(f"Error reading {file} with encoding {encoding}: {str(e)}")
                            continue
                    
                    if not file_loaded:
                        self.logger.warning(f"Could not read file {file} with any encoding")
                
                if not all_timelines:
                    self.logger.error("Could not read any timeline files with available encodings")
                    return False
                
                self.timeline_df = pd.concat(all_timelines)
                self.timeline_df['duration'] = pd.to_timedelta(self.timeline_df['duration'])
                self.timeline_df['date'] = self.timeline_df['start_time'].dt.date
                
                self.logger.info(f"Successfully loaded {len(self.timeline_df):,} timeline entries")
                self.logger.info(f"Date range: {self.timeline_df['date'].min()} to {self.timeline_df['date'].max()}")
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error loading timeline data: {str(e)}")
                return False
            
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
        
        # Create separate feature groups for all episode types
        layers = {
            'digital': folium.FeatureGroup(name='Digital Episodes'),
            'moving': folium.FeatureGroup(name='Moving Episodes'),
            'stationary': folium.FeatureGroup(name='Stationary Episodes'),
            'moving_digital': folium.FeatureGroup(name='Moving-Digital Overlap'),
            'gps': folium.FeatureGroup(name='GPS Points')
        }
        
        # Add episodes with enhanced styling
        for idx, episode in episode_data.iterrows():
            # Convert timestamps for comparison, handling timezone issue
            start_time = episode['start_time']
            end_time = episode['end_time']
            
            # Check if GPS timestamps need timezone handling
            is_gps_tz_aware = False
            if len(self.gps_df) > 0 and pd.api.types.is_datetime64_dtype(self.gps_df['timestamp']):
                is_gps_tz_aware = hasattr(self.gps_df['timestamp'].iloc[0], 'tzinfo') and self.gps_df['timestamp'].iloc[0].tzinfo is not None
            
            if is_gps_tz_aware:
                # Make episode timestamps timezone-aware to match GPS timestamps
                try:
                    if hasattr(start_time, 'tzinfo') and start_time.tzinfo is None:
                        start_time = pd.Timestamp(start_time).tz_localize('UTC')
                    if hasattr(end_time, 'tzinfo') and end_time.tzinfo is None:
                        end_time = pd.Timestamp(end_time).tz_localize('UTC')
                except Exception as e:
                    self.logger.warning(f"Error adjusting timezone: {str(e)}")
            else:
                # Make GPS timestamps naive to match episode timestamps
                if 'timestamp_naive' not in self.gps_df.columns:
                    try:
                        self.gps_df['timestamp_naive'] = self.gps_df['timestamp'].dt.tz_localize(None)
                    except Exception as e:
                        self.logger.warning(f"Error making timestamps naive: {str(e)}")
                        # If we can't fix the timestamps, skip this episode to avoid errors
                        continue
            
            # Filter GPS points based on episode time window, using the right timestamp column
            if is_gps_tz_aware and 'timestamp_naive' not in self.gps_df.columns:
                episode_points = self.gps_df[
                    (self.gps_df['timestamp'] >= start_time) &
                    (self.gps_df['timestamp'] <= end_time)
                ]
            elif 'timestamp_naive' in self.gps_df.columns:
                # Use naive timestamps for filtering
                if not hasattr(start_time, 'tzinfo'):
                    start_time = pd.Timestamp(start_time)
                if not hasattr(end_time, 'tzinfo'):
                    end_time = pd.Timestamp(end_time)
                
                naive_start = start_time.replace(tzinfo=None) if hasattr(start_time, 'tzinfo') else start_time
                naive_end = end_time.replace(tzinfo=None) if hasattr(end_time, 'tzinfo') else end_time
                
                episode_points = self.gps_df[
                    (self.gps_df['timestamp_naive'] >= naive_start) &
                    (self.gps_df['timestamp_naive'] <= naive_end)
                ]
            else:
                # Fallback to string comparison
                st_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
                et_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
                
                self.gps_df['timestamp_str'] = self.gps_df['timestamp'].astype(str)
                episode_points = self.gps_df[
                    (self.gps_df['timestamp_str'] >= st_str) &
                    (self.gps_df['timestamp_str'] <= et_str)
                ]
            
            if not episode_points.empty:
                # Subsample large point sets
                if len(episode_points) > 1000:
                    episode_points = episode_points.iloc[::len(episode_points)//1000]
                
                # Enhanced color selection
                color = self.colors.get(episode['episode_type'], '#333333')
                if episode['episode_type'] == 'movement' and \
                   episode.get('movement_state') == 'stationary':
                    color = self.colors['stationary']
                
                # Determine layer based on episode type and state
                if episode['episode_type'] == 'movement':
                    layer_key = 'moving' if episode.get('movement_state') == 'moving' else 'stationary'
                elif episode['episode_type'] == 'overlap':
                    layer_key = 'moving_digital'
                else:
                    layer_key = episode['episode_type'] if episode['episode_type'] in layers else 'digital'

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
                ).add_to(layers[layer_key])
        
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
                    Moving-Digital Overlap
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
        
        # Check if the GPS data needs timezone-naive duplicates
        if len(self.gps_df) > 0 and pd.api.types.is_datetime64_dtype(self.gps_df['timestamp']):
            if (hasattr(self.gps_df['timestamp'].iloc[0], 'tzinfo') and 
                self.gps_df['timestamp'].iloc[0].tzinfo is not None):
                try:
                    self.gps_df['timestamp_naive'] = self.gps_df['timestamp'].dt.tz_localize(None)
                    self.logger.info("Created timezone-naive version of timestamps for better compatibility")
                except Exception as e:
                    self.logger.warning(f"Could not create naive timestamps: {str(e)}")
        
        for date in self.timeline_df['date'].unique():
            self.logger.info(f"Processing visualizations for {date}")
            
            try:
                # Filter data for this day
                date_data = self.timeline_df[self.timeline_df['date'] == date]
                
                # Use date filtering that works with both timezone-aware and naive timestamps
                if 'timestamp_naive' in self.gps_df.columns:
                    # Use naive timestamps for date filtering
                    date_str = date.strftime('%Y-%m-%d')
                    date_gps = self.gps_df[
                        self.gps_df['timestamp_naive'].dt.strftime('%Y-%m-%d') == date_str
                    ]
                else:
                    # Try the original timestamp column
                    try:
                        date_gps = self.gps_df[
                            self.gps_df['timestamp'].dt.date == date
                        ]
                    except Exception:
                        # Fallback method
                        date_str = date.strftime('%Y-%m-%d')
                        date_gps = self.gps_df[
                            self.gps_df['timestamp'].astype(str).str[:10] == date_str
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
                
            except Exception as e:
                self.logger.error(f"Error processing {date}: {str(e)}", exc_info=True)
                continue
        
        self.logger.info(f"Completed processing all days for participant {self.participant_id}")

def main():
    """Main execution function with enhanced logging"""
    # Set up root logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = LOG_DIR / f'episode_visualization_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_filename)
        ]
    )
    logger = logging.getLogger("EpisodeVisualization")
    logger.info(f"Starting visualization process, logging to {log_filename}")
    
    try:
        # Get list of participants
        participant_dirs = [d for d in EPISODE_OUTPUT_DIR.iterdir() if d.is_dir()]
        logger.info(f"Found {len(participant_dirs)} participant directories")
        
        # Filter out macOS hidden files (starting with ._)
        participant_dirs = [d for d in participant_dirs if not d.name.startswith('._')]
        logger.info(f"Found {len(participant_dirs)} valid participant directories after filtering")
        
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