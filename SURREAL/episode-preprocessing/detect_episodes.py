#!/usr/bin/env python3
"""
Enhanced episode detection using Trackintel library
Focuses on identifying mobility between locations versus stationary periods
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import traceback
from pathlib import Path
import sys
import logging
from typing import Dict, List, Tuple, Optional
from collections import Counter
from tqdm import tqdm
import trackintel as ti
import geopandas as gpd
from shapely.geometry import Point

# Suppress pandas FutureWarnings related to inplace operations
import warnings
warnings.filterwarnings("ignore", 
                        message=".*inplace method.*", 
                        category=FutureWarning)
warnings.filterwarnings("ignore", 
                        message=".*Downcasting object dtype arrays.*", 
                        category=FutureWarning)

# Setup logging - File logging remains detailed but console output is reduced
file_handler = logging.FileHandler('episode_detection.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console by default
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)

# Create a special logger for summary statistics that will always print to console
summary_logger = logging.getLogger("summary")
summary_handler = logging.StreamHandler()
summary_handler.setFormatter(logging.Formatter('%(message)s'))
summary_logger.addHandler(summary_handler)
summary_logger.setLevel(logging.INFO)
summary_logger.propagate = False  # Don't propagate to root logger

# Add parent directory to path to find config
sys.path.append(str(Path(__file__).parent.parent))
from config.paths import GPS_PREP_DIR, EPISODE_OUTPUT_DIR, PROCESSED_DATA_DIR

# Trackintel configuration parameters
STAYPOINT_DISTANCE_THRESHOLD = 100  # meters - distance threshold for staypoint detection
STAYPOINT_TIME_THRESHOLD = 5.0  # minutes - time threshold for staypoint detection
STAYPOINT_GAP_THRESHOLD = 15.0  # minutes - max gap between consecutive positionfixes
LOCATION_EPSILON = 100  # meters - distance threshold for clustering staypoints into locations
LOCATION_MIN_SAMPLES = 1  # minimum number of staypoints to form a location
TRIP_GAP_THRESHOLD = 25  # minutes - maximum gap between consecutive staypoints
DIGITAL_USE_COL = 'action'  # column name for screen events

def ensure_tz_naive(datetime_series: pd.Series) -> pd.Series:
    """Convert a datetime series to timezone-naive if it has a timezone"""
    if datetime_series.empty:
        return datetime_series
        
    if hasattr(datetime_series.iloc[0], 'tz') and datetime_series.iloc[0].tz is not None:
        return datetime_series.dt.tz_localize(None)
    return datetime_series

class EpisodeProcessor:
    def __init__(self, participant_id: str):
        self.participant_id = participant_id
        self.logger = logging.getLogger(f"EpisodeProcessor_{participant_id}")
        self.output_dir = EPISODE_OUTPUT_DIR / participant_id
        
        # Skip creation if the path is a macOS hidden file
        if '._' in str(self.output_dir):
            self.logger.warning(f"Skipping macOS hidden file: {self.output_dir}")
            return
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _find_overlaps(self, digital_episodes: pd.DataFrame, 
                      movement_episodes: pd.DataFrame) -> pd.DataFrame:
        """Find temporal overlaps between digital and mobility episodes"""
        overlap_episodes = []
        
        # Check if dataframes are empty or missing required columns
        if digital_episodes.empty or movement_episodes.empty:
            self.logger.debug("Empty dataframe(s) provided to _find_overlaps, returning empty result")
            return pd.DataFrame()
        
        # Make copies to avoid modifying originals
        digital_episodes = digital_episodes.copy()
        movement_episodes = movement_episodes.copy()
        
        # Ensure timezone consistency - make all timestamps timezone-naive
        if 'start_time' in digital_episodes.columns:
            digital_episodes['start_time'] = ensure_tz_naive(digital_episodes['start_time'])
            digital_episodes['end_time'] = ensure_tz_naive(digital_episodes['end_time'])
                
        if 'started_at' in movement_episodes.columns:
            movement_episodes['started_at'] = ensure_tz_naive(movement_episodes['started_at'])
            movement_episodes['finished_at'] = ensure_tz_naive(movement_episodes['finished_at'])
        
        # Filter movement episodes to only mobility episodes (trips)
        for _, d_ep in digital_episodes.iterrows():
            for _, m_ep in movement_episodes.iterrows():
                start = max(d_ep['start_time'], m_ep['started_at'])
                end = min(d_ep['end_time'], m_ep['finished_at'])
                
                if start < end:  # There is an overlap
                    duration = end - start
                    if duration >= pd.Timedelta(minutes=1):
                        overlap_episodes.append({
                            'start_time': start,
                            'end_time': end,
                            'state': 'overlap',
                            'movement_state': 'mobility',
                            'latitude': m_ep['latitude'] if 'latitude' in m_ep else np.nan,
                            'longitude': m_ep['longitude'] if 'longitude' in m_ep else np.nan,
                            'duration': duration
                        })
        
        if overlap_episodes:
            return pd.DataFrame(overlap_episodes)
        return pd.DataFrame()

    def load_gps_data(self) -> Optional[ti.Positionfixes]:
        """Load GPS data with validation"""
        gps_path = GPS_PREP_DIR / f'{self.participant_id}_qstarz_prep.csv'
        self.logger.debug(f"Loading GPS data from {gps_path}")
        
        try:
            # First, check what columns are available
            gps_df = pd.read_csv(gps_path)
            
            # Check which datetime column is available
            datetime_col = None
            for col in ['tracked_at', 'UTC DATE TIME', 'timestamp']:
                if col in gps_df.columns:
                    datetime_col = col
                    break
            
            if datetime_col is None:
                raise ValueError(f"No datetime column found in {gps_path}")
                
            # Read again with parse_dates
            gps_df = pd.read_csv(gps_path, parse_dates=[datetime_col])
            
            # Check for different column name patterns
            lat_col = None
            lon_col = None
            
            for col in gps_df.columns:
                if col.upper() in ['LATITUDE', 'LAT'] or 'LAT' in col.upper():
                    lat_col = col
                if col.upper() in ['LONGITUDE', 'LON', 'LONG'] or 'LON' in col.upper():
                    lon_col = col
            
            if lat_col is None or lon_col is None:
                # If we didn't find lat/lon columns, check if this is already a Trackintel export
                if 'geometry' in gps_df.columns and 'user_id' in gps_df.columns:
                    # This is likely a GeoDataFrame that was saved to CSV
                    # We need to recreate the geometry from WKT
                    from shapely import wkt
                    gps_df['geometry'] = gps_df['geometry'].apply(wkt.loads)
                    gdf = gpd.GeoDataFrame(gps_df, geometry='geometry', crs="EPSG:4326")
                    
                    # Extract lat/lon if not available
                    if 'latitude' not in gps_df.columns:
                        gps_df['latitude'] = gdf.geometry.y
                    if 'longitude' not in gps_df.columns:
                        gps_df['longitude'] = gdf.geometry.x
                        
                    # Ensure required columns
                    if 'user_id' not in gps_df.columns:
                        gps_df['user_id'] = self.participant_id
                    if 'tracked_at' not in gps_df.columns:
                        gps_df['tracked_at'] = gps_df[datetime_col]
                    
                    # Make timezone aware
                    if not pd.api.types.is_datetime64_ns_dtype(gps_df['tracked_at']):
                        gps_df['tracked_at'] = pd.to_datetime(gps_df['tracked_at'])
                    
                    if hasattr(gps_df['tracked_at'].iloc[0], 'tz') and gps_df['tracked_at'].iloc[0].tz is None:
                        gps_df['tracked_at'] = gps_df['tracked_at'].dt.tz_localize('UTC', ambiguous='raise')
                        
                    # Return as Positionfixes
                    return ti.Positionfixes(gdf)
                else:
                    raise ValueError(f"Could not find latitude/longitude columns in {gps_path}")
            
            # Convert to trackintel's positionfixes format
            positionfixes = pd.DataFrame({
                'user_id': self.participant_id,
                'tracked_at': gps_df[datetime_col],
                'latitude': gps_df[lat_col],
                'longitude': gps_df[lon_col],
                'elevation': np.nan,  # Optional
                'accuracy': np.nan,   # Optional
            })
            
            # Make sure tracked_at is timezone aware (required by trackintel)
            if not pd.api.types.is_datetime64_ns_dtype(positionfixes['tracked_at']):
                positionfixes['tracked_at'] = pd.to_datetime(positionfixes['tracked_at'])
                
            if hasattr(positionfixes['tracked_at'].iloc[0], 'tz') and positionfixes['tracked_at'].iloc[0].tz is None:
                positionfixes['tracked_at'] = positionfixes['tracked_at'].dt.tz_localize('UTC', ambiguous='raise')
            
            # Convert to GeoDataFrame and set as trackintel Positionfixes
            geometry = [Point(lon, lat) for lon, lat in zip(positionfixes['longitude'], positionfixes['latitude'])]
            positionfixes = gpd.GeoDataFrame(positionfixes, geometry=geometry, crs="EPSG:4326")
            
            # Set as trackintel Positionfixes
            positionfixes = ti.Positionfixes(positionfixes)
            
            self.logger.debug(f"Loaded {len(positionfixes)} GPS points")
            return positionfixes
        except Exception as e:
            self.logger.error(f"Failed to load GPS data: {str(e)}")
            raise

    def load_app_data(self) -> pd.DataFrame:
        """Load app data with validation"""
        app_path = GPS_PREP_DIR / f'{self.participant_id}_app_prep.csv'
        self.logger.debug(f"Loading app data from {app_path}")
        
        try:
            app_df = pd.read_csv(app_path)
            
            # Check which column contains timestamps
            timestamp_col = None
            for col in ['timestamp', 'Timestamp', 'date', 'tracked_at']:
                if col in app_df.columns:
                    timestamp_col = col
                    break
                    
            if timestamp_col is None:
                # No direct timestamp column - check if we have date and time columns
                if 'date' in app_df.columns and 'time' in app_df.columns:
                    app_df['timestamp'] = pd.to_datetime(app_df['date'] + ' ' + app_df['time'], 
                                                      format='mixed', 
                                                      dayfirst=True)
                else:
                    raise ValueError(f"No timestamp column found in {app_path}")
            else:
                app_df['timestamp'] = pd.to_datetime(app_df[timestamp_col])
            
            # Ensure we have a date column
            app_df['date'] = app_df['timestamp'].dt.date
            
            # Ensure we have an action column for SCREEN ON/OFF
            action_col = None
            for col in app_df.columns:
                if col.lower() == 'action' or 'screen' in col.lower():
                    action_col = col
                    break
                    
            if action_col is not None and action_col != DIGITAL_USE_COL:
                app_df[DIGITAL_USE_COL] = app_df[action_col]
                
            self.logger.debug(f"Loaded {len(app_df)} app events")
            return app_df
        except Exception as e:
            self.logger.error(f"Failed to load app data: {str(e)}")
            raise

    def process_digital_episodes(self, app_df: pd.DataFrame) -> Dict[datetime.date, pd.DataFrame]:
        """Process digital episodes by day"""
        episodes_by_day = {}
        
        # Check if DIGITAL_USE_COL exists
        if DIGITAL_USE_COL not in app_df.columns:
            self.logger.warning(f"Digital use column '{DIGITAL_USE_COL}' not found in app data. Available columns: {app_df.columns.tolist()}")
            return episodes_by_day
            
        for date, day_data in app_df.groupby('date'):
            self.logger.debug(f"Processing digital episodes for {date}")
            
            # Try to identify screen events
            screen_on_values = ['SCREEN ON', 'screen_on', 'SCREEN_ON', 'on', 'ON']
            screen_off_values = ['SCREEN OFF', 'screen_off', 'SCREEN_OFF', 'off', 'OFF']
            
            screen_events = day_data[day_data[DIGITAL_USE_COL].isin(screen_on_values + screen_off_values)].copy()
            screen_events = screen_events.sort_values('timestamp')
            
            if len(screen_events) == 0:
                self.logger.debug(f"No screen events found for {date}")
                continue
                
            # Map values to standard format
            screen_events.loc[screen_events[DIGITAL_USE_COL].isin(screen_on_values), DIGITAL_USE_COL] = 'SCREEN ON'
            screen_events.loc[screen_events[DIGITAL_USE_COL].isin(screen_off_values), DIGITAL_USE_COL] = 'SCREEN OFF'
                
            episodes = []
            current_on = None
            
            for _, row in screen_events.iterrows():
                if row[DIGITAL_USE_COL] == 'SCREEN ON' and not current_on:
                    current_on = row['timestamp']
                elif row[DIGITAL_USE_COL] == 'SCREEN OFF' and current_on:
                    episodes.append({
                        'start_time': current_on,
                        'end_time': row['timestamp'],
                        'state': 'digital'
                    })
                    current_on = None
            
            if len(episodes) == 0:
                self.logger.debug(f"No digital episodes detected for {date}")
            else:
                episodes_df = pd.DataFrame(episodes)
                episodes_df['duration'] = episodes_df['end_time'] - episodes_df['start_time']
                
                # Remove timezone information
                episodes_df['start_time'] = ensure_tz_naive(episodes_df['start_time'])
                episodes_df['end_time'] = ensure_tz_naive(episodes_df['end_time'])
                
                episodes_by_day[date] = episodes_df
                self.logger.debug(f"Detected {len(episodes)} digital episodes for {date}")
                
        return episodes_by_day

    def process_mobility_episodes(self, positionfixes: ti.Positionfixes) -> Dict[datetime.date, pd.DataFrame]:
        """
        Process mobility episodes using the Trackintel library
        """
        try:
            # Generate staypoints from positionfixes
            self.logger.debug("Generating staypoints from positionfixes")
            pfs, staypoints = positionfixes.generate_staypoints(
                method='sliding',
                dist_threshold=STAYPOINT_DISTANCE_THRESHOLD,
                time_threshold=STAYPOINT_TIME_THRESHOLD,
                gap_threshold=STAYPOINT_GAP_THRESHOLD
            )
            
            # Generate triplegs from positionfixes and staypoints
            self.logger.debug("Generating triplegs from positionfixes and staypoints")
            pfs, triplegs = pfs.generate_triplegs(staypoints, gap_threshold=STAYPOINT_GAP_THRESHOLD)
            
            # Flag staypoints as activities (add is_activity column)
            self.logger.debug("Adding activity flag to staypoints")
            staypoints = staypoints.create_activity_flag()
            
            # Generate trips from staypoints and triplegs
            self.logger.debug("Generating trips from staypoints and triplegs")
            staypoints, triplegs, trips = staypoints.generate_trips(triplegs, gap_threshold=TRIP_GAP_THRESHOLD)
            
            # Convert trips into our format grouped by day
            trips['date'] = trips['started_at'].dt.date
            episodes_by_day = {}
            
            # Add centroid and distance for trips (since they don't have point geometry by default)
            if not trips.empty:
                trips['latitude'] = np.nan
                trips['longitude'] = np.nan
                
                # Try to get origin staypoint coordinates
                if 'origin_staypoint_id' in trips.columns and not staypoints.empty:
                    for idx, trip in trips.iterrows():
                        if pd.notna(trip['origin_staypoint_id']):
                            origin_sp = staypoints[staypoints.index == trip['origin_staypoint_id']]
                            if not origin_sp.empty:
                                trips.at[idx, 'latitude'] = origin_sp.iloc[0].geometry.y
                                trips.at[idx, 'longitude'] = origin_sp.iloc[0].geometry.x
            
            for date, day_trips in trips.groupby('date'):
                # Create a DataFrame in our expected format
                mobility_episodes = pd.DataFrame({
                    'started_at': day_trips['started_at'],
                    'finished_at': day_trips['finished_at'],
                    'duration': day_trips['finished_at'] - day_trips['started_at'],
                    'latitude': day_trips['latitude'],
                    'longitude': day_trips['longitude'],
                    'state': 'mobility'
                })
                
                # Remove timezone information
                mobility_episodes['started_at'] = ensure_tz_naive(mobility_episodes['started_at'])
                mobility_episodes['finished_at'] = ensure_tz_naive(mobility_episodes['finished_at'])
                
                episodes_by_day[date] = mobility_episodes
                self.logger.debug(f"Processed {len(mobility_episodes)} mobility episodes for {date}")
            
            return episodes_by_day
            
        except Exception as e:
            self.logger.error(f"Error processing mobility episodes: {str(e)}")
            traceback.print_exc()
            return {}

    def create_daily_timeline(self, digital_episodes: pd.DataFrame, 
                          mobility_episodes: pd.DataFrame,
                          overlap_episodes: pd.DataFrame) -> pd.DataFrame:
        """Create a chronological timeline of all episodes for a day"""
        # Add episode type column to each DataFrame
        if not digital_episodes.empty:
            digital_episodes = digital_episodes.copy()
            digital_episodes['episode_type'] = 'digital'
            digital_episodes['movement_state'] = None
            # Add empty location columns if they don't exist
            if 'latitude' not in digital_episodes.columns:
                digital_episodes['latitude'] = np.nan
            if 'longitude' not in digital_episodes.columns:
                digital_episodes['longitude'] = np.nan
                
            # Ensure timezone-naive datetimes
            digital_episodes['start_time'] = ensure_tz_naive(digital_episodes['start_time'])
            digital_episodes['end_time'] = ensure_tz_naive(digital_episodes['end_time'])
        
        if not mobility_episodes.empty:
            mobility_episodes = mobility_episodes.copy()
            mobility_episodes['episode_type'] = 'mobility'
            mobility_episodes['movement_state'] = mobility_episodes['state']
            mobility_episodes = mobility_episodes.drop(columns=['state'])
            
            # Rename columns to match digital_episodes
            mobility_episodes = mobility_episodes.rename(columns={
                'started_at': 'start_time',
                'finished_at': 'end_time'
            })
            
            # Ensure timezone-naive datetimes
            mobility_episodes['start_time'] = ensure_tz_naive(mobility_episodes['start_time'])
            mobility_episodes['end_time'] = ensure_tz_naive(mobility_episodes['end_time'])
        
        if not overlap_episodes.empty:
            overlap_episodes = overlap_episodes.copy()
            overlap_episodes['episode_type'] = 'overlap'
            
            # Ensure timezone-naive datetimes
            overlap_episodes['start_time'] = ensure_tz_naive(overlap_episodes['start_time'])
            overlap_episodes['end_time'] = ensure_tz_naive(overlap_episodes['end_time'])
        
        # Combine all episodes
        all_episodes = pd.concat([digital_episodes, mobility_episodes, overlap_episodes], 
                               ignore_index=True)
        
        # Sort chronologically
        if not all_episodes.empty:
            # Final check for all datetimes to be naive
            all_episodes['start_time'] = ensure_tz_naive(all_episodes['start_time'])
            all_episodes['end_time'] = ensure_tz_naive(all_episodes['end_time'])
            
            all_episodes = all_episodes.sort_values('start_time')
            
            # Add sequential episode number
            all_episodes['episode_number'] = range(1, len(all_episodes) + 1)
            
            # Calculate time since previous episode
            all_episodes['time_since_prev'] = all_episodes['start_time'].diff()
            
            # Define desired column order
            desired_columns = ['episode_number', 'episode_type', 'movement_state', 
                             'start_time', 'end_time', 'duration', 'time_since_prev',
                             'latitude', 'longitude']
            
            # Only select columns that exist in the dataframe
            existing_columns = [col for col in desired_columns if col in all_episodes.columns]
            all_episodes = all_episodes[existing_columns]
        
        return all_episodes

    def process_day(self, date: datetime.date, digital_episodes: pd.DataFrame, 
                   mobility_episodes: pd.DataFrame) -> dict:
        """Process a single day and generate statistics"""
        overlap_episodes = self._find_overlaps(digital_episodes, mobility_episodes)
        
        # Create daily timeline
        daily_timeline = self.create_daily_timeline(digital_episodes, mobility_episodes, overlap_episodes)
        
        # Save daily timeline
        if not daily_timeline.empty:
            timeline_file = self.output_dir / f"{date}_daily_timeline.csv"
            daily_timeline.to_csv(timeline_file, index=False)
            self.logger.debug(f"Saved daily timeline to {timeline_file}")
        
        # Count mobility and stationary episodes
        mobility_count = len(mobility_episodes) if not mobility_episodes.empty else 0
        mobility_duration = 0
        
        if not mobility_episodes.empty:
            # Calculate mobility duration in minutes
            duration_timedeltas = mobility_episodes['duration']
            mobility_duration = sum(dt.total_seconds() for dt in duration_timedeltas) / 60
        
        # Calculate statistics
        day_stats = {
            'user': self.participant_id,
            'date': date,
            'digital_episodes': len(digital_episodes),
            'mobility_episodes': mobility_count,
            'stationary_episodes': 0,  # Not using this with trackintel approach
            'overlap_episodes': len(overlap_episodes),
            'digital_duration': digital_episodes['duration'].sum().total_seconds() / 60 if not digital_episodes.empty and 'duration' in digital_episodes.columns else 0,
            'mobility_duration': mobility_duration,
            'stationary_duration': 0,  # Not using this with trackintel approach
            'overlap_duration': overlap_episodes['duration'].sum().total_seconds() / 60 if not overlap_episodes.empty and 'duration' in overlap_episodes.columns else 0,
        }
        
        # Save episodes
        for ep_type, episodes in [
            ('digital', digital_episodes),
            ('mobility', mobility_episodes),
            ('overlap', overlap_episodes)
        ]:
            if len(episodes) > 0:
                output_file = self.output_dir / f"{date}_{ep_type}_episodes.csv"
                episodes.to_csv(output_file, index=False)
                self.logger.debug(f"Saved {ep_type} episodes to {output_file}")
        
        return day_stats

    def process(self) -> List[dict]:
        """Main processing pipeline"""
        try:
            # Load data
            positionfixes = self.load_gps_data()
            app_df = self.load_app_data()
            
            # Process episodes
            digital_episodes = self.process_digital_episodes(app_df)
            mobility_episodes = self.process_mobility_episodes(positionfixes)
            
            # Process each day
            all_stats = []
            all_dates = sorted(set(digital_episodes.keys()) | set(mobility_episodes.keys()))
            
            for date in all_dates:
                digital_eps = digital_episodes.get(date, pd.DataFrame())
                mobility_eps = mobility_episodes.get(date, pd.DataFrame())
                
                if len(digital_eps) == 0:
                    self.logger.debug(f"No digital episodes for {date}")
                if len(mobility_eps) == 0:
                    self.logger.debug(f"No mobility episodes for {date}")
                
                day_stats = self.process_day(date, digital_eps, mobility_eps)
                all_stats.append(day_stats)
            
            # Save summary statistics
            summary_df = pd.DataFrame(all_stats)
            summary_file = self.output_dir / 'episode_summary.csv'
            summary_df.to_csv(summary_file, index=False)
            self.logger.debug(f"Saved summary statistics to {summary_file}")
            
            return all_stats
            
        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            return []

def main():
    # Find valid participants
    qstarz_files = {f.stem.replace('_qstarz_prep', ''): f 
                    for f in GPS_PREP_DIR.glob('*_qstarz_prep.csv')
                    if not f.stem.startswith('._')}  # Filter out macOS hidden files
    app_files = {f.stem.replace('_app_prep', ''): f 
                 for f in GPS_PREP_DIR.glob('*_app_prep.csv')
                 if not f.stem.startswith('._')}  # Filter out macOS hidden files
    
    common_ids = set(qstarz_files.keys()) & set(app_files.keys())
    # Filter out macOS hidden files (like '._005')
    common_ids = {pid for pid in common_ids if not pid.startswith('._')}
    logging.info(f"Found {len(common_ids)} participants with complete data")
    
    # Show a progress bar but suppress detailed logging during processing
    logging.getLogger().setLevel(logging.WARNING)
    
    # Track processing statistics
    all_stats = []
    participant_summaries = []
    processed_count = 0
    
    for pid in tqdm(common_ids, desc="Processing participants", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
        # Skip macOS hidden files
        if pid.startswith('._'):
            logging.warning(f"Skipping macOS hidden file participant: {pid}")
            continue
            
        try:
            processor = EpisodeProcessor(pid)
            participant_stats = processor.process()
            
            if participant_stats:
                all_stats.extend(participant_stats)
                
                # Calculate participant summary
                participant_df = pd.DataFrame(participant_stats)
                
                participant_summary = {
                    'participant_id': pid,
                    'days_of_data': len(participant_df),
                    'avg_digital_episodes': participant_df['digital_episodes'].mean(),
                    'avg_mobility_episodes': participant_df['mobility_episodes'].mean(),
                    'avg_stationary_episodes': participant_df['stationary_episodes'].mean(),
                    'avg_overlap_episodes': participant_df['overlap_episodes'].mean(),
                    'avg_digital_mins': participant_df['digital_duration'].mean(),
                    'avg_mobility_mins': participant_df['mobility_duration'].mean(),
                    'avg_stationary_mins': participant_df['stationary_duration'].mean(),
                    'avg_overlap_mins': participant_df['overlap_duration'].mean(),
                    'total_digital_mins': participant_df['digital_duration'].sum(),
                    'total_mobility_mins': participant_df['mobility_duration'].sum(),
                    'total_stationary_mins': participant_df['stationary_duration'].sum(),
                    'total_overlap_mins': participant_df['overlap_duration'].sum(),
                }
                
                participant_summaries.append(participant_summary)
                processed_count += 1
                
        except Exception as e:
            logging.error(f"Error processing participant {pid}: {str(e)}")
            continue
    
    # Restore logging level
    logging.getLogger().setLevel(logging.INFO)
    
    if all_stats:
        # Create overall summary
        all_summary = pd.DataFrame(all_stats)
        summary_file = EPISODE_OUTPUT_DIR / 'all_participants_summary.csv'
        all_summary.to_csv(summary_file, index=False)
        
        # Save participant summaries
        participant_summary_df = pd.DataFrame(participant_summaries)
        participant_summary_file = EPISODE_OUTPUT_DIR / 'participant_summaries.csv'
        participant_summary_df.to_csv(participant_summary_file, index=False)
        
        # Log concise summary to terminal
        summary_logger.info("\n" + "="*60)
        summary_logger.info(f"MOBILITY DETECTION SUMMARY (TRACKINTEL)")
        summary_logger.info("="*60)
        summary_logger.info(f"Successfully processed {processed_count}/{len(common_ids)} participants")
        summary_logger.info(f"Total days processed: {len(all_summary)}")
        
        # Overall statistics
        summary_logger.info("\nAVERAGE DAILY EPISODES PER PARTICIPANT:")
        summary_logger.info(f"Digital: {participant_summary_df['avg_digital_episodes'].mean():.2f}")
        summary_logger.info(f"Mobility: {participant_summary_df['avg_mobility_episodes'].mean():.2f}")
        summary_logger.info(f"Overlapping: {participant_summary_df['avg_overlap_episodes'].mean():.2f}")
        
        summary_logger.info("\nAVERAGE DAILY DURATION PER PARTICIPANT (minutes):")
        summary_logger.info(f"Digital: {participant_summary_df['avg_digital_mins'].mean():.2f}")
        summary_logger.info(f"Mobility: {participant_summary_df['avg_mobility_mins'].mean():.2f}")
        summary_logger.info(f"Overlapping: {participant_summary_df['avg_overlap_mins'].mean():.2f}")
        
        # Total study statistics
        summary_logger.info("\nSTUDY TOTALS:")
        summary_logger.info(f"Total digital episodes: {all_summary['digital_episodes'].sum()}")
        summary_logger.info(f"Total mobility episodes: {all_summary['mobility_episodes'].sum()}")
        summary_logger.info(f"Total overlap episodes: {all_summary['overlap_episodes'].sum()}")
        summary_logger.info(f"Total digital duration (hours): {all_summary['digital_duration'].sum()/60:.2f}")
        summary_logger.info(f"Total mobility duration (hours): {all_summary['mobility_duration'].sum()/60:.2f}")
        summary_logger.info(f"Total overlap duration (hours): {all_summary['overlap_duration'].sum()/60:.2f}")
        
        # Statistics by participant
        summary_logger.info("\nPARTICIPANT RANGE (min-max):")
        summary_logger.info(f"Days of data: {participant_summary_df['days_of_data'].min()}-"
                           f"{participant_summary_df['days_of_data'].max()}")
        summary_logger.info(f"Avg daily digital episodes: {participant_summary_df['avg_digital_episodes'].min():.1f}-"
                           f"{participant_summary_df['avg_digital_episodes'].max():.1f}")
        summary_logger.info(f"Avg daily mobility episodes: {participant_summary_df['avg_mobility_episodes'].min():.1f}-"
                           f"{participant_summary_df['avg_mobility_episodes'].max():.1f}")
        summary_logger.info(f"Avg daily digital duration (mins): {participant_summary_df['avg_digital_mins'].min():.1f}-"
                           f"{participant_summary_df['avg_digital_mins'].max():.1f}")
        summary_logger.info(f"Avg daily mobility duration (mins): {participant_summary_df['avg_mobility_mins'].min():.1f}-"
                           f"{participant_summary_df['avg_mobility_mins'].max():.1f}")
        
        # Output file locations
        summary_logger.info("\nOUTPUT FILES:")
        summary_logger.info(f"Detailed logs: episode_detection.log")
        summary_logger.info(f"All participants summary: {summary_file}")
        summary_logger.info(f"Participant-level summary: {participant_summary_file}")
        summary_logger.info("="*60)

if __name__ == "__main__":
    main()