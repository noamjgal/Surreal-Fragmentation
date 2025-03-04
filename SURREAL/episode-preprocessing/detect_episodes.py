#!/usr/bin/env python3
"""
Enhanced episode detection using Trackintel library
Focuses on identifying mobility between locations versus stationary periods
with improved data quality filtering and failure handling
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
from data_utils import DataCleaner

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

# Updated parameters with more permissive thresholds
STAYPOINT_DISTANCE_THRESHOLD = 50  # meters - REDUCED from 150 to be more sensitive to smaller movements
STAYPOINT_TIME_THRESHOLD = 5.0  # minutes - REDUCED from 10.0 to detect shorter stops
STAYPOINT_GAP_THRESHOLD = 60.0  # minutes - INCREASED from 30.0 to be more tolerant of gaps
LOCATION_EPSILON = 100  # meters - REDUCED from 200 to improve location detection
LOCATION_MIN_SAMPLES = 1  # minimum number of staypoints to form a significant location
TRIP_GAP_THRESHOLD = 45  # minutes - ADJUSTED for better trip segmentation

# Updated fallback detection parameters
MIN_MOVEMENT_SPEED = 20  # meters per minute (approx. 1.2 km/h) - REDUCED from 30 to detect slower walking
MAX_REASONABLE_SPEED = 2500  # meters per minute (approx. 150 km/h) for filtering

# More permissive data quality thresholds
MIN_GPS_POINTS_PER_DAY = 5  # REDUCED from 10 to be more permissive with sparse data
MAX_ACCEPTABLE_GAP_PERCENT = 60  # INCREASED from 40 to tolerate more gaps
MIN_TRACK_DURATION_HOURS = 1  # REDUCED from 2 hours to accept shorter valid periods

# Define digital usage column name based on app data sample
DIGITAL_USE_COL = 'action'  # Column containing screen events (SCREEN ON/OFF, etc.)

def ensure_tz_naive(datetime_series: pd.Series) -> pd.Series:
    """Convert a datetime series to timezone-naive if it has a timezone"""
    if datetime_series.empty:
        return datetime_series
        
    if hasattr(datetime_series.iloc[0], 'tz') and datetime_series.iloc[0].tz is not None:
        return datetime_series.dt.tz_localize(None)
    return datetime_series

def ensure_tz_aware(datetime_series: pd.Series) -> pd.Series:
    """Ensure a datetime series has timezone info (UTC)"""
    if datetime_series.empty:
        return datetime_series
        
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_dtype(datetime_series):
        datetime_series = pd.to_datetime(datetime_series)
        
    # Add timezone if missing
    if hasattr(datetime_series.iloc[0], 'tz') and datetime_series.iloc[0].tz is None:
        return datetime_series.dt.tz_localize('UTC')
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
        
        # Create DataCleaner instance
        self.data_cleaner = DataCleaner(self.logger)
        
        # Standardize ID
        self.participant_id_clean = self.data_cleaner.standardize_participant_id(participant_id)
        
        # Track processing status for each day
        self.day_processing_status = {}
        
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

    def _filter_gps_data(self, gps_df, datetime_col, lat_col, lon_col):
        """Filter GPS data to remove outliers and improve quality"""
        self.logger.debug(f"Filtering GPS data - original points: {len(gps_df)}")
        
        if len(gps_df) <= 1:
            return gps_df
            
        # Sort by timestamp
        gps_df = gps_df.sort_values(datetime_col)
        
        # Remove duplicate timestamps
        gps_df = gps_df.drop_duplicates(subset=[datetime_col])
        
        # NEW: Filter accuracy values if available
        if 'accuracy' in gps_df.columns:
            before_count = len(gps_df)
            gps_df = gps_df[(gps_df['accuracy'].isna()) | (gps_df['accuracy'] < 100)]
            filtered_count = before_count - len(gps_df)
            if filtered_count > 0:
                self.logger.debug(f"Filtered {filtered_count} points with poor accuracy")
        
        # NEW: Smooth GPS trajectories for noisy data
        if len(gps_df) >= 3:  # Need at least 3 points for a rolling window of 3
            gps_df['latitude_smooth'] = gps_df[lat_col].rolling(window=3, center=True).mean().fillna(gps_df[lat_col])
            gps_df['longitude_smooth'] = gps_df[lon_col].rolling(window=3, center=True).mean().fillna(gps_df[lon_col])
            # Use smoothed coordinates for further processing
            lat_col_proc = 'latitude_smooth'
            lon_col_proc = 'longitude_smooth'
        else:
            lat_col_proc = lat_col
            lon_col_proc = lon_col
        
        # Calculate speeds between consecutive points
        gps_df['prev_lat'] = gps_df[lat_col_proc].shift(1)
        gps_df['prev_lon'] = gps_df[lon_col_proc].shift(1)
        gps_df['time_diff'] = (gps_df[datetime_col].diff()).dt.total_seconds() / 60  # minutes
        
        # Only calculate speed where we have valid time differences
        mask = (gps_df['time_diff'] > 0)
        if mask.any():
            # Calculate rough distance in meters (using approximate conversion - 1 degree ~ 111km at equator)
            gps_df.loc[mask, 'distance'] = np.sqrt(
                ((gps_df.loc[mask, lat_col_proc] - gps_df.loc[mask, 'prev_lat']) * 111000)**2 + 
                ((gps_df.loc[mask, lon_col_proc] - gps_df.loc[mask, 'prev_lon']) * 
                 111000 * np.cos(np.radians(gps_df.loc[mask, lat_col_proc])))**2
            )
            
            # Calculate speed in meters per minute
            gps_df.loc[mask, 'speed'] = gps_df.loc[mask, 'distance'] / gps_df.loc[mask, 'time_diff']
            
            # Filter out unreasonable speeds
            gps_df = gps_df[(gps_df['speed'].isna()) | (gps_df['speed'] <= MAX_REASONABLE_SPEED)]
        
        # Copy smoothed coordinates back to original columns if used
        if 'latitude_smooth' in gps_df.columns:
            gps_df[lat_col] = gps_df['latitude_smooth']
            gps_df[lon_col] = gps_df['longitude_smooth']
        
        # Remove temporary columns
        temp_cols = ['prev_lat', 'prev_lon', 'time_diff', 'distance', 'speed', 
                     'latitude_smooth', 'longitude_smooth']
        gps_df = gps_df.drop(columns=[c for c in temp_cols if c in gps_df.columns])
        
        self.logger.debug(f"Filtered GPS data - remaining points: {len(gps_df)}")
        return gps_df

    def assess_day_quality(self, day_positionfixes: gpd.GeoDataFrame) -> Tuple[bool, dict]:
        """
        Assess the quality of GPS data for a single day
        Returns (is_valid, quality_stats)
        """
        quality_stats = {
            'total_points': len(day_positionfixes),
            'valid': False,
            'failure_reason': None
        }
        
        # Check minimum number of points
        if len(day_positionfixes) < MIN_GPS_POINTS_PER_DAY:
            quality_stats['failure_reason'] = f"Insufficient GPS points ({len(day_positionfixes)} < {MIN_GPS_POINTS_PER_DAY})"
            return False, quality_stats
            
        # Sort by timestamp
        day_positionfixes = day_positionfixes.sort_values('tracked_at')
        
        # Check day duration
        day_duration_hours = (day_positionfixes['tracked_at'].max() - 
                              day_positionfixes['tracked_at'].min()).total_seconds() / 3600
        quality_stats['duration_hours'] = day_duration_hours
        
        if day_duration_hours < MIN_TRACK_DURATION_HOURS:
            quality_stats['failure_reason'] = f"Day duration too short ({day_duration_hours:.1f} < {MIN_TRACK_DURATION_HOURS} hours)"
            return False, quality_stats
            
        # Check time gaps
        day_positionfixes['time_diff'] = day_positionfixes['tracked_at'].diff().dt.total_seconds() / 60  # minutes
        large_gaps = day_positionfixes['time_diff'] > 5  # gaps > 5 minutes
        percent_large_gaps = 100 * large_gaps.sum() / max(1, len(day_positionfixes) - 1)
        quality_stats['percent_large_gaps'] = percent_large_gaps
        quality_stats['median_gap_minutes'] = day_positionfixes['time_diff'].median()
        
        if percent_large_gaps > MAX_ACCEPTABLE_GAP_PERCENT:
            quality_stats['failure_reason'] = f"Too many large gaps ({percent_large_gaps:.1f}% > {MAX_ACCEPTABLE_GAP_PERCENT}%)"
            return False, quality_stats
            
        # Made it through all checks
        quality_stats['valid'] = True
        return True, quality_stats

    def load_gps_data(self) -> Optional[ti.Positionfixes]:
        """Load GPS data with validation"""
        gps_path = GPS_PREP_DIR / f'{self.participant_id}_gps_prep.csv'
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
                    
                    # Always ensure timestamps are timezone-aware
                    gps_df['tracked_at'] = ensure_tz_aware(gps_df['tracked_at'])
                        
                    # Return as Positionfixes
                    return ti.Positionfixes(gdf)
                else:
                    raise ValueError(f"Could not find latitude/longitude columns in {gps_path}")
            
            # Additional data quality filtering
            gps_df = self._filter_gps_data(gps_df, datetime_col, lat_col, lon_col)
            
            # Convert to trackintel's positionfixes format
            positionfixes = pd.DataFrame({
                'user_id': self.participant_id,
                'tracked_at': gps_df[datetime_col],
                'latitude': gps_df[lat_col],
                'longitude': gps_df[lon_col],
                'elevation': np.nan,  # Optional
                'accuracy': np.nan,   # Optional
            })
            
            # Always ensure tracked_at is timezone aware (required by trackintel)
            positionfixes['tracked_at'] = ensure_tz_aware(positionfixes['tracked_at'])
            
            # Convert to GeoDataFrame and set as trackintel Positionfixes
            geometry = [Point(lon, lat) for lon, lat in zip(positionfixes['longitude'], positionfixes['latitude'])]
            positionfixes = gpd.GeoDataFrame(positionfixes, geometry=geometry, crs="EPSG:4326")
            
            # Set as trackintel Positionfixes
            positionfixes = ti.Positionfixes(positionfixes)
            
            self.logger.debug(f"Loaded {len(positionfixes)} GPS points")
            
            # Debug GPS data quality
            self._debug_gps_quality(positionfixes)
            
            return positionfixes
        except Exception as e:
            self.logger.error(f"Failed to load GPS data: {str(e)}")
            raise

    def _debug_gps_quality(self, positionfixes):
        """Generate statistics about GPS data quality"""
        if len(positionfixes) <= 1:
            self.logger.warning(f"Insufficient GPS data for quality assessment: only {len(positionfixes)} points")
            return
            
        # Calculate time gaps between consecutive points
        pfs_df = positionfixes.copy()
        pfs_df = pfs_df.sort_values('tracked_at')
        pfs_df['time_diff'] = pfs_df['tracked_at'].diff().dt.total_seconds()
        
        # Calculate basic statistics
        stats = {
            'total_points': len(pfs_df),
            'unique_days': len(pfs_df['tracked_at'].dt.date.unique()),
            'median_time_gap_seconds': pfs_df['time_diff'].median(),
            'mean_time_gap_seconds': pfs_df['time_diff'].mean(),
            'max_time_gap_seconds': pfs_df['time_diff'].max(),
            'points_with_large_gaps': sum(pfs_df['time_diff'] > 300),  # > 5 min
            'percent_large_gaps': round(sum(pfs_df['time_diff'] > 300) / max(1, len(pfs_df) - 1) * 100, 1)
        }
        
        self.logger.info(f"GPS quality stats for {self.participant_id}: {stats}")
        
        # Warn if the data quality might cause issues with trackintel
        if stats['median_time_gap_seconds'] > STAYPOINT_GAP_THRESHOLD * 60:
            self.logger.warning(f"Median GPS time gap ({stats['median_time_gap_seconds']} sec) exceeds the staypoint gap threshold ({STAYPOINT_GAP_THRESHOLD*60} sec)")
            
        if stats['percent_large_gaps'] > 25:
            self.logger.warning(f"High percentage of large gaps in GPS data: {stats['percent_large_gaps']}%")

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
            
            # ENHANCED: Expanded list of screen event patterns
            screen_on_values = [
                'SCREEN ON', 'screen_on', 'SCREEN_ON', 'on', 'ON',
                'UNLOCK', 'unlock', 'STARTED', 'started'  # Added UNLOCK and STARTED
            ]
            screen_off_values = [
                'SCREEN OFF', 'screen_off', 'SCREEN_OFF', 'off', 'OFF',
                'LOCK', 'lock', 'LOCK SCREEN', 'PAUSED', 'paused'  # Added LOCK SCREEN and PAUSED
            ]
            
            # NEW: Special handling for rapid STARTED/PAUSED sequences
            # Group closely timed STARTED/PAUSED events (within 5 seconds) to avoid tiny episodes
            screen_events = day_data.copy()
            screen_events = screen_events.sort_values('timestamp')
            
            # Filter to only include relevant events
            screen_events = screen_events[screen_events[DIGITAL_USE_COL].isin(screen_on_values + screen_off_values)].copy()
            
            if len(screen_events) == 0:
                self.logger.debug(f"No screen events found for {date}")
                continue
            
            # Map values to standard format
            screen_events.loc[screen_events[DIGITAL_USE_COL].isin(screen_on_values), DIGITAL_USE_COL] = 'SCREEN ON'
            screen_events.loc[screen_events[DIGITAL_USE_COL].isin(screen_off_values), DIGITAL_USE_COL] = 'SCREEN OFF'
            
            # NEW: Filter out rapid on/off sequences (less than 3 seconds)
            screen_events['prev_time'] = screen_events['timestamp'].shift(1)
            screen_events['prev_action'] = screen_events[DIGITAL_USE_COL].shift(1)
            screen_events['time_diff'] = (screen_events['timestamp'] - screen_events['prev_time']).dt.total_seconds()
            
            # Mark events to remove (ON followed by OFF within 3 seconds)
            remove_mask = (screen_events[DIGITAL_USE_COL] == 'SCREEN OFF') & \
                         (screen_events['prev_action'] == 'SCREEN ON') & \
                         (screen_events['time_diff'] < 3)
            
            # Also mark the preceding ON events
            remove_indices = screen_events.index[remove_mask].tolist()
            prev_indices = [idx-1 for idx in remove_indices if idx-1 in screen_events.index]
            all_remove = remove_indices + prev_indices
            
            # Filter events
            if all_remove:
                self.logger.debug(f"Removing {len(all_remove)} rapid screen events")
                screen_events = screen_events.drop(all_remove)
            
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
    
    def _handle_stationary_data(self, positionfixes: gpd.GeoDataFrame) -> bool:
        """
        NEW: Handle cases where the device is completely stationary
        Returns True if data is stationary and was handled
        """
        if positionfixes.empty or len(positionfixes) < 3:
            return False
        
        # Calculate max distance between any points
        pfs = positionfixes.copy()
        
        # Create a centroid point
        center_lat = pfs['latitude'].mean()
        center_lon = pfs['longitude'].mean()
        
        # Calculate distance from each point to centroid (approximate in meters)
        pfs['dist_to_center'] = np.sqrt(
            ((pfs['latitude'] - center_lat) * 111000)**2 + 
            ((pfs['longitude'] - center_lon) * 111000 * np.cos(np.radians(pfs['latitude'])))**2
        )
        
        max_distance = pfs['dist_to_center'].max()
        duration_minutes = (pfs['tracked_at'].max() - pfs['tracked_at'].min()).total_seconds() / 60
        
        # Check if data represents a mostly stationary period
        if max_distance < 10 and duration_minutes > 30:  # Less than 10m movement over 30+ minutes
            self.logger.info(f"Detected stationary period ({max_distance:.1f}m max distance over {duration_minutes:.1f} minutes)")
            return True
        
        return False

    def _adaptive_parameter_selection(self, positionfixes: gpd.GeoDataFrame) -> dict:
        """
        NEW: Adaptively select parameters based on data characteristics
        """
        if positionfixes.empty:
            return {
                'dist_threshold': STAYPOINT_DISTANCE_THRESHOLD,
                'time_threshold': STAYPOINT_TIME_THRESHOLD,
                'gap_threshold': STAYPOINT_GAP_THRESHOLD
            }
        
        # Determine data source if possible (Qstarz vs smartphone)
        data_source = 'unknown'
        
        # Calculate sampling interval
        pfs = positionfixes.copy().sort_values('tracked_at')
        pfs['time_diff'] = pfs['tracked_at'].diff().dt.total_seconds()
        median_interval = pfs['time_diff'].median()
        
        # Determine data source based on sampling frequency
        if 0 < median_interval <= 10:  # High frequency (likely Qstarz)
            data_source = 'qstarz'
        elif 10 < median_interval <= 60:  # Medium frequency (likely smartphone)
            data_source = 'smartphone'
        else:  # Low frequency or irregular
            data_source = 'irregular'
        
        # Adjust parameters based on data source
        params = {}
        if data_source == 'qstarz':  # Higher frequency data
            params = {
                'dist_threshold': 30,  # More precise with high-freq data
                'time_threshold': 3.0,  # Can detect shorter stops
                'gap_threshold': 30.0   # Can use shorter gaps
            }
        elif data_source == 'smartphone':  # Lower frequency data
            params = {
                'dist_threshold': 70,   # More tolerant with low-freq data
                'time_threshold': 8.0,  # Need longer stops for confidence
                'gap_threshold': 45.0   # Moderate gap threshold
            }
        else:  # Irregular data
            params = {
                'dist_threshold': 50,   # Default
                'time_threshold': 5.0,  # Default
                'gap_threshold': 60.0   # More tolerant of gaps
            }
        
        self.logger.info(f"Adaptively selected parameters for {data_source} data source (median sampling: {median_interval:.1f}s): {params}")
        return params

    def _fallback_mobility_detection(self, positionfixes) -> Dict[datetime.date, pd.DataFrame]:
        """Enhanced fallback mobility detection with improved sensitivity"""
        self.logger.info("Using enhanced fallback mobility detection method")
        
        episodes_by_day = {}
        
        if positionfixes.empty or len(positionfixes) <= 1:
            self.logger.warning("Insufficient GPS points for fallback mobility detection")
            return episodes_by_day
            
        # Convert to pandas DataFrame for easier manipulation
        pfs = positionfixes.copy()
        
        # Ensure we're working with a DataFrame, not GeoDataFrame
        if isinstance(pfs, gpd.GeoDataFrame):
            pfs = pd.DataFrame(pfs.drop(columns='geometry'))
        
        # Sort by timestamp
        pfs = pfs.sort_values('tracked_at')
        
        # Add date column
        pfs['date'] = pfs['tracked_at'].dt.date
        
        # Calculate time differences and distances
        pfs['prev_lat'] = pfs['latitude'].shift(1)
        pfs['prev_lon'] = pfs['longitude'].shift(1)
        pfs['prev_time'] = pfs['tracked_at'].shift(1)
        pfs['time_diff'] = (pfs['tracked_at'] - pfs['prev_time']).dt.total_seconds() / 60  # minutes
        
        # Only calculate where we have consecutive points
        mask = (pfs['time_diff'] > 0) & (pfs['time_diff'] < STAYPOINT_GAP_THRESHOLD)  # Use the same gap threshold
        
        if mask.any():
            # Calculate approximate distance in meters
            pfs.loc[mask, 'distance'] = np.sqrt(
                ((pfs.loc[mask, 'latitude'] - pfs.loc[mask, 'prev_lat']) * 111000)**2 + 
                ((pfs.loc[mask, 'longitude'] - pfs.loc[mask, 'prev_lon']) * 
                 111000 * np.cos(np.radians(pfs.loc[mask, 'latitude'])))**2
            )
            
            # Calculate speed in meters per minute
            pfs.loc[mask, 'speed'] = pfs.loc[mask, 'distance'] / pfs.loc[mask, 'time_diff']
            
            # ENHANCED: Dynamic threshold for movement detection
            # Calculate median speed to adapt to individual movement patterns
            median_speed = pfs.loc[mask, 'speed'].median()
            speed_threshold = min(MAX_REASONABLE_SPEED, max(MIN_MOVEMENT_SPEED, median_speed * 0.7))
            
            self.logger.debug(f"Using dynamic speed threshold: {speed_threshold:.1f} m/min")
            
            # Mark points as moving if speed exceeds threshold
            pfs['moving'] = (pfs['speed'] > speed_threshold)
            
            # ENHANCED: Use distance thresholds for low-frequency data
            # If consecutive points are far apart, consider it movement even if time difference is large
            if mask.any() and pfs.loc[mask, 'distance'].max() > 100:  # At least 100m between points
                far_points = pfs['distance'] > 100
                pfs.loc[far_points, 'moving'] = True
            
            # Group consecutive moving points into trips
            pfs['trip_start'] = pfs['moving'] & ~pfs['moving'].shift(1, fill_value=False)
            pfs['trip_end'] = ~pfs['moving'] & pfs['moving'].shift(1, fill_value=False)
            
            trip_starts = pfs[pfs['trip_start']].copy()
            trip_ends = pfs[pfs['trip_end']].copy()
            
            # Create trips where we have both start and end
            if not trip_starts.empty and not trip_ends.empty:
                trips = []
                
                for _, start_row in trip_starts.iterrows():
                    # Find the next end after this start
                    end_candidates = trip_ends[trip_ends['tracked_at'] > start_row['tracked_at']]
                    
                    if not end_candidates.empty:
                        end_row = end_candidates.iloc[0]
                        
                        # UPDATED: Accept trips of any duration > 0
                        duration = (end_row['tracked_at'] - start_row['tracked_at']).total_seconds() / 60
                        if duration > 0:
                            trips.append({
                                'started_at': start_row['tracked_at'],
                                'finished_at': end_row['tracked_at'],
                                'latitude': start_row['latitude'],
                                'longitude': start_row['longitude'],
                                'date': start_row['date'],
                                'duration': duration
                            })
            
            # Group trips by day
            for trip in trips:
                date = trip['date']
                if date not in episodes_by_day:
                    episodes_by_day[date] = pd.DataFrame()
                
                new_trip = pd.DataFrame([{
                    'started_at': trip['started_at'],
                    'finished_at': trip['finished_at'],
                    'latitude': trip['latitude'],
                    'longitude': trip['longitude'],
                    'duration': pd.Timedelta(minutes=trip['duration']),
                    'state': 'mobility'
                }])
                
                episodes_by_day[date] = pd.concat([episodes_by_day[date], new_trip], ignore_index=True)
        
        # If we have any days with trips, log the results
        if episodes_by_day:
            total_trips = sum(len(df) for df in episodes_by_day.values())
            self.logger.info(f"Enhanced fallback method detected {total_trips} mobility episodes across {len(episodes_by_day)} days")
        else:
            self.logger.warning("Enhanced fallback method did not detect any mobility episodes")
            
        return episodes_by_day

    def process_mobility_episodes(self, positionfixes: ti.Positionfixes) -> Dict[datetime.date, pd.DataFrame]:
        """
        Process mobility episodes using the Trackintel library with enhanced fallback methods
        """
        try:
            # Skip if we have too few points
            if positionfixes.empty or len(positionfixes) <= 5:
                self.logger.warning(f"Insufficient GPS points ({len(positionfixes) if not positionfixes.empty else 0}) for mobility detection")
                return {}
                
            # Split the data by day and filter low-quality days
            pfs_by_day = {}
            valid_days = 0
            positionfixes_copy = positionfixes.copy()
            positionfixes_copy['date'] = positionfixes_copy['tracked_at'].dt.date
            
            for date, day_positionfixes in positionfixes_copy.groupby('date'):
                is_valid, quality_stats = self.assess_day_quality(day_positionfixes)
                self.day_processing_status[date] = {
                    'stage': 'data_quality',
                    'valid': is_valid,
                    'stats': quality_stats
                }
                
                if is_valid:
                    pfs_by_day[date] = day_positionfixes
                    valid_days += 1
                else:
                    self.logger.warning(f"Filtering out day {date} due to poor data quality: {quality_stats['failure_reason']}")
            
            if valid_days == 0:
                self.logger.warning("No days with valid GPS data quality")
                return {}
                
            self.logger.info(f"Processing {valid_days} days with valid GPS data")
            
            # Process each day separately and combine results
            all_mobility_episodes = {}
            
            for date, day_positionfixes in pfs_by_day.items():
                try:
                    # Check if this is stationary data first
                    if self._handle_stationary_data(day_positionfixes):
                        self.logger.info(f"Using stationary handling for {date}")
                        # Create a single stationary episode covering the whole period
                        mobility_episodes = pd.DataFrame([{
                            'started_at': day_positionfixes['tracked_at'].min(),
                            'finished_at': day_positionfixes['tracked_at'].max(),
                            'duration': day_positionfixes['tracked_at'].max() - day_positionfixes['tracked_at'].min(),
                            'latitude': day_positionfixes['latitude'].mean(),
                            'longitude': day_positionfixes['longitude'].mean(),
                            'state': 'stationary'  # Mark as stationary, not mobility
                        }])
                        
                        # Remove timezone information
                        mobility_episodes['started_at'] = ensure_tz_naive(mobility_episodes['started_at'])
                        mobility_episodes['finished_at'] = ensure_tz_naive(mobility_episodes['finished_at'])
                        
                        all_mobility_episodes[date] = mobility_episodes
                        self.day_processing_status[date] = {
                            'stage': 'completed', 
                            'valid': True,
                            'method': 'stationary_handler',
                            'episodes': len(mobility_episodes)
                        }
                        continue
                    
                    # Use adaptive parameter selection for trackintel
                    adaptive_params = self._adaptive_parameter_selection(day_positionfixes)
                    
                    # Generate staypoints from positionfixes
                    self.logger.debug(f"Generating staypoints for {date} with adaptive parameters")
                    
                    # Ensure tracked_at is timezone-aware
                    day_positionfixes['tracked_at'] = ensure_tz_aware(day_positionfixes['tracked_at'])
                    
                    # Create a fresh Positionfixes object for this day
                    day_pfs = ti.Positionfixes(day_positionfixes)
                    
                    # MULTI-LEVEL FALLBACK STRATEGY
                    # Level 1: Try with adaptive parameters
                    try:
                        day_pfs, staypoints = day_pfs.generate_staypoints(
                            method='sliding',
                            dist_threshold=adaptive_params['dist_threshold'],
                            time_threshold=adaptive_params['time_threshold'],
                            gap_threshold=adaptive_params['gap_threshold']
                        )
                    except Exception as e:
                        self.logger.warning(f"Error with adaptive parameters: {str(e)}")
                        staypoints = gpd.GeoDataFrame()
                    
                    # Level 2: If Level 1 failed, try with more permissive parameters
                    if staypoints.empty:
                        self.logger.info(f"Retrying with permissive parameters for {date}")
                        try:
                            # Refresh the positionfixes object
                            day_pfs = ti.Positionfixes(day_positionfixes)
                            day_pfs, staypoints = day_pfs.generate_staypoints(
                                method='sliding',
                                dist_threshold=100,  # Very permissive
                                time_threshold=3.0,  # Very permissive
                                gap_threshold=120.0  # Very permissive
                            )
                        except Exception as e:
                            self.logger.warning(f"Error with permissive parameters: {str(e)}")
                            staypoints = gpd.GeoDataFrame()
                    
                    # Level 3: If both previous methods failed, use fallback
                    if staypoints.empty:
                        self.logger.warning(f"No staypoints generated for {date} - using fallback method")
                        self.day_processing_status[date] = {
                            'stage': 'staypoints',
                            'valid': False,
                            'method': 'fallback',
                            'reason': 'No staypoints generated'
                        }
                        
                        # Use fallback method for this day
                        day_fallback = self._fallback_mobility_detection(day_positionfixes)
                        if date in day_fallback:
                            all_mobility_episodes[date] = day_fallback[date]
                            self.day_processing_status[date]['valid'] = True
                        continue
                    
                    # Log success
                    self.logger.debug(f"Generated {len(staypoints)} staypoints for {date}")
                    
                    # Ensure correct datetime handling
                    for dt_col in ['started_at', 'finished_at']:
                        if dt_col in staypoints.columns:
                            staypoints[dt_col] = ensure_tz_aware(staypoints[dt_col])
                    
                    # Generate triplegs
                    try:
                        day_pfs, triplegs = day_pfs.generate_triplegs(staypoints, gap_threshold=STAYPOINT_GAP_THRESHOLD)
                        
                        # Flag staypoints as activities
                        staypoints = staypoints.create_activity_flag()
                        
                        # Generate trips
                        staypoints, triplegs, trips = staypoints.generate_trips(triplegs, gap_threshold=TRIP_GAP_THRESHOLD)
                        
                        if trips.empty:
                            self.logger.warning(f"No trips generated for {date} - trying fallback method")
                            self.day_processing_status[date] = {
                                'stage': 'trips',
                                'valid': False,
                                'method': 'fallback',
                                'reason': 'No trips generated'
                            }
                            
                            # Use fallback method
                            day_fallback = self._fallback_mobility_detection(day_positionfixes)
                            if date in day_fallback:
                                all_mobility_episodes[date] = day_fallback[date]
                                self.day_processing_status[date]['valid'] = True
                            continue
                        
                        # Create a DataFrame in our expected format
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
                        
                        # Create mobility episodes
                        mobility_episodes = pd.DataFrame({
                            'started_at': trips['started_at'],
                            'finished_at': trips['finished_at'],
                            'duration': trips['finished_at'] - trips['started_at'],
                            'latitude': trips['latitude'],
                            'longitude': trips['longitude'],
                            'state': 'mobility'
                        })
                        
                        # Remove timezone information
                        mobility_episodes['started_at'] = ensure_tz_naive(mobility_episodes['started_at'])
                        mobility_episodes['finished_at'] = ensure_tz_naive(mobility_episodes['finished_at'])
                        
                        all_mobility_episodes[date] = mobility_episodes
                        self.day_processing_status[date] = {
                            'stage': 'completed', 
                            'valid': True,
                            'method': 'trackintel',
                            'episodes': len(mobility_episodes)
                        }
                        
                        self.logger.debug(f"Processed {len(mobility_episodes)} mobility episodes for {date}")
                        
                    except Exception as e:
                        self.logger.error(f"Error generating triplegs for {date}: {str(e)}")
                        self.day_processing_status[date] = {
                            'stage': 'triplegs',
                            'valid': False,
                            'method': 'fallback',
                            'reason': f"Error: {str(e)}"
                        }
                        
                        # Try fallback method
                        day_fallback = self._fallback_mobility_detection(day_positionfixes)
                        if date in day_fallback:
                            all_mobility_episodes[date] = day_fallback[date]
                            self.day_processing_status[date]['valid'] = True
                
                except Exception as day_error:
                    self.logger.error(f"Error processing mobility episodes for {date}: {str(day_error)}")
                    self.day_processing_status[date] = {
                        'stage': 'failed',
                        'valid': False,
                        'reason': f"Error: {str(day_error)}"
                    }
            
            return all_mobility_episodes
                
        except Exception as e:
            self.logger.error(f"Error processing mobility episodes with trackintel: {str(e)}")
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
        
        # Get processing status for this day
        day_status = self.day_processing_status.get(date, {'valid': False, 'reason': 'Unknown'})
        processing_method = day_status.get('method', 'unknown')
        
        # Calculate statistics
        day_stats = {
            'user': self.participant_id,
            'date': date,
            'valid_day': day_status.get('valid', False),
            'processing_method': processing_method,
            'digital_episodes': len(digital_episodes),
            'mobility_episodes': mobility_count,
            'stationary_episodes': 0,  # Not using this with trackintel approach
            'overlap_episodes': len(overlap_episodes),
            'digital_duration': digital_episodes['duration'].sum().total_seconds() / 60 if not digital_episodes.empty and 'duration' in digital_episodes.columns else 0,
            'mobility_duration': mobility_duration,
            'stationary_duration': 0,  # Not using this with trackintel approach
            'overlap_duration': overlap_episodes['duration'].sum().total_seconds() / 60 if not overlap_episodes.empty and 'duration' in overlap_episodes.columns else 0,
        }
        
        # Add failure reason if applicable
        if not day_stats['valid_day'] and 'reason' in day_status:
            day_stats['failure_reason'] = day_status['reason']
        
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
        
        # Add standardized participant ID to stats
        day_stats['participant_id_clean'] = self.participant_id_clean
        
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
            
            # Save day quality assessment
            quality_data = []
            for date, status in self.day_processing_status.items():
                row = {'date': date, 'valid': status.get('valid', False)}
                
                # Include all nested information flattened
                for key, value in status.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            row[f"{key}_{subkey}"] = subvalue
                    else:
                        row[key] = value
                        
                quality_data.append(row)
                
            if quality_data:
                quality_df = pd.DataFrame(quality_data)
                quality_file = self.output_dir / 'day_quality_assessment.csv'
                quality_df.to_csv(quality_file, index=False)
                self.logger.debug(f"Saved day quality assessment to {quality_file}")
            
            return all_stats
            
        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            return []

def main():
    # Find valid participants
    gps_files = {f.stem.replace('_gps_prep', ''): f 
                    for f in GPS_PREP_DIR.glob('*_gps_prep.csv')
                    if not f.stem.startswith('._')}  # Filter out macOS hidden files
    app_files = {f.stem.replace('_app_prep', ''): f 
                 for f in GPS_PREP_DIR.glob('*_app_prep.csv')
                 if not f.stem.startswith('._')}  # Filter out macOS hidden files
    
    common_ids = set(gps_files.keys()) & set(app_files.keys())
    # Filter out macOS hidden files (like '._005')
    common_ids = {pid for pid in common_ids if not pid.startswith('._')}
    logging.info(f"Found {len(common_ids)} participants with complete data")
    
    # Show a progress bar but suppress detailed logging during processing
    logging.getLogger().setLevel(logging.WARNING)
    
    # Track processing statistics
    all_stats = []
    participant_summaries = []
    processed_count = 0
    failed_count = 0
    
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
                
                # Count days with valid and invalid processing
                valid_days = sum(participant_df['valid_day']) if 'valid_day' in participant_df.columns else 0
                total_days = len(participant_df)
                invalid_days = total_days - valid_days
                
                participant_summary = {
                    'participant_id': pid,
                    'days_of_data': total_days,
                    'valid_days': valid_days,
                    'invalid_days': invalid_days,
                    'percent_valid': round(100 * valid_days / max(1, total_days), 1),
                    'avg_digital_episodes': participant_df['digital_episodes'].mean(),
                    'avg_mobility_episodes': participant_df['mobility_episodes'].mean(),
                    'avg_overlap_episodes': participant_df['overlap_episodes'].mean(),
                    'avg_digital_mins': participant_df['digital_duration'].mean(),
                    'avg_mobility_mins': participant_df['mobility_duration'].mean(),
                    'avg_overlap_mins': participant_df['overlap_duration'].mean(),
                    'total_digital_mins': participant_df['digital_duration'].sum(),
                    'total_mobility_mins': participant_df['mobility_duration'].sum(),
                    'total_overlap_mins': participant_df['overlap_duration'].sum(),
                }
                
                # Count processing methods
                if 'processing_method' in participant_df.columns:
                    method_counts = participant_df['processing_method'].value_counts().to_dict()
                    for method, count in method_counts.items():
                        participant_summary[f'method_{method}'] = count
                
                participant_summaries.append(participant_summary)
                processed_count += 1
                
                # Check if all days failed
                if valid_days == 0 and total_days > 0:
                    failed_count += 1
                    logging.error(f"All days failed for participant {pid}")
                
        except Exception as e:
            logging.error(f"Error processing participant {pid}: {str(e)}")
            failed_count += 1
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
        summary_logger.info(f"Failed to process {failed_count} participants")
        
        # Valid day stats
        if 'valid_day' in all_summary.columns:
            valid_days = all_summary[all_summary['valid_day'] == True]
            invalid_days = all_summary[all_summary['valid_day'] == False]
            total_days = len(all_summary)
            valid_percent = round(100 * len(valid_days) / total_days, 1) if total_days > 0 else 0
            
            summary_logger.info(f"\nDAY QUALITY ASSESSMENT:")
            summary_logger.info(f"Total days: {total_days}")
            summary_logger.info(f"Valid days: {len(valid_days)} ({valid_percent}%)")
            summary_logger.info(f"Invalid days: {len(invalid_days)} ({100-valid_percent}%)")
            
            # Processing method breakdown
            if 'processing_method' in all_summary.columns:
                method_counts = all_summary[all_summary['valid_day'] == True]['processing_method'].value_counts()
                summary_logger.info(f"\nPROCESSING METHODS (valid days):")
                for method, count in method_counts.items():
                    summary_logger.info(f"  {method}: {count} days ({round(100*count/len(valid_days), 1)}%)")
        
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
        summary_logger.info(f"Valid days: {participant_summary_df['valid_days'].min()}-"
                           f"{participant_summary_df['valid_days'].max()}")
        summary_logger.info(f"Percent valid: {participant_summary_df['percent_valid'].min():.1f}%-"
                           f"{participant_summary_df['percent_valid'].max():.1f}%")
        summary_logger.info(f"Avg daily digital episodes: {participant_summary_df['avg_digital_episodes'].min():.1f}-"
                           f"{participant_summary_df['avg_digital_episodes'].max():.1f}")
        summary_logger.info(f"Avg daily mobility episodes: {participant_summary_df['avg_mobility_episodes'].min():.1f}-"
                           f"{participant_summary_df['avg_mobility_episodes'].max():.1f}")
        
        # Output file locations
        summary_logger.info("\nOUTPUT FILES:")
        summary_logger.info(f"Detailed logs: episode_detection.log")
        summary_logger.info(f"All participants summary: {summary_file}")
        summary_logger.info(f"Participant-level summary: {participant_summary_file}")
        summary_logger.info("="*60)

if __name__ == "__main__":
    main()