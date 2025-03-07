#!/usr/bin/env python3
"""
Enhanced episode detection using Trackintel library
Identifies mobility between locations versus stationary periods
with improved data quality filtering
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
import trackintel as ti
import geopandas as gpd
from shapely.geometry import Point
from data_utils import DataCleaner

# Suppress pandas FutureWarnings
import warnings
warnings.filterwarnings("ignore", message=".*inplace method.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Downcasting object dtype arrays.*", category=FutureWarning)

# Setup logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Configure loggers
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_DIR / 'episode_detection.log'),
        logging.StreamHandler()
    ],
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Summary logger for statistics
summary_logger = logging.getLogger("summary")
summary_handler = logging.StreamHandler()
summary_logger.addHandler(summary_handler)
summary_logger.setLevel(logging.INFO)
summary_logger.propagate = False

# Add parent directory to path to find config
sys.path.append(str(Path(__file__).parent.parent))
from config.paths import GPS_PREP_DIR, EPISODE_OUTPUT_DIR, PROCESSED_DATA_DIR

# Parameters
STAYPOINT_DISTANCE_THRESHOLD = 75  # meters
STAYPOINT_TIME_THRESHOLD = 3.0     # minutes
STAYPOINT_GAP_THRESHOLD = 60.0     # minutes
LOCATION_EPSILON = 150             # meters
MIN_MOVEMENT_SPEED = 35            # meters per minute
MAX_REASONABLE_SPEED = 2500        # meters per minute
MIN_GPS_POINTS_PER_DAY = 5
MAX_ACCEPTABLE_GAP_PERCENT = 60
MIN_TRACK_DURATION_HOURS = 1
DIGITAL_USE_COL = 'action'         # Column containing screen events

def ensure_tz_naive(datetime_series: pd.Series) -> pd.Series:
    """Convert datetime series to timezone-naive if it has a timezone"""
    if datetime_series.empty:
        return datetime_series
    if hasattr(datetime_series.iloc[0], 'tz') and datetime_series.iloc[0].tz is not None:
        return datetime_series.dt.tz_localize(None)
    return datetime_series

def ensure_tz_aware(datetime_series: pd.Series) -> pd.Series:
    """Ensure datetime series has timezone info (UTC)"""
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
        self.data_cleaner = DataCleaner(self.logger)
        self.participant_id_clean = self.data_cleaner.standardize_participant_id(participant_id)
        self.day_processing_status = {}
    
    def _find_overlaps(self, digital_episodes: pd.DataFrame, movement_episodes: pd.DataFrame) -> pd.DataFrame:
        """Find temporal overlaps between digital and mobility episodes"""
        if digital_episodes.empty or movement_episodes.empty:
            return pd.DataFrame()
        
        # Make copies and ensure timezone consistency
        digital_episodes = digital_episodes.copy()
        movement_episodes = movement_episodes.copy()
        
        if 'start_time' in digital_episodes.columns:
            digital_episodes['start_time'] = ensure_tz_naive(digital_episodes['start_time'])
            digital_episodes['end_time'] = ensure_tz_naive(digital_episodes['end_time'])
        
        if 'started_at' in movement_episodes.columns:
            movement_episodes['started_at'] = ensure_tz_naive(movement_episodes['started_at'])
            movement_episodes['finished_at'] = ensure_tz_naive(movement_episodes['finished_at'])
        
        overlap_episodes = []
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
                            'latitude': m_ep.get('latitude', np.nan),
                            'longitude': m_ep.get('longitude', np.nan),
                            'duration': duration
                        })
        
        return pd.DataFrame(overlap_episodes) if overlap_episodes else pd.DataFrame()
    
    def _filter_gps_data(self, gps_df, datetime_col, lat_col, lon_col):
        """Filter GPS data to remove outliers and improve quality"""
        if len(gps_df) <= 1:
            return gps_df
        
        # Sort by timestamp and remove duplicates
        gps_df = gps_df.sort_values(datetime_col).drop_duplicates(subset=[datetime_col])
        
        # Filter accuracy values if available
        if 'accuracy' in gps_df.columns:
            gps_df = gps_df[(gps_df['accuracy'].isna()) | (gps_df['accuracy'] < 100)]
        
        # Smooth GPS trajectories for noisy data
        if len(gps_df) >= 3:
            gps_df['latitude_smooth'] = gps_df[lat_col].rolling(window=3, center=True).mean().fillna(gps_df[lat_col])
            gps_df['longitude_smooth'] = gps_df[lon_col].rolling(window=3, center=True).mean().fillna(gps_df[lon_col])
            lat_col_proc, lon_col_proc = 'latitude_smooth', 'longitude_smooth'
        else:
            lat_col_proc, lon_col_proc = lat_col, lon_col
        
        # Calculate speeds between consecutive points
        gps_df['prev_lat'] = gps_df[lat_col_proc].shift(1)
        gps_df['prev_lon'] = gps_df[lon_col_proc].shift(1)
        gps_df['time_diff'] = (gps_df[datetime_col].diff()).dt.total_seconds() / 60  # minutes
        
        # Only calculate speed where we have valid time differences
        mask = (gps_df['time_diff'] > 0)
        if mask.any():
            # Calculate rough distance in meters
            gps_df.loc[mask, 'distance'] = np.sqrt(
                ((gps_df.loc[mask, lat_col_proc] - gps_df.loc[mask, 'prev_lat']) * 111000)**2 + 
                ((gps_df.loc[mask, lon_col_proc] - gps_df.loc[mask, 'prev_lon']) * 
                 111000 * np.cos(np.radians(gps_df.loc[mask, lat_col_proc])))**2
            )
            
            # Calculate speed and filter out unreasonable speeds
            gps_df.loc[mask, 'speed'] = gps_df.loc[mask, 'distance'] / gps_df.loc[mask, 'time_diff']
            gps_df = gps_df[(gps_df['speed'].isna()) | (gps_df['speed'] <= MAX_REASONABLE_SPEED)]
        
        # Copy smoothed coordinates back to original columns if used
        if 'latitude_smooth' in gps_df.columns:
            gps_df[lat_col] = gps_df['latitude_smooth']
            gps_df[lon_col] = gps_df['longitude_smooth']
        
        # Remove temporary columns
        temp_cols = ['prev_lat', 'prev_lon', 'time_diff', 'distance', 'speed', 
                    'latitude_smooth', 'longitude_smooth']
        gps_df = gps_df.drop(columns=[c for c in temp_cols if c in gps_df.columns])
        
        return gps_df
    
    def assess_day_quality(self, day_positionfixes: gpd.GeoDataFrame) -> Tuple[bool, dict]:
        """Assess the quality of GPS data for a single day"""
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
        
        try:
            # Check what columns are available
            gps_df = pd.read_csv(gps_path)
            
            # Find datetime column
            datetime_col = next((col for col in ['tracked_at', 'UTC DATE TIME', 'timestamp'] 
                               if col in gps_df.columns), None)
            
            if datetime_col is None:
                raise ValueError(f"No datetime column found in {gps_path}")
            
            # Read again with parse_dates
            gps_df = pd.read_csv(gps_path, parse_dates=[datetime_col])
            
            # Find lat/lon columns
            lat_col = next((col for col in gps_df.columns 
                          if col.upper() in ['LATITUDE', 'LAT'] or 'LAT' in col.upper()), None)
            lon_col = next((col for col in gps_df.columns 
                          if col.upper() in ['LONGITUDE', 'LON', 'LONG'] or 'LON' in col.upper()), None)
            
            if lat_col is None or lon_col is None:
                # Check if this is already a Trackintel export with geometry
                if 'geometry' in gps_df.columns and 'user_id' in gps_df.columns:
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
                    gps_df['tracked_at'] = ensure_tz_aware(gps_df['tracked_at'])
                    
                    # Return as Positionfixes
                    return ti.Positionfixes(gdf)
                else:
                    raise ValueError(f"Could not find latitude/longitude columns in {gps_path}")
            
            # Filter GPS data for quality
            gps_df = self._filter_gps_data(gps_df, datetime_col, lat_col, lon_col)
            
            # Convert to trackintel's positionfixes format
            positionfixes = pd.DataFrame({
                'user_id': self.participant_id,
                'tracked_at': gps_df[datetime_col],
                'latitude': gps_df[lat_col],
                'longitude': gps_df[lon_col],
                'elevation': np.nan,
                'accuracy': np.nan,
            })
            
            # Ensure tracked_at is timezone aware (required by trackintel)
            positionfixes['tracked_at'] = ensure_tz_aware(positionfixes['tracked_at'])
            
            # Convert to GeoDataFrame
            geometry = [Point(lon, lat) for lon, lat in zip(positionfixes['longitude'], positionfixes['latitude'])]
            positionfixes = gpd.GeoDataFrame(positionfixes, geometry=geometry, crs="EPSG:4326")
            
            return ti.Positionfixes(positionfixes)
            
        except Exception as e:
            self.logger.error(f"Failed to load GPS data: {str(e)}")
            raise
    
    def load_app_data(self) -> pd.DataFrame:
        """Load app data with validation"""
        app_path = GPS_PREP_DIR / f'{self.participant_id}_app_prep.csv'
        
        try:
            app_df = pd.read_csv(app_path)
            
            # Find timestamp column
            timestamp_col = next((col for col in ['timestamp', 'Timestamp', 'date', 'tracked_at'] 
                                if col in app_df.columns), None)
            
            if timestamp_col is None:
                # Check if we have date and time columns
                if 'date' in app_df.columns and 'time' in app_df.columns:
                    app_df['timestamp'] = pd.to_datetime(app_df['date'] + ' ' + app_df['time'], 
                                                      format='mixed', dayfirst=True)
                else:
                    raise ValueError(f"No timestamp column found in {app_path}")
            else:
                app_df['timestamp'] = pd.to_datetime(app_df[timestamp_col])
            
            # Ensure we have a date column
            app_df['date'] = app_df['timestamp'].dt.date
            
            # Find action column for SCREEN ON/OFF
            action_col = next((col for col in app_df.columns 
                             if col.lower() == 'action' or 'screen' in col.lower()), None)
            
            if action_col is not None and action_col != DIGITAL_USE_COL:
                app_df[DIGITAL_USE_COL] = app_df[action_col]
            
            return app_df
            
        except Exception as e:
            self.logger.error(f"Failed to load app data: {str(e)}")
            raise
    
    def process_digital_episodes(self, app_df: pd.DataFrame) -> Dict[datetime.date, pd.DataFrame]:
        """Process digital episodes by day focusing on system-level screen events"""
        episodes_by_day = {}
        
        if DIGITAL_USE_COL not in app_df.columns:
            self.logger.warning(f"Digital use column '{DIGITAL_USE_COL}' not found")
            return episodes_by_day
        
        # Define screen event patterns - MODIFIED to focus on actual screen events
        screen_on_values = ['SCREEN ON', 'screen_on', 'SCREEN_ON', 'on', 'ON']
        screen_off_values = ['SCREEN OFF', 'screen_off', 'SCREEN_OFF', 'off', 'OFF']
        
        for date, day_data in app_df.groupby('date'):
            # Filter to only include system-level screen events
            screen_events = day_data.sort_values('timestamp')
            
            # Add check for 'package name' column
            if 'package name' in screen_events.columns:
                # Only include Android system screen events
                screen_events = screen_events[
                    (screen_events['package name'] == 'android') & 
                    (screen_events[DIGITAL_USE_COL].isin(screen_on_values + screen_off_values))
                ].copy()
            else:
                # Fallback to just filtering by action when package name is not available
                screen_events = screen_events[screen_events[DIGITAL_USE_COL].isin(screen_on_values + screen_off_values)].copy()
            
            if len(screen_events) == 0:
                continue
            
            # Map values to standard format
            screen_events.loc[screen_events[DIGITAL_USE_COL].isin(screen_on_values), DIGITAL_USE_COL] = 'SCREEN ON'
            screen_events.loc[screen_events[DIGITAL_USE_COL].isin(screen_off_values), DIGITAL_USE_COL] = 'SCREEN OFF'
            
            # Filter out rapid on/off sequences
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
                screen_events = screen_events.drop(all_remove)
            
            # Create episodes
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
            
            if episodes:
                episodes_df = pd.DataFrame(episodes)
                episodes_df['duration'] = episodes_df['end_time'] - episodes_df['start_time']
                
                # Remove timezone information
                episodes_df['start_time'] = ensure_tz_naive(episodes_df['start_time'])
                episodes_df['end_time'] = ensure_tz_naive(episodes_df['end_time'])
                
                episodes_by_day[date] = episodes_df
        
        return episodes_by_day
    
    def _fallback_mobility_detection(self, positionfixes) -> Dict[datetime.date, pd.DataFrame]:
        """Fallback mobility detection when trackintel fails"""
        episodes_by_day = {}
        
        if positionfixes.empty or len(positionfixes) <= 1:
            return episodes_by_day
        
        # Convert to pandas DataFrame for easier manipulation
        pfs = positionfixes.copy()
        if isinstance(pfs, gpd.GeoDataFrame):
            pfs = pd.DataFrame(pfs.drop(columns='geometry'))
        
        # Sort by timestamp and add date column
        pfs = pfs.sort_values('tracked_at')
        pfs['date'] = pfs['tracked_at'].dt.date
        
        # Calculate distances and time differences
        pfs['prev_lat'] = pfs['latitude'].shift(1)
        pfs['prev_lon'] = pfs['longitude'].shift(1)
        pfs['prev_time'] = pfs['tracked_at'].shift(1)
        pfs['time_diff'] = (pfs['tracked_at'] - pfs['prev_time']).dt.total_seconds() / 60  # minutes
        
        # Only calculate where we have consecutive points
        mask = (pfs['time_diff'] > 0) & (pfs['time_diff'] < STAYPOINT_GAP_THRESHOLD)
        
        if mask.any():
            # Calculate distance in meters
            pfs.loc[mask, 'distance'] = np.sqrt(
                ((pfs.loc[mask, 'latitude'] - pfs.loc[mask, 'prev_lat']) * 111000)**2 + 
                ((pfs.loc[mask, 'longitude'] - pfs.loc[mask, 'prev_lon']) * 
                 111000 * np.cos(np.radians(pfs.loc[mask, 'latitude'])))**2
            )
            
            # Calculate speed and mark movement
            pfs.loc[mask, 'speed'] = pfs.loc[mask, 'distance'] / pfs.loc[mask, 'time_diff']
            median_speed = pfs.loc[mask, 'speed'].median()
            speed_threshold = min(MAX_REASONABLE_SPEED, max(MIN_MOVEMENT_SPEED, median_speed * 0.7))
            pfs['moving'] = (pfs['speed'] > speed_threshold)
            
            # Mark distant points as moving
            if mask.any() and pfs.loc[mask, 'distance'].max() > 100:
                far_points = pfs['distance'] > 100
                pfs.loc[far_points, 'moving'] = True
            
            # Identify trip starts and ends
            pfs['trip_start'] = pfs['moving'] & ~pfs['moving'].shift(1, fill_value=False)
            pfs['trip_end'] = ~pfs['moving'] & pfs['moving'].shift(1, fill_value=False)
            
            trip_starts = pfs[pfs['trip_start']].copy()
            trip_ends = pfs[pfs['trip_end']].copy()
            
            # Create trips where we have both start and end
            trips = []
            for _, start_row in trip_starts.iterrows():
                # Find the next end after this start
                end_candidates = trip_ends[trip_ends['tracked_at'] > start_row['tracked_at']]
                
                if not end_candidates.empty:
                    end_row = end_candidates.iloc[0]
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
        
        return episodes_by_day
    
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
            all_episodes = all_episodes.sort_values('start_time')
            all_episodes['episode_number'] = range(1, len(all_episodes) + 1)
            all_episodes['time_since_prev'] = all_episodes['start_time'].diff()
            
            # Select relevant columns
            cols = ['episode_number', 'episode_type', 'movement_state', 
                   'start_time', 'end_time', 'duration', 'time_since_prev',
                   'latitude', 'longitude']
            all_episodes = all_episodes[[c for c in cols if c in all_episodes.columns]]
        
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
        
        # Calculate statistics
        day_status = self.day_processing_status.get(date, {'valid': False, 'reason': 'Unknown'})
        processing_method = day_status.get('method', 'unknown')
        
        day_stats = {
            'user': self.participant_id,
            'date': date,
            'valid_day': day_status.get('valid', False),
            'processing_method': processing_method,
            'digital_episodes': len(digital_episodes),
            'mobility_episodes': len(mobility_episodes) if not mobility_episodes.empty else 0,
            'overlap_episodes': len(overlap_episodes),
            'digital_duration': digital_episodes['duration'].sum().total_seconds() / 60 if not digital_episodes.empty else 0,
            'mobility_duration': sum(dt.total_seconds() for dt in mobility_episodes['duration']) / 60 if not mobility_episodes.empty else 0,
            'overlap_duration': overlap_episodes['duration'].sum().total_seconds() / 60 if not overlap_episodes.empty else 0,
            'participant_id_clean': self.participant_id_clean
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
                day_stats = self.process_day(date, digital_eps, mobility_eps)
                all_stats.append(day_stats)
            
            # Save summary statistics
            if all_stats:
                summary_df = pd.DataFrame(all_stats)
                summary_file = self.output_dir / 'episode_summary.csv'
                summary_df.to_csv(summary_file, index=False)
            
            return all_stats
            
        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}")
            traceback.print_exc()
            return []
    
    def process_mobility_episodes(self, positionfixes: ti.Positionfixes) -> Dict[datetime.date, pd.DataFrame]:
        """Process mobility episodes using Trackintel with fallback methods"""
        if positionfixes.empty or len(positionfixes) <= 5:
            return {}
        
        # Split by day and filter low-quality days
        pfs_by_day = {}
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
        
        if not pfs_by_day:
            return {}
        
        # Process each day
        all_mobility_episodes = {}
        
        for date, day_positionfixes in pfs_by_day.items():
            try:
                # Ensure tracked_at is timezone-aware
                day_positionfixes['tracked_at'] = ensure_tz_aware(day_positionfixes['tracked_at'])
                day_pfs = ti.Positionfixes(day_positionfixes)
                
                # Try to generate staypoints with standard parameters
                try:
                    day_pfs, staypoints = day_pfs.generate_staypoints(
                        method='sliding',
                        dist_threshold=STAYPOINT_DISTANCE_THRESHOLD,
                        time_threshold=STAYPOINT_TIME_THRESHOLD,
                        gap_threshold=STAYPOINT_GAP_THRESHOLD
                    )
                except Exception:
                    staypoints = gpd.GeoDataFrame()
                
                # If standard parameters failed, try fallback
                if staypoints.empty:
                    day_fallback = self._fallback_mobility_detection(day_positionfixes)
                    if date in day_fallback:
                        all_mobility_episodes[date] = day_fallback[date]
                        self.day_processing_status[date] = {
                            'stage': 'completed',
                            'valid': True,
                            'method': 'fallback'
                        }
                    continue
                
                # Generate triplegs and trips
                try:
                    day_pfs, triplegs = day_pfs.generate_triplegs(staypoints, gap_threshold=STAYPOINT_GAP_THRESHOLD)
                    staypoints = staypoints.create_activity_flag()
                    staypoints, triplegs, trips = staypoints.generate_trips(triplegs, gap_threshold=STAYPOINT_GAP_THRESHOLD)
                    
                    if trips.empty:
                        # Fallback if no trips
                        day_fallback = self._fallback_mobility_detection(day_positionfixes)
                        if date in day_fallback:
                            all_mobility_episodes[date] = day_fallback[date]
                            self.day_processing_status[date] = {
                                'stage': 'completed',
                                'valid': True,
                                'method': 'fallback'
                            }
                        continue
                    
                    # Create mobility episodes
                    trips['latitude'], trips['longitude'] = np.nan, np.nan
                    
                    # Try to get coordinates from origin staypoints
                    if 'origin_staypoint_id' in trips.columns and not staypoints.empty:
                        for idx, trip in trips.iterrows():
                            if pd.notna(trip['origin_staypoint_id']):
                                origin_sp = staypoints[staypoints.index == trip['origin_staypoint_id']]
                                if not origin_sp.empty:
                                    trips.at[idx, 'latitude'] = origin_sp.iloc[0].geometry.y
                                    trips.at[idx, 'longitude'] = origin_sp.iloc[0].geometry.x
                    
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
                        'method': 'trackintel'
                    }
                
                except Exception as e:
                    self.logger.error(f"Error generating trips for {date}: {str(e)}")
                    # Try fallback method
                    day_fallback = self._fallback_mobility_detection(day_positionfixes)
                    if date in day_fallback:
                        all_mobility_episodes[date] = day_fallback[date]
                        self.day_processing_status[date] = {
                            'stage': 'completed',
                            'valid': True,
                            'method': 'fallback'
                        }
            
            except Exception as day_error:
                self.logger.error(f"Error processing day {date}: {str(day_error)}")
                self.day_processing_status[date] = {
                    'stage': 'failed',
                    'valid': False,
                    'reason': f"Error: {str(day_error)}"
                }
        
        return all_mobility_episodes

def main():
    """Main execution function"""
    # Find valid participants
    gps_files = {f.stem.replace('_gps_prep', ''): f 
                for f in GPS_PREP_DIR.glob('*_gps_prep.csv')
                if not f.stem.startswith('._')}
    app_files = {f.stem.replace('_app_prep', ''): f 
               for f in GPS_PREP_DIR.glob('*_app_prep.csv')
               if not f.stem.startswith('._')}
    
    common_ids = set(gps_files.keys()) & set(app_files.keys())
    common_ids = {pid for pid in common_ids if not pid.startswith('._')}
    logging.info(f"Found {len(common_ids)} participants with complete data")
    
    # Suppress detailed logging during processing
    from tqdm import tqdm
    logging.getLogger().setLevel(logging.WARNING)
    
    # Process participants
    all_stats = []
    participant_summaries = []
    processed_count = 0
    failed_count = 0
    
    for pid in tqdm(common_ids, desc="Processing participants"):
        if pid.startswith('._'):
            continue
            
        try:
            processor = EpisodeProcessor(pid)
            participant_stats = processor.process()
            
            if participant_stats:
                all_stats.extend(participant_stats)
                
                # Calculate participant summary
                participant_df = pd.DataFrame(participant_stats)
                valid_days = sum(participant_df['valid_day']) if 'valid_day' in participant_df.columns else 0
                total_days = len(participant_df)
                
                participant_summary = {
                    'participant_id': pid,
                    'days_of_data': total_days,
                    'valid_days': valid_days,
                    'invalid_days': total_days - valid_days,
                    'percent_valid': round(100 * valid_days / max(1, total_days), 1),
                    'avg_digital_episodes': participant_df['digital_episodes'].mean(),
                    'avg_mobility_episodes': participant_df['mobility_episodes'].mean(),
                    'avg_overlap_episodes': participant_df['overlap_episodes'].mean(),
                    'avg_digital_mins': participant_df['digital_duration'].mean(),
                    'avg_mobility_mins': participant_df['mobility_duration'].mean(),
                    'avg_overlap_mins': participant_df['overlap_duration'].mean(),
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
        except Exception as e:
            logging.error(f"Error processing participant {pid}: {str(e)}")
            failed_count += 1
    
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
        
        # Log summary information
        summary_logger.info("\n" + "="*60)
        summary_logger.info(f"MOBILITY DETECTION SUMMARY")
        summary_logger.info("="*60)
        summary_logger.info(f"Successfully processed {processed_count}/{len(common_ids)} participants")
        summary_logger.info(f"Failed to process {failed_count} participants")
        
        # Valid day stats
        if 'valid_day' in all_summary.columns:
            valid_days = all_summary[all_summary['valid_day'] == True]
            total_days = len(all_summary)
            valid_percent = round(100 * len(valid_days) / total_days, 1) if total_days > 0 else 0
            
            summary_logger.info(f"\nDAY QUALITY ASSESSMENT:")
            summary_logger.info(f"Total days: {total_days}")
            summary_logger.info(f"Valid days: {len(valid_days)} ({valid_percent}%)")
            
            # Processing method breakdown
            if 'processing_method' in all_summary.columns:
                method_counts = all_summary[all_summary['valid_day'] == True]['processing_method'].value_counts()
                summary_logger.info(f"\nPROCESSING METHODS:")
                for method, count in method_counts.items():
                    summary_logger.info(f"  {method}: {count} days ({round(100*count/len(valid_days), 1)}%)")
            
            # Episode count and duration statistics
            summary_logger.info("\nEPISODE STATISTICS (Valid Days Only):")
            for ep_type in ['digital', 'mobility', 'overlap']:
                ep_count = valid_days[f'{ep_type}_episodes'].sum()
                avg_count = valid_days[f'{ep_type}_episodes'].mean()
                ep_duration = valid_days[f'{ep_type}_duration'].sum()
                avg_duration = valid_days[f'{ep_type}_duration'].mean()
                
                summary_logger.info(f"  {ep_type.capitalize()} Episodes:")
                summary_logger.info(f"    Total: {int(ep_count)} episodes ({round(ep_duration/60, 1)} hours)")
                summary_logger.info(f"    Per Day: {round(avg_count, 1)} episodes ({round(avg_duration, 1)} minutes)")

if __name__ == "__main__":
    main()