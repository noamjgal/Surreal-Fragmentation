#!/usr/bin/env python3
"""
Enhanced episode detection using Trackintel library
Identifies mobility between locations versus stationary periods
with consistent timezone handling and improved integration with preprocessing
"""
import pandas as pd
import numpy as np
import os
from datetime import timedelta, datetime, date as datetime_date
import traceback
from pathlib import Path
import sys
import logging
from typing import Dict, List, Tuple, Optional
import trackintel as ti
import geopandas as gpd
from shapely.geometry import Point
import argparse
import warnings
import warnings


# Suppress pandas FutureWarnings
pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings("ignore", message=".*inplace method.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Downcasting object dtype arrays.*", category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning, message=".*observed=False.*")

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

# Import preprocessing module for direct integration
try:
    from preprocess_gps import process_participant
except ImportError:
    logging.warning("Could not import preprocess_gps module. Preprocessing may not be available.")

# Parameters for trackintel processing
STAYPOINT_DISTANCE_THRESHOLD = 75  # meters (standard academic value)
STAYPOINT_TIME_THRESHOLD = 5.0     # minutes (standard minimum in literature)
STAYPOINT_GAP_THRESHOLD = 45.0     # minutes (slightly reduced from default)
LOCATION_EPSILON = 100             # meters (slightly reduced for better precision)
MIN_GPS_POINTS_PER_DAY = 5
MAX_ACCEPTABLE_GAP_PERCENT = 60
MIN_TRACK_DURATION_HOURS = 1
DIGITAL_USE_COL = 'action'         # Column containing screen events
MAX_REASONABLE_TRIP_DURATION = 120  # Maximum reasonable trip duration in minutes

def ensure_tz_aware(datetime_series: pd.Series) -> pd.Series:
    """Ensure datetime series has timezone info (UTC)"""
    if datetime_series.empty:
        return datetime_series
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_dtype(datetime_series):
        datetime_series = pd.to_datetime(datetime_series)
    
    # Check if timezone info is present
    if datetime_series.dt.tz is None:
        return datetime_series.dt.tz_localize('UTC')
    return datetime_series

def ensure_tz_naive(datetime_series: pd.Series) -> pd.Series:
    """Convert datetime series to timezone-naive if it has a timezone"""
    if datetime_series.empty:
        return datetime_series
    
    if datetime_series.dt.tz is not None:
        return datetime_series.dt.tz_localize(None)
    return datetime_series

class ParticipantData:
    """Class to handle participant-specific data loading and preparation"""
    
    def __init__(self, participant_id: str, logger):
        self.participant_id = str(participant_id)
        self.logger = logger
    
    def standardize_participant_id(self) -> str:
        """Standardize participant ID format"""
        try:
            # Remove leading zeros but keep as string
            return str(int(self.participant_id))
        except ValueError:
            # If can't convert to int, return as is
            return self.participant_id
    
    def load_gps_data(self, gps_path: Path) -> pd.DataFrame:
        """Load GPS data from preprocessed file"""
        try:
            self.logger.info(f"Loading GPS data from {gps_path}")
            gps_df = pd.read_csv(gps_path, parse_dates=['tracked_at'])
            
            # Ensure required columns exist
            required_cols = ['user_id', 'tracked_at', 'latitude', 'longitude', 'date']
            missing_cols = [col for col in required_cols if col not in gps_df.columns]
            
            if missing_cols:
                self.logger.error(f"Missing required columns in GPS data: {missing_cols}")
                if 'date' in missing_cols and 'tracked_at' in gps_df.columns:
                    # Create date column from tracked_at
                    gps_df['date'] = pd.to_datetime(gps_df['tracked_at']).dt.date
                    missing_cols.remove('date')
                
                if missing_cols:
                    raise ValueError(f"Required columns missing: {missing_cols}")
            
            # Ensure tracked_at is timezone aware (required by trackintel)
            gps_df['tracked_at'] = ensure_tz_aware(gps_df['tracked_at'])
            
            # Log data summary
            self.logger.info(f"Loaded {len(gps_df)} GPS points covering {gps_df['date'].nunique()} days")
            return gps_df
            
        except Exception as e:
            self.logger.error(f"Error loading GPS data: {str(e)}")
            raise
    
    def load_app_data(self, app_path: Path) -> pd.DataFrame:
        """Load app usage data from preprocessed file"""
        try:
            self.logger.info(f"Loading app data from {app_path}")
            app_df = pd.read_csv(app_path, parse_dates=['timestamp'])
            
            # Ensure required columns exist
            required_cols = ['timestamp', 'action', 'date']
            missing_cols = [col for col in required_cols if col not in app_df.columns]
            
            if missing_cols:
                self.logger.error(f"Missing required columns in app data: {missing_cols}")
                if 'date' in missing_cols and 'timestamp' in app_df.columns:
                    # Create date column from timestamp
                    app_df['date'] = pd.to_datetime(app_df['timestamp']).dt.date
                    missing_cols.remove('date')
                
                if missing_cols:
                    raise ValueError(f"Required columns missing: {missing_cols}")
            
            # Log data summary
            self.logger.info(f"Loaded {len(app_df)} app events covering {app_df['date'].nunique()} days")
            return app_df
            
        except Exception as e:
            self.logger.error(f"Error loading app data: {str(e)}")
            raise

class EpisodeProcessor:
    def __init__(self, participant_id: str, preprocess_data: bool = False):
        self.participant_id = participant_id
        self.logger = logging.getLogger(f"EpisodeProcessor_{participant_id}")
        self.output_dir = EPISODE_OUTPUT_DIR / participant_id
        self.preprocess_data = preprocess_data
        self.home_location = (np.nan, np.nan)  # Initialize home location
        
        # Skip creation if the path is a macOS hidden file
        if '._' in str(self.output_dir):
            self.logger.warning(f"Skipping macOS hidden file: {self.output_dir}")
            return
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_handler = ParticipantData(participant_id, self.logger)
        self.participant_id_clean = self.data_handler.standardize_participant_id()
        self.day_processing_status = {}
        
        # Log initialization
        self.logger.info(f"Initialized episode processor for participant {participant_id}")
    
    def _find_overlaps(self, digital_episodes: pd.DataFrame, 
                     mobility_episodes: pd.DataFrame) -> pd.DataFrame:
        """Find temporal overlaps between digital and mobility episodes"""
        if digital_episodes.empty or mobility_episodes.empty:
            return pd.DataFrame()
        
        # Make copies and ensure timezone consistency
        digital_episodes = digital_episodes.copy()
        mobility_episodes = mobility_episodes.copy()
        
        # Ensure timezone-naive datetimes for consistent comparison
        if 'start_time' in digital_episodes.columns:
            digital_episodes['start_time'] = ensure_tz_naive(digital_episodes['start_time'])
            digital_episodes['end_time'] = ensure_tz_naive(digital_episodes['end_time'])
        
        if 'started_at' in mobility_episodes.columns:
            mobility_episodes['started_at'] = ensure_tz_naive(mobility_episodes['started_at'])
            mobility_episodes['finished_at'] = ensure_tz_naive(mobility_episodes['finished_at'])
        
        # Find overlaps by comparing each pair of episodes
        overlap_episodes = []
        for _, d_ep in digital_episodes.iterrows():
            for _, m_ep in mobility_episodes.iterrows():
                # Find potential overlap
                start = max(d_ep['start_time'], m_ep['started_at'])
                end = min(d_ep['end_time'], m_ep['finished_at'])
                
                if start < end:  # There is an overlap
                    duration = end - start
                    if duration >= pd.Timedelta(minutes=1):  # Minimum 1 minute overlap
                        overlap_data = {
                            'start_time': start,
                            'end_time': end,
                            'state': 'overlap',
                            'movement_state': 'mobility',
                            'latitude': m_ep.get('latitude', np.nan),
                            'longitude': m_ep.get('longitude', np.nan),
                            'duration': duration
                        }
                        
                        # Copy transport type and mode if available
                        if 'transport_type' in m_ep:
                            overlap_data['transport_type'] = m_ep['transport_type']
                        if 'primary_mode' in m_ep:
                            overlap_data['primary_mode'] = m_ep['primary_mode']
                        if 'location_type' in m_ep:
                            overlap_data['location_type'] = m_ep['location_type']
                            
                        overlap_episodes.append(overlap_data)
        
        return pd.DataFrame(overlap_episodes) if overlap_episodes else pd.DataFrame()
    
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
        quality_stats['median_gap_minutes'] = day_positionfixes['time_diff'].median() if 'time_diff' in day_positionfixes else 0
        
        if percent_large_gaps > MAX_ACCEPTABLE_GAP_PERCENT:
            quality_stats['failure_reason'] = f"Too many large gaps ({percent_large_gaps:.1f}% > {MAX_ACCEPTABLE_GAP_PERCENT}%)"
            return False, quality_stats
        
        # Made it through all checks
        quality_stats['valid'] = True
        return True, quality_stats
    
    def identify_home_location(self, positionfixes: ti.Positionfixes) -> Tuple[float, float]:
        """Identify home location based on 3-6 AM positions"""
        if positionfixes.empty:
            return (np.nan, np.nan)
        
        # Create a copy with hour information
        pfs = positionfixes.copy()
        pfs['hour'] = pfs['tracked_at'].dt.hour
        
        # Filter for nighttime hours (3-6 AM)
        night_pfs = pfs[(pfs['hour'] >= 3) & (pfs['hour'] < 6)]
        
        if night_pfs.empty:
            self.logger.warning("No nighttime (3-6 AM) data found for home location detection")
            # Use most common daytime location as fallback
            all_pfs = pfs.copy()
            
            # Generate staypoints for all data
            try:
                _, staypoints = all_pfs.generate_staypoints(
                    method='sliding',
                    dist_threshold=STAYPOINT_DISTANCE_THRESHOLD,
                    time_threshold=STAYPOINT_TIME_THRESHOLD * 2,  # Use longer threshold for home detection
                    gap_threshold=STAYPOINT_GAP_THRESHOLD
                )
                
                if staypoints.empty:
                    self.logger.warning("No staypoints available for home detection")
                    return (np.nan, np.nan)
                
                # Find most common staypoint by duration
                staypoints['duration'] = (staypoints['finished_at'] - staypoints['started_at']).dt.total_seconds()
                longest_staypoint = staypoints.loc[staypoints['duration'].idxmax()]
                home_lat = longest_staypoint.geometry.y
                home_lon = longest_staypoint.geometry.x
                
                self.logger.info(f"Home location determined from longest staypoint: {home_lat:.6f}, {home_lon:.6f}")
                return (home_lat, home_lon)
                
            except Exception as e:
                self.logger.error(f"Error in home location detection fallback: {str(e)}")
                return (np.nan, np.nan)
        
        # Generate staypoints from nighttime data
        try:
            _, night_staypoints = night_pfs.generate_staypoints(
                method='sliding',
                dist_threshold=STAYPOINT_DISTANCE_THRESHOLD,
                time_threshold=STAYPOINT_TIME_THRESHOLD,
                gap_threshold=STAYPOINT_GAP_THRESHOLD
            )
            
            if night_staypoints.empty:
                self.logger.warning("No nighttime staypoints found for home location")
                return (np.nan, np.nan)
            
            # Calculate duration for each staypoint
            night_staypoints['duration'] = (night_staypoints['finished_at'] - night_staypoints['started_at']).dt.total_seconds()
            
            # Find the most common nighttime staypoint location (by total duration)
            longest_staypoint = night_staypoints.loc[night_staypoints['duration'].idxmax()]
            home_lat = longest_staypoint.geometry.y
            home_lon = longest_staypoint.geometry.x
            
            self.logger.info(f"Home location determined: {home_lat:.6f}, {home_lon:.6f}")
            return (home_lat, home_lon)
            
        except Exception as e:
            self.logger.error(f"Error in nighttime home location detection: {str(e)}")
            return (np.nan, np.nan)

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate haversine distance between two points in meters"""
        if np.isnan(lat1) or np.isnan(lon1) or np.isnan(lat2) or np.isnan(lon2):
            return np.inf
            
        # Earth radius in meters
        R = 6371000
        
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        # Haversine formula
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c

    def process_staypoints_with_location(self, staypoints, home_location):
        """Process staypoints and classify as home or out_of_home"""
        if staypoints.empty:
            return staypoints
            
        staypoints = staypoints.copy()
        home_lat, home_lon = home_location
        
        # Add location_type column
        staypoints['location_type'] = 'unknown'
        
        # Calculate distance to home for each staypoint
        for idx, sp in staypoints.iterrows():
            distance = self.haversine_distance(
                sp.geometry.y, sp.geometry.x, 
                home_lat, home_lon
            )
            
            # Classify based on distance (100m threshold)
            if distance <= 100:
                staypoints.at[idx, 'location_type'] = 'home'
            else:
                staypoints.at[idx, 'location_type'] = 'out_of_home'
        
        return staypoints
    
    def load_gps_data(self) -> Optional[ti.Positionfixes]:
        """Load GPS data with validation or trigger preprocessing if needed"""
        gps_path = GPS_PREP_DIR / f'{self.participant_id}_processed_gps.csv'
        
        # If preprocessing is requested or file doesn't exist, run preprocessing
        if self.preprocess_data or not gps_path.exists():
            self.logger.info(f"Preprocessing data for participant {self.participant_id}")
            try:
                # This assumes process_participant from preprocess_gps.py has been imported
                # Find applicable qstarz and app files for this participant
                from config.paths import RAW_DATA_DIR
                
                # Search for files using patterns from preprocess_gps.py
                qstarz_files = {}
                app_files = {}
                app_gps_files = {}
                
                # Simplified file search - in production, extend this to match the original script
                for f in RAW_DATA_DIR.glob(f"**/*{self.participant_id}*.csv"):
                    if "Qstarz" in f.stem or "qstarz" in f.stem:
                        qstarz_files[self.participant_id] = f
                    elif "app" in f.stem.lower() or "usage" in f.stem.lower():
                        app_files[self.participant_id] = f
                    elif "gps" in f.stem.lower() or "location" in f.stem.lower():
                        app_gps_files[self.participant_id] = f
                
                # Run preprocessing for this participant
                result = process_participant(self.participant_id, qstarz_files, app_files, app_gps_files)
                
                if not result.get('success', False):
                    self.logger.error(f"Preprocessing failed: {result.get('reason', 'Unknown error')}")
                    raise ValueError(f"Preprocessing failed: {result.get('reason', 'Unknown error')}")
                
            except Exception as e:
                self.logger.error(f"Error during preprocessing: {str(e)}")
                raise
        
        try:
            # Load GPS data
            gps_df = self.data_handler.load_gps_data(gps_path)
            
            # Convert to trackintel's Positionfixes format
            geometry = [Point(lon, lat) for lon, lat in zip(gps_df['longitude'], gps_df['latitude'])]
            positionfixes = gpd.GeoDataFrame(
                data={
                    'user_id': gps_df['user_id'],
                    'tracked_at': gps_df['tracked_at'],
                    'elevation': np.nan,
                    'accuracy': gps_df['accuracy'] if 'accuracy' in gps_df.columns else np.nan,
                },
                geometry=geometry,
                crs="EPSG:4326"
            )
            
            return ti.Positionfixes(positionfixes)
            
        except Exception as e:
            self.logger.error(f"Failed to load or convert GPS data: {str(e)}")
            raise
    
    def load_app_data(self) -> pd.DataFrame:
        """Load app data with validation"""
        app_path = GPS_PREP_DIR / f'{self.participant_id}_app_prep.csv'
        return self.data_handler.load_app_data(app_path)
    
    def process_digital_episodes(self, app_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Process digital episodes by day focusing on system-level screen events"""
        episodes_by_day = {}
        
        if DIGITAL_USE_COL not in app_df.columns:
            self.logger.warning(f"Digital use column '{DIGITAL_USE_COL}' not found in app data")
            return episodes_by_day
        
        # Define screen event patterns
        screen_on_patterns = ['SCREEN ON', 'screen_on', 'SCREEN_ON', 'on', 'ON']
        screen_off_patterns = ['SCREEN OFF', 'screen_off', 'SCREEN_OFF', 'off', 'OFF']
        
        # Ensure date column is datetime.date type
        if 'date' in app_df.columns:
            if pd.api.types.is_datetime64_dtype(app_df['date']):
                app_df['date'] = app_df['date'].dt.date
            elif not pd.api.types.is_object_dtype(app_df['date']):
                app_df['date'] = pd.to_datetime(app_df['date']).dt.date
                
        # Log date types for debugging
        self.logger.info(f"App data date column type: {app_df['date'].dtype}")
        self.logger.info(f"Sample dates: {app_df['date'].head(3).tolist()}")
            
        for date, day_data in app_df.groupby('date'):
            self.logger.info(f"Processing digital episodes for date {date}")
            
            # Filter to only include screen events
            screen_events = day_data.sort_values('timestamp')
            
            # Check for package name column (Android events)
            if 'package name' in screen_events.columns:
                # Only include Android system screen events
                screen_events = screen_events[
                    (screen_events['package name'] == 'android') & 
                    (screen_events[DIGITAL_USE_COL].str.upper().isin(
                        [p.upper() for p in screen_on_patterns + screen_off_patterns]))
                ].copy()
            else:
                # Fallback to just filtering by action when package name is not available
                screen_events = screen_events[
                    screen_events[DIGITAL_USE_COL].str.upper().isin(
                        [p.upper() for p in screen_on_patterns + screen_off_patterns])
                ].copy()
            
            if len(screen_events) == 0:
                self.logger.warning(f"No screen events found for date {date}")
                continue
            
            # Map values to standard format
            screen_events['action_type'] = 'UNKNOWN'
            
            # Standardize ON events
            for pattern in screen_on_patterns:
                mask = screen_events[DIGITAL_USE_COL].str.upper() == pattern.upper()
                screen_events.loc[mask, 'action_type'] = 'SCREEN ON'
            
            # Standardize OFF events
            for pattern in screen_off_patterns:
                mask = screen_events[DIGITAL_USE_COL].str.upper() == pattern.upper()
                screen_events.loc[mask, 'action_type'] = 'SCREEN OFF'
            
            # Remove unknown events
            screen_events = screen_events[screen_events['action_type'] != 'UNKNOWN']
            
            # Filter out rapid on/off sequences
            screen_events['prev_time'] = screen_events['timestamp'].shift(1)
            screen_events['prev_action'] = screen_events['action_type'].shift(1)
            screen_events['time_diff'] = (screen_events['timestamp'] - screen_events['prev_time']).dt.total_seconds()
            
            # Mark events to remove (ON followed by OFF within 3 seconds)
            remove_mask = (screen_events['action_type'] == 'SCREEN OFF') & \
                          (screen_events['prev_action'] == 'SCREEN ON') & \
                          (screen_events['time_diff'] < 3)
            
            # Also mark the preceding ON events
            remove_indices = screen_events.index[remove_mask].tolist()
            prev_indices = [idx-1 for idx in remove_indices if idx-1 in screen_events.index]
            all_remove = remove_indices + prev_indices
            
            # Filter events
            if all_remove:
                self.logger.info(f"Removing {len(all_remove)} rapid toggle events")
                screen_events = screen_events.drop(all_remove)
            
            # Create episodes
            episodes = []
            current_on = None
            
            for _, row in screen_events.iterrows():
                if row['action_type'] == 'SCREEN ON' and current_on is None:
                    current_on = row['timestamp']
                elif row['action_type'] == 'SCREEN OFF' and current_on is not None:
                    episodes.append({
                        'start_time': current_on,
                        'end_time': row['timestamp'],
                        'state': 'digital'
                    })
                    current_on = None
            
            # Handle case where last event is SCREEN ON (add episode ending at midnight)
            if current_on is not None:
                # End at 23:59:59 of the same day
                end_time = pd.Timestamp(date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                episodes.append({
                    'start_time': current_on,
                    'end_time': end_time,
                    'state': 'digital'
                })
            
            if episodes:
                episodes_df = pd.DataFrame(episodes)
                episodes_df['duration'] = episodes_df['end_time'] - episodes_df['start_time']
                
                # Ensure timezone-naive datetimes for consistent processing
                episodes_df['start_time'] = ensure_tz_naive(episodes_df['start_time'])
                episodes_df['end_time'] = ensure_tz_naive(episodes_df['end_time'])
                
                # Store episodes with string date to ensure consistency
                date_str = str(date)
                self.logger.info(f"Created {len(episodes_df)} digital episodes for date {date_str} (type of date: {type(date)})")
                episodes_by_day[date_str] = episodes_df
        
        return episodes_by_day
    
    def process_mobility_episodes(self, positionfixes: ti.Positionfixes) -> Dict[str, pd.DataFrame]:
        """Process mobility episodes using Trackintel with transport mode detection and home location"""
        if positionfixes.empty or len(positionfixes) <= 5:
            self.logger.warning("Insufficient position fixes for mobility processing")
            return {}
        
        # Identify home location first
        home_location = self.identify_home_location(positionfixes)
        self.home_location = home_location  # Store for later use
        self.logger.info(f"Home location identified as: {home_location}")
        
        # Split by day and filter low-quality days
        pfs_by_day = {}
        positionfixes_copy = positionfixes.copy()
        
        # Ensure tracked_at is timezone-aware (required by trackintel)
        positionfixes_copy['tracked_at'] = ensure_tz_aware(positionfixes_copy['tracked_at'])
        
        # Extract date as string for consistent keys
        positionfixes_copy['date'] = positionfixes_copy['tracked_at'].dt.date
        
        for date, day_positionfixes in positionfixes_copy.groupby('date'):
            is_valid, quality_stats = self.assess_day_quality(day_positionfixes)
            self.day_processing_status[date] = {
                'stage': 'data_quality',
                'valid': is_valid,
                'stats': quality_stats
            }
            
            if is_valid:
                self.logger.info(f"Date {date} passed quality checks: {len(day_positionfixes)} points")
                pfs_by_day[date] = day_positionfixes
            else:
                self.logger.warning(f"Date {date} failed quality checks: {quality_stats['failure_reason']}")
        
        if not pfs_by_day:
            self.logger.warning("No days with valid GPS data found")
            return {}
        
        # Process each day with trackintel
        all_mobility_episodes = {}
        all_staypoints_by_day = {}  # Store staypoints by day for location analysis
        
        for date, day_positionfixes in pfs_by_day.items():
            try:
                self.logger.info(f"Processing mobility for date {date} with trackintel")
                
                # Ensure tracked_at is timezone-aware (required by trackintel)
                day_positionfixes['tracked_at'] = ensure_tz_aware(day_positionfixes['tracked_at'])
                day_pfs = ti.Positionfixes(day_positionfixes)
                
                # Generate staypoints
                self.logger.info(f"Generating staypoints with distance={STAYPOINT_DISTANCE_THRESHOLD}m, " +
                                f"time={STAYPOINT_TIME_THRESHOLD}min, gap={STAYPOINT_GAP_THRESHOLD}min")
                
                day_pfs, staypoints = day_pfs.generate_staypoints(
                    method='sliding',
                    dist_threshold=STAYPOINT_DISTANCE_THRESHOLD,
                    time_threshold=STAYPOINT_TIME_THRESHOLD,
                    gap_threshold=STAYPOINT_GAP_THRESHOLD
                )
                
                if staypoints.empty:
                    self.logger.warning(f"No staypoints detected for date {date}")
                    continue
                
                self.logger.info(f"Generated {len(staypoints)} staypoints")
                
                # Classify staypoints as home or out_of_home
                staypoints = self.process_staypoints_with_location(staypoints, home_location)
                
                # Store staypoints for later use
                all_staypoints_by_day[date] = staypoints
                
                # Generate triplegs (movement segments)
                day_pfs, triplegs = day_pfs.generate_triplegs(staypoints, gap_threshold=STAYPOINT_GAP_THRESHOLD)
                
                if triplegs.empty:
                    self.logger.warning(f"No triplegs detected for date {date}")
                    continue
                
                self.logger.info(f"Generated {len(triplegs)} triplegs")
                
                # ----------------- TRANSPORT MODE PREDICTION CODE -----------------
                
                # Predict transport modes for triplegs
                try:
                    self.logger.info("Predicting transport modes for triplegs")
                    triplegs = triplegs.predict_transport_mode(method="simple-coarse")
                    
                    # Add a field to classify as "active" or "automated" transport
                    if 'mode' in triplegs.columns:
                        # Define mappings for trackintel modes to our transport types
                        # The trackintel library uses these specific mode names
                        mode_to_type = {
                            'slow_mobility': 'active',       # Walking, cycling
                            'motorized_mobility': 'automated', # Cars, buses, etc.
                            'fast_mobility': 'automated'      # High-speed rail, etc.
                        }
                        
                        # Create a new column for transport type
                        triplegs['transport_type'] = triplegs['mode'].map(mode_to_type).fillna('unknown')
                        
                        # Log modes found
                        mode_counts = triplegs['mode'].value_counts()
                        self.logger.info(f"Transport modes found: {dict(mode_counts)}")
                        type_counts = triplegs['transport_type'].value_counts()
                        self.logger.info(f"Transport types: {dict(type_counts)}")
                    else:
                        self.logger.warning("Mode prediction succeeded but 'mode' column not found in triplegs")
                        triplegs['transport_type'] = 'unknown'
                except Exception as e:
                    self.logger.warning(f"Transport mode prediction failed: {str(e)}")
                    # Create default columns to continue processing
                    triplegs['mode'] = 'unknown'
                    triplegs['transport_type'] = 'unknown'
                
                # ----------------- END TRANSPORT MODE PREDICTION -----------------
                
                # Flag activities and generate trips with stricter parameters
                staypoints = staypoints.create_activity_flag()
                staypoints, triplegs, trips = staypoints.generate_trips(
                    triplegs, 
                    gap_threshold=STAYPOINT_GAP_THRESHOLD/2  # More sensitive to gaps
                )
                
                if trips.empty:
                    self.logger.warning(f"No trips detected for date {date}")
                    continue
                
                self.logger.info(f"Generated {len(trips)} trips")
                
                # ----------------- TRANSPORT MODE AGGREGATION CODE -----------------
                
                # Aggregate transport modes at the trip level
                if 'mode' in triplegs.columns and 'transport_type' in triplegs.columns:
                    # Create columns for transport info in trips
                    trips['modes'] = None
                    trips['primary_mode'] = 'unknown'
                    trips['transport_type'] = 'unknown'
                    
                    # Group triplegs by trip_id and aggregate mode information
                    for trip_id, trip_row in trips.iterrows():
                        trip_triplegs = triplegs[triplegs['trip_id'] == trip_id]
                        
                        if not trip_triplegs.empty:
                            # Store all modes used in this trip
                            all_modes = trip_triplegs['mode'].dropna().unique().tolist()
                            trips.at[trip_id, 'modes'] = ', '.join(all_modes) if all_modes else 'unknown'
                            
                            # Determine primary mode based on longest duration
                            if len(trip_triplegs) > 0:
                                # Calculate duration for each tripleg
                                trip_triplegs['duration'] = (trip_triplegs['finished_at'] - trip_triplegs['started_at']).dt.total_seconds()
                                
                                # Group by mode and sum durations
                                mode_durations = trip_triplegs.groupby('mode')['duration'].sum()
                                if not mode_durations.empty:
                                    primary_mode = mode_durations.idxmax()
                                    trips.at[trip_id, 'primary_mode'] = primary_mode
                                    
                                    # Map primary mode to transport type
                                    mode_to_type = {
                                        'slow_mobility': 'active',
                                        'motorized_mobility': 'automated',
                                        'fast_mobility': 'automated'
                                    }
                                    trips.at[trip_id, 'transport_type'] = mode_to_type.get(primary_mode, 'unknown')
                
                # ----------------- END TRANSPORT MODE AGGREGATION -----------------
                
                # Add trip duration validation and correction
                trip_durations = (trips['finished_at'] - trips['started_at']).dt.total_seconds() / 60
                MAX_REASONABLE_TRIP_MINUTES = 120  # 2 hours
                original_trips_count = len(trips)
                
                # Split unreasonably long trips
                reasonable_trips = trips[trip_durations <= MAX_REASONABLE_TRIP_MINUTES].copy()
                long_trips = trips[trip_durations > MAX_REASONABLE_TRIP_MINUTES].copy()
                
                if not long_trips.empty:
                    self.logger.info(f"Splitting {len(long_trips)} overly long trips")
                    
                    # For each long trip, create shorter segments
                    for _, long_trip in long_trips.iterrows():
                        trip_duration = (long_trip['finished_at'] - long_trip['started_at']).total_seconds() / 60
                        num_segments = max(2, int(trip_duration / MAX_REASONABLE_TRIP_MINUTES))
                        
                        segment_duration = (long_trip['finished_at'] - long_trip['started_at']) / num_segments
                        
                        for i in range(num_segments):
                            segment_start = long_trip['started_at'] + (i * segment_duration)
                            segment_end = segment_start + segment_duration
                            
                            # Create new trip segment
                            new_segment = long_trip.copy()
                            new_segment['started_at'] = segment_start
                            new_segment['finished_at'] = segment_end
                            
                            # Add to reasonable trips
                            reasonable_trips = pd.concat([reasonable_trips, pd.DataFrame([new_segment])], ignore_index=True)
                    
                    # Replace original trips with processed ones
                    trips = reasonable_trips
                    self.logger.info(f"Split long trips: {original_trips_count} → {len(trips)} trips")
                
                # Create mobility episodes from trips
                trips['latitude'], trips['longitude'] = np.nan, np.nan
                
                # Try to get coordinates from origin staypoints
                if 'origin_staypoint_id' in trips.columns and not staypoints.empty:
                    for idx, trip in trips.iterrows():
                        if pd.notna(trip['origin_staypoint_id']):
                            origin_sp = staypoints[staypoints.index == trip['origin_staypoint_id']]
                            if not origin_sp.empty:
                                trips.at[idx, 'latitude'] = origin_sp.iloc[0].geometry.y
                                trips.at[idx, 'longitude'] = origin_sp.iloc[0].geometry.x
                                # Add location type info from staypoint
                                if 'location_type' in origin_sp.columns:
                                    trips.at[idx, 'origin_location_type'] = origin_sp.iloc[0]['location_type']
                                    # Also set the main location_type for the trip
                                    trips.at[idx, 'location_type'] = origin_sp.iloc[0]['location_type']
                
                # Add destination location type
                if 'destination_staypoint_id' in trips.columns and not staypoints.empty:
                    for idx, trip in trips.iterrows():
                        if pd.notna(trip['destination_staypoint_id']):
                            dest_sp = staypoints[staypoints.index == trip['destination_staypoint_id']]
                            if not dest_sp.empty and 'location_type' in dest_sp.columns:
                                trips.at[idx, 'destination_location_type'] = dest_sp.iloc[0]['location_type']
                                # Update the main location_type if needed (prioritize destination)
                                if 'location_type' not in trips.columns or pd.isna(trips.at[idx, 'location_type']):
                                    trips.at[idx, 'location_type'] = dest_sp.iloc[0]['location_type']
                
                # Create mobility episodes
                mobility_episodes = pd.DataFrame({
                    'started_at': trips['started_at'],
                    'finished_at': trips['finished_at'],
                    'duration': trips['finished_at'] - trips['started_at'],
                    'latitude': trips['latitude'],
                    'longitude': trips['longitude'],
                    'state': 'mobility',
                    'primary_mode': trips['primary_mode'] if 'primary_mode' in trips.columns else 'unknown',
                    'transport_type': trips['transport_type'] if 'transport_type' in trips.columns else 'unknown',
                    'modes': trips['modes'] if 'modes' in trips.columns else None,
                    'origin_location_type': trips['origin_location_type'] if 'origin_location_type' in trips.columns else 'unknown',
                    'destination_location_type': trips['destination_location_type'] if 'destination_location_type' in trips.columns else 'unknown',
                    'location_type': trips['location_type'] if 'location_type' in trips.columns else 'unknown'
                })
                
                # Ensure timezone-naive datetimes for consistent comparison
                mobility_episodes['started_at'] = ensure_tz_naive(mobility_episodes['started_at'])
                mobility_episodes['finished_at'] = ensure_tz_naive(mobility_episodes['finished_at'])
                
                # Log transport modes summary
                if 'transport_type' in mobility_episodes.columns:
                    type_counts = mobility_episodes['transport_type'].value_counts().to_dict()
                    self.logger.info(f"Mobility episode transport types: {type_counts}")
                    
                    if 'primary_mode' in mobility_episodes.columns:
                        mode_counts = mobility_episodes['primary_mode'].value_counts().to_dict()
                        self.logger.info(f"Mobility episode primary modes: {mode_counts}")
                
                self.logger.info(f"Created {len(mobility_episodes)} mobility episodes for date {date}")
                all_mobility_episodes[date] = mobility_episodes
                
                # Update processing status
                self.day_processing_status[date] = {
                    'stage': 'completed',
                    'valid': True,
                    'method': 'trackintel'
                }
                
            except Exception as e:
                self.logger.error(f"Error processing mobility for date {date}: {str(e)}")
                self.logger.error(traceback.format_exc())
                self.day_processing_status[date] = {
                    'stage': 'failed',
                    'valid': False,
                    'reason': f"Error: {str(e)}"
                }
        
        # Process staypoint episodes (stationary periods)
        all_location_episodes = self.process_staypoint_episodes(all_staypoints_by_day)
        
        # Combine location episodes with mobility episodes for all dates
        all_combined_episodes = {}
        all_dates = set(all_mobility_episodes.keys()) | set(all_location_episodes.keys())
        
        for date in all_dates:
            mobility_eps = all_mobility_episodes.get(date, pd.DataFrame())
            location_eps = all_location_episodes.get(date, pd.DataFrame())
            
            if not mobility_eps.empty and not location_eps.empty:
                # Combine both types of episodes
                combined_eps = pd.concat([mobility_eps, location_eps], ignore_index=True)
                all_combined_episodes[date] = combined_eps
            elif not mobility_eps.empty:
                all_combined_episodes[date] = mobility_eps
            elif not location_eps.empty:
                all_combined_episodes[date] = location_eps
        
        return all_combined_episodes

    def process_staypoint_episodes(self, staypoints_by_day):
        """Create location episodes (home vs out_of_home) from staypoints"""
        location_episodes_by_day = {}
        
        for date, staypoints in staypoints_by_day.items():
            if staypoints.empty:
                continue
                
            # Create episodes from staypoints
            episodes = []
            for idx, sp in staypoints.iterrows():
                # Skip staypoints that are part of trips (but keep activity staypoints)
                if (pd.notna(sp.get('trip_id')) or 
                    sp.get('is_activity', False) == False or 
                    sp.get('activity_flag', False) == False):
                    continue
                    
                episodes.append({
                    'started_at': sp['started_at'],
                    'finished_at': sp['finished_at'],
                    'duration': sp['finished_at'] - sp['started_at'],
                    'latitude': sp.geometry.y,
                    'longitude': sp.geometry.x,
                    'state': 'stationary',
                    'location_type': sp.get('location_type', 'unknown'),
                    'primary_mode': 'stationary',
                    'transport_type': 'stationary'
                })
            
            if episodes:
                location_episodes = pd.DataFrame(episodes)
                
                # Ensure timezone-naive datetimes
                location_episodes['started_at'] = ensure_tz_naive(location_episodes['started_at'])
                location_episodes['finished_at'] = ensure_tz_naive(location_episodes['finished_at'])
                
                # Log summary
                home_episodes = location_episodes[location_episodes['location_type'] == 'home']
                out_episodes = location_episodes[location_episodes['location_type'] == 'out_of_home']
                
                self.logger.info(f"Created {len(location_episodes)} location episodes for date {date} " +
                            f"({len(home_episodes)} home, {len(out_episodes)} out of home)")
                
                location_episodes_by_day[date] = location_episodes
        
        return location_episodes_by_day

    
    def create_daily_timeline(self, digital_episodes: pd.DataFrame, 
                           mobility_episodes: pd.DataFrame,
                           overlap_episodes: pd.DataFrame) -> pd.DataFrame:
        """Create a chronological timeline of all episodes for a day"""
        # Add episode type column to each DataFrame
        if not digital_episodes.empty:
            digital_episodes = digital_episodes.copy()
            digital_episodes['episode_type'] = 'digital'
            digital_episodes['movement_state'] = None
            digital_episodes['transport_type'] = None
            digital_episodes['primary_mode'] = None
            digital_episodes['location_type'] = None
            # Add empty location columns if they don't exist
            if 'latitude' not in digital_episodes.columns:
                digital_episodes['latitude'] = np.nan
            if 'longitude' not in digital_episodes.columns:
                digital_episodes['longitude'] = np.nan
            
            # Ensure timezone-naive datetimes
            digital_episodes['start_time'] = ensure_tz_naive(digital_episodes['start_time'])
            digital_episodes['end_time'] = ensure_tz_naive(digital_episodes['end_time'])
            
            # Try to determine location_type for digital episodes
            if hasattr(self, 'home_location') and self.home_location[0] is not np.nan:
                home_lat, home_lon = self.home_location
                for idx, ep in digital_episodes.iterrows():
                    if not pd.isna(ep['latitude']) and not pd.isna(ep['longitude']):
                        distance = self.haversine_distance(ep['latitude'], ep['longitude'], home_lat, home_lon)
                        digital_episodes.at[idx, 'location_type'] = 'home' if distance <= 100 else 'out_of_home'
        
        if not mobility_episodes.empty:
            mobility_episodes = mobility_episodes.copy()
            mobility_episodes['episode_type'] = 'mobility'
            
            if 'state' in mobility_episodes.columns:
                mobility_episodes['movement_state'] = mobility_episodes['state']
                
                # If state is 'stationary', use location_type for movement_state
                stationary_mask = mobility_episodes['state'] == 'stationary'
                if stationary_mask.any() and 'location_type' in mobility_episodes.columns:
                    # Create more descriptive movement state for stationary episodes
                    mobility_episodes.loc[stationary_mask & (mobility_episodes['location_type'] == 'home'), 'movement_state'] = 'at_home'
                    mobility_episodes.loc[stationary_mask & (mobility_episodes['location_type'] == 'out_of_home'), 'movement_state'] = 'out_of_home'
                
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
            
            # Add transport information to overlap episodes if available in mobility
            for field in ['transport_type', 'primary_mode', 'location_type']:
                if not mobility_episodes.empty and field in mobility_episodes.columns:
                    overlap_episodes[field] = 'unknown'
                    
                    # Match transport info from mobility episodes to overlaps
                    for idx, overlap in overlap_episodes.iterrows():
                        # Find mobility episodes that overlap with this overlap episode
                        for _, mobility in mobility_episodes.iterrows():
                            # Check if there's an overlap between this mobility episode and the overlap episode
                            if (max(overlap['start_time'], mobility['start_time']) < 
                                min(overlap['end_time'], mobility['end_time'])):
                                # Transfer info
                                overlap_episodes.at[idx, field] = mobility.get(field, 'unknown')
                                break
            
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
            cols = ['episode_number', 'episode_type', 'movement_state', 'transport_type', 'primary_mode', 
                   'location_type', 'start_time', 'end_time', 'duration', 'time_since_prev',
                   'latitude', 'longitude']
            all_episodes = all_episodes[[c for c in cols if c in all_episodes.columns]]
        
        return all_episodes
        
    def process_day(self, date: datetime_date, digital_episodes: pd.DataFrame, 
                mobility_episodes: pd.DataFrame) -> dict:
        """Process a single day and generate statistics"""
        self.logger.info(f"Processing day {date}")
        
        # Find overlaps between digital and mobility
        overlap_episodes = self._find_overlaps(digital_episodes, mobility_episodes)
        
        # Create daily timeline
        daily_timeline = self.create_daily_timeline(digital_episodes, mobility_episodes, overlap_episodes)
        
        # Save daily timeline
        if not daily_timeline.empty:
            timeline_file = self.output_dir / f"{date}_daily_timeline.csv"
            daily_timeline.to_csv(timeline_file, index=False)
            self.logger.info(f"Saved daily timeline to {timeline_file}")
        
        # Calculate statistics
        day_status = self.day_processing_status.get(date, {'valid': False, 'reason': 'Unknown'})
        processing_method = day_status.get('method', 'unknown')
        
        # Calculate transport type statistics
        active_transport_duration = 0
        automated_transport_duration = 0
        active_transport_episodes = 0
        automated_transport_episodes = 0
        
        # Calculate location type statistics
        home_duration = 0
        out_of_home_duration = 0
        home_episodes = 0
        out_of_home_episodes = 0
        
        if not mobility_episodes.empty:
            # Transport type stats
            if 'transport_type' in mobility_episodes.columns:
                # Classify episodes by transport type
                active_mask = mobility_episodes['transport_type'] == 'active'
                auto_mask = mobility_episodes['transport_type'] == 'automated'
                
                # Count episodes by type
                active_transport_episodes = active_mask.sum()
                automated_transport_episodes = auto_mask.sum()
                
                # Calculate durations by type (in minutes)
                if active_mask.any():
                    active_transport_duration = mobility_episodes.loc[active_mask, 'duration'].sum().total_seconds() / 60
                
                if auto_mask.any():
                    automated_transport_duration = mobility_episodes.loc[auto_mask, 'duration'].sum().total_seconds() / 60
                
                # Log the breakdown for debugging
                self.logger.info(f"Transport breakdown - Active: {active_transport_episodes} episodes ({active_transport_duration:.1f} min), " +
                            f"Automated: {automated_transport_episodes} episodes ({automated_transport_duration:.1f} min)")
            
            # Location type stats
            if 'location_type' in mobility_episodes.columns:
                # Classify episodes by location type 
                home_mask = mobility_episodes['location_type'] == 'home'
                out_mask = mobility_episodes['location_type'] == 'out_of_home'
                
                # Count episodes by type
                home_episodes = home_mask.sum()
                out_of_home_episodes = out_mask.sum()
                
                # Calculate durations by type (in minutes)
                if home_mask.any():
                    home_duration = mobility_episodes.loc[home_mask, 'duration'].sum().total_seconds() / 60
                
                if out_mask.any():
                    out_of_home_duration = mobility_episodes.loc[out_mask, 'duration'].sum().total_seconds() / 60
                
                # Log the breakdown for debugging
                self.logger.info(f"Location breakdown - Home: {home_episodes} episodes ({home_duration:.1f} min), " +
                            f"Out of home: {out_of_home_episodes} episodes ({out_of_home_duration:.1f} min)")
            
            # Add statistics for stationary episodes specifically
            stationary_mask = mobility_episodes['state'] == 'stationary'
            if stationary_mask.any():
                stationary_home_mask = stationary_mask & (mobility_episodes['location_type'] == 'home')
                stationary_out_mask = stationary_mask & (mobility_episodes['location_type'] == 'out_of_home')
                
                if stationary_home_mask.any():
                    home_episodes += stationary_home_mask.sum()
                    home_duration += mobility_episodes.loc[stationary_home_mask, 'duration'].sum().total_seconds() / 60
                
                if stationary_out_mask.any():
                    out_of_home_episodes += stationary_out_mask.sum()
                    out_of_home_duration += mobility_episodes.loc[stationary_out_mask, 'duration'].sum().total_seconds() / 60
                
                self.logger.info(f"Stationary episodes - Home: {stationary_home_mask.sum()}, Out of home: {stationary_out_mask.sum()}")
        
        day_stats = {
            'user': self.participant_id,
            'date': date,
            'valid_day': day_status.get('valid', False),
            'processing_method': processing_method,
            'digital_episodes': len(digital_episodes),
            'mobility_episodes': len(mobility_episodes) if not mobility_episodes.empty else 0,
            'overlap_episodes': len(overlap_episodes),
            'digital_duration': digital_episodes['duration'].sum().total_seconds() / 60 if not digital_episodes.empty else 0,
            'mobility_duration': mobility_episodes['duration'].sum().total_seconds() / 60 if not mobility_episodes.empty else 0,
            'overlap_duration': overlap_episodes['duration'].sum().total_seconds() / 60 if not overlap_episodes.empty else 0,
            'active_transport_duration': active_transport_duration,
            'automated_transport_duration': automated_transport_duration,
            'active_transport_episodes': active_transport_episodes,
            'automated_transport_episodes': automated_transport_episodes,
            'home_duration': home_duration,
            'out_of_home_duration': out_of_home_duration,
            'home_episodes': home_episodes,
            'out_of_home_episodes': out_of_home_episodes,
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
                self.logger.info(f"Saved {len(episodes)} {ep_type} episodes to {output_file}")
        
        return day_stats
    
    def process(self) -> List[dict]:
        """Main processing pipeline"""
        try:
            self.logger.info(f"Starting processing for participant {self.participant_id}")
            
            # Load data
            positionfixes = self.load_gps_data()
            app_df = self.load_app_data()
            
            if positionfixes.empty:
                self.logger.error("No valid GPS data loaded")
                return []
            
            if app_df.empty:
                self.logger.error("No valid app data loaded")
                return []
            
            # Process episodes
            digital_episodes = self.process_digital_episodes(app_df)
            mobility_episodes = self.process_mobility_episodes(positionfixes)
            
            # Process each day - ensure consistent key types (convert all to string)
            all_stats = []
            
            # Add extensive logging for debugging
            self.logger.info("Debugging digital_episodes keys types:")
            for key in digital_episodes.keys():
                self.logger.info(f"Key: {key}, Type: {type(key)}")
                
            self.logger.info("Debugging mobility_episodes keys types:")
            for key in mobility_episodes.keys():
                self.logger.info(f"Key: {key}, Type: {type(key)}")
            
            # Convert all keys to strings for consistent comparison
            digital_dates = set()
            for key in digital_episodes.keys():
                if isinstance(key, datetime_date):
                    digital_dates.add(str(key))
                elif isinstance(key, str):
                    digital_dates.add(key)
                else:
                    self.logger.warning(f"Unexpected key type in digital_episodes: {type(key)}")
                    digital_dates.add(str(key))
            
            mobility_dates = set()
            for key in mobility_episodes.keys():
                if isinstance(key, datetime_date):
                    mobility_dates.add(str(key))
                elif isinstance(key, str):
                    mobility_dates.add(key)
                else:
                    self.logger.warning(f"Unexpected key type in mobility_episodes: {type(key)}")
                    mobility_dates.add(str(key))
            
            all_dates = sorted(digital_dates | mobility_dates)
            
            self.logger.info(f"Processing {len(all_dates)} days of data")
            
            for date_str in all_dates:
                self.logger.info(f"Processing date string: {date_str} (type: {type(date_str)})")
                
                # Convert string date back to datetime.date for lookup
                date_obj = None
                try:
                    if isinstance(date_str, str):
                        # Try parsing as yyyy-mm-dd
                        if '-' in date_str:
                            date_parts = date_str.split('-')
                            if len(date_parts) == 3:
                                self.logger.info(f"Parsed date parts: {date_parts}")
                                date_obj = datetime_date(int(date_parts[0]), int(date_parts[1]), int(date_parts[2]))
                        else:
                            # Try other formats
                            date_obj = datetime.strptime(date_str, '%Y%m%d').date()
                    else:
                        # It's already a date object or something else
                        if hasattr(date_str, 'date') and callable(getattr(date_str, 'date')):
                            date_obj = date_str.date()
                        elif isinstance(date_str, datetime_date):
                            date_obj = date_str
                        else:
                            # Last resort - try to convert to string and parse
                            self.logger.warning(f"Unusual date type: {type(date_str)}, value: {date_str}")
                            date_obj = datetime.strptime(str(date_str), '%Y-%m-%d').date()
                            
                    self.logger.info(f"Converted to date object: {date_obj}")
                except (ValueError, IndexError, TypeError) as e:
                    self.logger.error(f"Unable to parse date string: {date_str}, error: {str(e)}")
                    continue
                
                # Get episodes using date object or string, depending on what's in the dictionaries
                digital_eps = pd.DataFrame()
                mobility_eps = pd.DataFrame()
                
                # Try multiple key formats for maximum compatibility
                for key_format in [date_obj, date_str, str(date_obj) if date_obj else None]:
                    if key_format is None:
                        continue
                    if key_format in digital_episodes:
                        digital_eps = digital_episodes[key_format]
                        break
                
                for key_format in [date_obj, date_str, str(date_obj) if date_obj else None]:
                    if key_format is None:
                        continue
                    if key_format in mobility_episodes:
                        mobility_eps = mobility_episodes[key_format]
                        break
                
                if date_obj:
                    day_stats = self.process_day(date_obj, digital_eps, mobility_eps)
                    all_stats.append(day_stats)
            
            # Save summary statistics
            if all_stats:
                summary_df = pd.DataFrame(all_stats)
                summary_file = self.output_dir / 'episode_summary.csv'
                summary_df.to_csv(summary_file, index=False)
                self.logger.info(f"Saved summary statistics to {summary_file}")
            
            return all_stats
            
        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            return []

def main():
    """Main execution function"""
    # Parse command line arguments for optional preprocessing
    parser = argparse.ArgumentParser(description='Episode detection with optional preprocessing')
    parser.add_argument('--preprocess', action='store_true', help='Run preprocessing before episode detection')
    parser.add_argument('--participant', type=str, help='Process a specific participant ID')
    args = parser.parse_args()
    
    # Find valid participants
    if args.preprocess:
        # If preprocessing requested, use looser criteria to find participants
        try:
            from preprocess_gps import main as preprocess_main
            logging.info("Running preprocessing for all participants...")
            preprocess_main()
            logging.info("Preprocessing complete, continuing to episode detection")
        except ImportError:
            logging.error("Could not import preprocess_gps.main, skipping preprocessing step")
    
    # Find processed GPS and app files
    gps_files = {f.stem.split('_processed_gps')[0]: f 
                for f in GPS_PREP_DIR.glob('*_processed_gps.csv')
                if not f.stem.startswith('._')}
    
    app_files = {f.stem.split('_app_prep')[0]: f 
               for f in GPS_PREP_DIR.glob('*_app_prep.csv')
               if not f.stem.startswith('._')}
    
    common_ids = set(gps_files.keys()) & set(app_files.keys())
    common_ids = {pid for pid in common_ids if not pid.startswith('._')}
    
    # Filter to specific participant if requested
    if args.participant:
        if args.participant in common_ids:
            common_ids = {args.participant}
            logging.info(f"Processing only participant {args.participant}")
        else:
            logging.error(f"Requested participant {args.participant} not found in processed data")
            if args.preprocess:
                logging.error("Try running without --participant flag first to preprocess all data")
            return
    
    logging.info(f"Found {len(common_ids)} participants with complete data")
    
    # Suppress detailed logging during processing
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        logging.warning("tqdm not installed, progress will not be displayed")
    
    # Process participants
    all_stats = []
    participant_summaries = []
    processed_count = 0
    failed_count = 0
    
    # Process each participant
    participant_iterator = tqdm(common_ids, desc="Processing participants") if use_tqdm else common_ids
    
    for pid in participant_iterator:
        if pid.startswith('._'):
            continue
            
        try:
            # Create processor with preprocess flag
            processor = EpisodeProcessor(pid, preprocess_data=args.preprocess)
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
    # In main() function, update the summary reporting section to include new metrics
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
            valid_percent = round(100 * len(valid_days) / max(1, total_days), 1) if total_days > 0 else 0
            
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
            
            # Transport mode statistics
            if 'active_transport_episodes' in valid_days.columns:
                summary_logger.info("\nTRANSPORT MODE STATISTICS (Valid Days Only):")
                active_count = valid_days['active_transport_episodes'].sum()
                active_duration = valid_days['active_transport_duration'].sum()
                auto_count = valid_days['automated_transport_episodes'].sum()
                auto_duration = valid_days['automated_transport_duration'].sum()
                
                summary_logger.info(f"  Active Transport (Walking, Cycling):")
                summary_logger.info(f"    Total: {int(active_count)} episodes ({round(active_duration/60, 1)} hours)")
                summary_logger.info(f"    Per Day: {round(valid_days['active_transport_episodes'].mean(), 1)} episodes ({round(valid_days['active_transport_duration'].mean(), 1)} minutes)")
                
                summary_logger.info(f"  Automated Transport (Car, Bus, Train):")
                summary_logger.info(f"    Total: {int(auto_count)} episodes ({round(auto_duration/60, 1)} hours)")
                summary_logger.info(f"    Per Day: {round(valid_days['automated_transport_episodes'].mean(), 1)} episodes ({round(valid_days['automated_transport_duration'].mean(), 1)} minutes)")
            
            # Location type statistics if available
            if 'home_episodes' in valid_days.columns:
                summary_logger.info("\nLOCATION STATISTICS (Valid Days Only):")
                home_count = valid_days['home_episodes'].sum()
                home_duration = valid_days['home_duration'].sum()
                out_count = valid_days['out_of_home_episodes'].sum()
                out_duration = valid_days['out_of_home_duration'].sum()
                
                summary_logger.info(f"  At Home:")
                summary_logger.info(f"    Total: {int(home_count)} episodes ({round(home_duration/60, 1)} hours)")
                summary_logger.info(f"    Per Day: {round(valid_days['home_episodes'].mean(), 1)} episodes ({round(valid_days['home_duration'].mean(), 1)} minutes)")
                
                summary_logger.info(f"  Out of Home:")
                summary_logger.info(f"    Total: {int(out_count)} episodes ({round(out_duration/60, 1)} hours)")
                summary_logger.info(f"    Per Day: {round(valid_days['out_of_home_episodes'].mean(), 1)} episodes ({round(valid_days['out_of_home_duration'].mean(), 1)} minutes)")

if __name__ == "__main__":
    main()