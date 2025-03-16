#!/usr/bin/env python3
"""
Streamlined episode detection script combining mobility and digital event processing
with improved efficiency and better organization
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import traceback
from pathlib import Path
import sys
import logging
from typing import Dict, List, Tuple, Optional, Union, Set
import warnings
import time
from dataclasses import dataclass

# Import the mobility detector module (properly separated now)
from mobility_detector import MobilityDetector, FallbackProcessor

# Suppress pandas warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Define constants
DIGITAL_USE_COL = 'action'  # Column containing screen events
MAX_MOBILITY_DURATION_MINUTES = 120  # Max realistic duration for a mobility episode
MAX_DIGITAL_DURATION_MINUTES = 240   # Max realistic duration for a digital episode

# Setup logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f'episode_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

# Configure loggers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

print(f"Starting episode detection script. Logging to {log_file}")
logging.info(f"Initializing episode detection pipeline")

# Summary logger for statistics
summary_logger = logging.getLogger("summary")
summary_handler = logging.StreamHandler()
summary_logger.addHandler(summary_handler)
summary_logger.setLevel(logging.INFO)
summary_logger.propagate = False

# Add parent directory to path to find config
sys.path.append(str(Path(__file__).parent.parent))
from config.paths import GPS_PREP_DIR, EPISODE_OUTPUT_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

@dataclass
class EpisodeStats:
    """Data class for tracking episode statistics"""
    total_days: int = 0
    valid_days: int = 0
    days_with_mobility: int = 0
    days_with_digital: int = 0
    days_with_overlap: int = 0
    total_mobility_episodes: int = 0
    total_digital_episodes: int = 0
    total_overlap_episodes: int = 0
    total_mobility_duration_mins: float = 0.0
    total_digital_duration_mins: float = 0.0
    total_overlap_duration_mins: float = 0.0
    days_with_fallback_data: int = 0
    successful_fallback_days: int = 0


class TimeUtil:
    """Utility class for time-related operations"""
    
    @staticmethod
    def ensure_tz_naive(datetime_series: pd.Series) -> pd.Series:
        """Convert datetime series to timezone-naive if it has a timezone"""
        if datetime_series.empty:
            return datetime_series
        if hasattr(datetime_series.iloc[0], 'tz') and datetime_series.iloc[0].tz is not None:
            return datetime_series.dt.tz_localize(None)
        return datetime_series
    
    @staticmethod
    def ensure_consistent_timezones(*dataframes, time_cols=None):
        """Ensure all time columns in all dataframes have consistent timezone handling"""
        if time_cols is None:
            time_cols = ['start_time', 'end_time', 'tracked_at', 'started_at', 'finished_at']
        
        result_dfs = []
        for df in dataframes:
            if df is None or df.empty:
                result_dfs.append(df)
                continue
                
            df_copy = df.copy()
            for col in time_cols:
                if col in df_copy.columns and pd.api.types.is_datetime64_dtype(df_copy[col]):
                    # Check if the column has timezone info
                    if len(df_copy) > 0 and hasattr(df_copy[col].iloc[0], 'tz') and df_copy[col].iloc[0].tz is not None:
                        # Convert to timezone-naive
                        df_copy[col] = df_copy[col].dt.tz_localize(None)
            
            result_dfs.append(df_copy)
        
        if len(result_dfs) == 1:
            return result_dfs[0]
        return result_dfs


class DigitalEpisodeProcessor:
    """Handles the processing of digital episodes (screen on/off events)"""
    
    def __init__(self, logger):
        self.logger = logger
    
    def process_digital_events(self, app_df: pd.DataFrame) -> Tuple[Dict[datetime.date, pd.DataFrame], List[Tuple[datetime.date, str]]]:
        """Process digital episodes by day with efficient handling of screen events"""
        episodes_by_day = {}
        
        if app_df.empty or DIGITAL_USE_COL not in app_df.columns:
            self.logger.warning(f"Digital use column '{DIGITAL_USE_COL}' not found or empty data")
            return episodes_by_day, []
        
        # Define screen event patterns 
        screen_on_values = ['SCREEN ON', 'screen_on', 'SCREEN_ON', 'on', 'ON']
        screen_off_values = ['SCREEN OFF', 'screen_off', 'SCREEN_OFF', 'off', 'OFF']
        
        # Track issues for summary logging
        issue_counter = {
            'missing_off': 0, 
            'unclosed_session': 0, 
            'dropped_long': 0,
            'empty_days': 0
        }
        problem_days = []
        
        # Process each day
        for date, day_data in app_df.groupby('date'):
            screen_events = self._prepare_screen_events(day_data, screen_on_values, screen_off_values)
            
            if len(screen_events) == 0:
                self.logger.debug(f"No screen events found for {date}")
                problem_days.append((date, "No screen events found"))
                issue_counter['empty_days'] += 1
                continue
            
            # Filter out rapid on/off sequences (likely spurious)
            screen_events = self._filter_rapid_sequences(screen_events)
            
            # Extract episodes
            episodes, day_issues = self._extract_episodes(screen_events)
            
            # Update issue counts
            for issue_type, count in day_issues.items():
                issue_counter[issue_type] += count
            
            if episodes:
                episodes_df = pd.DataFrame(episodes)
                
                # Remove timezone information
                episodes_df['start_time'] = TimeUtil.ensure_tz_naive(episodes_df['start_time'])
                episodes_df['end_time'] = TimeUtil.ensure_tz_naive(episodes_df['end_time'])
                
                episodes_by_day[date] = episodes_df
            else:
                self.logger.debug(f"No valid digital episodes for {date}")
                problem_days.append((date, "No valid digital episodes"))
        
        # Log summary of issues
        self._log_issues_summary(issue_counter)
        
        return episodes_by_day, problem_days
    
    def _prepare_screen_events(self, day_data: pd.DataFrame, screen_on_values: List[str], 
                              screen_off_values: List[str]) -> pd.DataFrame:
        """Prepare screen events by filtering and standardizing formats"""
        # Filter to only include relevant screen events
        screen_events = day_data.sort_values('timestamp')
        
        # Add check for 'package name' column
        if 'package name' in screen_events.columns:
            # Only include Android system screen events
            screen_events = screen_events[
                (screen_events['package name'] == 'android') & 
                (screen_events[DIGITAL_USE_COL].isin(screen_on_values + screen_off_values))
            ].copy()
        else:
            # Fallback to just filtering by action
            screen_events = screen_events[
                screen_events[DIGITAL_USE_COL].isin(screen_on_values + screen_off_values)
            ].copy()
        
        # Map values to standard format
        if not screen_events.empty:
            screen_events.loc[screen_events[DIGITAL_USE_COL].isin(screen_on_values), DIGITAL_USE_COL] = 'SCREEN ON'
            screen_events.loc[screen_events[DIGITAL_USE_COL].isin(screen_off_values), DIGITAL_USE_COL] = 'SCREEN OFF'
        
        return screen_events
    
    def _filter_rapid_sequences(self, screen_events: pd.DataFrame) -> pd.DataFrame:
        """Filter out rapid on/off sequences (likely spurious)"""
        if screen_events.empty:
            return screen_events
            
        # Calculate time differences between events
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
        
        # Drop temporary columns
        screen_events = screen_events.drop(columns=['prev_time', 'prev_action', 'time_diff'], errors='ignore')
        
        return screen_events
    
    def _extract_episodes(self, screen_events: pd.DataFrame) -> Tuple[List[Dict], Dict[str, int]]:
        """Extract digital episodes from screen events sequence with improved handling"""
        episodes = []
        current_on = None
        issues = {'missing_off': 0, 'unclosed_session': 0, 'dropped_long': 0}
        
        for _, row in screen_events.iterrows():
            if row[DIGITAL_USE_COL] == 'SCREEN ON' and current_on is None:
                current_on = row['timestamp']
            elif row[DIGITAL_USE_COL] == 'SCREEN OFF' and current_on is not None:
                # Check for unreasonably long sessions (likely missing OFF event)
                duration_mins = (row['timestamp'] - current_on).total_seconds() / 60
                if duration_mins <= MAX_DIGITAL_DURATION_MINUTES:
                    episodes.append({
                        'start_time': current_on,
                        'end_time': row['timestamp'],
                        'state': 'digital',
                        'duration': row['timestamp'] - current_on
                    })
                else:
                    # Drop suspiciously long episodes
                    issues['dropped_long'] += 1
                current_on = None
            elif row[DIGITAL_USE_COL] == 'SCREEN ON' and current_on is not None:
                # Missing OFF event
                issues['missing_off'] += 1
                current_on = row['timestamp']
        
        # Handle unclosed session at end of day
        if current_on is not None:
            issues['unclosed_session'] += 1
        
        return episodes, issues
    
    def _log_issues_summary(self, issues: Dict[str, int]):
        """Log summary of issues encountered during processing"""
        if issues['missing_off'] > 0:
            self.logger.warning(f"Dropped {issues['missing_off']} digital sessions with missing SCREEN OFF events")
        if issues['unclosed_session'] > 0:
            self.logger.warning(f"Dropped {issues['unclosed_session']} unclosed digital sessions at end of days")
        if issues['dropped_long'] > 0:
            self.logger.warning(f"Dropped {issues['dropped_long']} suspiciously long digital episodes (>{MAX_DIGITAL_DURATION_MINUTES} minutes)")
        if issues['empty_days'] > 0:
            self.logger.info(f"Found {issues['empty_days']} days with no valid screen events")


class EpisodeOverlapAnalyzer:
    """Handles finding overlaps between mobility and digital episodes"""
    
    def __init__(self, logger):
        self.logger = logger
    
    def find_overlaps(self, digital_episodes: pd.DataFrame, mobility_episodes: pd.DataFrame) -> pd.DataFrame:
        """Find temporal overlaps between digital and mobility episodes"""
        if digital_episodes.empty or mobility_episodes.empty:
            return pd.DataFrame()
        
        # Standardize data
        digital_episodes, mobility_episodes = self._standardize_episode_data(digital_episodes, mobility_episodes)
        
        overlap_episodes = []
        
        # Find overlaps with vectorized operations where possible
        for _, d_ep in digital_episodes.iterrows():
            # Filter potential overlapping mobility episodes
            potential_overlaps = mobility_episodes[
                (mobility_episodes['start_time'] <= d_ep['end_time']) & 
                (mobility_episodes['end_time'] >= d_ep['start_time'])
            ]
            
            for _, m_ep in potential_overlaps.iterrows():
                start = max(d_ep['start_time'], m_ep['start_time'])
                end = min(d_ep['end_time'], m_ep['end_time'])
                
                if start < end:  # There is an overlap
                    duration = end - start
                    if duration >= pd.Timedelta(seconds=30):  # Minimum meaningful overlap
                        overlap_episodes.append({
                            'start_time': start,
                            'end_time': end,
                            'state': 'overlap',
                            'movement_state': 'mobility',
                            'duration': duration
                        })
                        
                        # Add coordinates if available
                        for coord in ['start_lat', 'start_lon', 'end_lat', 'end_lon']:
                            if coord in m_ep:
                                overlap_episodes[-1][coord] = m_ep[coord]
        
        return pd.DataFrame(overlap_episodes) if overlap_episodes else pd.DataFrame()
    
    def _standardize_episode_data(self, digital_episodes: pd.DataFrame, 
                                 mobility_episodes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Standardize episode data for consistent processing"""
        # Make copies and ensure timezone consistency
        digital_episodes = digital_episodes.copy()
        mobility_episodes = mobility_episodes.copy()
        
        # Ensure consistent column names
        if 'started_at' in mobility_episodes.columns:
            mobility_episodes = mobility_episodes.rename(columns={
                'started_at': 'start_time',
                'finished_at': 'end_time'
            })
        
        # Ensure datetime columns are timezone-naive
        digital_episodes['start_time'] = TimeUtil.ensure_tz_naive(digital_episodes['start_time'])
        digital_episodes['end_time'] = TimeUtil.ensure_tz_naive(digital_episodes['end_time'])
        mobility_episodes['start_time'] = TimeUtil.ensure_tz_naive(mobility_episodes['start_time'])
        mobility_episodes['end_time'] = TimeUtil.ensure_tz_naive(mobility_episodes['end_time'])
        
        return digital_episodes, mobility_episodes
    
    def create_daily_timeline(self, digital_episodes: pd.DataFrame, mobility_episodes: pd.DataFrame, 
                             overlap_episodes: pd.DataFrame) -> pd.DataFrame:
        """Create a combined timeline of all episodes for visualization and analysis"""
        # Handle empty dataframes
        if digital_episodes.empty and mobility_episodes.empty and overlap_episodes.empty:
            return pd.DataFrame()
        
        # Create copies and standardize data
        dfs_to_combine = []
        
        if not digital_episodes.empty:
            digital_copy = digital_episodes.copy()
            digital_copy['start_time'] = TimeUtil.ensure_tz_naive(digital_copy['start_time'])
            digital_copy['end_time'] = TimeUtil.ensure_tz_naive(digital_copy['end_time'])
            dfs_to_combine.append(digital_copy)
            
        if not mobility_episodes.empty:
            mobility_copy = mobility_episodes.copy()
            if 'start_time' not in mobility_copy.columns and 'started_at' in mobility_copy.columns:
                mobility_copy = mobility_copy.rename(columns={
                    'started_at': 'start_time',
                    'finished_at': 'end_time'
                })
            mobility_copy['start_time'] = TimeUtil.ensure_tz_naive(mobility_copy['start_time'])
            mobility_copy['end_time'] = TimeUtil.ensure_tz_naive(mobility_copy['end_time'])
            dfs_to_combine.append(mobility_copy)
            
        if not overlap_episodes.empty:
            overlap_copy = overlap_episodes.copy()
            overlap_copy['start_time'] = TimeUtil.ensure_tz_naive(overlap_copy['start_time'])
            overlap_copy['end_time'] = TimeUtil.ensure_tz_naive(overlap_copy['end_time'])
            dfs_to_combine.append(overlap_copy)
        
        if not dfs_to_combine:
            return pd.DataFrame()
        
        # Combine and sort
        combined = pd.concat(dfs_to_combine, ignore_index=True)
        if 'start_time' in combined.columns and not combined.empty:
            combined = combined.sort_values('start_time')
        
        return combined


class DataLoader:
    """Handles loading and validating input data"""
    
    def __init__(self, participant_id: str, logger):
        self.participant_id = participant_id
        self.logger = logger
    
    def load_gps_data(self) -> pd.DataFrame:
        """Load GPS data with validation"""
        gps_path = GPS_PREP_DIR / f'{self.participant_id}_gps_prep.csv'
        
        try:
            # Check if file exists
            if not gps_path.exists():
                self.logger.error(f"GPS file not found: {gps_path}")
                return pd.DataFrame()
            
            # Read CSV with datetime parsing
            gps_df = pd.read_csv(gps_path, parse_dates=['tracked_at'])
            
            # Verify required columns
            required_cols = ['tracked_at', 'latitude', 'longitude', 'user_id']
            missing_cols = [col for col in required_cols if col not in gps_df.columns]
            
            if missing_cols:
                self.logger.error(f"Missing required columns in GPS data: {missing_cols}")
                
                # Try to fix missing user_id
                if 'user_id' in missing_cols and len(missing_cols) == 1:
                    gps_df['user_id'] = self.participant_id
                    missing_cols = []
            
            if missing_cols:
                return pd.DataFrame()
            
            return gps_df
            
        except Exception as e:
            self.logger.error(f"Failed to load GPS data: {str(e)}")
            return pd.DataFrame()
    
    def load_app_data(self) -> pd.DataFrame:
        """Load app data with validation"""
        app_path = GPS_PREP_DIR / f'{self.participant_id}_app_prep.csv'
        
        try:
            # Check if file exists
            if not app_path.exists():
                self.logger.error(f"App data file not found: {app_path}")
                return pd.DataFrame()
            
            # Read CSV
            app_df = pd.read_csv(app_path)
            
            # Find timestamp column
            timestamp_col = self._identify_timestamp_column(app_df)
            
            if timestamp_col is None:
                self.logger.error(f"No timestamp column found in {app_path}")
                return pd.DataFrame()
            
            # Ensure we have a date column
            app_df['date'] = app_df['timestamp'].dt.date
            
            # Find action column for SCREEN ON/OFF
            action_col = self._identify_action_column(app_df)
            
            if DIGITAL_USE_COL not in app_df.columns:
                self.logger.error(f"Digital use column '{DIGITAL_USE_COL}' not found in app data")
                return pd.DataFrame()
            
            return app_df
            
        except Exception as e:
            self.logger.error(f"Failed to load app data: {str(e)}")
            return pd.DataFrame()
    
    def _identify_timestamp_column(self, app_df: pd.DataFrame) -> Optional[str]:
        """Identify timestamp column in app data with improved handling"""
        # Check common timestamp column names
        timestamp_col = next((col for col in ['timestamp', 'Timestamp', 'date', 'tracked_at'] 
                            if col in app_df.columns), None)
        
        if timestamp_col is not None:
            app_df['timestamp'] = pd.to_datetime(app_df[timestamp_col])
            return timestamp_col
        
        # Check if we have date and time columns
        if 'date' in app_df.columns and 'time' in app_df.columns:
            app_df['timestamp'] = pd.to_datetime(app_df['date'] + ' ' + app_df['time'], 
                                          format='mixed', dayfirst=True)
            return 'timestamp'
        
        return None
    
    def _identify_action_column(self, app_df: pd.DataFrame) -> Optional[str]:
        """Identify action column for screen events"""
        action_col = next((col for col in app_df.columns 
                        if col.lower() == 'action' or 'screen' in col.lower()), None)
        
        if action_col is not None and action_col != DIGITAL_USE_COL:
            app_df[DIGITAL_USE_COL] = app_df[action_col]
            return action_col
            
        return None
    
    def find_smartphone_gps(self):
        """Find smartphone GPS data for this participant"""
        # Add more logging to debug path issues
        self.logger.info(f"Looking for smartphone GPS for participant {self.participant_id}")
        
        # Try different possible locations and naming patterns
        possible_paths = [
            GPS_PREP_DIR / f'{self.participant_id}_smartphone_gps.csv',
            RAW_DATA_DIR / "Participants" / f"Pilot_{self.participant_id}" / "9 - Smartphone Tracking App" / f"{self.participant_id.lstrip('0')}-gps.csv",
            RAW_DATA_DIR / "Participants" / f"Pilot_{self.participant_id}" / "9 - Smartphone Tracking App" / f"{self.participant_id}-gps.csv"
        ]
        
        # Check each possible path
        for path in possible_paths:
            self.logger.info(f"Checking path: {path}")
            if path.exists() and not path.name.startswith('._'):
                self.logger.info(f"Found smartphone GPS at {path}")
                return path
        
        # Try to find any gps.csv file in the smartphone tracking app folder
        smartphone_dir = RAW_DATA_DIR / "Participants" / f"Pilot_{self.participant_id}" / "9 - Smartphone Tracking App"
        self.logger.info(f"Checking directory: {smartphone_dir}")
        if smartphone_dir.exists():
            gps_files = list(smartphone_dir.glob("*-gps.csv"))
            if gps_files and not gps_files[0].name.startswith('._'):
                self.logger.info(f"Found smartphone GPS via glob: {gps_files[0]}")
                return gps_files[0]
            else:
                self.logger.info(f"Directory exists but no matching files found. Contents: {list(smartphone_dir.glob('*'))}")
        else:
            self.logger.info(f"Smartphone directory does not exist")
        
        return None


class IntegratedEpisodeProcessor:
    """
    Main processor class that combines mobility detection with digital episode detection
    for comprehensive participant activity analysis with improved organization
    """
    
    def __init__(self, participant_id: str):
        """Initialize the integrated episode processor for a participant"""
        self.participant_id = participant_id
        self.logger = logging.getLogger(f"Processor_{participant_id}")
        self.output_dir = EPISODE_OUTPUT_DIR / participant_id
        
        # Skip creation if the path is a macOS hidden file
        if '._' in str(self.output_dir):
            self.logger.warning(f"Skipping macOS hidden file: {self.output_dir}")
            return
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize helper components
        self.data_loader = DataLoader(participant_id, self.logger)
        self.digital_processor = DigitalEpisodeProcessor(self.logger)
        self.overlap_analyzer = EpisodeOverlapAnalyzer(self.logger)
        self.mobility_detector = MobilityDetector(participant_id, self.logger)
        
        # Track issues
        self.problem_days: List[Tuple[datetime.date, str]] = []
        self.fallback_days: Set[datetime.date] = set()
    
    def process(self) -> List[dict]:
        """Main processing pipeline with improved structure and error handling"""
        try:
            # 1. Load data
            gps_df = self.data_loader.load_gps_data()
            app_df = self.data_loader.load_app_data()
            
            if app_df.empty:
                self.logger.error("No valid app data found")
                return []
            
            # 2. Process mobility episodes
            mobility_episodes_by_day = self._process_mobility(gps_df)
            
            # 3. Process digital episodes
            digital_episodes_by_day, digital_problem_days = self.digital_processor.process_digital_events(app_df)
            self.problem_days.extend(digital_problem_days)
            
            # 4. Process each day to find overlaps and generate stats
            all_stats = self._process_all_days(digital_episodes_by_day, mobility_episodes_by_day)
            
            # 5. Save summary files
            self._save_summary_files(all_stats)
            
            return all_stats
            
        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}")
            traceback.print_exc()
            return []
    
    def _process_mobility(self, gps_df: pd.DataFrame) -> Dict[datetime.date, pd.DataFrame]:
        """Process GPS data to detect mobility episodes with improved error handling"""
        if gps_df.empty:
            self.logger.error("No valid GPS data found")
            return {}
        
        # Detect mobility episodes
        mobility_episodes_by_day = self.mobility_detector.process_participant(gps_df)
        
        # Get problem days from mobility detection
        mobility_problem_days = self.mobility_detector.problem_days
        self.problem_days.extend(mobility_problem_days)
        
        # Try smartphone GPS for problem days
        if mobility_problem_days:
            self._try_smartphone_fallback(mobility_problem_days, gps_df, mobility_episodes_by_day)
        
        return mobility_episodes_by_day
    
    def _try_smartphone_fallback(self, problem_days: List[Tuple[datetime.date, str]], 
                               gps_df: pd.DataFrame, 
                               mobility_episodes_by_day: Dict[datetime.date, pd.DataFrame]):
        """Try to use smartphone GPS data as fallback for problematic days"""
        smartphone_gps_path = self.data_loader.find_smartphone_gps()
        
        if smartphone_gps_path:
            self.logger.info(f"Found smartphone GPS at {smartphone_gps_path}")
            
            try:
                # Process smartphone GPS as fallback
                smartphone_gps = self.mobility_detector.process_smartphone_gps(smartphone_gps_path)
                
                if smartphone_gps is not None:
                    # Process smartphone GPS data
                    fallback_processor = FallbackProcessor(self.participant_id, self.logger)
                    fallback_days = fallback_processor.get_fallback_days(
                        problem_days, 
                        gps_df, 
                        smartphone_gps
                    )
                    
                    # Process each fallback day
                    for date, day_data in fallback_days.items():
                        fallback_episodes, _ = self.mobility_detector.process_day(date, day_data)
                        
                        if not fallback_episodes.empty:
                            # Mark these as smartphone-derived
                            fallback_episodes['data_source'] = 'smartphone'
                            
                            # Add to mobility episodes
                            mobility_episodes_by_day[date] = fallback_episodes
                            
                            # Keep track of fallback days for reporting
                            self.fallback_days.add(date)
                            
                            # Remove from problem days
                            self.problem_days = [
                                (d, r) for d, r in self.problem_days 
                                if d != date
                            ]
            except Exception as e:
                self.logger.error(f"Error processing smartphone GPS: {str(e)}")
        else:
            self.logger.warning("No smartphone GPS data found for fallback")
    
    def _process_all_days(self, digital_episodes_by_day: Dict[datetime.date, pd.DataFrame],
                        mobility_episodes_by_day: Dict[datetime.date, pd.DataFrame]) -> List[dict]:
        """Process each day to find overlaps and generate statistics"""
        all_stats = []
        all_dates = sorted(set(list(digital_episodes_by_day.keys()) + list(mobility_episodes_by_day.keys())))
        
        for date in all_dates:
            digital_eps = digital_episodes_by_day.get(date, pd.DataFrame())
            mobility_eps = mobility_episodes_by_day.get(date, pd.DataFrame())
            
            # Skip days with neither digital nor mobility data
            if digital_eps.empty and mobility_eps.empty:
                self.logger.warning(f"No episodes for {date}")
                continue
            
            # Filter excessively long mobility episodes
            if not mobility_eps.empty:
                mobility_eps = self._filter_long_mobility_episodes(date, mobility_eps)
                if mobility_eps.empty:
                    continue  # Skip if all mobility episodes were filtered out
            
            # Find overlaps
            overlap_eps = self.overlap_analyzer.find_overlaps(digital_eps, mobility_eps)
            
            # Create daily timeline
            daily_timeline = self.overlap_analyzer.create_daily_timeline(digital_eps, mobility_eps, overlap_eps)
            
            # Save daily timeline
            if not daily_timeline.empty:
                timeline_file = self.output_dir / f"{date}_daily_timeline.csv"
                daily_timeline.to_csv(timeline_file, index=False)
            
            # Get day status and calculate statistics
            day_stats = self._calculate_day_stats(date, digital_eps, mobility_eps, overlap_eps)
            all_stats.append(day_stats)
            
            # Save individual episode files
            self._save_episode_files(date, digital_eps, mobility_eps, overlap_eps)
        
        return all_stats
    
    def _filter_long_mobility_episodes(self, date: datetime.date, 
                                     mobility_episodes: pd.DataFrame) -> pd.DataFrame:
        """Filter out unrealistically long mobility episodes"""
        # Calculate duration in minutes for each episode
        if 'duration' not in mobility_episodes.columns:
            if 'start_time' in mobility_episodes.columns and 'end_time' in mobility_episodes.columns:
                mobility_episodes['duration'] = mobility_episodes['end_time'] - mobility_episodes['start_time']
            elif 'started_at' in mobility_episodes.columns and 'finished_at' in mobility_episodes.columns:
                mobility_episodes['duration'] = mobility_episodes['finished_at'] - mobility_episodes['started_at']
        
        # Filter out unrealistically long episodes
        if 'duration' in mobility_episodes.columns:
            original_count = len(mobility_episodes)
            
            # Convert timedelta to minutes and filter
            mobility_duration_mins = mobility_episodes['duration'].dt.total_seconds() / 60
            mobility_episodes = mobility_episodes[mobility_duration_mins <= MAX_MOBILITY_DURATION_MINUTES]
            
            if len(mobility_episodes) < original_count:
                self.logger.warning(f"Filtered out {original_count - len(mobility_episodes)} unrealistically long mobility episodes on {date}")
                
                # If we've filtered out all episodes, log it
                if mobility_episodes.empty and original_count > 0:
                    self.logger.warning(f"All mobility episodes on {date} were unrealistically long")
                    self.problem_days.append((date, "All mobility episodes were too long"))
        
        return mobility_episodes
    
    def _calculate_day_stats(self, date: datetime.date, digital_episodes: pd.DataFrame, 
                           mobility_episodes: pd.DataFrame, overlap_episodes: pd.DataFrame) -> dict:
        """Calculate statistics for a day's episodes"""
        # Get day quality status from mobility detector
        day_status = self.mobility_detector.day_stats.get(date, {})
        quality_stats = day_status.get('quality', {})
        is_valid_day = day_status.get('valid', False)
        detection_method = day_status.get('detection_method', 'unknown')
        
        # Track if this was a fallback data day
        is_fallback_day = date in self.fallback_days
        
        # Calculate statistics
        day_stats = {
            'user': self.participant_id,
            'date': date,
            'is_valid_day': is_valid_day,
            'detection_method': detection_method,
            'used_fallback_data': is_fallback_day,
            'digital_episodes': len(digital_episodes) if not digital_episodes.empty else 0,
            'mobility_episodes': len(mobility_episodes) if not mobility_episodes.empty else 0,
            'overlap_episodes': len(overlap_episodes) if not overlap_episodes.empty else 0,
            'digital_duration_mins': digital_episodes['duration'].sum().total_seconds() / 60 if not digital_episodes.empty else 0,
            'mobility_duration_mins': mobility_episodes['duration'].dt.total_seconds().sum() / 60 if not mobility_episodes.empty else 0,
            'overlap_duration_mins': overlap_episodes['duration'].sum().total_seconds() / 60 if not overlap_episodes.empty else 0,
        }
        
        # Add quality metrics
        for key, value in quality_stats.items():
            if key not in ['valid', 'failure_reason']:
                day_stats[f'quality_{key}'] = value
        
        # Add failure reason if applicable
        if not is_valid_day and 'failure_reason' in quality_stats:
            day_stats['failure_reason'] = quality_stats['failure_reason']
        
        return day_stats
    
    def _save_episode_files(self, date: datetime.date, digital_episodes: pd.DataFrame,
                          mobility_episodes: pd.DataFrame, overlap_episodes: pd.DataFrame):
        """Save individual episode files for each type"""
        for ep_type, episodes in [
            ('digital', digital_episodes),
            ('mobility', mobility_episodes),
            ('overlap', overlap_episodes)
        ]:
            if len(episodes) > 0:
                output_file = self.output_dir / f"{date}_{ep_type}_episodes.csv"
                episodes.to_csv(output_file, index=False)
    
    def _save_summary_files(self, all_stats: List[dict]):
        """Save summary statistics and problem days report"""
        if all_stats:
            # Save summary statistics
            summary_df = pd.DataFrame(all_stats)
            summary_file = self.output_dir / 'episode_summary.csv'
            summary_df.to_csv(summary_file, index=False)
        
        # Save problem days report
        problem_file = self.output_dir / "problem_days_report.txt"
        with open(problem_file, 'w') as f:
            f.write(f"Problem Days Report for Participant {self.participant_id}:\n")
            f.write("="*50 + "\n")
            
            if not self.problem_days:
                f.write("No problem days identified.")
            else:
                for date, reason in sorted(self.problem_days):
                    f.write(f"{date}: {reason}\n")


def generate_comprehensive_report(stats, regular_stats, fallback_stats, problem_days_all, all_summary):
    """
    Generate a comprehensive report with better fallback comparison and reasons for invalid days
    """
    summary_logger.info("\n" + "="*60)
    summary_logger.info(f"EPISODE DETECTION SUMMARY")
    summary_logger.info("="*60)
    
    # Overall statistics
    total_days = stats.total_days
    valid_days = stats.valid_days
    valid_percent = round(100 * valid_days / max(1, total_days), 1)
    
    summary_logger.info(f"\nDAY QUALITY ASSESSMENT:")
    summary_logger.info(f"Total days: {total_days}")
    summary_logger.info(f"Valid days: {valid_days} ({valid_percent}%)")
    summary_logger.info(f"Invalid days: {total_days - valid_days} ({100 - valid_percent:.1f}%)")
    
    # Fallback vs. Regular data comparison with safe division
    fallback_total = fallback_stats.total_days  # Will be 0 if no fallback days
    fallback_percent = round(100 * fallback_total / max(1, total_days), 1)
    
    summary_logger.info(f"\nFALLBACK vs. REGULAR DATA:")
    summary_logger.info(f"Days with fallback data: {fallback_total} ({fallback_percent}% of all days)")
    
    if fallback_total > 0:
        fallback_success_percent = round(100 * fallback_stats.successful_fallback_days / fallback_total, 1)
        summary_logger.info(f"Successful fallback days: {fallback_stats.successful_fallback_days} ({fallback_success_percent}% of fallback days)")
    
    # Calculate averages for regular and fallback data safely
    reg_avg_mobility_eps = 0
    reg_avg_mobility_mins = 0
    fb_avg_mobility_eps = 0
    fb_avg_mobility_mins = 0
    
    if regular_stats.days_with_mobility > 0:
        reg_avg_mobility_eps = regular_stats.total_mobility_episodes / regular_stats.days_with_mobility
        reg_avg_mobility_mins = regular_stats.total_mobility_duration_mins / regular_stats.days_with_mobility
    
    if fallback_stats.days_with_mobility > 0:
        fb_avg_mobility_eps = fallback_stats.total_mobility_episodes / fallback_stats.days_with_mobility
        fb_avg_mobility_mins = fallback_stats.total_mobility_duration_mins / fallback_stats.days_with_mobility
    
    # Only show mobility comparison if we have data for at least one source
    if regular_stats.days_with_mobility > 0 or fallback_stats.days_with_mobility > 0:
        summary_logger.info(f"\nMOBILITY COMPARISON:")
        if regular_stats.days_with_mobility > 0:
            summary_logger.info(f"  Regular data: {reg_avg_mobility_eps:.1f} episodes/day, {reg_avg_mobility_mins:.1f} mins/day")
        if fallback_stats.days_with_mobility > 0:
            summary_logger.info(f"  Fallback data: {fb_avg_mobility_eps:.1f} episodes/day, {fb_avg_mobility_mins:.1f} mins/day")
    
    # Episode statistics
    summary_logger.info("\nEPISODE STATISTICS (Valid Days Only):")
    for ep_type, days_count, total_count, total_duration in [
        ('Digital', stats.days_with_digital, stats.total_digital_episodes, stats.total_digital_duration_mins),
        ('Mobility', stats.days_with_mobility, stats.total_mobility_episodes, stats.total_mobility_duration_mins),
        ('Overlap', stats.days_with_overlap, stats.total_overlap_episodes, stats.total_overlap_duration_mins)
    ]:
        if days_count > 0:
            avg_count = total_count / days_count
            avg_duration = total_duration / days_count
            avg_episode_duration = total_duration / max(1, total_count)  # Average duration per episode
            
            summary_logger.info(f"  {ep_type} Episodes:")
            summary_logger.info(f"    Total: {int(total_count)} episodes ({round(total_duration/60, 1)} hours)")
            summary_logger.info(f"    Per Day: {round(avg_count, 1)} episodes ({round(avg_duration, 1)} minutes)")
            summary_logger.info(f"    Avg Episode Duration: {round(avg_episode_duration, 1)} minutes")
    
    # If we have all_summary, provide mobility duration distribution
    if not all_summary.empty and 'mobility_duration_mins' in all_summary.columns:
        valid_mobility_days = all_summary[(all_summary['is_valid_day']==True) & (all_summary['mobility_episodes'] > 0)]
        
        if not valid_mobility_days.empty:
            durations = valid_mobility_days['mobility_duration_mins'].values
            summary_logger.info("\nMOBILITY DURATION DISTRIBUTION:")
            summary_logger.info(f"  Min: {np.min(durations):.1f} minutes")
            summary_logger.info(f"  25th percentile: {np.percentile(durations, 25):.1f} minutes")
            summary_logger.info(f"  Median: {np.median(durations):.1f} minutes")
            summary_logger.info(f"  75th percentile: {np.percentile(durations, 75):.1f} minutes")
            summary_logger.info(f"  Max: {np.max(durations):.1f} minutes")
            summary_logger.info(f"  Mean: {np.mean(durations):.1f} minutes")
            
            # Count days with suspiciously high mobility
            high_mobility_days = np.sum(durations > 600)  # More than 10 hours of mobility
            if high_mobility_days > 0:
                high_pct = 100*high_mobility_days/len(durations)
                summary_logger.info(f"  Days with >10 hours mobility: {high_mobility_days} ({high_pct:.1f}%)")
    
    # Enhanced problem days analysis
    if problem_days_all:
        # Categorize problems with improved categorization
        categories = {}
        reason_counts = {}
        
        for _, _, reason in problem_days_all:
            # Add to specific reason counts
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            
            # Improved categorization logic with more specific categories
            category = "Other Issues"
            
            if "GPS" in reason or "gps" in reason or "point" in reason:
                if "insufficient" in reason.lower() or "too few" in reason.lower():
                    category = "Insufficient GPS Points"
                elif "quality" in reason.lower():
                    category = "Poor GPS Quality"
                else:
                    category = "GPS Data Issues"
            elif "screen" in reason.lower() or "digital" in reason.lower():
                category = "Digital Data Issues"
            elif "staypoint" in reason.lower():
                category = "Staypoint Detection Issues"
            elif "mobility" in reason.lower() or "episode" in reason.lower() or "no trip" in reason.lower():
                if "no" in reason.lower():
                    category = "No Mobility Detected"
                else:
                    category = "Mobility Detection Issues"
            elif any(term in reason.lower() for term in ["long", "duration", "unrealistic"]):
                category = "Duration Issues"
            elif "gap" in reason.lower():
                category = "Temporal Gaps"
            
            categories[category] = categories.get(category, 0) + 1
        
        summary_logger.info(f"\nPROBLEM DAYS BREAKDOWN:")
        summary_logger.info(f"  Total problem days: {len(problem_days_all)}")
        
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            percent = round(100 * count / len(problem_days_all), 1)
            summary_logger.info(f"  {category}: {count} days ({percent}%)")
        
        # Report top specific reasons
        summary_logger.info(f"\nTOP SPECIFIC ISSUES:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            percent = round(100 * count / len(problem_days_all), 1)
            summary_logger.info(f"  {reason}: {count} days ({percent}%)")
        
        # Detection method statistics, if available
        if not all_summary.empty and 'detection_method' in all_summary.columns:
            summary_logger.info("\nDETECTION METHOD STATISTICS:")
            method_counts = all_summary['detection_method'].value_counts()
            
            for method, count in method_counts.items():
                percent = round(100 * count / len(all_summary), 1)
                success_rate = 0
                if 'is_valid_day' in all_summary.columns:
                    method_valid = all_summary[(all_summary['detection_method'] == method) & 
                                            (all_summary['is_valid_day'] == True)]
                    if count > 0:
                        success_rate = round(100 * len(method_valid) / count, 1)
                
                summary_logger.info(f"  {method}: {count} days ({percent}% of total), {success_rate}% success rate")
    
    return "Comprehensive report generated"


def main():
    """Main execution function with streamlined organization"""
    start_time = time.time()
    logging.info("Episode detection started")
    
    # 1. Find valid participants
    participants = find_valid_participants()
    if not participants:
        return
    
    # 2. Ensure output directory exists
    prepare_output_directory()
    
    # 3. Process each participant and gather statistics
    stats, participant_summaries, problem_days = process_all_participants(participants)
    
    # 4. Generate and save summary statistics
    save_summary_statistics(stats, participant_summaries, problem_days)
    
    # 5. Log completion
    elapsed_time = time.time() - start_time
    logging.info(f"Episode detection completed in {elapsed_time:.2f} seconds")
    
    return "Episode detection completed successfully."


def find_valid_participants():
    """Find valid participants with required data files"""
    # Find GPS and app files
    gps_files = {f.stem.replace('_gps_prep', ''): f 
                for f in GPS_PREP_DIR.glob('*_gps_prep.csv')
                if not f.stem.startswith('._')}
    app_files = {f.stem.replace('_app_prep', ''): f 
               for f in GPS_PREP_DIR.glob('*_app_prep.csv')
               if not f.stem.startswith('._')}
    
    # Log found files
    logging.info(f"Looking for data in: {GPS_PREP_DIR}")
    logging.info(f"Found {len(gps_files)} GPS files: {list(gps_files.keys())[:5]}{'...' if len(gps_files) > 5 else ''}")
    logging.info(f"Found {len(app_files)} app files: {list(app_files.keys())[:5]}{'...' if len(app_files) > 5 else ''}")
    
    # Find participants with both GPS and app data
    common_ids = set(gps_files.keys()) & set(app_files.keys())
    common_ids = {pid for pid in common_ids if not pid.startswith('._')}
    logging.info(f"Found {len(common_ids)} participants with complete data")
    
    if len(common_ids) == 0:
        logging.warning("No participants found with complete data. Check that GPS_PREP_DIR is correct.")
        logging.warning(f"Current GPS_PREP_DIR: {GPS_PREP_DIR}")
        logging.warning("Directory contents:")
        for f in GPS_PREP_DIR.glob('*'):
            logging.warning(f"  - {f.name}")
        return None
    
    return common_ids


def prepare_output_directory():
    """Ensure output directory exists and is writable"""
    if not EPISODE_OUTPUT_DIR.exists():
        EPISODE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created output directory: {EPISODE_OUTPUT_DIR}")
    
    # Check if we can write to output directory
    try:
        test_file = EPISODE_OUTPUT_DIR / "test_write_access.txt"
        with open(test_file, 'w') as f:
            f.write("Testing write access")
        test_file.unlink()  # Delete the test file
        logging.info(f"Successfully verified write access to {EPISODE_OUTPUT_DIR}")
        return True
    except Exception as e:
        logging.error(f"Cannot write to output directory {EPISODE_OUTPUT_DIR}: {str(e)}")
        return False


def process_all_participants(participants):
    """Process all participants and collect statistics"""
    # Initialize statistics trackers
    stats = EpisodeStats()
    fallback_stats = EpisodeStats()
    regular_stats = EpisodeStats()
    
    participant_summaries = []
    problem_days_all = []
    processed_count = 0
    failed_count = 0
    
    # Set up progress tracking
    try:
        from tqdm import tqdm
        participants_iter = tqdm(participants, desc="Processing participants")
    except ImportError:
        logging.info("tqdm not installed, progress bar will not be shown")
        participants_iter = participants
    
    logging.info("Beginning participant processing")
    
    # Process each participant
    for participant_id in participants_iter:
        if participant_id.startswith('._'):
            continue
            
        logging.info(f"Processing participant: {participant_id}")
        try:
            # Create processor and process participant
            processor = IntegratedEpisodeProcessor(participant_id)
            participant_stats = processor.process()
            
            if participant_stats:
                # Calculate participant summary
                participant_summary = calculate_participant_summary(participant_id, participant_stats)
                participant_summaries.append(participant_summary)
                processed_count += 1
                
                # Add problem days
                for date, reason in processor.problem_days:
                    problem_days_all.append((participant_id, date, reason))
                
                # Update statistics
                update_statistics(participant_stats, stats, regular_stats, fallback_stats)
        except Exception as e:
            logging.error(f"Error processing participant {participant_id}: {str(e)}")
            logging.error(traceback.format_exc())
            failed_count += 1
    
    # Log completion summary
    logging.info(f"Processed {processed_count} participants successfully")
    logging.info(f"Failed to process {failed_count} participants")
    
    return (stats, fallback_stats, regular_stats), participant_summaries, problem_days_all


def calculate_participant_summary(participant_id, participant_stats):
    """Calculate summary statistics for a participant"""
    participant_df = pd.DataFrame(participant_stats)
    valid_days = sum(participant_df['is_valid_day']) if 'is_valid_day' in participant_df.columns else 0
    total_days = len(participant_df)
    
    participant_summary = {
        'participant_id': participant_id,
        'days_of_data': total_days,
        'valid_days': valid_days,
        'invalid_days': total_days - valid_days,
        'percent_valid': round(100 * valid_days / max(1, total_days), 1),
        'avg_digital_episodes': participant_df['digital_episodes'].mean(),
        'avg_mobility_episodes': participant_df['mobility_episodes'].mean(),
        'avg_overlap_episodes': participant_df['overlap_episodes'].mean(),
        'avg_digital_mins': participant_df['digital_duration_mins'].mean(),
        'avg_mobility_mins': participant_df['mobility_duration_mins'].mean(),
        'avg_overlap_mins': participant_df['overlap_duration_mins'].mean(),
    }
    
    # Count fallback days
    if 'used_fallback_data' in participant_df.columns:
        fallback_days = participant_df['used_fallback_data'].sum()
        participant_summary['fallback_days'] = fallback_days
    
    # Count detection methods
    if 'detection_method' in participant_df.columns:
        method_counts = participant_df['detection_method'].value_counts().to_dict()
        for method, count in method_counts.items():
            participant_summary[f'method_{method}'] = count
    
    return participant_summary


def update_statistics(participant_stats, overall_stats, regular_stats, fallback_stats):
    """Update statistics from a participant's data"""
    # Convert to DataFrame for easier processing
    participant_df = pd.DataFrame(participant_stats)
    
    # Update statistics for each day
    for _, day in participant_df.iterrows():
        # Update overall stats
        overall_stats.total_days += 1
        if day['is_valid_day']:
            overall_stats.valid_days += 1
        
        # Count episodes
        if day['digital_episodes'] > 0:
            overall_stats.days_with_digital += 1
            overall_stats.total_digital_episodes += day['digital_episodes']
            overall_stats.total_digital_duration_mins += day['digital_duration_mins']
        
        if day['mobility_episodes'] > 0:
            overall_stats.days_with_mobility += 1
            overall_stats.total_mobility_episodes += day['mobility_episodes']
            overall_stats.total_mobility_duration_mins += day['mobility_duration_mins']
        
        if day['overlap_episodes'] > 0:
            overall_stats.days_with_overlap += 1
            overall_stats.total_overlap_episodes += day['overlap_episodes']
            overall_stats.total_overlap_duration_mins += day['overlap_duration_mins']
        
        # Separate fallback and regular statistics
        if day.get('used_fallback_data', False):
            fallback_stats.total_days += 1
            if day['is_valid_day']:
                fallback_stats.valid_days += 1
                fallback_stats.successful_fallback_days += 1
            
            if day['digital_episodes'] > 0:
                fallback_stats.days_with_digital += 1
                fallback_stats.total_digital_episodes += day['digital_episodes']
                fallback_stats.total_digital_duration_mins += day['digital_duration_mins']
            
            if day['mobility_episodes'] > 0:
                fallback_stats.days_with_mobility += 1
                fallback_stats.total_mobility_episodes += day['mobility_episodes']
                fallback_stats.total_mobility_duration_mins += day['mobility_duration_mins']
            
            if day['overlap_episodes'] > 0:
                fallback_stats.days_with_overlap += 1
                fallback_stats.total_overlap_episodes += day['overlap_episodes']
                fallback_stats.total_overlap_duration_mins += day['overlap_duration_mins']
        else:
            # Regular day (non-fallback)
            regular_stats.total_days += 1
            if day['is_valid_day']:
                regular_stats.valid_days += 1
            
            if day['digital_episodes'] > 0:
                regular_stats.days_with_digital += 1
                regular_stats.total_digital_episodes += day['digital_episodes']
                regular_stats.total_digital_duration_mins += day['digital_duration_mins']
            
            if day['mobility_episodes'] > 0:
                regular_stats.days_with_mobility += 1
                regular_stats.total_mobility_episodes += day['mobility_episodes']
                regular_stats.total_mobility_duration_mins += day['mobility_duration_mins']
            
            if day['overlap_episodes'] > 0:
                regular_stats.days_with_overlap += 1
                regular_stats.total_overlap_episodes += day['overlap_episodes']
                regular_stats.total_overlap_duration_mins += day['overlap_duration_mins']


def save_summary_statistics(stats_tuple, participant_summaries, problem_days_all):
    """Save summary statistics to files and log report"""
    stats, fallback_stats, regular_stats = stats_tuple
    
    # Create overall dataset for analysis
    all_stats = []
    for participant in participant_summaries:
        participant_id = participant['participant_id']
        days = participant.get('days', [])
        all_stats.extend(days)
    
    all_summary = pd.DataFrame(all_stats) if all_stats else pd.DataFrame()
    
    # Save summary files
    if participant_summaries:
        # Save overall participant summary
        if not all_summary.empty:
            summary_file = EPISODE_OUTPUT_DIR / 'all_participants_summary.csv'
            all_summary.to_csv(summary_file, index=False)
        
        # Save participant summaries
        participant_summary_df = pd.DataFrame(participant_summaries)
        participant_summary_file = EPISODE_OUTPUT_DIR / 'participant_summaries.csv'
        participant_summary_df.to_csv(participant_summary_file, index=False)
        
        # Save problem days master list
        if problem_days_all:
            problem_days_df = pd.DataFrame(problem_days_all, columns=['participant_id', 'date', 'reason'])
            problem_days_file = EPISODE_OUTPUT_DIR / "all_problem_days.csv"
            problem_days_df.to_csv(problem_days_file, index=False)
    
    # Generate comprehensive report
    generate_comprehensive_report(stats, regular_stats, fallback_stats, problem_days_all, all_summary)


if __name__ == "__main__":
    logging.info("Initializing episode detection pipeline")
    print(f"Starting episode detection script. Logging to {LOG_DIR}/episode_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    main()
    print("Episode detection completed")