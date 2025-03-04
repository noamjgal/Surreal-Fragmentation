import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import json
import re
import sys
from data_utils import DataCleaner

# Get the current file's directory and add parent directory to path if needed
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))
    
# Import the EpisodeFragmentationAnalyzer class
from daily_fragmentation import EpisodeFragmentationAnalyzer

class FragmentationEMAConnector:
    def __init__(self, 
                 analyzer: EpisodeFragmentationAnalyzer,
                 window_hours: float = 4.0,
                 debug_mode: bool = False):
        """
        Initialize the connector that links EMA responses with fragmentation metrics
        
        Args:
            analyzer: Initialized EpisodeFragmentationAnalyzer instance
            window_hours: Number of hours prior to EMA to analyze for fragmentation
            debug_mode: Whether to enable detailed debug logging
        """
        self.analyzer = analyzer
        self.window_hours = window_hours
        self.debug_mode = debug_mode
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging"""
        log_level = logging.DEBUG if self.debug_mode else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def filter_episodes_by_time_window(self, 
                                      episodes_df: pd.DataFrame, 
                                      ema_timestamp: pd.Timestamp,
                                      hours_prior: float = 4.0) -> pd.DataFrame:
        """
        Filter episodes to only include those within X hours prior to EMA timestamp
        
        Args:
            episodes_df: DataFrame containing episodes
            ema_timestamp: Timestamp of the EMA response
            hours_prior: Number of hours prior to EMA to include
            
        Returns:
            Filtered DataFrame with episodes in the specified time window
        """
        if episodes_df.empty:
            return episodes_df
        
        # Create a copy to avoid modifying the original
        episodes_df = episodes_df.copy()
        
        # Ensure datetime columns are properly formatted
        for col in ['start_time', 'end_time']:
            if col in episodes_df.columns:
                if not pd.api.types.is_datetime64_any_dtype(episodes_df[col]):
                    episodes_df[col] = pd.to_datetime(episodes_df[col], errors='coerce')
        
        # Calculate window start time
        window_start = ema_timestamp - pd.Timedelta(hours=hours_prior)
        
        # Filter episodes that overlap with the window
        # An episode overlaps if:
        # 1. It starts before the EMA and ends after the window start, OR
        # 2. It starts within the window
        filtered_df = episodes_df[
            ((episodes_df['start_time'] <= ema_timestamp) & 
             (episodes_df['end_time'] >= window_start)) |
            ((episodes_df['start_time'] >= window_start) & 
             (episodes_df['start_time'] <= ema_timestamp))
        ]
        
        return filtered_df
    
    def calculate_pre_ema_fragmentation(self, 
                                        participant_dir: Path,
                                        ema_data: pd.DataFrame,
                                        participant_id: str) -> pd.DataFrame:
        """
        Calculate fragmentation metrics for time windows prior to each EMA response
        
        Args:
            participant_dir: Path to the participant's episode data directory
            ema_data: DataFrame with EMA responses
            participant_id: Identifier for the participant
            
        Returns:
            DataFrame with EMA responses and corresponding fragmentation metrics
        """
        # Create a copy of EMA data to add fragmentation metrics
        ema_with_metrics = ema_data.copy()
        
        # Initialize fragmentation metric columns
        for prefix in ['digital', 'mobility', 'overlap']:
            ema_with_metrics[f'{prefix}_fragmentation_index'] = np.nan
            ema_with_metrics[f'{prefix}_episode_count'] = np.nan
            ema_with_metrics[f'{prefix}_total_duration'] = np.nan
            ema_with_metrics[f'{prefix}_mean_duration'] = np.nan
            ema_with_metrics[f'{prefix}_std_duration'] = np.nan
            ema_with_metrics[f'{prefix}_cv'] = np.nan
        
        # Process each EMA response
        for idx, ema_row in ema_with_metrics.iterrows():
            try:
                # Get EMA timestamp
                ema_timestamp = pd.to_datetime(ema_row['datetime'])
                ema_date = ema_timestamp.strftime('%Y-%m-%d')
                
                # Define possible episode file patterns for the date
                digital_file_patterns = [
                    participant_dir / f"{ema_date}_digital_episodes.csv",
                    participant_dir / f"digital_episodes_{ema_date}.csv",
                    participant_dir / f"digital_{ema_date}_episodes.csv",
                ]
                
                mobility_file_patterns = [
                    participant_dir / f"{ema_date}_mobility_episodes.csv",
                    participant_dir / f"mobility_episodes_{ema_date}.csv",
                    participant_dir / f"{ema_date}_moving_episodes.csv",
                    participant_dir / f"moving_episodes_{ema_date}.csv",
                ]
                
                overlap_file_patterns = [
                    participant_dir / f"{ema_date}_overlap_episodes.csv",
                    participant_dir / f"overlap_episodes_{ema_date}.csv",
                ]
                
                # Find existing files
                digital_file = next((f for f in digital_file_patterns if f.exists()), None)
                mobility_file = next((f for f in mobility_file_patterns if f.exists()), None)
                overlap_file = next((f for f in overlap_file_patterns if f.exists()), None)
                
                # Skip if required files don't exist
                if not digital_file or not mobility_file:
                    self.logger.warning(f"Missing episode files for {participant_id} on {ema_date} for EMA at {ema_timestamp}")
                    continue
                
                # Load episode data
                digital_df = pd.read_csv(digital_file)
                mobility_df = pd.read_csv(mobility_file)
                overlap_df = pd.read_csv(overlap_file) if overlap_file else pd.DataFrame()
                
                # Skip if dataframes are empty
                if digital_df.empty and mobility_df.empty:
                    self.logger.warning(f"Empty episode data for {participant_id} on {ema_date}")
                    continue
                
                # Filter episodes to the time window before EMA
                digital_window = self.filter_episodes_by_time_window(
                    digital_df, ema_timestamp, self.window_hours
                )
                mobility_window = self.filter_episodes_by_time_window(
                    mobility_df, ema_timestamp, self.window_hours
                )
                overlap_window = self.filter_episodes_by_time_window(
                    overlap_df, ema_timestamp, self.window_hours
                ) if not overlap_df.empty else pd.DataFrame()
                
                # Calculate fragmentation metrics for the filtered episodes
                digital_metrics = self.analyzer.calculate_fragmentation_index(
                    digital_window, 'digital', participant_id, ema_date
                )
                mobility_metrics = self.analyzer.calculate_fragmentation_index(
                    mobility_window, 'mobility', participant_id, ema_date
                )
                overlap_metrics = self.analyzer.calculate_fragmentation_index(
                    overlap_window, 'overlap', participant_id, ema_date
                ) if not overlap_window.empty else {
                    'fragmentation_index': np.nan,
                    'episode_count': 0,
                    'total_duration': 0,
                    'status': 'no_overlap_episodes'
                }
                
                # Store metrics in the EMA dataframe
                for metrics_dict, prefix in zip(
                    [digital_metrics, mobility_metrics, overlap_metrics],
                    ['digital', 'mobility', 'overlap']
                ):
                    for key, column in [
                        ('fragmentation_index', f'{prefix}_fragmentation_index'),
                        ('episode_count', f'{prefix}_episode_count'),
                        ('total_duration', f'{prefix}_total_duration'),
                        ('mean_duration', f'{prefix}_mean_duration'),
                        ('std_duration', f'{prefix}_std_duration'),
                        ('cv', f'{prefix}_cv')
                    ]:
                        if key in metrics_dict:
                            ema_with_metrics.at[idx, column] = metrics_dict[key]
                
            except Exception as e:
                self.logger.error(f"Error processing EMA at {ema_timestamp} for {participant_id}: {str(e)}")
        
        return ema_with_metrics

def get_all_participant_dirs(episode_dir: Path) -> dict:
    """Get all participant directories and print their contents for debugging"""
    participant_dirs = {}
    
    # List all folders in the episode directory
    logging.info(f"Listing all directories in episode dir: {episode_dir}")
    
    try:
        all_dirs = [d for d in episode_dir.iterdir() if d.is_dir() and not d.name.startswith('._')]
        
        for dir_path in all_dirs:
            dir_name = dir_path.name
            participant_dirs[dir_name] = dir_path
            
            # List a sample of files in this directory
            files = list(dir_path.glob('*_episodes.csv'))
            if files:
                logging.info(f"Directory '{dir_name}' contains {len(files)} episode files")
                logging.info(f"Sample files: {[f.name for f in files[:3]]}")
            else:
                logging.warning(f"Directory '{dir_name}' has no episode files")
                
        logging.info(f"Found {len(participant_dirs)} participant directories: {list(participant_dirs.keys())}")
        return participant_dirs
    except Exception as e:
        logging.error(f"Error listing participant directories: {str(e)}")
        return {}

def extract_participant_number(participant_id: str) -> str:
    """Extract just the numeric portion from a participant ID"""
    # Remove common prefixes
    id_clean = participant_id.lower()
    for prefix in ['surreal', 'surreal_', 'surreal-']:
        if id_clean.startswith(prefix):
            id_clean = id_clean[len(prefix):]
    
    # Remove 'p' suffix if present
    if id_clean.endswith('p'):
        id_clean = id_clean[:-1]
        
    # Extract only digits
    digits = ''.join(c for c in id_clean if c.isdigit())
    
    # Remove leading zeros
    if digits:
        return digits.lstrip('0')
    
    return ""

def match_participant_id(ema_participant_id: str, participant_dirs: dict) -> Optional[Path]:
    """Match EMA participant ID to episode directory using just the numeric portion"""
    # Extract numeric portion from EMA ID
    ema_number = extract_participant_number(ema_participant_id)
    
    if not ema_number:
        logging.warning(f"Could not extract numeric ID from {ema_participant_id}")
        return None
    
    # Try direct match with directory name
    if ema_number in participant_dirs:
        logging.info(f"Direct match: EMA ID {ema_participant_id} → directory {ema_number}")
        return participant_dirs[ema_number]
    
    # Try matching with leading zeros
    for dir_name, dir_path in participant_dirs.items():
        # Extract digits only from directory name
        dir_digits = ''.join(c for c in dir_name if c.isdigit())
        
        # Compare without leading zeros
        if dir_digits.lstrip('0') == ema_number:
            logging.info(f"Numeric match: EMA ID {ema_participant_id} → directory {dir_name}")
            return dir_path
    
    logging.warning(f"No matching directory found for EMA ID {ema_participant_id} (numeric: {ema_number})")
    return None

def find_episodes_in_window(
    participant_dir: Path,
    dates: List[str],
    episode_type: str,
    window_start_time: pd.Timestamp,
    window_end_time: pd.Timestamp,
) -> pd.DataFrame:
    """
    Find all episodes of a specific type that overlap with a given time window.
    
    Args:
        participant_dir: Directory containing the participant's data
        dates: List of dates to look for episodes (in YYYY-MM-DD format)
        episode_type: Type of episodes to look for (digital, mobility, overlap)
        window_start_time: Start of the time window
        window_end_time: End of the time window
        
    Returns:
        DataFrame containing all episodes that overlap with the time window
    """
    logger = logging.getLogger(__name__)
    all_episodes = []
    file_pattern = f"*_{episode_type}_episodes.csv"
    
    # Try each date to find matching episode files
    for date in dates:
        try:
            # Convert date to string format if it's not already a string
            if not isinstance(date, str):
                # If date is a datetime object or timestamp
                if hasattr(date, 'strftime'):
                    date_str = date.strftime('%Y-%m-%d')
                else:
                    # If it's some other type, convert to string
                    date_str = str(date)
            else:
                date_str = date
                
            # Try different date formats that might exist in the filenames
            date_formats = [date_str]
            if '-' in date_str:
                date_formats.append(date_str.replace("-", ""))
            
            found_files = []
            for date_format in date_formats:
                # Look for exact date match files
                pattern = f"*{date_format}_{episode_type}_episodes.csv"
                files = list(participant_dir.glob(pattern))
                
                # Also try alternative filename patterns
                alt_patterns = [
                    f"*_{episode_type}_episodes_{date_format}.csv",
                    f"*{date_format}*_{episode_type}*.csv",
                ]
                
                for alt_pattern in alt_patterns:
                    files.extend(list(participant_dir.glob(alt_pattern)))
                
                if files:
                    found_files.extend(files)
                    break
                    
            # If no exact date match, look for all files of this type
            if not found_files:
                found_files = list(participant_dir.glob(file_pattern))
            
            # Process each found file
            for file_path in found_files:
                # Skip macOS system files (starting with ._)
                if file_path.name.startswith('._'):
                    continue
                    
                try:
                    # Read the episodes file
                    episodes = pd.read_csv(file_path)
                    
                    # Handle different column naming conventions
                    if episode_type in ["digital", "mobility"]:
                        start_col = "started_at" if "started_at" in episodes.columns else "start_time"
                        end_col = "finished_at" if "finished_at" in episodes.columns else "end_time"
                    else:  # overlap
                        start_col = "start_time"
                        end_col = "end_time"
                    
                    # Skip if required columns are missing
                    if start_col not in episodes.columns or end_col not in episodes.columns:
                        logger.warning(f"Required columns missing in {file_path}")
                        continue
                    
                    # Convert string timestamps to datetime objects
                    for col in [start_col, end_col]:
                        if episodes[col].dtype == object:  # If column contains strings
                            episodes[col] = pd.to_datetime(episodes[col], errors='coerce')
                    
                    # Convert window boundaries to numpy datetime64 if dataframe times are in that format
                    window_start = window_start_time
                    window_end = window_end_time
                    
                    if episodes[start_col].dtype == 'datetime64[ns]':
                        # Convert window boundaries to numpy datetime64
                        window_start = pd.Timestamp(window_start_time).to_datetime64()
                        window_end = pd.Timestamp(window_end_time).to_datetime64()
                    else:
                        # Convert dataframe times to pandas Timestamp
                        episodes[start_col] = episodes[start_col].apply(lambda x: pd.Timestamp(x) if pd.notna(x) else x)
                        episodes[end_col] = episodes[end_col].apply(lambda x: pd.Timestamp(x) if pd.notna(x) else x)
                    
                    # Filter episodes that overlap with the window
                    # An episode overlaps if:
                    # - It starts before the window ends AND
                    # - It ends after the window starts
                    mask = episodes[start_col].notna() & episodes[end_col].notna()
                    overlap_mask = (
                        mask & 
                        (episodes[start_col] < window_end) & 
                        (episodes[end_col] > window_start)
                    )
                    
                    filtered_episodes = episodes[overlap_mask].copy()
                    if not filtered_episodes.empty:
                        # Add info about the file source
                        filtered_episodes['source_file'] = file_path.name
                        filtered_episodes['date'] = date_str
                        all_episodes.append(filtered_episodes)
                        
                except Exception as e:
                    logger.error(f"Error reading episodes from {file_path}: {str(e)}")
                    continue
        except Exception as e:
            logger.error(f"Error processing date {date}: {str(e)}")
            continue
    
    # Combine all found episodes
    if all_episodes:
        return pd.concat(all_episodes, ignore_index=True)
    else:
        # Return empty DataFrame with appropriate columns based on episode type
        if episode_type in ["digital", "mobility"]:
            return pd.DataFrame(columns=['started_at', 'finished_at', 'duration', 'movement_state'])
        else:  # overlap
            return pd.DataFrame(columns=['start_time', 'end_time', 'duration', 'state', 'movement_state'])

def process_ema_with_episodes(
    ema_file: Path,
    participant_episode_dir: Path,
    window_hours: float,
    analyzer: EpisodeFragmentationAnalyzer
) -> List[Dict]:
    """Process EMA responses with episode fragmentation metrics from preceding windows"""
    results = []
    
    try:
        # Use DataCleaner to standardize data
        data_cleaner = DataCleaner(logging.getLogger())
        
        # Load EMA data
        ema_data = pd.read_csv(ema_file)
        
        # Standardize participant ID
        participant_id = ema_file.stem.replace('normalized_participant_', '')
        participant_id_clean = data_cleaner.standardize_participant_id(participant_id)
        
        # Standardize timestamps in EMA data
        if 'datetime' in ema_data.columns:
            ema_data = data_cleaner.standardize_timestamps(ema_data, ['datetime'])
        
        # Process each EMA response
        for _, row in ema_data.iterrows():
            ema_datetime = row['datetime']
            
            # Calculate window start time
            window_start = ema_datetime - pd.Timedelta(hours=window_hours)
            
            # Look for episode files that might contain data in this window
            window_date = window_start.date()
            possible_dates = [window_date, ema_datetime.date()]
            
            # Find digital episodes in this window
            digital_episodes = find_episodes_in_window(
                participant_episode_dir, 
                possible_dates, 
                'digital',
                window_start, 
                ema_datetime
            )
            
            # Find mobility episodes in this window
            mobility_episodes = find_episodes_in_window(
                participant_episode_dir, 
                possible_dates, 
                'mobility',
                window_start, 
                ema_datetime
            )
            
            # Find overlap episodes in this window
            overlap_episodes = find_episodes_in_window(
                participant_episode_dir, 
                possible_dates, 
                'overlap',
                window_start, 
                ema_datetime
            )
            
            # Calculate fragmentation metrics
            digital_metrics = analyzer.calculate_fragmentation_index(
                digital_episodes, 
                'digital', 
                participant_id
            )
            
            mobility_metrics = analyzer.calculate_fragmentation_index(
                mobility_episodes, 
                'mobility', 
                participant_id
            )
            
            overlap_metrics = analyzer.calculate_fragmentation_index(
                overlap_episodes, 
                'overlap', 
                participant_id
            )
            
            # Create result entry with standardized participant ID
            result = {
                'participant_id': participant_id,
                'participant_id_clean': participant_id_clean,
                'ema_datetime': ema_datetime,
                'window_hours': window_hours,
                'window_start': window_start,
                'digital_fragmentation_index': digital_metrics.get('fragmentation_index', np.nan),
                'digital_episode_count': digital_metrics.get('episode_count', 0),
                'digital_total_duration': digital_metrics.get('total_duration', 0),
                'mobility_fragmentation_index': mobility_metrics.get('fragmentation_index', np.nan),
                'mobility_episode_count': mobility_metrics.get('episode_count', 0),
                'mobility_total_duration': mobility_metrics.get('total_duration', 0),
                'overlap_fragmentation_index': overlap_metrics.get('fragmentation_index', np.nan),
                'overlap_episode_count': overlap_metrics.get('episode_count', 0),
                'overlap_total_duration': overlap_metrics.get('total_duration', 0),
            }
            
            # Add EMA variable data from the row
            for col in row.index:
                if col not in ['datetime', 'Participant_ID']:
                    result[f'ema_{col}'] = row[col]
            
            results.append(result)
            
    except Exception as e:
        logging.error(f"Error processing {ema_file}: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        
    return results

def process_ema_fragmentation(
    episode_dir: Path,
    ema_dir: Path,
    output_dir: Path,
    window_hours: float = 4.0,
    min_episodes: int = 2,
    entropy_based: bool = True,
    debug_mode: bool = False
) -> pd.DataFrame:
    """
    Process EMA data with fragmentation metrics from preceding time windows
    """
    # Setup logging
    log_level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('window_fragmentation.log'),
            logging.StreamHandler()
        ]
    )
    
    # Initialize the fragmentation analyzer
    analyzer = EpisodeFragmentationAnalyzer(
        min_episodes=min_episodes,
        entropy_based=entropy_based,
        debug_mode=debug_mode
    )
    
    # Print debug info about directories
    logging.info(f"Episode directory: {episode_dir}")
    logging.info(f"Episode directory exists: {episode_dir.exists()}")
    logging.info(f"EMA directory: {ema_dir}")
    logging.info(f"EMA directory exists: {ema_dir.exists()}")
    
    # Get all participant directories with debugging info
    participant_dirs = get_all_participant_dirs(episode_dir)
    
    # Find EMA data files
    ema_files = list(ema_dir.glob("normalized_participant_*.csv"))
    logging.info(f"Found {len(ema_files)} EMA data files")
    
    # Process each participant's EMA data
    all_results = []
    
    for ema_file in ema_files:
        # Extract participant ID from filename
        participant_id = ema_file.stem.replace('normalized_participant_', '')
        logging.info(f"Processing EMA data for participant {participant_id}")
        
        # Find matching episode directory
        participant_episode_dir = match_participant_id(participant_id, participant_dirs)
        
        if not participant_episode_dir:
            logging.warning(f"Episode directory not found for participant {participant_id}")
            continue
            
        # Process this participant's EMA data with episodes from the matching directory
        participant_results = process_ema_with_episodes(
            ema_file=ema_file,
            participant_episode_dir=participant_episode_dir,
            window_hours=window_hours,
            analyzer=analyzer
        )
        
        if participant_results:
            all_results.extend(participant_results)
            logging.info(f"Generated {len(participant_results)} results for participant {participant_id}")
        else:
            logging.warning(f"No results generated for participant {participant_id}")
    
    # Combine all results
    if all_results:
        # Convert to DataFrame
        combined_results = pd.DataFrame(all_results)
        
        # Apply data cleaning on final output
        data_cleaner = DataCleaner(logging.getLogger())
        combined_results = data_cleaner.standardize_missing_values(combined_results)
        
        # Save combined results
        output_file = output_dir / "ema_fragmentation_all.csv"
        combined_results.to_csv(output_file, index=False)
        logging.info(f"Saved {len(combined_results)} rows to {output_file}")
        
        # Generate summary statistics
        generate_summary_stats(combined_results, output_dir)
        
        return combined_results
    else:
        logging.warning("No results were generated")
        return pd.DataFrame()

def generate_summary_stats(combined_results: pd.DataFrame, output_dir: Path):
    """Generate summary statistics and save to files"""
    # Create participant-level summary
    participant_summary = combined_results.groupby('participant_id').agg({
        'digital_fragmentation_index': ['mean', 'std', 'count'],
        'mobility_fragmentation_index': ['mean', 'std', 'count'],
        'overlap_fragmentation_index': ['mean', 'std', 'count'],
        'digital_episode_count': 'mean',
        'mobility_episode_count': 'mean',
        'overlap_episode_count': 'mean',
        'digital_total_duration': 'mean',
        'mobility_total_duration': 'mean',
        'overlap_total_duration': 'mean'
    })
    
    # Flatten multi-index columns
    participant_summary.columns = ['_'.join(col).strip() for col in participant_summary.columns.values]
    
    # Save participant summary
    participant_summary.to_csv(output_dir / 'participant_summary.csv')
    logging.info(f"Saved participant summary to {output_dir / 'participant_summary.csv'}")
    
    # Create daily summary if date information is available
    if 'ema_datetime' in combined_results.columns:
        combined_results['date'] = combined_results['ema_datetime'].dt.date
        daily_summary = combined_results.groupby('date').agg({
            'digital_fragmentation_index': ['mean', 'std', 'count'],
            'mobility_fragmentation_index': ['mean', 'std', 'count'],
            'overlap_fragmentation_index': ['mean', 'std', 'count'],
            'digital_episode_count': 'mean',
            'mobility_episode_count': 'mean',
            'overlap_episode_count': 'mean',
            'digital_total_duration': 'mean',
            'mobility_total_duration': 'mean',
            'overlap_total_duration': 'mean',
            'participant_id': 'nunique'
        })
        
        # Flatten multi-index columns
        daily_summary.columns = ['_'.join(col).strip() for col in daily_summary.columns.values]
        
        # Save daily summary
        daily_summary.to_csv(output_dir / 'daily_summary.csv')
        logging.info(f"Saved daily summary to {output_dir / 'daily_summary.csv'}")

def main():
    # Import paths from central configuration
    from config.paths import EPISODE_OUTPUT_DIR, EMA_NORMALIZED_DIR, EMA_FRAGMENTATION_DIR
    
    # Use paths from the centralized configuration
    episode_dir = EPISODE_OUTPUT_DIR  # Episodes from external drive
    ema_dir = EMA_NORMALIZED_DIR      # EMA data from local directory
    output_dir = EMA_FRAGMENTATION_DIR  # Output to local directory
    
    # Add debug prints
    print(f"Episode directory: {episode_dir}")
    print(f"Episode directory exists: {episode_dir.exists()}")
    print(f"EMA directory: {ema_dir}")
    print(f"EMA directory exists: {ema_dir.exists()}")
    
    # Check if directories exist
    if not episode_dir.exists():
        logging.error(f"Episode directory not found at: {episode_dir}")
        return
    
    if not ema_dir.exists():
        logging.error(f"EMA directory not found at: {ema_dir}")
        return
        
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process EMA data with fragmentation metrics
    combined_data = process_ema_fragmentation(
        episode_dir=episode_dir,
        ema_dir=ema_dir,
        output_dir=output_dir,
        window_hours=4.0,  # 4-hour window prior to EMA
        min_episodes=2,    # Minimum episodes needed for fragmentation
        entropy_based=True,  # Use entropy-based fragmentation
        debug_mode=True   # Enable debug mode for verbose logging
    )
    
    if combined_data is not None and not combined_data.empty:
        logging.info("\n" + "="*50)
        logging.info("EMA FRAGMENTATION ANALYSIS SUMMARY")
        logging.info("="*50)
        
        # Log basic statistics
        logging.info(f"\nTotal EMA responses processed: {len(combined_data)}")
        
        # Check if participant_id exists before accessing
        if 'participant_id' in combined_data.columns:
            logging.info(f"Total participants: {combined_data['participant_id'].nunique()}")
        else:
            logging.info("No participant ID data available")
        
        # Log coverage statistics for fragmentation metrics
        for prefix in ['digital', 'mobility', 'overlap']:
            metric_col = f'{prefix}_fragmentation_index'
            if metric_col in combined_data.columns:
                valid_count = combined_data[metric_col].notna().sum()
                total_count = len(combined_data)
                coverage = (valid_count / total_count) * 100 if total_count > 0 else 0
                
                logging.info(f"\n{prefix.capitalize()} Fragmentation Coverage:")
                logging.info(f"  Valid measurements: {valid_count} of {total_count} ({coverage:.1f}%)")
                
                if valid_count > 0:
                    logging.info(f"  Mean: {combined_data[metric_col].mean():.4f}")
                    logging.info(f"  Median: {combined_data[metric_col].median():.4f}")
                    logging.info(f"  Std Dev: {combined_data[metric_col].std():.4f}")
        
        # Log file locations
        logging.info("\nOutput Files:")
        logging.info(f"  Combined EMA with fragmentation data: {output_dir / 'ema_fragmentation_all.csv'}")
        logging.info(f"  Participant summary: {output_dir / 'participant_summary.csv'}")
        logging.info(f"  Daily summary: {output_dir / 'daily_summary.csv'}")
        
        logging.info("\nAnalysis complete!")
    else:
        logging.error("Analysis failed - no results were generated.")

if __name__ == "__main__":
    main()