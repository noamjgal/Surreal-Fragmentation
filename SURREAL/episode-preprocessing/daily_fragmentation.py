# daily fragmentation script 
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Tuple, Optional
import json
import re
from data_utils import DataCleaner
import sys

# Add parent directory to path to find config
sys.path.append(str(Path(__file__).parent.parent))
from config.paths import EPISODE_OUTPUT_DIR, PROCESSED_DATA_DIR

class EpisodeFragmentationAnalyzer:
    def __init__(self, 
                 min_episodes: int = 1,
                 max_episode_duration: float = 24 * 60,  # 24 hours in minutes
                 outlier_threshold: float = 3.0,  # Standard deviations for outlier detection
                 entropy_based: bool = True,
                 debug_mode: bool = False):
        """
        Initialize fragmentation analyzer with configurable settings
        
        Args:
            min_episodes: Minimum number of episodes required for fragmentation calculation
            max_episode_duration: Maximum allowed episode duration in minutes
            outlier_threshold: Number of standard deviations for outlier detection
            entropy_based: Whether to use entropy-based (True) or HHI-based (False) fragmentation
            debug_mode: Whether to enable detailed debug logging
        """
        self.min_episodes = min_episodes
        self.max_episode_duration = max_episode_duration
        self.outlier_threshold = outlier_threshold
        self.entropy_based = True  # Always use entropy-based, ignore parameter
        self.debug_mode = debug_mode
        self._setup_logging()
        
        # Initialize statistics tracking
        self.stats = {
            'digital': {'success': 0, 'insufficient_episodes': 0, 'invalid_duration': 0},
            'mobility': {'success': 0, 'insufficient_episodes': 0, 'invalid_duration': 0, 'missing_file': 0, 'column_mismatch': 0},
            'overlap': {'success': 0, 'insufficient_episodes': 0, 'invalid_duration': 0, 'missing_file': 0}
        }
        
        # Track failure reasons
        self.failure_reasons = {
            'digital': {},
            'mobility': {},
            'overlap': {}
        }
        
    def _setup_logging(self):
        """Configure logging"""
        log_level = logging.DEBUG if self.debug_mode else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def parse_duration_string(self, duration_str):
        """Parse duration string in various formats to minutes"""
        try:
            # Handle NumPy types
            if hasattr(duration_str, 'item'):
                return float(duration_str.item())
                
            # For numeric values already
            if isinstance(duration_str, (int, float)):
                return float(duration_str)
                
            # For timedelta objects
            if isinstance(duration_str, timedelta):
                return duration_str.total_seconds() / 60
                
            # Convert to string if it's not already
            duration_str = str(duration_str).strip()
            
            # Handle format like "0 days 00:01:23" or "0 days 00"
            if 'days' in duration_str:
                # Extract days part
                days_match = re.search(r'(\d+)\s*days', duration_str)
                days = int(days_match.group(1)) if days_match else 0
                
                # Extract time part
                time_part = duration_str.split('days')[1].strip()
                
                # Handle with colons (HH:MM:SS)
                if ':' in time_part:
                    parts = time_part.split(':')
                    if len(parts) == 3:  # HH:MM:SS
                        h, m, s = map(float, parts)
                        return days * 24 * 60 + h * 60 + m + s / 60
                    elif len(parts) == 2:  # MM:SS
                        m, s = map(float, parts)
                        return days * 24 * 60 + m + s / 60
                else:
                    # Just hours like "0 days 01" or hours with no colons
                    try:
                        hours = float(time_part)
                        return days * 24 * 60 + hours * 60
                    except ValueError:
                        return 1.0
            
            # Handle HH:MM:SS format
            elif ':' in duration_str:
                parts = duration_str.split(':')
                if len(parts) == 3:  # HH:MM:SS
                    h, m, s = map(float, parts)
                    return h * 60 + m + s / 60
                elif len(parts) == 2:  # MM:SS
                    m, s = map(float, parts)
                    return m + s / 60
            
            # Try direct conversion for simple numeric strings
            else:
                return float(duration_str)
        except Exception:
            # If all parsing fails, return a default value
            return 1.0

    def calculate_fragmentation_index(self, episodes_df: pd.DataFrame, episode_type: str, 
                                    participant_id: str, date_str: str) -> Dict:
        """Calculate fragmentation index for a set of episodes"""
        result = {
            'fragmentation_index': np.nan,
            'episode_count': 0,
            'total_duration': 0,
            'mean_duration': np.nan,
            'std_duration': np.nan,
            'cv': np.nan,  # Coefficient of variation
            'status': 'no_data'
        }
        
        # Return empty results if no data
        if episodes_df.empty:
            if self.debug_mode:
                self.logger.debug(f"No {episode_type} episodes for {participant_id} on {date_str}")
            
            reason = 'missing_file'
            self.stats[episode_type]['missing_file'] = self.stats[episode_type].get('missing_file', 0) + 1
            self._update_failure_reason(episode_type, reason)
            result['status'] = reason
            return result
        
        # Add more detailed logging only in debug mode
        if self.debug_mode:
            self.logger.debug(f"Processing {len(episodes_df)} {episode_type} episodes for {participant_id} on {date_str}")
            self.logger.debug(f"Columns available: {episodes_df.columns.tolist()}")
        
        # Handle missing columns based on episode_type
        time_cols = ['start_time', 'end_time']
        
        # Verify we have the required columns
        missing_cols = [col for col in time_cols if col not in episodes_df.columns]
        if missing_cols:
            reason = 'column_mismatch'
            self.logger.warning(f"Missing required columns {missing_cols} in {episode_type} episodes")
            self.stats[episode_type]['column_mismatch'] = self.stats[episode_type].get('column_mismatch', 0) + 1
            self._update_failure_reason(episode_type, reason)
            result['status'] = reason
            return result
        
        # Filter out invalid rows (missing timestamps)
        valid_data = episodes_df.dropna(subset=time_cols)
        
        if len(valid_data) == 0:
            reason = 'no_valid_episodes'
            self.logger.warning(f"No valid {episode_type} episodes after filtering NaN timestamps")
            self.stats[episode_type]['invalid_duration'] += 1
            self._update_failure_reason(episode_type, reason)
            result['status'] = reason
            return result
        
        # Calculate durations if not already in the dataset
        if 'duration' not in valid_data.columns:
            valid_data['duration'] = (valid_data['end_time'] - valid_data['start_time']).dt.total_seconds()
        else:
            # Convert duration to numeric if it's a string - reduced logging
            try:
                # Try to convert duration column to numeric
                sample_val = valid_data['duration'].iloc[0] if len(valid_data) > 0 else None
                
                # Check if the duration values need parsing
                if isinstance(sample_val, str):
                    # Only log in debug mode
                    if self.debug_mode:
                        self.logger.debug(f"Converting string durations to numeric for {episode_type} episodes")
                    # Use the parse_duration_string method which handles various formats
                    valid_data['duration'] = valid_data['duration'].apply(self.parse_duration_string)
                else:
                    # Ensure all values are numeric using pandas to_numeric
                    valid_data['duration'] = pd.to_numeric(valid_data['duration'], errors='coerce')
            except Exception as e:
                self.logger.error(f"Error converting duration to numeric: {str(e)}")
                # Create a safe numeric duration column
                self.logger.info("Falling back to calculated durations based on timestamps")
                valid_data['duration_numeric'] = (valid_data['end_time'] - valid_data['start_time']).dt.total_seconds()
                valid_data['duration'] = valid_data['duration_numeric']
        
        # Handle and log any negative or zero durations
        try:
            # Ensure durations are numeric before comparison
            numeric_durations = pd.to_numeric(valid_data['duration'], errors='coerce')
            invalid_durations = (numeric_durations <= 0).sum()
            if invalid_durations > 0:
                self.logger.warning(f"{invalid_durations} {episode_type} episodes have invalid durations <= 0")
            
            # Filter to valid durations
            valid_data = valid_data[numeric_durations > 0]
        except Exception as e:
            self.logger.error(f"Error filtering invalid durations: {str(e)}")
            # Try to recover
            valid_data['safe_duration'] = (valid_data['end_time'] - valid_data['start_time']).dt.total_seconds()
            valid_data = valid_data[valid_data['safe_duration'] > 0]
            valid_data['duration'] = valid_data['safe_duration']
        
        # Check if we still have enough data
        if len(valid_data) < self.min_episodes:
            reason = f'insufficient_episodes_{len(valid_data)}'
            self.logger.warning(f"Too few valid {episode_type} episodes: {len(valid_data)} < {self.min_episodes}")
            self.stats[episode_type]['insufficient_episodes'] += 1
            self._update_failure_reason(episode_type, reason)
            result['status'] = reason
            result['episode_count'] = len(valid_data)
            return result
        
        # NEW CODE: Detect and remove outliers
        try:
            duration_values = pd.to_numeric(valid_data['duration'], errors='coerce')
            mean_duration = duration_values.mean()
            std_duration = duration_values.std()
            
            # Filter outliers based on standard deviation threshold
            if len(valid_data) > 2:  # Need at least 3 points for meaningful outlier detection
                outlier_mask = np.abs(duration_values - mean_duration) <= (self.outlier_threshold * std_duration)
                valid_data = valid_data[outlier_mask]
                
                # Log outlier removal only in debug mode
                if self.debug_mode:
                    removed_count = (~outlier_mask).sum()
                    if removed_count > 0:
                        self.logger.debug(f"Removed {removed_count} outliers from {episode_type} episodes")
            
            # Check if we still have enough data after outlier removal
            if len(valid_data) < self.min_episodes:
                reason = f'insufficient_episodes_after_outlier_removal_{len(valid_data)}'
                self.logger.warning(f"Too few valid {episode_type} episodes after outlier removal: {len(valid_data)}")
                self.stats[episode_type]['insufficient_episodes'] += 1
                self._update_failure_reason(episode_type, reason)
                result['status'] = reason
                result['episode_count'] = len(valid_data)
                return result
        except Exception as e:
            self.logger.warning(f"Error during outlier detection: {str(e)}, proceeding without outlier removal")
        
        # Record episode counts and total duration
        result['episode_count'] = len(valid_data)
        result['total_duration'] = valid_data['duration'].sum()
        
        # Calculate basic statistics
        result['mean_duration'] = valid_data['duration'].mean()
        result['std_duration'] = valid_data['duration'].std() if len(valid_data) > 1 else 0
        result['cv'] = result['std_duration'] / result['mean_duration'] if result['mean_duration'] > 0 else 0
        
        # ALWAYS use entropy-based fragmentation
        try:
            # Entropy-based fragmentation using episode durations
            durations = valid_data['duration'].values
            total_time = durations.sum()
            
            # Calculate probabilities (proportion of time spent in each episode)
            probabilities = durations / total_time
            
            # Calculate entropy
            entropy = -np.sum(probabilities * np.log(probabilities))
            
            # Normalize to [0, 1] - higher is more fragmented
            max_entropy = np.log(len(durations))
            index = entropy / max_entropy if max_entropy > 0 else 0
            
            result['fragmentation_index'] = index
            result['status'] = 'success'
            self.stats[episode_type]['success'] += 1
            
            if self.debug_mode:
                self.logger.debug(f"Successfully calculated {episode_type} fragmentation: {index:.4f}")
            
        except Exception as e:
            reason = f'calculation_error_{str(e)}'
            self.logger.error(f"Error calculating entropy-based fragmentation: {str(e)}")
            self._update_failure_reason(episode_type, reason)
            result['status'] = reason
        
        return result

    def _update_failure_reason(self, episode_type: str, reason: str):
        """Update failure reason tracking"""
        if reason not in self.failure_reasons[episode_type]:
            self.failure_reasons[episode_type][reason] = 0
        self.failure_reasons[episode_type][reason] += 1

    def process_daily_episodes(self, participant_dir: Path, date_str: str, participant_id: str) -> Optional[Dict]:
        """Process episodes for a single day for one participant"""
        try:
            # Look for digital episode file
            digital_pattern = f"{date_str}_digital_episodes.csv"
            digital_pattern2 = f"digital_episodes_{date_str}.csv"
            digital_file = next((f for f in participant_dir.glob(f"*{digital_pattern}*")), None)
            if not digital_file:
                digital_file = next((f for f in participant_dir.glob(f"*{digital_pattern2}*")), None)
            
            # Look for mobility/movement episode file
            mobility_pattern = f"{date_str}_mobility_episodes.csv"
            mobility_pattern2 = f"mobility_episodes_{date_str}.csv"
            movement_pattern = f"{date_str}_movement_episodes.csv"
            movement_pattern2 = f"movement_episodes_{date_str}.csv"
            
            mobility_file = next((f for f in participant_dir.glob(f"*{mobility_pattern}*")), None)
            if not mobility_file:
                mobility_file = next((f for f in participant_dir.glob(f"*{mobility_pattern2}*")), None)
            if not mobility_file:
                mobility_file = next((f for f in participant_dir.glob(f"*{movement_pattern}*")), None)
            if not mobility_file:
                mobility_file = next((f for f in participant_dir.glob(f"*{movement_pattern2}*")), None)
            
            # Look for overlap episode file
            overlap_pattern = f"{date_str}_overlap_episodes.csv"
            overlap_pattern2 = f"overlap_episodes_{date_str}.csv"
            overlap_file = next((f for f in participant_dir.glob(f"*{overlap_pattern}*")), None)
            if not overlap_file:
                overlap_file = next((f for f in participant_dir.glob(f"*{overlap_pattern2}*")), None)
            
            # Only log in debug mode - less verbose
            if self.debug_mode:
                log_message = f"Participant {participant_id}, Date {date_str} - Files found: "
                log_message += f"Digital: {'Yes' if digital_file else 'No'}, "
                log_message += f"Mobility/Movement: {'Yes' if mobility_file else 'No'}, "
                log_message += f"Overlap: {'Yes' if overlap_file else 'No'}"
                self.logger.debug(log_message)
            
            if not digital_file and not mobility_file and not overlap_file:
                self.logger.warning(f"Missing all required episode files for {participant_id} on {date_str}")
                return None
            
            # Read episode data if files exist
            digital_episodes = pd.read_csv(digital_file) if digital_file else pd.DataFrame()
            mobility_episodes = pd.read_csv(mobility_file) if mobility_file else pd.DataFrame()
            overlap_episodes = pd.read_csv(overlap_file) if overlap_file else pd.DataFrame()
            
            # Display column names only in debug mode
            if self.debug_mode:
                if not digital_episodes.empty:
                    self.logger.debug(f"Digital episode columns: {digital_episodes.columns.tolist()}")
                if not mobility_episodes.empty:
                    self.logger.debug(f"Mobility episode columns: {mobility_episodes.columns.tolist()}")
                if not overlap_episodes.empty:
                    self.logger.debug(f"Overlap episode columns: {overlap_episodes.columns.tolist()}")
            
            # Check if we have enough data to calculate anything
            if (digital_episodes.empty and mobility_episodes.empty and overlap_episodes.empty):
                logging.warning(f"All episode files for {participant_id} on {date_str} are empty")
                return None
            
            # Standardize column names for mobility episodes
            if not mobility_episodes.empty:
                # Map different column names to standard format
                col_mapping = {
                    'started_at': 'start_time',
                    'finished_at': 'end_time'
                }
                mobility_episodes = mobility_episodes.rename(columns={k: v for k, v in col_mapping.items() if k in mobility_episodes.columns})
                
                # Verify required columns exist after standardization
                if 'start_time' not in mobility_episodes.columns or 'end_time' not in mobility_episodes.columns:
                    self.logger.warning(f"Missing required time columns in mobility file for {participant_id} on {date_str}")
                    self.stats['mobility']['column_mismatch'] = self.stats['mobility'].get('column_mismatch', 0) + 1
                    self._update_failure_reason('mobility', 'column_mismatch')
                    mobility_episodes = pd.DataFrame()  # Empty it so we don't try to process it
            
            # Standardize column names for overlap episodes
            if not overlap_episodes.empty:
                # Check/fix columns if needed
                if 'started_at' in overlap_episodes.columns and 'start_time' not in overlap_episodes.columns:
                    overlap_episodes = overlap_episodes.rename(columns={'started_at': 'start_time', 'finished_at': 'end_time'})
                
                # Verify required columns exist after standardization
                if 'start_time' not in overlap_episodes.columns or 'end_time' not in overlap_episodes.columns:
                    self.logger.warning(f"Missing required time columns in overlap file for {participant_id} on {date_str}")
                    self.stats['overlap']['column_mismatch'] = self.stats['overlap'].get('column_mismatch', 0) + 1
                    self._update_failure_reason('overlap', 'column_mismatch')
                    overlap_episodes = pd.DataFrame()  # Empty it
            
            # Only log in debug mode
            if self.debug_mode:
                logging.debug(f"Digital episodes: {len(digital_episodes)}, "
                            f"Mobility episodes: {len(mobility_episodes)}, "
                            f"Overlap episodes: {len(overlap_episodes)}")
            
            # Convert time columns to datetime
            for df, name in [(digital_episodes, 'digital'), (mobility_episodes, 'mobility'), (overlap_episodes, 'overlap')]:
                if not df.empty:
                    for col in ['start_time', 'end_time']:
                        if col in df.columns:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                            
                            # Only log in warning cases
                            nat_count = df[col].isna().sum()
                            if nat_count > 0:
                                self.logger.warning(f"{nat_count} NaT values after datetime conversion in {name} {col}")
            
            # Calculate fragmentation metrics for each episode type
            digital_metrics = self.calculate_fragmentation_index(
                digital_episodes, 'digital', participant_id, date_str
            )
            mobility_metrics = self.calculate_fragmentation_index(
                mobility_episodes, 'mobility', participant_id, date_str
            )
            overlap_metrics = self.calculate_fragmentation_index(
                overlap_episodes, 'overlap', participant_id, date_str
            ) if not overlap_episodes.empty else {
                'fragmentation_index': np.nan,
                'episode_count': 0,
                'total_duration': 0,
                'status': 'no_overlap_episodes'
            }
            
            # Add identifying prefixes to metrics keys
            digital_metrics_prefixed = {f"digital_{k}": v for k, v in digital_metrics.items()}
            mobility_metrics_prefixed = {f"mobility_{k}": v for k, v in mobility_metrics.items()}
            overlap_metrics_prefixed = {f"overlap_{k}": v for k, v in overlap_metrics.items()}
            
            # Add cleaned ID to result
            result = {
                'participant_id': participant_id,
                'date': date_str,
                **digital_metrics_prefixed,
                **mobility_metrics_prefixed,
                **overlap_metrics_prefixed
            }
            
            # Only log detailed metrics in debug mode
            if self.debug_mode:
                self.logger.debug(f"Calculated metrics for {participant_id} on {date_str}: "
                               f"Digital: {digital_metrics.get('fragmentation_index', 'N/A')}, "
                               f"Mobility: {mobility_metrics.get('fragmentation_index', 'N/A')}, "
                               f"Overlap: {overlap_metrics.get('fragmentation_index', 'N/A')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing episodes for {participant_id} on {date_str}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def print_failure_summary(self):
        """Print a summary of failure reasons"""
        self.logger.info("\nFAILURE SUMMARY BY EPISODE TYPE")
        self.logger.info("=" * 40)
        
        for episode_type in ['digital', 'mobility', 'overlap']:
            self.logger.info(f"\n{episode_type.capitalize()} Episode Failures:")
            
            # Success rate
            total = sum(self.stats[episode_type].values())
            success = self.stats[episode_type].get('success', 0)
            if total > 0:
                success_rate = (success / total) * 100
                self.logger.info(f"  Success Rate: {success}/{total} ({success_rate:.1f}%)")
            else:
                self.logger.info("  No data processed")
            
            # Detailed failure reasons
            if self.failure_reasons[episode_type]:
                self.logger.info("  Failure Reasons:")
                for reason, count in sorted(self.failure_reasons[episode_type].items(), key=lambda x: x[1], reverse=True):
                    self.logger.info(f"    - {reason}: {count} instances")
            else:
                self.logger.info("  No failures recorded")

    def generate_analysis_plots(self, data_df, output_dir):
        """Generate analysis plots for fragmentation metrics"""
        if data_df is None or data_df.empty:
            self.logger.warning("No data available for visualization - skipping plot generation")
            return
        
        # Create output directory for plots
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        
        # Select only numeric columns for correlation
        numeric_cols = data_df.select_dtypes(include=['float64', 'int64']).columns
        
        # Check if we have enough numeric data
        if len(numeric_cols) < 2:
            self.logger.warning(f"Not enough numeric columns for correlation analysis: {numeric_cols}")
            return
        
        # Filter out columns with all NaN values
        valid_cols = [col for col in numeric_cols if data_df[col].notna().sum() > 0]
        
        # Check if we have enough valid columns after filtering
        if len(valid_cols) < 2:
            self.logger.warning(f"Not enough valid data for correlation analysis. Valid columns: {valid_cols}")
            return
        
        # Compute correlation only for valid columns
        correlation = data_df[valid_cols].corr()
        
        # Check if correlation matrix has any valid data
        if correlation.size == 0 or correlation.isna().all().all():
            self.logger.warning("Correlation matrix is empty or contains only NaN values")
            return
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Matrix of Fragmentation Metrics', fontsize=16)
        plt.tight_layout()
        plt.savefig(plot_dir / 'correlation_matrix.png', dpi=300)
        plt.close()
        
        # The rest of your visualization code with similar checks...
        
        # Example for fragmentation index distributions
        frag_cols = [col for col in data_df.columns if 'fragmentation_index' in col]
        if frag_cols and any(data_df[col].notna().sum() > 0 for col in frag_cols):
            plt.figure(figsize=(12, 6))
            for col in frag_cols:
                if data_df[col].notna().sum() > 0:  # Only plot if we have valid data
                    sns.histplot(data_df[col].dropna(), kde=True, label=col)
            plt.title('Distribution of Fragmentation Indices', fontsize=16)
            plt.xlabel('Fragmentation Index Value', fontsize=14)
            plt.ylabel('Frequency', fontsize=14)
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / 'fragmentation_distribution.png', dpi=300)
            plt.close()
        else:
            self.logger.warning("No valid fragmentation index data for distribution plot")

        # Add similar checks for other plots...

def process_episodes_data(
    episode_dir: Path,
    output_dir: Path,
    min_episodes: int = 1,
    entropy_based: bool = True,
    debug_mode: bool = False
):
    """Process all participants' data to calculate fragmentation metrics"""
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer and DataCleaner
    analyzer = EpisodeFragmentationAnalyzer(
        min_episodes=min_episodes, 
        entropy_based=True,  # Force true
        outlier_threshold=3.0,  # Add explicit outlier threshold  
        debug_mode=debug_mode
    )
    data_cleaner = DataCleaner(logging.getLogger())
    
    # Get list of participant directories, filtering out hidden files
    participant_dirs = [d for d in episode_dir.iterdir() 
                       if d.is_dir() and not d.name.startswith('.')]
    logging.info(f"Found {len(participant_dirs)} participant directories")
    
    all_results = []
    
    # Process each participant
    for participant_dir in participant_dirs:
        participant_id = participant_dir.name
        
        # Get list of episode files - try both potential patterns
        episode_files = list(participant_dir.glob('*_digital_episodes.csv'))
        episode_files.extend(list(participant_dir.glob('digital_episodes_*.csv')))
        
        # Filter out hidden files
        episode_files = [f for f in episode_files if not f.name.startswith('.')]
        
        if not episode_files:
            logging.warning(f"No episode files found for participant {participant_id}")
            continue
        
        # Extract dates from filenames
        dates = set()
        for file in episode_files:
            for pattern in ['%Y-%m-%d', '%Y%m%d']:
                try:
                    # Try to extract date using different patterns
                    date_str = re.search(r'(\d{4}-\d{2}-\d{2}|\d{8})', file.name)
                    if date_str:
                        date_obj = datetime.strptime(date_str.group(1), pattern)
                        dates.add(date_obj.strftime('%Y-%m-%d'))
                        break
                except (ValueError, AttributeError):
                    continue
        
        if not dates:
            logging.warning(f"Could not extract dates from filenames for {participant_id}")
            continue
        
        logging.info(f"Processing {len(dates)} days for participant {participant_id}")
        
        # Process each date
        participant_results = []
        for date_str in sorted(dates):
            result = analyzer.process_daily_episodes(participant_dir, date_str, participant_id)
            if result:
                participant_results.append(result)
        
        # Add to overall results
        all_results.extend(participant_results)
    
    # Convert to DataFrame
    combined_results = pd.DataFrame(all_results) if all_results else pd.DataFrame()
    
    # Log the column names only once in normal mode
    logging.info(f"DataFrame columns: {combined_results.columns.tolist() if not combined_results.empty else 'None'}")
    
    # Check for empty dataframe
    if combined_results.empty:
        logging.error("No valid data was processed. Check episode files and logs.")
        # Create empty files with correctly named columns to avoid future processing errors
        empty_df = pd.DataFrame(columns=[
            'participant_id', 'date', 
            'digital_fragmentation_index', 'digital_episode_count', 'digital_total_duration',
            'mobility_fragmentation_index', 'mobility_episode_count', 'mobility_total_duration', 
            'overlap_fragmentation_index', 'overlap_episode_count', 'overlap_total_duration'
        ])
        empty_df.to_csv(output_dir / 'fragmentation_all_metrics.csv', index=False)
        empty_df.to_csv(output_dir / 'participant_summaries.csv', index=False)
        return empty_df
    
    # Verify that the key metrics columns exist
    logging.info("Checking for expected columns in dataset...")
    expected_columns = [
        'digital_fragmentation_index', 'mobility_fragmentation_index', 'overlap_fragmentation_index'
    ]
    
    # Check if any of the expected columns are missing
    missing_columns = [col for col in expected_columns if col not in combined_results.columns]
    if missing_columns:
        logging.warning(f"Missing expected metric columns: {missing_columns}")
    
    # Calculate participant summaries
    available_cols = combined_results.columns.tolist()
    
    # Create a dynamic aggregation dictionary based on available columns
    agg_dict = {}
    metrics_to_check = [
        'digital_episode_count', 'digital_fragmentation_index', 'digital_total_duration',
        'mobility_episode_count', 'mobility_fragmentation_index', 'mobility_total_duration',
        'overlap_episode_count', 'overlap_fragmentation_index', 'overlap_total_duration'
    ]
    
    for col in metrics_to_check:
        if col in available_cols:
            if 'count' in col:
                agg_dict[col] = ['mean', 'min', 'max', 'sum']
            else:
                agg_dict[col] = ['mean', 'min', 'max']
    
    # Only attempt to aggregate if we have metrics columns
    if agg_dict:
        try:
            participant_summary = combined_results.groupby('participant_id').agg(agg_dict)
            
            # Flatten column structure
            participant_summary.columns = ['_'.join(col).strip() for col in participant_summary.columns.values]
            
            # Add days_processed column
            days_per_participant = combined_results.groupby('participant_id').size()
            participant_summary['days_processed'] = days_per_participant
            
            # Reset index for easier handling
            participant_summary = participant_summary.reset_index()
            
            # Save summary
            participant_summary.to_csv(output_dir / 'participant_summaries.csv', index=False)
        except Exception as e:
            logging.error(f"Error creating participant summary: {str(e)}")
            # Create a basic participant summary
            participant_summary = combined_results.groupby('participant_id').size().reset_index()
            participant_summary.columns = ['participant_id', 'days_processed']
            participant_summary.to_csv(output_dir / 'participant_summaries.csv', index=False)
    else:
        logging.warning("No metrics columns available for aggregation")
        # Create a basic participant summary
        participant_summary = combined_results.groupby('participant_id').size().reset_index()
        participant_summary.columns = ['participant_id', 'days_processed']
        participant_summary.to_csv(output_dir / 'participant_summaries.csv', index=False)
    
    # Save full results
    combined_results.to_csv(output_dir / 'fragmentation_all_metrics.csv', index=False)
    
    # Generate plots
    analyzer.generate_analysis_plots(combined_results, output_dir)
    
    # Print failure reasons summary
    analyzer.print_failure_summary()
    
    return combined_results

def main():
    # Configure paths using the same config as detect_episodes.py
    episode_dir = EPISODE_OUTPUT_DIR
    output_dir = PROCESSED_DATA_DIR / 'fragmentation'
    
    # Check if episode directory exists
    if not episode_dir.exists():
        print(f"Episode directory not found at: {episode_dir}")
        print("Please check the path and try again.")
        return
    
    print(f"Found episode directory at: {episode_dir}")
    print(f"Participant directories found: {len([d for d in episode_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])}")
    
    # Process all episode data
    results_df = process_episodes_data(
        episode_dir=episode_dir,
        output_dir=output_dir,
        min_episodes=1,  # Changed from 2 to 1
        entropy_based=True,  # Keep parameter but it's ignored
        debug_mode=False    # Disable verbose debugging for cleaner output
    )
    
    # Log summary statistics and file locations
    if results_df is not None:
        logging.info("\n" + "="*50)
        logging.info("FRAGMENTATION ANALYSIS SUMMARY")
        logging.info("="*50)
        
        # Summary statistics - with column existence checks
        logging.info("\nSummary Statistics:")
        for metric in ['digital_fragmentation_index', 'mobility_fragmentation_index', 'overlap_fragmentation_index']:
            if metric in results_df.columns:  # Check if column exists
                valid_values = results_df[metric].dropna()
                logging.info(f"\n{metric.split('_')[0].capitalize()} Fragmentation:")
                logging.info(f"  Valid measurements: {len(valid_values)} of {len(results_df)} ({len(valid_values)/len(results_df)*100:.1f}% if available)")
                if not valid_values.empty:
                    logging.info(f"  Mean: {valid_values.mean():.4f}")
                    logging.info(f"  Median: {valid_values.median():.4f}")
                    logging.info(f"  Std Dev: {valid_values.std():.4f}")
                    logging.info(f"  Range: [{valid_values.min():.4f}, {valid_values.max():.4f}]")
                else:
                    logging.info("  No valid measurements available")
            else:
                logging.info(f"\n{metric.split('_')[0].capitalize()} Fragmentation:")
                logging.info(f"  No data available for this metric")
        
        # Participant coverage - safer implementation
        if 'participant_id' in results_df.columns:
            total_participants = results_df['participant_id'].nunique()
            logging.info(f"\nParticipant Coverage:")
            logging.info(f"  Total participants processed: {total_participants}")
            avg_days = len(results_df) / total_participants if total_participants > 0 else 0
            logging.info(f"  Average days per participant: {avg_days:.1f}")
        else:
            logging.info("\nParticipant Coverage: No participant data available")
        
        # File locations
        logging.info("\nOutput Files:")
        logging.info(f"  Full metrics data: {output_dir / 'fragmentation_all_metrics.csv'}")
        logging.info(f"  Participant summary: {output_dir / 'participant_summaries.csv'}")
        logging.info(f"  Visualization plots: {output_dir / 'plots/'}")
        
        logging.info("\nAnalysis complete!")
    else:
        logging.error("Analysis failed - no results were generated.")

if __name__ == "__main__":
    main()