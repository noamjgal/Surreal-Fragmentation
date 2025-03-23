import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple, Optional

class FragmentationAnalyzer:
    def __init__(self, 
                 min_episodes: int = 1,
                 max_episode_duration: float = 24 * 60,  # 24 hours in minutes
                 outlier_threshold: float = 3.0):  # Standard deviations for outlier detection
        """
        Initialize fragmentation analyzer with configurable settings
        
        Args:
            min_episodes: Minimum number of episodes required for fragmentation calculation
            max_episode_duration: Maximum allowed episode duration in minutes
            outlier_threshold: Number of standard deviations for outlier detection
        """
        self.min_episodes = min_episodes
        self.max_episode_duration = max_episode_duration
        self.outlier_threshold = outlier_threshold
        self._setup_logging()
        
        # Initialize statistics tracking
        self.stats = {
            'digital': {'success': 0, 'insufficient_episodes': 0, 'invalid_duration': 0},
            'mobility': {'success': 0, 'insufficient_episodes': 0, 'invalid_duration': 0},
            'digital_home': {'success': 0, 'insufficient_episodes': 0, 'invalid_duration': 0},
            'overlap': {'success': 0, 'insufficient_episodes': 0, 'invalid_duration': 0}
        }
        
    def _setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def calculate_fragmentation_index(self, 
                                    episodes_df: pd.DataFrame, 
                                    episode_type: str) -> Dict:
        """
        Calculate fragmentation index with improved validation and outlier detection
        
        Args:
            episodes_df: DataFrame with episode data
            episode_type: Type of episodes ('digital' or 'mobility')
            
        Returns:
            Dictionary with fragmentation metrics and status
        """
        # Rename duration_minutes to duration for internal processing
        episodes_df = episodes_df.rename(columns={'duration_minutes': 'duration'})
        
        if len(episodes_df) < self.min_episodes:
            self.stats[episode_type]['insufficient_episodes'] += 1
            return {
                'fragmentation_index': np.nan,
                'episode_count': len(episodes_df),
                'total_duration': episodes_df['duration'].sum() if not episodes_df.empty else 0,
                'status': 'insufficient_episodes'
            }
            
        # Validate durations
        valid_episodes = episodes_df[
            (episodes_df['duration'] > 0) &
            (episodes_df['duration'] <= self.max_episode_duration)
        ].copy()
        
        if len(valid_episodes) < self.min_episodes:
            self.stats[episode_type]['invalid_duration'] += 1
            return {
                'fragmentation_index': np.nan,
                'episode_count': len(valid_episodes),
                'total_duration': valid_episodes['duration'].sum(),
                'status': 'invalid_duration'
            }
            
        # Detect and handle outliers
        mean_duration = valid_episodes['duration'].mean()
        std_duration = valid_episodes['duration'].std()
        outlier_mask = np.abs(valid_episodes['duration'] - mean_duration) <= (self.outlier_threshold * std_duration)
        valid_episodes = valid_episodes[outlier_mask]
        
        if len(valid_episodes) < self.min_episodes:
            self.stats[episode_type]['insufficient_episodes'] += 1
            return {
                'fragmentation_index': np.nan,
                'episode_count': len(valid_episodes),
                'total_duration': valid_episodes['duration'].sum(),
                'status': 'insufficient_episodes_after_outlier_removal'
            }
            
        # Calculate normalized durations
        total_duration = valid_episodes['duration'].sum()
        normalized_durations = valid_episodes['duration'] / total_duration
        
        # Calculate entropy-based index only
        S = len(valid_episodes)
        
        # Entropy-based calculation
        index = 0.0
        if S > 1:
            entropy = -np.sum(normalized_durations * np.log(normalized_durations))
            index = entropy / np.log(S)
        
        self.stats[episode_type]['success'] += 1
        
        return {
            'fragmentation_index': index,
            'episode_count': S,
            'total_duration': total_duration,
            'mean_duration': valid_episodes['duration'].mean(),
            'std_duration': valid_episodes['duration'].std(),
            'cv': valid_episodes['duration'].std() / valid_episodes['duration'].mean() if valid_episodes['duration'].mean() > 0 else np.nan,
            'status': 'success'
        }

    def calculate_digital_frag_during_mobility(self, 
                                             digital_df: pd.DataFrame, 
                                             mobility_df: pd.DataFrame) -> Dict:
        """
        Calculate fragmentation of digital use during mobility periods
        
        Args:
            digital_df: DataFrame with digital episodes
            mobility_df: DataFrame with mobility episodes
            
        Returns:
            Dictionary with fragmentation metrics for digital use during mobility
        """
        if digital_df.empty or mobility_df.empty:
            return {
                'fragmentation_index': np.nan,
                'episode_count': 0,
                'total_duration': 0,
                'status': 'no_episodes'
            }
            
        # Convert times to datetime if needed
        for df in [digital_df, mobility_df]:
            for col in ['start_time', 'end_time']:
                if df[col].dtype != 'datetime64[ns]':
                    df[col] = pd.to_datetime(df[col])
        
        # Find overlapping episodes
        overlapping_episodes = []
        for _, digital in digital_df.iterrows():
            for _, mobility in mobility_df.iterrows():
                if (digital['start_time'] < mobility['end_time'] and 
                    digital['end_time'] > mobility['start_time']):
                    # Calculate overlap duration
                    overlap_start = max(digital['start_time'], mobility['start_time'])
                    overlap_end = min(digital['end_time'], mobility['end_time'])
                    duration = (overlap_end - overlap_start).total_seconds() / 60
                    
                    if duration > 0:
                        overlapping_episodes.append({
                            'start_time': overlap_start,
                            'end_time': overlap_end,
                            'duration': duration
                        })
        
        if not overlapping_episodes:
            return {
                'fragmentation_index': np.nan,
                'episode_count': 0,
                'total_duration': 0,
                'status': 'no_overlapping_episodes'
            }
            
        overlap_df = pd.DataFrame(overlapping_episodes)
        return self.calculate_fragmentation_index(overlap_df, 'digital')

    def calculate_digital_frag_during_home(self, 
                                         digital_df: pd.DataFrame, 
                                         home_df: pd.DataFrame) -> Dict:
        """
        Calculate fragmentation of digital use during home periods
        
        Args:
            digital_df: DataFrame with digital episodes
            home_df: DataFrame with home episodes
            
        Returns:
            Dictionary with fragmentation metrics for digital use at home
        """
        if digital_df.empty or home_df.empty:
            return {
                'fragmentation_index': np.nan,
                'episode_count': 0,
                'total_duration': 0,
                'status': 'no_episodes'
            }
        
        # Convert times to datetime if needed
        for df in [digital_df, home_df]:
            for col in ['start_time', 'end_time']:
                if df[col].dtype != 'datetime64[ns]':
                    df[col] = pd.to_datetime(df[col])
        
        # Find overlapping episodes
        overlapping_episodes = []
        for _, digital in digital_df.iterrows():
            for _, home in home_df.iterrows():
                if (digital['start_time'] < home['end_time'] and 
                    digital['end_time'] > home['start_time']):
                    # Calculate overlap duration
                    overlap_start = max(digital['start_time'], home['start_time'])
                    overlap_end = min(digital['end_time'], home['end_time'])
                    duration = (overlap_end - overlap_start).total_seconds() / 60
                    
                    if duration > 0:
                        overlapping_episodes.append({
                            'start_time': overlap_start,
                            'end_time': overlap_end,
                            'duration': duration
                        })
        
        if not overlapping_episodes:
            return {
                'fragmentation_index': np.nan,
                'episode_count': 0,
                'total_duration': 0,
                'status': 'no_overlapping_episodes'
            }
        
        overlap_df = pd.DataFrame(overlapping_episodes)
        return self.calculate_fragmentation_index(overlap_df, 'digital_home')

    def process_episode_summary(self, 
                              digital_file_path: str, 
                              mobility_file_path: str,
                              home_file_path: str = None,  # NEW: Optional home episodes file
                              print_sample: bool = False) -> Optional[Dict]:
        """Process episode data and calculate all fragmentation metrics"""
        try:
            digital_df = pd.read_csv(digital_file_path)
            mobility_df = pd.read_csv(mobility_file_path)
            
            # Load home episodes if available
            home_df = None
            if home_file_path and os.path.exists(home_file_path):
                home_df = pd.read_csv(home_file_path)
                if print_sample:
                    self.logger.info(f"\nSample data from {os.path.basename(home_file_path)}:")
                    self.logger.info(home_df.head())
            else:
                self.logger.warning(f"Home episodes file not found: {home_file_path}")
            
            for df in [digital_df, mobility_df]:
                df['start_time'] = pd.to_datetime(df['start_time'])
                df['end_time'] = pd.to_datetime(df['end_time'])
            
            if home_df is not None:
                home_df['start_time'] = pd.to_datetime(home_df['start_time'])
                home_df['end_time'] = pd.to_datetime(home_df['end_time'])
            
            if print_sample:
                self.logger.info(f"\nSample data from {os.path.basename(digital_file_path)}:")
                self.logger.info(digital_df.head())
                self.logger.info(f"\nSample data from {os.path.basename(mobility_file_path)}:")
                self.logger.info(mobility_df.head())
            
            participant_id, date = self._extract_info_from_filename(os.path.basename(digital_file_path))
            
            # Calculate various fragmentation metrics
            digital_metrics = self.calculate_fragmentation_index(digital_df, 'digital')
            mobility_metrics = self.calculate_fragmentation_index(mobility_df, 'mobility')
            overlap_metrics = self.calculate_digital_frag_during_mobility(digital_df, mobility_df)
            
            # NEW: Calculate digital during home fragmentation if home data is available
            digital_home_metrics = {
                'fragmentation_index': np.nan,
                'episode_count': 0,
                'total_duration': 0,
                'status': 'no_home_data'
            }
            
            if home_df is not None and not home_df.empty:
                digital_home_metrics = self.calculate_digital_frag_during_home(digital_df, home_df)
            
            # NEW: Calculate delta between digital-home and digital-mobility fragmentation
            digital_home_frag = digital_home_metrics.get('fragmentation_index', np.nan)
            digital_mobility_frag = overlap_metrics.get('fragmentation_index', np.nan)
            
            if not np.isnan(digital_home_frag) and not np.isnan(digital_mobility_frag):
                digital_home_mobility_delta = digital_home_frag - digital_mobility_frag
            else:
                digital_home_mobility_delta = np.nan
            
            result = {
                'participant_id': participant_id,
                'date': date,
                'digital_fragmentation_index': digital_metrics['fragmentation_index'],
                'mobility_fragmentation_index': mobility_metrics['fragmentation_index'],
                'overlap_fragmentation_index': overlap_metrics['fragmentation_index'],
                'digital_home_fragmentation_index': digital_home_metrics['fragmentation_index'],  # NEW
                'digital_home_mobility_delta': digital_home_mobility_delta,  # NEW
                'digital_episode_count': digital_metrics['episode_count'],
                'mobility_episode_count': mobility_metrics['episode_count'],
                'digital_home_episode_count': digital_home_metrics['episode_count'],  # NEW
                'overlap_episode_count': overlap_metrics['episode_count'],  # NEW
                'digital_total_duration': digital_metrics['total_duration'],
                'mobility_total_duration': mobility_metrics['total_duration'],
                'digital_home_total_duration': digital_home_metrics['total_duration'],  # NEW
                'overlap_total_duration': overlap_metrics.get('total_duration', 0),  # NEW
                'digital_status': digital_metrics['status'],
                'mobility_status': mobility_metrics['status'],
                'overlap_status': overlap_metrics['status'],
                'digital_home_status': digital_home_metrics['status']  # NEW
            }
            
            # Add additional metrics if available
            for metrics_type in [digital_metrics, mobility_metrics, digital_home_metrics, overlap_metrics]:
                if 'cv' in metrics_type:
                    if metrics_type == digital_metrics:
                        prefix = 'digital_'
                    elif metrics_type == mobility_metrics:
                        prefix = 'mobility_'
                    elif metrics_type == digital_home_metrics:
                        prefix = 'digital_home_'
                    else:
                        prefix = 'overlap_'
                    result[f'{prefix}cv'] = metrics_type['cv']
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing files {digital_file_path} and {mobility_file_path}: {str(e)}")
            return None

    def _extract_info_from_filename(self, filename: str) -> Tuple[str, datetime.date]:
        """Extract participant ID and date from filename with more flexible parsing"""
        try:
            # Remove .csv extension
            base_name = filename.replace('.csv', '')
            
            # Split by underscore
            parts = base_name.split('_')
            
            # Look for the date part (contains hyphens)
            date_part = next((p for p in parts if '-' in p), None)
            if not date_part:
                raise ValueError(f"Could not find date in filename: {filename}")
            
            # Find the user ID part (usually after the date)
            date_idx = parts.index(date_part)
            if date_idx < len(parts) - 1:
                user_id = parts[date_idx + 1]
            else:
                user_id = parts[date_idx - 1]
            
            # Parse the date
            date = datetime.strptime(date_part, "%Y-%m-%d").date()
            
            self.logger.debug(f"Extracted date={date}, user_id={user_id} from {filename}")
            return user_id, date
        
        except Exception as e:
            self.logger.error(f"Error extracting info from filename {filename}: {str(e)}")
            # Fallback to a default extraction
            parts = filename.split('_')
            if len(parts) >= 4:
                date_str = parts[2]
                user_id = parts[3].split('.')[0]
                try:
                    date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    return user_id, date
                except:
                    self.logger.error(f"Failed to parse date from {date_str}")
            
            # Last resort fallback
            self.logger.warning(f"Using placeholder values for unparseable filename: {filename}")
            return "unknown", datetime.now().date()

    def generate_analysis_plots(self, df: pd.DataFrame, output_dir: Path):
        """Generate visualization plots for fragmentation analysis"""
        metrics = ['digital_fragmentation_index', 'mobility_fragmentation_index', 
                  'overlap_fragmentation_index', 'digital_home_fragmentation_index']  # Added digital_home
        
        for metric in metrics:
            if metric not in df.columns or df[metric].isna().all():
                self.logger.warning(f"Skipping plot for {metric} - no valid data")
                continue
            
            plt.figure(figsize=(10, 6))
            
            # Create histogram
            plt.hist(df[metric].dropna(), bins=50, alpha=0.7)
            plt.title(f'Distribution of {metric}')
            plt.xlabel('Fragmentation Index')
            plt.ylabel('Frequency')
            
            # Add vertical lines for quartiles
            quartiles = df[metric].quantile([0.25, 0.5, 0.75])
            for q, q_value in quartiles.items():
                plt.axvline(q_value, color='r', linestyle='--', alpha=0.5)
                plt.text(q_value, plt.ylim()[1]*0.9, f'Q{int(q*4)}={q_value:.2f}', 
                        rotation=90, verticalalignment='top')
            
            plt.savefig(output_dir / f'{metric}_distribution.png')
            plt.close()
        
        # NEW: Add plot for digital fragmentation by location type
        if 'digital_home_fragmentation_index' in df.columns and 'overlap_fragmentation_index' in df.columns:
            # Create paired data for valid comparisons
            valid_mask = df['digital_home_fragmentation_index'].notna() & df['overlap_fragmentation_index'].notna()
            
            if valid_mask.sum() > 0:
                plt.figure(figsize=(10, 6))
                
                # Prepare data for boxplot
                compare_data = pd.DataFrame({
                    'Digital at Home': df.loc[valid_mask, 'digital_home_fragmentation_index'],
                    'Digital during Mobility': df.loc[valid_mask, 'overlap_fragmentation_index']
                })
                
                # Calculate means for annotation
                means = compare_data.mean()
                
                # Create boxplot
                import seaborn as sns
                sns.boxplot(data=compare_data)
                
                # Add mean lines
                for i, col in enumerate(compare_data.columns):
                    plt.axhline(means[col], color='red', linestyle='--', alpha=0.5)
                    plt.text(i, means[col]+0.01, f'Mean: {means[col]:.3f}', 
                            ha='center', va='bottom', fontsize=12)
                
                plt.title('Digital Fragmentation by Location Type')
                plt.ylabel('Fragmentation Index')
                plt.tight_layout()
                plt.savefig(output_dir / 'digital_fragmentation_by_location.png')
                plt.close()
            else:
                self.logger.warning("Not enough data for digital fragmentation by location comparison")
        
        # NEW: Add histogram for delta metric
        if 'digital_home_mobility_delta' in df.columns:
            delta_data = df['digital_home_mobility_delta'].dropna()
            
            if len(delta_data) > 0:
                plt.figure(figsize=(10, 6))
                
                # Create histogram with density
                plt.hist(delta_data, bins=30, alpha=0.7, density=True)
                
                # Add vertical line at zero (no difference)
                plt.axvline(0, color='black', linestyle='-', alpha=0.5)
                
                # Add vertical line at mean
                mean_delta = delta_data.mean()
                plt.axvline(mean_delta, color='red', linestyle='--', alpha=0.7)
                plt.text(mean_delta, plt.ylim()[1]*0.9, f'Mean: {mean_delta:.2f}', 
                        rotation=90, verticalalignment='top')
                
                plt.title('Delta between Digital Fragmentation at Home vs. Mobility')
                plt.xlabel('Fragmentation Index Delta (Home - Mobility)')
                plt.ylabel('Density')
                plt.tight_layout()
                plt.savefig(output_dir / 'digital_home_mobility_delta.png')
                plt.close()
            else:
                self.logger.warning("Not enough data for digital fragmentation delta histogram")

def main():
    # Try multiple possible paths for flexibility
    possible_base_dirs = [
        Path('/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon'),
        Path('/Users/noamgal/DSProjects/Fragmentation/TLV/data-processing'),
        Path('/Users/noamgal/DSProjects/Fragmentation/TLV'),
        Path('.')  # Current directory as fallback
    ]
    
    # Find the first existing directory with episode files
    input_dir = None
    for base_dir in possible_base_dirs:
        possible_input = base_dir / 'episodes'
        if possible_input.exists() and any(possible_input.glob('*_episodes_*.csv')):
            input_dir = possible_input
            logging.info(f"Found episodes directory at: {input_dir}")
            break
    
    if not input_dir:
        logging.error("Could not find valid episodes directory. Please specify the correct path.")
        return
    
    # Set output directory in the same parent as input
    output_dir = input_dir.parent / 'fragmentation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = FragmentationAnalyzer()
    
    # Get all CSV files in directory
    all_files = list(input_dir.glob('*.csv'))
    logging.info(f"Found {len(all_files)} CSV files in {input_dir}")
    
    # Extract all episode files with more flexible patterns
    digital_files = []
    mobility_files = []
    home_files = []
    
    for file_path in all_files:
        filename = file_path.name.lower()
        if 'digital_episodes' in filename or 'digital_episode' in filename:
            digital_files.append(file_path)
        elif any(pattern in filename for pattern in ['mobility_episodes', 'moving_episodes', 'mobility_episode', 'moving_episode']):
            mobility_files.append(file_path)
        elif 'home_episodes' in filename or 'home_episode' in filename:
            home_files.append(file_path)
    
    logging.info(f"Found {len(digital_files)} digital episode files")
    logging.info(f"Found {len(mobility_files)} mobility episode files")
    logging.info(f"Found {len(home_files)} home episode files")
    
    if not digital_files:
        logging.error("No digital episode files found. Check the episodes directory.")
        return
    
    # Map files by user and date
    episode_files = {}
    
    # Helper function to extract date and user from filename
    def extract_key_from_file(file_path):
        try:
            parts = file_path.stem.split('_')
            date_part = next((p for p in parts if '-' in p or p.isdigit() and len(p) == 8), None)
            if not date_part:
                return None
                
            # Find user part - typically after episodes and before/after date
            # Handle different formats: 
            # 1. episode_type_episodes_date_user
            # 2. episode_type_episodes_user_date
            ep_idx = next((i for i, p in enumerate(parts) if 'episode' in p), -1)
            
            if date_part in parts and ep_idx >= 0:
                date_idx = parts.index(date_part)
                if date_idx > ep_idx + 1:  # Format: episode_type_episodes_user_date
                    user_idx = date_idx - 1
                else:  # Format: episode_type_episodes_date_user
                    user_idx = date_idx + 1
                    
                if user_idx < len(parts):
                    user_id = parts[user_idx]
                    return (date_part, user_id)
            
            return None
        except Exception as e:
            logging.warning(f"Error extracting key from {file_path.name}: {str(e)}")
            return None
    
    # Map files to (date, user) keys
    for file_list, file_type in [
        (digital_files, 'digital'),
        (mobility_files, 'mobility'),
        (home_files, 'home')
    ]:
        for file_path in file_list:
            key = extract_key_from_file(file_path)
            if key:
                date_part, user_id = key
                if key not in episode_files:
                    episode_files[key] = {}
                episode_files[key][file_type] = file_path
                logging.info(f"Mapped {file_type} file: {file_path.name} -> ({date_part}, {user_id})")
    
    logging.info(f"Successfully mapped {len(episode_files)} unique (date, user) pairs")
    
    # Now process files that have at least digital and mobility data
    all_results = []
    
    for key, files in tqdm(episode_files.items(), desc="Processing episodes"):
        if 'digital' in files and 'mobility' in files:
            digital_path = files['digital']
            mobility_path = files['mobility']
            home_path = files.get('home', None)
            
            results = analyzer.process_episode_summary(
                str(digital_path),
                str(mobility_path),
                home_file_path=str(home_path) if home_path else None,
                print_sample=(len(all_results) == 0)  # Print sample for first one only
            )
            
            if results is not None:
                all_results.append(results)
        else:
            logging.warning(f"Skipping incomplete data for {key} - missing required episode types")
    
    # Create summary DataFrame
    if all_results:
        combined_results = pd.DataFrame(all_results)
        
        # Generate and save analysis files
        analyzer.generate_analysis_plots(combined_results, output_dir)
        
        # Save cleaned results
        combined_results.to_csv(output_dir / 'fragmentation_summary.csv', index=False)
        logging.info(f"\nSaved fragmentation summary to {output_dir}/fragmentation_summary.csv")
        
        # Print statistics
        logging.info("\nProcessing Statistics:")
        for episode_type in ['digital', 'mobility', 'digital_home', 'overlap']:
            stats = analyzer.stats[episode_type]
            total = sum(stats.values())
            logging.info(f"\n{episode_type.capitalize()} Episodes:")
            for status, count in stats.items():
                percentage = (count/total*100) if total > 0 else 0
                logging.info(f"  {status}: {count} ({percentage:.1f}%)")
        
        # Print summary statistics
        logging.info("\nSummary Statistics:")
        for col in combined_results.select_dtypes(include=[np.number]).columns:
            stats = combined_results[col].describe()
            logging.info(f"\n{col}:")
            logging.info(f"  Mean: {stats['mean']:.3f}")
            logging.info(f"  Std: {stats['std']:.3f}")
            logging.info(f"  Min: {stats['min']:.3f}")
            logging.info(f"  Max: {stats['max']:.3f}")
    else:
        logging.warning("No valid results were generated")

if __name__ == "__main__":
    main()