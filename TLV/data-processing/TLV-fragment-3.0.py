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
            'moving': {'success': 0, 'insufficient_episodes': 0, 'invalid_duration': 0}
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
            episode_type: Type of episodes ('digital' or 'moving')
            
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
        
        # Calculate fragmentation metrics using normalized entropy
        S = len(valid_episodes)
        
        if S == 1:
            frag_index = 0.0  # Single episode means no fragmentation
        else:
            # Calculate Shannon entropy
            entropy = -np.sum(normalized_durations * np.log(normalized_durations))
            # Normalize by maximum possible entropy (ln(S))
            frag_index = entropy / np.log(S)
        
        self.stats[episode_type]['success'] += 1
        
        return {
            'fragmentation_index': frag_index,
            'episode_count': S,
            'total_duration': total_duration,
            'mean_duration': valid_episodes['duration'].mean(),
            'std_duration': valid_episodes['duration'].std(),
            'cv': valid_episodes['duration'].std() / valid_episodes['duration'].mean() if valid_episodes['duration'].mean() > 0 else np.nan,
            'status': 'success'
        }

    def calculate_digital_frag_during_mobility(self, 
                                             digital_df: pd.DataFrame, 
                                             moving_df: pd.DataFrame) -> Dict:
        """
        Calculate fragmentation of digital use during mobility periods
        
        Args:
            digital_df: DataFrame with digital episodes
            moving_df: DataFrame with mobility episodes
            
        Returns:
            Dictionary with fragmentation metrics for digital use during mobility
        """
        if digital_df.empty or moving_df.empty:
            return {
                'fragmentation_index': np.nan,
                'episode_count': 0,
                'total_duration': 0,
                'status': 'no_episodes'
            }
            
        # Convert times to datetime if needed
        for df in [digital_df, moving_df]:
            for col in ['start_time', 'end_time']:
                if df[col].dtype != 'datetime64[ns]':
                    df[col] = pd.to_datetime(df[col])
        
        # Find overlapping episodes
        overlapping_episodes = []
        for _, digital in digital_df.iterrows():
            for _, mobility in moving_df.iterrows():
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

    def process_episode_summary(self, 
                              digital_file_path: str, 
                              moving_file_path: str,
                              print_sample: bool = False) -> Optional[Dict]:
        """Process episode data and calculate all fragmentation metrics"""
        try:
            digital_df = pd.read_csv(digital_file_path)
            moving_df = pd.read_csv(moving_file_path)
            
            for df in [digital_df, moving_df]:
                df['start_time'] = pd.to_datetime(df['start_time'])
                df['end_time'] = pd.to_datetime(df['end_time'])
            
            if print_sample:
                self.logger.info(f"\nSample data from {os.path.basename(digital_file_path)}:")
                self.logger.info(digital_df.head())
                self.logger.info(f"\nSample data from {os.path.basename(moving_file_path)}:")
                self.logger.info(moving_df.head())
            
            participant_id, date = self._extract_info_from_filename(os.path.basename(digital_file_path))
            
            # Calculate various fragmentation metrics
            digital_metrics = self.calculate_fragmentation_index(digital_df, 'digital')
            moving_metrics = self.calculate_fragmentation_index(moving_df, 'moving')
            overlap_metrics = self.calculate_digital_frag_during_mobility(digital_df, moving_df)
            
            result = {
                'participant_id': participant_id,
                'date': date,
                'digital_fragmentation_index': digital_metrics['fragmentation_index'],
                'moving_fragmentation_index': moving_metrics['fragmentation_index'],
                'digital_frag_during_mobility': overlap_metrics['fragmentation_index'],
                'digital_episode_count': digital_metrics['episode_count'],
                'moving_episode_count': moving_metrics['episode_count'],
                'digital_total_duration': digital_metrics['total_duration'],
                'moving_total_duration': moving_metrics['total_duration'],
                'digital_status': digital_metrics['status'],
                'moving_status': moving_metrics['status'],
                'overlap_status': overlap_metrics['status']
            }
            
            # Add additional metrics if available
            for metrics_type in [digital_metrics, moving_metrics]:
                if 'cv' in metrics_type:
                    prefix = 'digital_' if metrics_type == digital_metrics else 'moving_'
                    result[f'{prefix}cv'] = metrics_type['cv']
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing files {digital_file_path} and {moving_file_path}: {str(e)}")
            return None

    def _extract_info_from_filename(self, filename: str) -> Tuple[str, datetime.date]:
        """Extract participant ID and date from filename"""
        # Original format: "digital_episodes_2022-06-13_29.csv"
        # Parts will be: ['digital', 'episodes', '2022-06-13', '29.csv']
        parts = filename.split('_')
        
        # Get date from third part
        date_str = parts[2]
        # Get participant ID from fourth part (remove .csv)
        participant_id = parts[3].split('.')[0]
        
        date = datetime.strptime(date_str, "%Y-%m-%d").date()
        return participant_id, date

    def generate_analysis_plots(self, df: pd.DataFrame, output_dir: Path):
        """Generate visualization plots for fragmentation analysis"""
        metrics = ['digital_fragmentation_index', 'moving_fragmentation_index', 
                  'digital_frag_during_mobility']
        
        for metric in metrics:
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

def main():
    # Configure paths
    input_dir = Path('/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/episodes')
    output_dir = Path('/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/fragmentation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = FragmentationAnalyzer()
    
    # Process all episode files
    digital_files = sorted([f for f in os.listdir(input_dir) 
                          if f.startswith('digital_episodes_') and f.endswith('.csv')])
    moving_files = sorted([f for f in os.listdir(input_dir) 
                         if f.startswith('moving_episodes_') and f.endswith('.csv')])
    
    if len(digital_files) != len(moving_files):
        logging.warning("Mismatch in number of digital and moving episode files")
    
    all_results = []
    for i, (digital_file, moving_file) in enumerate(tqdm(zip(digital_files, moving_files), 
                                                       desc="Processing episodes")):
        digital_path = input_dir / digital_file
        moving_path = input_dir / moving_file
        
        results = analyzer.process_episode_summary(digital_path, moving_path, 
                                                 print_sample=(i==0))
        if results is not None:
            all_results.append(results)
    
    # Create summary DataFrame
    if all_results:
        combined_results = pd.DataFrame(all_results)
        
        # Save results
        output_file = output_dir / 'fragmentation_summary.csv'
        combined_results.to_csv(output_file, index=False)
        logging.info(f"\nSaved fragmentation summary to {output_file}")
        
        # Generate analysis plots
        analyzer.generate_analysis_plots(combined_results, output_dir)
        
        # Print statistics
        logging.info("\nProcessing Statistics:")
        for episode_type in ['digital', 'moving']:
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