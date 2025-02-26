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

class EpisodeFragmentationAnalyzer:
    def __init__(self, 
                 min_episodes: int = 2,
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
        self.entropy_based = entropy_based
        self.debug_mode = debug_mode
        self._setup_logging()
        
        # Initialize statistics tracking
        self.stats = {
            'digital': {'success': 0, 'insufficient_episodes': 0, 'invalid_duration': 0},
            'mobility': {'success': 0, 'insufficient_episodes': 0, 'invalid_duration': 0},
            'overlap': {'success': 0, 'insufficient_episodes': 0, 'invalid_duration': 0}
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

    def calculate_fragmentation_index(self, 
                                      episodes_df: pd.DataFrame, 
                                      episode_type: str,
                                      participant_id: str = "",
                                      date_str: str = "") -> Dict:
        """Calculate fragmentation index with improved validation and outlier detection"""
        if episodes_df.empty:
            return {
                'fragmentation_index': np.nan,
                'episode_count': 0,
                'total_duration': 0,
                'status': 'empty_dataframe'
            }
        
        # Create a copy to avoid modifying the original
        episodes_df = episodes_df.copy()
        
        # Rename columns if needed and ensure we have a duration column
        if 'duration_minutes' in episodes_df.columns:
            episodes_df = episodes_df.rename(columns={'duration_minutes': 'duration'})
        
        # Process duration column depending on its type
        if 'duration' in episodes_df.columns:
            episodes_df['duration'] = episodes_df['duration'].apply(self.parse_duration_string)
        # If no duration column exists, calculate from start/end times
        elif 'start_time' in episodes_df.columns and 'end_time' in episodes_df.columns:
            # Convert to datetime if needed
            for col in ['start_time', 'end_time']:
                if not pd.api.types.is_datetime64_any_dtype(episodes_df[col]):
                    episodes_df[col] = pd.to_datetime(episodes_df[col], errors='coerce')
            
            # Calculate duration in minutes
            episodes_df['duration'] = (episodes_df['end_time'] - episodes_df['start_time']).dt.total_seconds() / 60
        # If we still have no duration, create a default
        if 'duration' not in episodes_df.columns:
            episodes_df['duration'] = 1.0
            self.logger.warning(f"No duration information found for {episode_type} episodes, using default values")
        
        # Check minimum episode count
        if len(episodes_df) < self.min_episodes:
            self.stats[episode_type]['insufficient_episodes'] += 1
            return {
                'fragmentation_index': np.nan,
                'episode_count': len(episodes_df),
                'total_duration': episodes_df['duration'].sum() if not episodes_df.empty else 0,
                'status': 'insufficient_episodes'
            }
            
        # Validate durations - remove zeros and very large values
        valid_episodes = episodes_df[
            (episodes_df['duration'] > 0) &
            (episodes_df['duration'] <= self.max_episode_duration)
        ].copy()
        
        if len(valid_episodes) < self.min_episodes:
            self.stats[episode_type]['invalid_duration'] += 1
            return {
                'fragmentation_index': np.nan,
                'episode_count': len(valid_episodes),
                'total_duration': valid_episodes['duration'].sum() if not valid_episodes.empty else 0,
                'status': 'invalid_duration'
            }
            
        # Detect and handle outliers
        mean_duration = valid_episodes['duration'].mean()
        std_duration = valid_episodes['duration'].std()
        if std_duration > 0:  # Only apply outlier detection if there's variation
            outlier_mask = np.abs(valid_episodes['duration'] - mean_duration) <= (self.outlier_threshold * std_duration)
            valid_episodes = valid_episodes[outlier_mask]
        
        if len(valid_episodes) < self.min_episodes:
            self.stats[episode_type]['insufficient_episodes'] += 1
            return {
                'fragmentation_index': np.nan,
                'episode_count': len(valid_episodes),
                'total_duration': valid_episodes['duration'].sum() if not valid_episodes.empty else 0,
                'status': 'insufficient_episodes_after_outlier_removal'
            }
            
        # Calculate normalized durations
        total_duration = valid_episodes['duration'].sum()
        normalized_durations = valid_episodes['duration'] / total_duration
        
        # Calculate fragmentation index
        S = len(valid_episodes)
        
        if self.entropy_based:
            # Entropy-based calculation
            index = 0.0
            if S > 1:
                # Avoid log(0) by handling zero durations
                nonzero_normalized = normalized_durations[normalized_durations > 0]
                if len(nonzero_normalized) > 1:
                    entropy = -np.sum(nonzero_normalized * np.log(nonzero_normalized))
                    index = entropy / np.log(S)
        else:
            # HHI-based calculation (Herfindahl-Hirschman Index)
            index = 0.0
            if S > 1:
                hhi = np.sum(normalized_durations**2)
                index = (1 - hhi) / (1 - 1/S)
        
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

    def process_daily_episodes(self, participant_dir: Path, date_str: str, participant_id: str) -> Optional[Dict]:
        """Process all episode types for a single day"""
        try:
            # Define file paths for each episode type - check multiple patterns
            digital_file_patterns = [
                participant_dir / f"{date_str}_digital_episodes.csv",  # Pattern 1
                participant_dir / f"digital_episodes_{date_str}.csv",  # Pattern 2
                participant_dir / f"digital_{date_str}_episodes.csv",  # Pattern 3
            ]
            
            mobility_file_patterns = [
                participant_dir / f"{date_str}_mobility_episodes.csv",  # Pattern 1
                participant_dir / f"mobility_episodes_{date_str}.csv",  # Pattern 2
                participant_dir / f"{date_str}_moving_episodes.csv",    # Pattern 3
                participant_dir / f"moving_episodes_{date_str}.csv",    # Pattern 4
            ]
            
            overlap_file_patterns = [
                participant_dir / f"{date_str}_overlap_episodes.csv",   # Pattern 1
                participant_dir / f"overlap_episodes_{date_str}.csv",   # Pattern 2
            ]
            
            # Find existing files
            digital_file = next((f for f in digital_file_patterns if f.exists()), None)
            mobility_file = next((f for f in mobility_file_patterns if f.exists()), None)
            overlap_file = next((f for f in overlap_file_patterns if f.exists()), None)
            
            # Check if required files exist
            if not digital_file or not mobility_file:
                self.logger.warning(f"Missing required episode files for {participant_id} on {date_str}")
                return None
            
            # Load episode files
            digital_df = pd.read_csv(digital_file)
            mobility_df = pd.read_csv(mobility_file)
            overlap_df = pd.read_csv(overlap_file) if overlap_file else pd.DataFrame()
            
            # Check if dataframes are empty
            if digital_df.empty or mobility_df.empty:
                self.logger.warning(f"Empty episode data for {participant_id} on {date_str}")
                return None
            
            # Convert time columns to datetime
            for df in [digital_df, mobility_df, overlap_df]:
                if not df.empty:
                    for col in ['start_time', 'end_time']:
                        if col in df.columns:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Calculate fragmentation metrics for each episode type
            digital_metrics = self.calculate_fragmentation_index(
                digital_df, 'digital', participant_id, date_str
            )
            mobility_metrics = self.calculate_fragmentation_index(
                mobility_df, 'mobility', participant_id, date_str
            )
            overlap_metrics = self.calculate_fragmentation_index(
                overlap_df, 'overlap', participant_id, date_str
            ) if not overlap_df.empty else {
                'fragmentation_index': np.nan,
                'episode_count': 0,
                'total_duration': 0,
                'status': 'no_overlap_episodes'
            }
            
            result = {
                'participant_id': participant_id,
                'date': date_str,
                'digital_fragmentation_index': digital_metrics['fragmentation_index'],
                'mobility_fragmentation_index': mobility_metrics['fragmentation_index'],
                'overlap_fragmentation_index': overlap_metrics['fragmentation_index'],
                'digital_episode_count': digital_metrics['episode_count'],
                'mobility_episode_count': mobility_metrics['episode_count'],
                'overlap_episode_count': overlap_metrics['episode_count'],
                'digital_total_duration': digital_metrics['total_duration'],
                'mobility_total_duration': mobility_metrics['total_duration'],
                'overlap_total_duration': overlap_metrics['total_duration'],
                'digital_status': digital_metrics['status'],
                'mobility_status': mobility_metrics['status'],
                'overlap_status': overlap_metrics['status']
            }
            
            # Add additional metrics if available
            for metrics_dict, prefix in zip(
                [digital_metrics, mobility_metrics, overlap_metrics],
                ['digital_', 'mobility_', 'overlap_']
            ):
                for key in ['mean_duration', 'std_duration', 'cv']:
                    if key in metrics_dict:
                        result[f'{prefix}{key}'] = metrics_dict[key]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing episodes for {participant_id} on {date_str}: {str(e)}")
            return None

    def generate_analysis_plots(self, df: pd.DataFrame, output_dir: Path):
        """Generate visualization plots for fragmentation analysis"""
        # Set seaborn style for better plots
        sns.set(style="whitegrid")
        
        # Ensure output directory exists
        plots_dir = output_dir / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Create correlation matrix
        corr_cols = [col for col in df.columns if col.endswith('_fragmentation_index') 
                    or col.endswith('_episode_count') or col.endswith('_total_duration')]
        
        corr_df = df[corr_cols].copy()
        correlation = corr_df.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Matrix of Fragmentation Metrics')
        plt.tight_layout()
        plt.savefig(plots_dir / 'correlation_matrix.png')
        plt.close()
        
        # Plot histograms for each fragmentation index
        metrics = ['digital_fragmentation_index', 'mobility_fragmentation_index', 'overlap_fragmentation_index']
        
        plt.figure(figsize=(12, 6))
        for i, metric in enumerate(metrics):
            if df[metric].notna().sum() > 0:  # Only plot if we have data
                plt.subplot(1, 3, i+1)
                sns.histplot(df[metric].dropna(), kde=True)
                plt.title(f'{metric.split("_")[0].capitalize()} Fragmentation')
                plt.xlabel('Fragmentation Index')
                plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'fragmentation_distributions.png')
        plt.close()
        
        # Create a participant summary plot
        participant_means = df.groupby('participant_id')[metrics].mean().reset_index()
        
        plt.figure(figsize=(12, 6))
        participant_means_melted = pd.melt(
            participant_means, 
            id_vars=['participant_id'],
            value_vars=metrics,
            var_name='Metric', 
            value_name='Fragmentation Index'
        )
        
        # Replace metric names for cleaner display
        participant_means_melted['Metric'] = participant_means_melted['Metric'].apply(
            lambda x: x.split('_')[0].capitalize()
        )
        
        sns.barplot(x='participant_id', y='Fragmentation Index', hue='Metric', data=participant_means_melted)
        plt.title('Average Fragmentation Index by Participant')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / 'participant_comparison.png')
        plt.close()
        
        # Create time series plot for selected participants
        # Get top 5 participants with most data points
        top_participants = df['participant_id'].value_counts().head(5).index.tolist()
        
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            metric_name = metric.split('_')[0].capitalize()
            
            for participant in top_participants:
                participant_data = df[df['participant_id'] == participant].copy()
                participant_data['date'] = pd.to_datetime(participant_data['date'])
                participant_data = participant_data.sort_values('date')
                
                if len(participant_data) > 1:  # Only plot if we have enough data points
                    plt.plot(participant_data['date'], participant_data[metric], 
                            label=f'Participant {participant}', marker='o')
            
            plt.title(f'{metric_name} Fragmentation Over Time')
            plt.xlabel('Date')
            plt.ylabel('Fragmentation Index')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plots_dir / f'{metric_name.lower()}_time_series.png')
            plt.close()

def process_episodes_data(
    episode_dir: Path,
    output_dir: Path,
    min_episodes: int = 2,
    entropy_based: bool = True,
    debug_mode: bool = False
):
    """Process all participants' data to calculate fragmentation metrics"""
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = EpisodeFragmentationAnalyzer(
        min_episodes=min_episodes,
        entropy_based=entropy_based,
        debug_mode=debug_mode
    )
    
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
            
        logging.info(f"Processing {len(episode_files)} days for participant {participant_id}")
        
        # Process each file
        for file in episode_files:
            # Extract date from filename
            try:
                # First try pattern "*_digital_episodes.csv"
                if '_digital_episodes.csv' in file.name:
                    date_str = file.name.split('_digital_episodes.csv')[0]
                # Then try pattern "digital_episodes_*.csv" 
                elif 'digital_episodes_' in file.name:
                    date_str = file.name.split('digital_episodes_')[1].split('.csv')[0]
                else:
                    # Try to extract any date-like part (YYYY-MM-DD)
                    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', file.name)
                    if date_match:
                        date_str = date_match.group(1)
                    else:
                        logging.warning(f"Could not extract date from filename: {file.name}")
                        continue
                
                # Skip files with invalid dates (like empty strings)
                if not date_str or not date_str[0].isdigit():
                    logging.warning(f"Invalid date format in filename: {file.name}")
                    continue
                
                # Process episodes for this day
                daily_metrics = analyzer.process_daily_episodes(participant_dir, date_str, participant_id)
                
                if daily_metrics:
                    all_results.append(daily_metrics)
            except Exception as e:
                logging.error(f"Error processing file {file}: {str(e)}")
    
    # Create summary DataFrame
    if all_results:
        combined_results = pd.DataFrame(all_results)
        
        # Save full results
        combined_results.to_csv(output_dir / 'fragmentation_all_metrics.csv', index=False)
        
        # Generate and save analysis plots
        analyzer.generate_analysis_plots(combined_results, output_dir)
        
        # Generate summary stats by participant
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
        participant_summary.to_csv(output_dir / 'participant_fragmentation_summary.csv')
        
        # Print processing statistics
        logging.info("\nProcessing Statistics:")
        for episode_type in ['digital', 'mobility', 'overlap']:
            stats = analyzer.stats[episode_type]
            total = sum(stats.values())
            logging.info(f"\n{episode_type.capitalize()} Episodes:")
            for status, count in stats.items():
                percentage = (count/total*100) if total > 0 else 0
                logging.info(f"  {status}: {count} ({percentage:.1f}%)")
        
        # Generate analysis report
        metrics_by_type = {}
        for episode_type in ['digital', 'mobility', 'overlap']:
            col = f'{episode_type}_fragmentation_index'
            if col in combined_results.columns:
                metrics_by_type[episode_type] = {
                    'mean': combined_results[col].mean(),
                    'median': combined_results[col].median(),
                    'std': combined_results[col].std(),
                    'min': combined_results[col].min(),
                    'max': combined_results[col].max(),
                    'count': combined_results[col].count(),
                    'missing': combined_results[col].isna().sum(),
                }
        
        # Save metrics summary as JSON - convert NumPy types to Python native types
        metrics_by_type_serializable = {}
        for episode_type, metrics in metrics_by_type.items():
            metrics_by_type_serializable[episode_type] = {
                key: float(value) if isinstance(value, (np.float64, np.float32, np.int64, np.int32)) else value
                for key, value in metrics.items()
            }
            
        with open(output_dir / 'fragmentation_metrics_summary.json', 'w') as f:
            json.dump(metrics_by_type_serializable, f, indent=4)
            
        return combined_results
    else:
        logging.warning("No valid results were generated")
        return None

def main():
    # Configure paths - adjust these to match your directory structure
    episode_dir = Path('/Volumes/Extreme SSD/SURREAL-DataBackup/HUJI_data-main/processed/episodes')
    output_dir = Path('/Volumes/Extreme SSD/SURREAL-DataBackup/HUJI_data-main/processed/fragmentation')
    
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
        min_episodes=2,  # Minimum episode count for calculation
        entropy_based=True,  # Use entropy-based fragmentation
        debug_mode=False     # Disable verbose debugging
    )
    
    # Log summary statistics and file locations
    if results_df is not None:
        logging.info("\n" + "="*50)
        logging.info("FRAGMENTATION ANALYSIS SUMMARY")
        logging.info("="*50)
        
        # Summary statistics
        logging.info("\nSummary Statistics:")
        for metric in ['digital_fragmentation_index', 'mobility_fragmentation_index', 'overlap_fragmentation_index']:
            valid_values = results_df[metric].dropna()
            logging.info(f"\n{metric.split('_')[0].capitalize()} Fragmentation:")
            logging.info(f"  Valid measurements: {len(valid_values)} of {len(results_df)} ({len(valid_values)/len(results_df)*100:.1f}%)")
            if not valid_values.empty:
                logging.info(f"  Mean: {valid_values.mean():.4f}")
                logging.info(f"  Median: {valid_values.median():.4f}")
                logging.info(f"  Std Dev: {valid_values.std():.4f}")
                logging.info(f"  Range: [{valid_values.min():.4f}, {valid_values.max():.4f}]")
        
        # Participant coverage
        total_participants = results_df['participant_id'].nunique()
        logging.info(f"\nParticipant Coverage:")
        logging.info(f"  Total participants processed: {total_participants}")
        avg_days = len(results_df) / total_participants if total_participants > 0 else 0
        logging.info(f"  Average days per participant: {avg_days:.1f}")
        
        # File locations
        logging.info("\nOutput Files:")
        logging.info(f"  Full metrics data: {output_dir / 'fragmentation_all_metrics.csv'}")
        logging.info(f"  Participant summary: {output_dir / 'participant_fragmentation_summary.csv'}")
        logging.info(f"  Metrics summary JSON: {output_dir / 'fragmentation_metrics_summary.json'}")
        logging.info(f"  Visualization plots: {output_dir / 'plots/'}")
        
        logging.info("\nAnalysis complete!")
    else:
        logging.error("Analysis failed - no results were generated.")

if __name__ == "__main__":
    main()