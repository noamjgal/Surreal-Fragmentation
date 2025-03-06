import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import json

class MetricsProcessor:
    """
    Consolidated metrics processor that combines EMA scores, fragmentation metrics,
    and demographic data while respecting early morning handling from preprocessing.
    """
    def __init__(self, base_dir: str):
        """
        Initialize processor with base directory containing all input files.
        
        Args:
            base_dir: Base directory containing all subdirectories with data files
        """
        self.base_dir = Path(base_dir)
        self.output_dir = self.base_dir / 'metrics'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data quality thresholds
        self.quality_thresholds = {
            'min_digital_episodes': 2,
            'min_mobility_episodes': 2,
            'min_digital_duration': 30,  # minutes
            'min_coverage_hours': 6,
            'min_overlap_episodes': 1
        }
        
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging with both file and console output"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'metrics_processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _validate_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out rows with insufficient data quality based on episode counts and durations.
        
        Args:
            df: DataFrame with combined metrics
            
        Returns:
            DataFrame with only valid rows meeting quality thresholds
        """
        # Create mask for valid data
        valid_data = (
            (df['digital_episode_count'] >= self.quality_thresholds['min_digital_episodes']) &
            (df['mobility_episode_count'] >= self.quality_thresholds['min_mobility_episodes']) &
            (df['digital_total_duration'] >= self.quality_thresholds['min_digital_duration']) &
            (df['coverage_hours'] >= self.quality_thresholds['min_coverage_hours']) &
            (df['overlap_num_episodes'] >= self.quality_thresholds['min_overlap_episodes'])  # new condition
        )
        
        # Log excluded data
        excluded = ~valid_data
        if excluded.any():
            self.logger.warning(f"\nExcluded {excluded.sum()} rows due to insufficient data quality:")
            self.logger.warning(f"Low digital episodes: {(df['digital_episode_count'] < self.quality_thresholds['min_digital_episodes']).sum()}")
            self.logger.warning(f"Low mobility episodes: {(df['mobility_episode_count'] < self.quality_thresholds['min_mobility_episodes']).sum()}")
            self.logger.warning(f"Low digital duration: {(df['digital_total_duration'] < self.quality_thresholds['min_digital_duration']).sum()}")
            self.logger.warning(f"Low coverage hours: {(df['coverage_hours'] < self.quality_thresholds['min_coverage_hours']).sum()}")
            self.logger.warning(f"Low overlap episodes: {(df['overlap_num_episodes'] < self.quality_thresholds['min_overlap_episodes']).sum()}")  # new log
            
            # Save excluded data for analysis
            excluded_df = df[excluded].copy()
            excluded_df.to_csv(self.output_dir / 'excluded_data.csv', index=False)
            
        return df[valid_data].copy()

    def process_ema_scores(self, ema_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process EMA responses and calculate STAI-6 index score, preserving individual responses
        """
        df = ema_df.copy()
        
        # Reverse scoring for positive items (5-point scale)
        reverse_items = ['PEACE', 'RELAXATION', 'SATISFACTION']
        for item in reverse_items:
            df[f'{item}_R'] = 6 - df[item]
        
        # Calculate STAI-6 score (1-5 range)
        anxiety_items = ['TENSE', 'RELAXATION_R', 'WORRY', 
                        'PEACE_R', 'IRRITATION', 'SATISFACTION_R']
        
        # Add validation checks and logging
        for item in anxiety_items:
            if df[item].max() > 5 or df[item].min() < 1:
                self.logger.warning(f"Invalid values found in {item}: range [{df[item].min()}, {df[item].max()}]")
        
        # Calculate mean score (keeping it in 1-5 range)
        df['STAI6_score'] = df[anxiety_items].mean(axis=1)
        
        # Log some sample calculations
        sample_rows = df.head(3)
        for idx, row in sample_rows.iterrows():
            self.logger.info(f"\nSample calculation for row {idx}:")
            for item in anxiety_items:
                self.logger.info(f"{item}: {row[item]}")
            self.logger.info(f"Final score (1-5 range): {row['STAI6_score']:.2f}")
        
        # Keep necessary columns
        cols_to_keep = ['Participant_ID', 'StartDate', 'EndDate'] + \
                    ['Gender', 'School', 'Class'] + \
                    anxiety_items + \
                    ['STAI6_score', 'HAPPY']
        
        return df[cols_to_keep]

    def combine_metrics(self) -> pd.DataFrame:
        """
        Combine all metrics using data_quality_summary as the backbone
        """
        try:
            # Load all required datasets
            quality_df = pd.read_csv(self.base_dir / 'preprocessed_summaries/data_quality_summary.csv')
            frag_df = pd.read_csv(self.base_dir / 'fragmentation/fragmentation_summary.csv')
            ema_df = pd.read_csv(self.base_dir / 'Survey/csv/End_of_the_day_questionnaire.csv')
            episode_df = pd.read_csv(self.base_dir / 'episodes/episode_summary.csv')
            
            self.logger.info("Loaded all input datasets")
            
            # Process EMA scores first
            processed_ema = self.process_ema_scores(ema_df)
            
            # Convert timestamps in EMA data and quality_df
            processed_ema['StartDate'] = pd.to_datetime(processed_ema['StartDate'])
            processed_ema['EndDate'] = pd.to_datetime(processed_ema['EndDate'])
            
            # Ensure timestamp formats match before merging
            if 'ema_timestamp' in quality_df.columns:
                quality_df['ema_timestamp'] = pd.to_datetime(quality_df['ema_timestamp'])
            
            # Merge quality summary with fragmentation data
            merged_df = pd.merge(
                quality_df,
                frag_df,
                left_on=['user', 'associated_data_date'],
                right_on=['participant_id', 'date'],
                how='inner'
            )
            
            # Handle episode summary metrics
            episode_df['unique_id'] = (
                episode_df['user'].astype(str) + '_' + 
                episode_df['date'].astype(str) + '_' + 
                episode_df['episode_type'].astype(str)
            )
            
            episode_df = episode_df.drop_duplicates(subset=['unique_id', 'num_episodes', 'total_duration_minutes', 'mean_duration_minutes'])
            
            # Create separate pivots for each metric
            episode_metrics = ['num_episodes', 'total_duration_minutes', 'mean_duration_minutes']
            pivoted_dfs = []
            
            for metric in episode_metrics:
                try:
                    pivot_df = episode_df.pivot(
                        index=['user', 'date'],
                        columns='episode_type',
                        values=metric
                    )
                    pivot_df.columns = [f'{col}_{metric}'.lower() for col in pivot_df.columns]
                    pivoted_dfs.append(pivot_df)
                except Exception as e:
                    self.logger.warning(f"Error pivoting {metric}: {str(e)}")
                    continue
            
            # Combine all pivoted metrics
            episode_pivot = pd.concat(pivoted_dfs, axis=1).reset_index()
            
            # Merge episode data
            merged_df = pd.merge(
                merged_df,
                episode_pivot,
                left_on=['user', 'associated_data_date'],
                right_on=['user', 'date'],
                how='left',
                suffixes=('', '_episode')
            )
            
            # Merge EMA data
            merged_df = pd.merge(
                merged_df,
                processed_ema,
                left_on=['user', 'ema_timestamp'],
                right_on=['Participant_ID', 'StartDate'],
                how='left'
            )
            
            # Clean up redundant columns
            cols_to_drop = [
                'date_episode', 'participant_id', 'Participant_ID',
                'StartDate', 'EndDate'
            ]
            merged_df = merged_df.drop(columns=[
                col for col in cols_to_drop if col in merged_df.columns
            ])
            
            # Add weekday and time-based features
            merged_df['weekday'] = pd.to_datetime(merged_df['associated_data_date']).dt.dayofweek
            merged_df['is_weekend'] = merged_df['weekday'].isin([5, 6]).astype(int)
            
            # Apply data quality validation
            merged_df = self._validate_data_quality(merged_df)
            
            # Calculate z-scores for key metrics
            metric_cols = [
                'digital_fragmentation_index',
                'mobility_fragmentation_index',
                'overlap_fragmentation_index',
                'digital_total_duration',
                'mobility_total_duration',
                'STAI6_score'
            ]
            
            # Calculate z-scores only on validated data
            for col in metric_cols:
                if col in merged_df.columns:
                    valid_data = merged_df[col].dropna()
                    if len(valid_data) > 0:
                        mean = valid_data.mean()
                        std = valid_data.std()
                        if std > 0:
                            merged_df[f'{col}_zscore'] = (merged_df[col] - mean) / std
                        else:
                            self.logger.warning(f"Zero standard deviation for {col}, skipping z-score calculation")
            
            # Save the filtered dataset
            output_path = self.output_dir / 'combined_metrics.csv'
            merged_df.to_csv(output_path, index=False)
            
            # Generate summary statistics
            self._generate_summary_statistics(merged_df)
            
            self.logger.info(f"Successfully combined and filtered metrics. Saved to {output_path}")
            self.logger.info(f"Final dataset contains {len(merged_df)} rows")
            
            return merged_df
            
        except Exception as e:
            self.logger.error(f"Error combining metrics: {str(e)}", exc_info=True)
            raise

    def _generate_summary_statistics(self, df: pd.DataFrame):
        """Generate summary statistics for the combined metrics"""
        stats = {}
        
        # Basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats['numeric_summary'] = df[numeric_cols].describe()
        
        # Demographic distributions
        for col in ['Gender', 'School', 'Class']:
            if col in df.columns:
                stats[f'{col}_distribution'] = df[col].value_counts()
        
        # Early morning response statistics
        stats['early_morning_counts'] = df['is_early_morning'].value_counts()
        
        # Coverage statistics
        stats['coverage_quality'] = df['data_quality'].value_counts()
        stats['coverage_hours'] = df['coverage_hours'].describe()
        
        # Save statistics to Excel
        with pd.ExcelWriter(self.output_dir / 'metrics_summary.xlsx') as writer:
            for name, stat_df in stats.items():
                stat_df.to_excel(writer, sheet_name=name[:31])
        
        # Log key statistics
        self.logger.info("\nKey Statistics:")
        self.logger.info(f"Total participants: {df['user'].nunique()}")
        self.logger.info(f"Total days: {len(df)}")
        self.logger.info(f"Days with good quality data: {(df['data_quality'] == 'good').sum()}")
        self.logger.info(f"Early morning responses: {df['is_early_morning'].sum()}")

   
def main():
    # Define base directory
    base_dir = Path('/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon')  # Update this path
    
    # Initialize processor
    processor = MetricsProcessor(base_dir)
    
    # Combine all metrics
    try:
        combined_data = processor.combine_metrics()
        print("Successfully combined all metrics")
    except Exception as e:
        print(f"Error processing metrics: {str(e)}")

if __name__ == "__main__":
    main()