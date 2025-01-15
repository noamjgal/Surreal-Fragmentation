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
            
            # Merge EMA data using Participant_ID and timestamp matching
            # Convert timestamps to the same format before merging
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
            
            # Calculate z-scores for key metrics with better error handling
            metric_cols = [
                'digital_fragmentation_index',
                'moving_fragmentation_index',
                'digital_frag_during_mobility',
                'digital_total_duration',
                'moving_total_duration',
                'STAI6_score'
            ]
            
            # Log metrics availability
            self.logger.info("\nMetrics availability check:")
            for col in metric_cols:
                if col in merged_df.columns:
                    non_null = merged_df[col].notna().sum()
                    total = len(merged_df)
                    self.logger.info(f"{col}: {non_null}/{total} non-null values ({(non_null/total)*100:.1f}%)")
                else:
                    self.logger.warning(f"Missing metric column: {col}")

            # Calculate z-scores with proper handling of NaN values
            for col in metric_cols:
                if col in merged_df.columns:
                    valid_data = merged_df[col].dropna()
                    if len(valid_data) > 0:
                        mean = valid_data.mean()
                        std = valid_data.std()
                        if std > 0:  # Avoid division by zero
                            merged_df[f'{col}_zscore'] = (merged_df[col] - mean) / std
                        else:
                            self.logger.warning(f"Zero standard deviation for {col}, skipping z-score calculation")
                    else:
                        self.logger.warning(f"No valid data for {col}, skipping z-score calculation")

            # Verify z-score columns
            zscore_cols = [col for col in merged_df.columns if col.endswith('_zscore')]
            self.logger.info("\nCreated z-score columns:")
            for col in zscore_cols:
                non_null = merged_df[col].notna().sum()
                total = len(merged_df)
                self.logger.info(f"{col}: {non_null}/{total} non-null values ({(non_null/total)*100:.1f}%)")

            # Save the combined dataset
            output_path = self.output_dir / 'combined_metrics.csv'
            merged_df.to_csv(output_path, index=False)
            
            # Generate summary statistics
            self._generate_summary_statistics(merged_df)
            
            self.logger.info(f"Successfully combined metrics and saved to {output_path}")
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

