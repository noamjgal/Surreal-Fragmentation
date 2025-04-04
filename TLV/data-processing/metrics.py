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
        
        # Define transport mode mappings
        self.active_transport_types = ['AT', 'Walking']
        self.mechanized_transport_types = ['PT']
        
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
            (df['overlap_episode_count'] >= self.quality_thresholds['min_overlap_episodes'])
        )
        
        # Log excluded data
        excluded = ~valid_data
        if excluded.any():
            self.logger.warning(f"\nExcluded {excluded.sum()} rows due to insufficient data quality:")
            self.logger.warning(f"Low digital episodes: {(df['digital_episode_count'] < self.quality_thresholds['min_digital_episodes']).sum()}")
            self.logger.warning(f"Low mobility episodes: {(df['mobility_episode_count'] < self.quality_thresholds['min_mobility_episodes']).sum()}")
            self.logger.warning(f"Low digital duration: {(df['digital_total_duration'] < self.quality_thresholds['min_digital_duration']).sum()}")
            self.logger.warning(f"Low coverage hours: {(df['coverage_hours'] < self.quality_thresholds['min_coverage_hours']).sum()}")
            self.logger.warning(f"Low overlap episodes: {(df['overlap_episode_count'] < self.quality_thresholds['min_overlap_episodes']).sum()}")
            
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

    def calculate_location_durations(self, episode_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate total durations for home and other location types based on episode data.
        
        Args:
            episode_df: DataFrame containing episode information
            
        Returns:
            DataFrame with additional location duration columns
        """
        # Create a copy to avoid modifying the original
        df = episode_df.copy()
        
        try:
            # Check if episode_type column exists
            if 'episode_type' not in df.columns:
                self.logger.warning("episode_type column not found in episode data")
                return pd.DataFrame(columns=['user', 'date', 'home_duration', 'out_of_home_duration'])
                
            # Create a user-date index for grouping
            df['user_date'] = df['user'].astype(str) + '_' + df['date'].astype(str)
            
            # Calculate location durations by type
            location_durations = {}
            
            # Process each user-date combination
            for user_date in df['user_date'].unique():
                user_date_df = df[df['user_date'] == user_date]
                
                # Extract home durations
                home_df = user_date_df[user_date_df['episode_type'] == 'home']
                home_duration = home_df['total_duration_minutes'].sum() if not home_df.empty else 0
                
                # Calculate total observed time from first and last timestamps
                if not user_date_df.empty:
                    first_timestamp = pd.to_datetime(user_date_df['first_timestamp'].iloc[0])
                    last_timestamp = pd.to_datetime(user_date_df['last_timestamp'].iloc[0])
                    total_observed_time = (last_timestamp - first_timestamp).total_seconds() / 60
                    
                    # Calculate out-of-home duration as total time minus home time
                    out_of_home_duration = total_observed_time - home_duration
                else:
                    out_of_home_duration = 0
                
                # Store the results
                location_durations[user_date] = {
                    'home_duration': home_duration,
                    'out_of_home_duration': out_of_home_duration
                }
            
            # Convert the dictionary to a DataFrame
            location_df = pd.DataFrame.from_dict(location_durations, orient='index')
            location_df.index.name = 'user_date'
            location_df = location_df.reset_index()
            
            # Split user_date back to separate columns
            location_df[['user', 'date']] = location_df['user_date'].str.split('_', n=1, expand=True)
            location_df = location_df.drop(columns=['user_date'])
            
            # Log some sample calculations
            self.logger.info("\nSample location duration calculations:")
            self.logger.info(location_df.head())
            
            return location_df
            
        except Exception as e:
            self.logger.error(f"Error calculating location durations: {str(e)}", exc_info=True)
            # Return empty DataFrame with necessary columns
            return pd.DataFrame(columns=['user', 'date', 'home_duration', 'out_of_home_duration'])
            
    def calculate_transport_durations(self, episode_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate total durations for active and mechanized transport based on episode data.
        
        Args:
            episode_df: DataFrame containing episode information
            
        Returns:
            DataFrame with additional transport duration columns
        """
        # Create a copy to avoid modifying the original
        df = episode_df.copy()
        
        try:
            # Check if episode_type column exists
            if 'episode_type' not in df.columns:
                self.logger.warning("episode_type column not found in episode data")
                return pd.DataFrame(columns=['user', 'date', 'active_transport_duration', 'mechanized_transport_duration'])
                
            # Create a user-date index for grouping
            df['user_date'] = df['user'].astype(str) + '_' + df['date'].astype(str)
            
            # Calculate transport durations by type
            transport_durations = {}
            
            # Process each user-date combination
            for user_date in df['user_date'].unique():
                user_date_df = df[df['user_date'] == user_date]
                
                # Extract active and mechanized transport directly from episode types
                active_df = user_date_df[user_date_df['episode_type'] == 'active_transport']
                mechanized_df = user_date_df[user_date_df['episode_type'] == 'mechanized_transport']
                
                active_duration = active_df['total_duration_minutes'].sum() if not active_df.empty else 0
                mechanized_duration = mechanized_df['total_duration_minutes'].sum() if not mechanized_df.empty else 0
                
                # Store the results
                transport_durations[user_date] = {
                    'active_transport_duration': active_duration,
                    'mechanized_transport_duration': mechanized_duration
                }
            
            # Convert the dictionary to a DataFrame
            transport_df = pd.DataFrame.from_dict(transport_durations, orient='index')
            transport_df.index.name = 'user_date'
            transport_df = transport_df.reset_index()
            
            # Split user_date back to separate columns
            transport_df[['user', 'date']] = transport_df['user_date'].str.split('_', n=1, expand=True)
            transport_df = transport_df.drop(columns=['user_date'])
            
            # Log some sample calculations
            self.logger.info("\nSample transport duration calculations:")
            self.logger.info(transport_df.head())
            
            return transport_df
            
        except Exception as e:
            self.logger.error(f"Error calculating transport durations: {str(e)}", exc_info=True)
            # Return empty DataFrame with necessary columns
            return pd.DataFrame(columns=['user', 'date', 'active_transport_duration', 'mechanized_transport_duration'])

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
            
            # Ensure consistent data types for merge columns
            # Convert 'user' columns to string in all dataframes
            quality_df['user'] = quality_df['user'].astype(str)
            frag_df['participant_id'] = frag_df['participant_id'].astype(str)
            episode_df['user'] = episode_df['user'].astype(str)
            
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
            
            # Calculate transport durations using updated method that reads from active/mechanized episode types
            transport_durations = self.calculate_transport_durations(episode_df)
            
            # Calculate home durations
            home_durations = self.calculate_location_durations(episode_df)
            
            # Make sure user column is string type in transport and home dataframes
            transport_durations['user'] = transport_durations['user'].astype(str)
            home_durations['user'] = home_durations['user'].astype(str)
            
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
            
            # Convert user column to string in the pivot dataframe
            episode_pivot['user'] = episode_pivot['user'].astype(str)
            
            # Merge episode data
            merged_df = pd.merge(
                merged_df,
                episode_pivot,
                left_on=['user', 'associated_data_date'],
                right_on=['user', 'date'],
                how='left',
                suffixes=('', '_episode')
            )
            
            # Log merge operations for debugging
            self.logger.info(f"After episode merge: {len(merged_df)} rows")
            
            # Merge transport durations
            merged_df = pd.merge(
                merged_df,
                transport_durations,
                left_on=['user', 'associated_data_date'],
                right_on=['user', 'date'],
                how='left',
                suffixes=('', '_transport')
            )
            
            self.logger.info(f"After transport merge: {len(merged_df)} rows")
            
            # Merge home durations
            merged_df = pd.merge(
                merged_df,
                home_durations,
                left_on=['user', 'associated_data_date'],
                right_on=['user', 'date'],
                how='left',
                suffixes=('', '_location')
            )
            
            self.logger.info(f"After home duration merge: {len(merged_df)} rows")
            
            # Fill NaN values in duration columns with 0
            duration_columns = ['active_transport_duration', 'mechanized_transport_duration', 'home_duration', 'out_of_home_duration']
            for col in duration_columns:
                if col in merged_df.columns:
                    missing_count = merged_df[col].isna().sum()
                    if missing_count > 0:
                        self.logger.warning(f"Filling {missing_count} missing values in {col} with 0")
                    merged_df[col] = merged_df[col].fillna(0)
            
            # Merge EMA data
            processed_ema['Participant_ID'] = processed_ema['Participant_ID'].astype(str)
            merged_df = pd.merge(
                merged_df,
                processed_ema,
                left_on=['user', 'ema_timestamp'],
                right_on=['Participant_ID', 'StartDate'],
                how='left'
            )
            
            # Clean up redundant columns
            cols_to_drop = [
                'date_episode', 'date_transport', 'date_location', 'participant_id', 'Participant_ID',
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
            
            # Log column names to verify home_duration is included
            self.logger.info(f"Columns in final dataframe: {merged_df.columns.tolist()}")
            
            # Calculate z-scores for key metrics including the new ones
            metric_cols = [
                'digital_fragmentation_index',
                'mobility_fragmentation_index',
                'overlap_fragmentation_index',
                'digital_home_fragmentation_index',
                'digital_home_mobility_delta',
                'digital_total_duration',
                'mobility_total_duration',
                'active_transport_duration',
                'mechanized_transport_duration',
                'home_duration',
                'out_of_home_duration',
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
            self.logger.info(f"Transport metrics: active_transport_duration mean={merged_df['active_transport_duration'].mean():.2f}, " +
                           f"mechanized_transport_duration mean={merged_df['mechanized_transport_duration'].mean():.2f}")
            self.logger.info(f"Home duration mean={merged_df['home_duration'].mean():.2f} minutes")
            self.logger.info(f"Out of home duration mean={merged_df['out_of_home_duration'].mean():.2f} minutes")
            
            # Log new metrics if available
            if 'digital_home_fragmentation_index' in merged_df.columns:
                self.logger.info(f"Digital home fragmentation mean={merged_df['digital_home_fragmentation_index'].mean():.4f}")
            if 'digital_home_mobility_delta' in merged_df.columns:
                self.logger.info(f"Digital home-mobility delta mean={merged_df['digital_home_mobility_delta'].mean():.4f}")
            
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
        
        # Transport and location statistics
        stats['transport_location_duration'] = df[['active_transport_duration', 
                                                 'mechanized_transport_duration',
                                                 'home_duration',
                                                 'out_of_home_duration']].describe()
        
        # NEW: Add fragmentation metrics statistics
        fragmentation_cols = [col for col in df.columns if 'fragmentation' in col]
        if fragmentation_cols:
            stats['fragmentation_metrics'] = df[fragmentation_cols].describe()
        
        # NEW: Add delta metric statistics if available
        if 'digital_home_mobility_delta' in df.columns:
            stats['digital_home_mobility_delta'] = df[['digital_home_mobility_delta']].describe()
        
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
        self.logger.info(f"Average active transport duration: {df['active_transport_duration'].mean():.2f} minutes")
        self.logger.info(f"Average mechanized transport duration: {df['mechanized_transport_duration'].mean():.2f} minutes")
        self.logger.info(f"Average home duration: {df['home_duration'].mean():.2f} minutes")
        self.logger.info(f"Average out of home duration: {df['out_of_home_duration'].mean():.2f} minutes")
        
        # NEW: Log the new metrics
        if 'digital_home_fragmentation_index' in df.columns:
            valid_dhf = df['digital_home_fragmentation_index'].dropna()
            if len(valid_dhf) > 0:
                self.logger.info(f"Average digital-home fragmentation: {valid_dhf.mean():.4f}")
                self.logger.info(f"Valid digital-home fragmentation measurements: {len(valid_dhf)}/{len(df)} ({len(valid_dhf)/len(df)*100:.1f}%)")
        
        if 'digital_home_mobility_delta' in df.columns:
            valid_delta = df['digital_home_mobility_delta'].dropna()
            if len(valid_delta) > 0:
                self.logger.info(f"Average digital home-mobility delta: {valid_delta.mean():.4f}")
                self.logger.info(f"Valid delta measurements: {len(valid_delta)}/{len(df)} ({len(valid_delta)/len(df)*100:.1f}%)")

   
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