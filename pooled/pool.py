#!/usr/bin/env python3
"""
Pooled Analysis of STAI Anxiety Data across SURREAL and TLV Datasets

This script standardizes STAI anxiety data from both SURREAL and TLV datasets,
performs quality checks, and outputs a cleaned pooled dataset for further analysis.

Usage:
    python analysis.py [--output_dir /path/to/output] [--debug]
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
from datetime import datetime
from scipy import stats
import warnings


class PooledSTAIAnalysis:
    def __init__(self, output_dir=None, debug=False, standardization_type='population'):
        """Initialize the pooled STAI analysis.
        
        Args:
            output_dir (str): Directory to save outputs
            debug (bool): Enable debug logging
            standardization_type (str): Type of standardization to use ('participant' or 'population')
        """
        # Set standardization type
        self.standardization_type = standardization_type
        
        # Hardcoded paths to data files - will be adjusted based on standardization type
        
        if self.standardization_type == 'population':
            self.surreal_path = Path('pooled/data/ema_fragmentation_demographics_population_norm.csv')
        else:  # participant level
            self.surreal_path = Path('pooled/data/ema_fragmentation_demographics_participant_norm.csv')
            
        self.tlv_path = Path('pooled/data/combined_metrics.csv')
        
        # Set output directory relative to script location
        script_dir = Path(__file__).parent
        self.output_dir = script_dir / "processed" 
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug = debug
        
        # Setup logging
        self._setup_logging()
        
        # Define variable mappings between datasets
        self.variable_mappings = {
            'surreal': {
                'id': 'participant_id_clean',
                'anxiety': 'ema_STAI-Y-A-6_zstd',
                'anxiety_raw': 'ema_STAI-Y-A-6_raw',
                'depression': 'ema_CES-D-8_zstd',
                'depression_raw': 'ema_CES-D-8_raw',
                'gender': 'Gender',
                'location': 'City.center',  # Yes = city center, No = suburb
                'fragmentation': {
                    'digital': 'frag_digital_fragmentation_index',
                    'mobility': 'frag_mobility_fragmentation_index',
                    'overlap': 'frag_overlap_fragmentation_index'
                }
            },
            'tlv': {
                'id': 'user',
                'anxiety': 'STAI6_score',
                'happiness': 'HAPPY',
                'gender': 'Gender',
                'location': 'School',  # Assuming 'suburb' or other values
                'fragmentation': {
                    'digital': 'digital_fragmentation_index',
                    'mobility': 'moving_fragmentation_index',
                    'overlap': 'digital_frag_during_mobility'
                }
            },
            'standardized': {
                'id': 'participant_id',
                'dataset': 'dataset_source',
                'anxiety': 'anxiety_score_std',
                'anxiety_raw': 'anxiety_score_raw',
                'mood': 'mood_score_std',  # Depression for SURREAL, inverted happiness for TLV
                'mood_raw': 'mood_score_raw',
                'gender': 'gender_standardized',
                'location': 'location_type',  # city_center or suburb
                'age_group': 'age_group',  # adult or adolescent
                'fragmentation': {
                    'digital': 'digital_fragmentation',
                    'mobility': 'mobility_fragmentation',
                    'overlap': 'overlap_fragmentation'
                }
            }
        }

    def _setup_logging(self):
        """Set up logging configuration."""
        log_dir = self.output_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'pooled_stai_analysis_{self.standardization_type}_{timestamp}.log'
        
        log_level = logging.DEBUG if self.debug else logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing pooled STAI analysis with {self.standardization_type} standardization")
        
        self.logger.info(f"SURREAL data: {self.surreal_path}")
        self.logger.info(f"TLV data: {self.tlv_path}")
        self.logger.info(f"Output directory: {self.output_dir}")

    def load_surreal_data(self):
        """Load SURREAL data and extract relevant variables."""
        if not self.surreal_path or not self.surreal_path.exists():
            self.logger.warning("SURREAL data path not provided or file doesn't exist")
            return None
            
        try:
            self.logger.info(f"Loading SURREAL data from {self.surreal_path}")
            df = pd.read_csv(self.surreal_path)
            self.logger.info(f"SURREAL data loaded with shape: {df.shape}")
            
            # Check for required columns
            required_cols = [self.variable_mappings['surreal']['id'], 
                            self.variable_mappings['surreal']['anxiety']]
            for col in required_cols:
                if col not in df.columns:
                    self.logger.error(f"Required column '{col}' not found in SURREAL data")
                    return None
            
            # Extract participant ID
            df['participant_id'] = df[self.variable_mappings['surreal']['id']].astype(str)
            
            # Extract anxiety score (already z-standardized in SURREAL)
            anxiety_col = self.variable_mappings['surreal']['anxiety']
            anxiety_raw_col = self.variable_mappings['surreal']['anxiety_raw']
            df[self.variable_mappings['standardized']['anxiety']] = df[anxiety_col]
            df[self.variable_mappings['standardized']['anxiety_raw']] = df[anxiety_raw_col]
            
            # Extract depression score if available
            if self.variable_mappings['surreal']['depression'] in df.columns:
                depression_col = self.variable_mappings['surreal']['depression']
                depression_raw_col = self.variable_mappings['surreal']['depression_raw']
                df[self.variable_mappings['standardized']['mood']] = df[depression_col]
                df[self.variable_mappings['standardized']['mood_raw']] = df[depression_raw_col]
            
            # Extract gender and standardize
            if self.variable_mappings['surreal']['gender'] in df.columns:
                gender_col = self.variable_mappings['surreal']['gender']
                # Map gender to standardized format (female/male)
                df[self.variable_mappings['standardized']['gender']] = df[gender_col].apply(
                    lambda x: 'female' if x.strip().lower() in ['f', 'female', 'נקבה'] else 'male' 
                    if x.strip().lower() in ['m', 'male', 'זכר'] else 'other'
                )
            
            # Extract location type and standardize
            if self.variable_mappings['surreal']['location'] in df.columns:
                location_col = self.variable_mappings['surreal']['location']
                # Map location to standardized format (city_center/suburb)
                df[self.variable_mappings['standardized']['location']] = df[location_col].apply(
                    lambda x: 'city_center' if x == 'Yes' else 'suburb'
                )
            
            # Add age group (all SURREAL participants are adults)
            df[self.variable_mappings['standardized']['age_group']] = 'adult'
            
            # Extract fragmentation metrics if available
            for frag_type, col_name in self.variable_mappings['surreal']['fragmentation'].items():
                if col_name in df.columns:
                    std_col = self.variable_mappings['standardized']['fragmentation'][frag_type]
                    df[std_col] = df[col_name]
            
            # Add dataset source identifier
            df[self.variable_mappings['standardized']['dataset']] = 'surreal'
            
            # Log statistics
            self._log_data_stats(df, 'SURREAL')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading SURREAL data: {str(e)}")
            if self.debug:
                self.logger.exception("Detailed error:")
            return None

    def load_tlv_data(self):
        """Load TLV data and extract relevant variables."""
        if not self.tlv_path or not self.tlv_path.exists():
            self.logger.warning("TLV data path not provided or file doesn't exist")
            return None
            
        try:
            self.logger.info(f"Loading TLV data from {self.tlv_path}")
            df = pd.read_csv(self.tlv_path)
            self.logger.info(f"TLV data loaded with shape: {df.shape}")
            
            # Check for required columns
            required_cols = [self.variable_mappings['tlv']['id'], 
                            self.variable_mappings['tlv']['anxiety']]
            for col in required_cols:
                if col not in df.columns:
                    self.logger.error(f"Required column '{col}' not found in TLV data")
                    return None
            
            # Extract participant ID
            df['participant_id'] = df[self.variable_mappings['tlv']['id']].astype(str)
            
            # Extract raw anxiety score
            anxiety_col = self.variable_mappings['tlv']['anxiety']
            df[self.variable_mappings['standardized']['anxiety_raw']] = df[anxiety_col]
            
            # Z-standardize the anxiety score based on standardization type
            if self.standardization_type == 'participant':
                # Participant-level standardization (within each participant)
                participant_groups = df.groupby('participant_id')[anxiety_col]
                std_scores = []
                
                for participant_id, group in participant_groups:
                    if len(group) > 1 and group.std() > 0:
                        # Only standardize if we have >1 data point and std > 0
                        std_scores.extend((group - group.mean()) / group.std())
                    else:
                        # If only one data point, set z-score to 0
                        std_scores.extend([0] * len(group))
                
                df[self.variable_mappings['standardized']['anxiety']] = std_scores
            else:
                # Population-level standardization (across all participants)
                df[self.variable_mappings['standardized']['anxiety']] = stats.zscore(
                    df[anxiety_col], nan_policy='omit'
                )
            
            # Extract happiness score if available and invert it for mood comparison
            # (higher happiness = lower depression, so we invert for consistency)
            if self.variable_mappings['tlv']['happiness'] in df.columns:
                happiness_col = self.variable_mappings['tlv']['happiness']
                # Store raw happiness score
                df[self.variable_mappings['standardized']['mood_raw']] = df[happiness_col]
                
                # Z-standardize and invert happiness score based on standardization type
                if self.standardization_type == 'participant':
                    # Participant-level standardization
                    participant_groups = df.groupby('participant_id')[happiness_col]
                    std_scores = []
                    
                    for participant_id, group in participant_groups:
                        if len(group) > 1 and group.std() > 0:
                            # Invert and standardize
                            std_scores.extend(-1 * (group - group.mean()) / group.std())
                        else:
                            # If only one data point, set z-score to 0
                            std_scores.extend([0] * len(group))
                    
                    df[self.variable_mappings['standardized']['mood']] = std_scores
                else:
                    # Population-level standardization
                    df[self.variable_mappings['standardized']['mood']] = -1 * stats.zscore(
                        df[happiness_col], nan_policy='omit'
                    )
            
            # Extract gender and standardize
            if self.variable_mappings['tlv']['gender'] in df.columns:
                gender_col = self.variable_mappings['tlv']['gender']
                # Map gender to standardized format (female/male)
                df[self.variable_mappings['standardized']['gender']] = df[gender_col].apply(
                    lambda x: 'female' if str(x).strip().lower() in ['f', 'female', 'נקבה'] else 'male' 
                    if str(x).strip().lower() in ['m', 'male', 'זכר'] else 'other'
                )
            
            # Extract location type and standardize
            if self.variable_mappings['tlv']['location'] in df.columns:
                location_col = self.variable_mappings['tlv']['location']
                # Map location to standardized format (city_center/suburb)
                df[self.variable_mappings['standardized']['location']] = df[location_col].apply(
                    lambda x: 'city_center' if str(x).strip().lower() != 'suburb' else 'suburb'
                )
            
            # Add age group (all TLV participants are adolescents)
            df[self.variable_mappings['standardized']['age_group']] = 'adolescent'
            
            # Extract fragmentation metrics if available
            for frag_type, col_name in self.variable_mappings['tlv']['fragmentation'].items():
                if col_name in df.columns:
                    std_col = self.variable_mappings['standardized']['fragmentation'][frag_type]
                    df[std_col] = df[col_name]
            
            # Add dataset source identifier
            df[self.variable_mappings['standardized']['dataset']] = 'tlv'
            
            # Log statistics
            self._log_data_stats(df, 'TLV')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading TLV data: {str(e)}")
            if self.debug:
                self.logger.exception("Detailed error:")
            return None

    def _log_data_stats(self, df, dataset_name):
        """Log basic statistics about the data."""
        # Get standardized column names
        std_id = 'participant_id'
        std_anxiety = self.variable_mappings['standardized']['anxiety']
        std_anxiety_raw = self.variable_mappings['standardized']['anxiety_raw']
        std_mood = self.variable_mappings['standardized']['mood']
        std_mood_raw = self.variable_mappings['standardized']['mood_raw']
        std_gender = self.variable_mappings['standardized']['gender']
        std_location = self.variable_mappings['standardized']['location']
        
        # Count participants and observations
        n_participants = df[std_id].nunique()
        n_observations = len(df)
        n_valid_anxiety = df[std_anxiety].notna().sum()
        
        self.logger.info(f"{dataset_name} dataset statistics:")
        self.logger.info(f"  Participants: {n_participants}")
        self.logger.info(f"  Total observations: {n_observations}")
        self.logger.info(f"  Valid anxiety scores: {n_valid_anxiety} ({n_valid_anxiety/n_observations:.1%})")
        
        # Log anxiety score summary if available
        if n_valid_anxiety > 0:
            self.logger.info(f"  Anxiety scores (raw):")
            self.logger.info(f"    Mean: {df[std_anxiety_raw].mean():.2f}")
            self.logger.info(f"    Std: {df[std_anxiety_raw].std():.2f}")
            self.logger.info(f"    Min: {df[std_anxiety_raw].min():.2f}")
            self.logger.info(f"    Max: {df[std_anxiety_raw].max():.2f}")
            
        # Log mood score summary if available
        if std_mood in df.columns and df[std_mood].notna().sum() > 0:
            n_valid_mood = df[std_mood].notna().sum()
            self.logger.info(f"  Mood scores:")
            self.logger.info(f"    Valid scores: {n_valid_mood} ({n_valid_mood/n_observations:.1%})")
            self.logger.info(f"    Mean: {df[std_mood].mean():.2f}")
            self.logger.info(f"    Std: {df[std_mood].std():.2f}")
            self.logger.info(f"    Min: {df[std_mood].min():.2f}")
            self.logger.info(f"    Max: {df[std_mood].max():.2f}")
        
        # Log demographic summaries
        if std_gender in df.columns:
            gender_counts = df[std_gender].value_counts()
            self.logger.info(f"  Gender distribution:")
            for gender, count in gender_counts.items():
                self.logger.info(f"    {gender}: {count} ({count/n_observations:.1%})")
                
        if std_location in df.columns:
            location_counts = df[std_location].value_counts()
            self.logger.info(f"  Location distribution:")
            for location, count in location_counts.items():
                self.logger.info(f"    {location}: {count} ({count/n_observations:.1%})")
            
        # Log average observations per participant
        obs_per_participant = df.groupby(std_id).size()
        self.logger.info(f"  Observations per participant:")
        self.logger.info(f"    Mean: {obs_per_participant.mean():.1f}")
        self.logger.info(f"    Min: {obs_per_participant.min()}")
        self.logger.info(f"    Max: {obs_per_participant.max()}")
        
        # Log fragmentation metrics if available
        for frag_type, std_col in self.variable_mappings['standardized']['fragmentation'].items():
            if std_col in df.columns and df[std_col].notna().sum() > 0:
                self.logger.info(f"  {frag_type.capitalize()} fragmentation:")
                self.logger.info(f"    Mean: {df[std_col].mean():.3f}")
                self.logger.info(f"    Std: {df[std_col].std():.3f}")
                self.logger.info(f"    Valid observations: {df[std_col].notna().sum()} ({df[std_col].notna().sum()/n_observations:.1%})")

    def merge_datasets(self):
        """Load, preprocess, and merge both datasets."""
        # Load datasets
        surreal_df = self.load_surreal_data()
        tlv_df = self.load_tlv_data()
        
        # Track available datasets
        available_datasets = []
        if surreal_df is not None:
            available_datasets.append('surreal')
            self.surreal_data = surreal_df
            
        if tlv_df is not None:
            available_datasets.append('tlv')
            self.tlv_data = tlv_df
        
        if not available_datasets:
            self.logger.error("No datasets are available for analysis")
            return None
            
        self.logger.info(f"Available datasets: {', '.join(available_datasets)}")
        
        # If both datasets are available, merge them
        if len(available_datasets) == 2:
            # Get subset of relevant columns for each dataset
            std_columns = [
                'participant_id',
                self.variable_mappings['standardized']['dataset'],
                self.variable_mappings['standardized']['anxiety'],
                self.variable_mappings['standardized']['anxiety_raw'],
                self.variable_mappings['standardized']['mood'],
                self.variable_mappings['standardized']['mood_raw'],
                self.variable_mappings['standardized']['gender'],
                self.variable_mappings['standardized']['location'],
                self.variable_mappings['standardized']['age_group']
            ]
            
            # Add fragmentation columns if available
            for std_col in self.variable_mappings['standardized']['fragmentation'].values():
                if (std_col in surreal_df.columns) or (std_col in tlv_df.columns):
                    if std_col not in std_columns:
                        std_columns.append(std_col)
            
            # Filter columns to only those present in each dataset
            surreal_columns = [col for col in std_columns if col in surreal_df.columns]
            tlv_columns = [col for col in std_columns if col in tlv_df.columns]
            
            # Get relevant data subsets
            surreal_subset = surreal_df[surreal_columns]
            tlv_subset = tlv_df[tlv_columns]
            
            # Combine the datasets
            # For each dataset, add any missing columns with NaN values
            for col in std_columns:
                if col not in surreal_subset.columns:
                    surreal_subset[col] = np.nan
                if col not in tlv_subset.columns:
                    tlv_subset[col] = np.nan
                    
            combined_df = pd.concat([surreal_subset, tlv_subset], ignore_index=True)
            self.logger.info(f"Combined dataset shape: {combined_df.shape}")
            
            # Check for missing values in key columns
            std_anxiety = self.variable_mappings['standardized']['anxiety']
            missing_anxiety = combined_df[std_anxiety].isna().sum()
            self.logger.info(f"Missing standardized anxiety scores: {missing_anxiety} ({missing_anxiety/len(combined_df):.1%})")
            
            # Calculate basic statistics by dataset
            self.logger.info("Statistics by dataset in combined data:")
            for dataset in ['surreal', 'tlv']:
                dataset_data = combined_df[combined_df[self.variable_mappings['standardized']['dataset']] == dataset]
                if len(dataset_data) > 0:
                    self.logger.info(f"  {dataset.upper()}:")
                    self.logger.info(f"    Participants: {dataset_data['participant_id'].nunique()}")
                    self.logger.info(f"    Observations: {len(dataset_data)}")
                    self.logger.info(f"    Valid anxiety scores: {dataset_data[std_anxiety].notna().sum()}")
            
            self.pooled_data = combined_df
            return combined_df
            
        elif len(available_datasets) == 1:
            # Only one dataset available, use it as is
            dataset_name = available_datasets[0]
            if dataset_name == 'surreal':
                self.pooled_data = surreal_df
            else:
                self.pooled_data = tlv_df
                
            self.logger.info(f"Only {dataset_name} data available, using as pooled dataset")
            self.logger.info(f"Pooled dataset shape: {self.pooled_data.shape}")
            
            return self.pooled_data
        
        return None

    def save_pooled_data(self):
        """Save the pooled dataset to CSV."""
        if not hasattr(self, 'pooled_data') or self.pooled_data is None:
            self.logger.warning("No pooled data available to save")
            return
            
        output_file = self.output_dir / f"pooled_stai_data_{self.standardization_type}.csv"
        self.pooled_data.to_csv(output_file, index=False)
        self.logger.info(f"Pooled data saved to {output_file}")

    def run_quality_checks(self):
        """Run quality checks on the pooled dataset."""
        if not hasattr(self, 'pooled_data') or self.pooled_data is None:
            self.logger.warning("No pooled data available for quality checks")
            return
            
        self.logger.info("Running quality checks on pooled data...")
        
        try:
            # Get standardized column names
            std_anxiety = self.variable_mappings['standardized']['anxiety']
            std_mood = self.variable_mappings['standardized']['mood']
            std_gender = self.variable_mappings['standardized']['gender']
            std_location = self.variable_mappings['standardized']['location']
            std_age_group = self.variable_mappings['standardized']['age_group']
            std_dataset = self.variable_mappings['standardized']['dataset']
            
            # Check 1: Missing values in key variables
            missing_report = {
                col: self.pooled_data[col].isna().sum() 
                for col in self.pooled_data.columns 
                if self.pooled_data[col].isna().sum() > 0
            }
            
            if missing_report:
                self.logger.info("Missing value report:")
                for col, count in missing_report.items():
                    self.logger.info(f"  {col}: {count} missing values ({count/len(self.pooled_data):.1%})")
            else:
                self.logger.info("No missing values found in any columns")
            
            # Check 2: Outliers in anxiety scores (beyond 3 standard deviations)
            mean = self.pooled_data[std_anxiety].mean()
            std = self.pooled_data[std_anxiety].std()
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
            
            outliers = self.pooled_data[(self.pooled_data[std_anxiety] < lower_bound) | 
                                       (self.pooled_data[std_anxiety] > upper_bound)]
            
            self.logger.info(f"Outlier check for anxiety scores (beyond ±3 std dev):")
            self.logger.info(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
            self.logger.info(f"  Found {len(outliers)} outliers ({len(outliers)/len(self.pooled_data):.1%} of data)")
            
            # Check 3: Participants with too few observations (less than 3)
            participant_counts = self.pooled_data.groupby('participant_id').size()
            few_obs_participants = participant_counts[participant_counts < 3]
            
            self.logger.info(f"Participants with fewer than 3 observations:")
            self.logger.info(f"  Count: {len(few_obs_participants)} out of {len(participant_counts)} participants")
            
            # Check 4: Data balance between datasets
            if std_dataset in self.pooled_data.columns:
                dataset_counts = self.pooled_data[std_dataset].value_counts()
                self.logger.info(f"Dataset balance:")
                for dataset, count in dataset_counts.items():
                    self.logger.info(f"  {dataset}: {count} observations ({count/len(self.pooled_data):.1%})")
            
            # Check 5: Gender balance
            if std_gender in self.pooled_data.columns:
                gender_counts = self.pooled_data[std_gender].value_counts()
                self.logger.info(f"Gender balance:")
                for gender, count in gender_counts.items():
                    self.logger.info(f"  {gender}: {count} observations ({count/len(self.pooled_data):.1%})")
                
                # By dataset
                self.logger.info(f"Gender balance by dataset:")
                for dataset in self.pooled_data[std_dataset].unique():
                    dataset_data = self.pooled_data[self.pooled_data[std_dataset] == dataset]
                    if len(dataset_data) > 0:
                        self.logger.info(f"  {dataset}:")
                        ds_gender_counts = dataset_data[std_gender].value_counts()
                        for gender, count in ds_gender_counts.items():
                            self.logger.info(f"    {gender}: {count} ({count/len(dataset_data):.1%})")
            
            # Check 6: Location balance
            if std_location in self.pooled_data.columns:
                location_counts = self.pooled_data[std_location].value_counts()
                self.logger.info(f"Location balance:")
                for location, count in location_counts.items():
                    self.logger.info(f"  {location}: {count} observations ({count/len(self.pooled_data):.1%})")
                
                # By dataset
                self.logger.info(f"Location balance by dataset:")
                for dataset in self.pooled_data[std_dataset].unique():
                    dataset_data = self.pooled_data[self.pooled_data[std_dataset] == dataset]
                    if len(dataset_data) > 0:
                        self.logger.info(f"  {dataset}:")
                        ds_location_counts = dataset_data[std_location].value_counts()
                        for location, count in ds_location_counts.items():
                            if pd.notna(location):
                                self.logger.info(f"    {location}: {count} ({count/len(dataset_data):.1%})")
            
            # Check 7: Age group balance
            if std_age_group in self.pooled_data.columns:
                age_counts = self.pooled_data[std_age_group].value_counts()
                self.logger.info(f"Age group balance:")
                for age, count in age_counts.items():
                    self.logger.info(f"  {age}: {count} observations ({count/len(self.pooled_data):.1%})")
            
            # Check 8: Fragmentation metrics availability
            frag_cols = [col for col in self.pooled_data.columns 
                         if col in self.variable_mappings['standardized']['fragmentation'].values()]
            
            if frag_cols:
                self.logger.info(f"Fragmentation metrics availability:")
                for col in frag_cols:
                    valid_count = self.pooled_data[col].notna().sum()
                    valid_percent = valid_count / len(self.pooled_data) * 100
                    self.logger.info(f"  {col}: {valid_count} valid values ({valid_percent:.1f}%)")
                    
                    # Check by dataset
                    for dataset in self.pooled_data[std_dataset].unique():
                        dataset_data = self.pooled_data[self.pooled_data[std_dataset] == dataset]
                        valid_in_dataset = dataset_data[col].notna().sum()
                        if len(dataset_data) > 0:
                            valid_percent = valid_in_dataset / len(dataset_data) * 100
                            self.logger.info(f"    {dataset}: {valid_in_dataset} valid values ({valid_percent:.1f}%)")
            
            # Check 9: Correlation between standardized anxiety/mood scores and fragmentation
            self.logger.info(f"Correlations between anxiety and fragmentation metrics:")
            for col in frag_cols:
                if self.pooled_data[col].notna().sum() > 10:
                    corr = self.pooled_data[[std_anxiety, col]].corr().iloc[0, 1]
            
            # Check mood correlations if available
            if std_mood in self.pooled_data.columns:
                self.logger.info(f"Correlations between mood and fragmentation metrics:")
                for col in frag_cols:
                    if self.pooled_data[col].notna().sum() > 10 and self.pooled_data[std_mood].notna().sum() > 10:
                        valid_data = self.pooled_data[[std_mood, col]].dropna()
                        if len(valid_data) > 10:
                            corr = valid_data.corr().iloc[0, 1]
                            self.logger.info(f"  {std_mood} vs {col}: r = {corr:.3f}")
                            
                            # Check by dataset
                            for dataset in self.pooled_data[std_dataset].unique():
                                dataset_data = self.pooled_data[self.pooled_data[std_dataset] == dataset]
                                if dataset_data[col].notna().sum() > 5 and dataset_data[std_mood].notna().sum() > 5:
                                    valid_ds_data = dataset_data[[std_mood, col]].dropna()
                                    if len(valid_ds_data) > 5:
                                        dataset_corr = valid_ds_data.corr().iloc[0, 1]
                                        self.logger.info(f"    {dataset}: r = {dataset_corr:.3f}")
            
            # Check 10: Anxiety comparison between datasets
            if std_dataset in self.pooled_data.columns:
                self.logger.info(f"Anxiety score comparison between datasets:")
                datasets = self.pooled_data[std_dataset].unique()
                if len(datasets) > 1:
                    anxiety_by_dataset = {
                        dataset: self.pooled_data[self.pooled_data[std_dataset] == dataset][std_anxiety].dropna()
                        for dataset in datasets
                    }
                    
                    for dataset, values in anxiety_by_dataset.items():
                        self.logger.info(f"  {dataset}: n={len(values)}, mean={values.mean():.3f}, sd={values.std():.3f}")
                    
                    # T-test between datasets
                    if len(anxiety_by_dataset) == 2 and all(len(v) > 5 for v in anxiety_by_dataset.values()):
                        d1, d2 = datasets
                        t_stat, p_val = stats.ttest_ind(
                            anxiety_by_dataset[d1], 
                            anxiety_by_dataset[d2],
                            equal_var=False
                        )
                        self.logger.info(f"  T-test {d1} vs {d2}: t={t_stat:.3f}, p={p_val:.4f}")
            
            # Check 11: Mood comparison between datasets (if available)
            if std_dataset in self.pooled_data.columns and std_mood in self.pooled_data.columns:
                valid_mood = self.pooled_data[std_mood].notna().sum()
                if valid_mood > 10:
                    self.logger.info(f"Mood score comparison between datasets:")
                    datasets = self.pooled_data[std_dataset].unique()
                    mood_by_dataset = {
                        dataset: self.pooled_data[self.pooled_data[std_dataset] == dataset][std_mood].dropna()
                        for dataset in datasets
                    }
                    
                    for dataset, values in mood_by_dataset.items():
                        if len(values) > 5:
                            self.logger.info(f"  {dataset}: n={len(values)}, mean={values.mean():.3f}, sd={values.std():.3f}")
                    
                    # T-test between datasets
                    if len(mood_by_dataset) == 2 and all(len(v) > 5 for v in mood_by_dataset.values()):
                        d1, d2 = datasets
                        t_stat, p_val = stats.ttest_ind(
                            mood_by_dataset[d1], 
                            mood_by_dataset[d2],
                            equal_var=False
                        )
                        self.logger.info(f"  T-test {d1} vs {d2}: t={t_stat:.3f}, p={p_val:.4f}")
            
            self.logger.info("Quality checks completed")
            
        except Exception as e:
            self.logger.error(f"Error running quality checks: {str(e)}")
            if self.debug:
                self.logger.exception("Detailed error:")

    def print_summary(self):
        """Print a concise summary of the pooled dataset for reference."""
        if not hasattr(self, 'pooled_data') or self.pooled_data is None:
            self.logger.warning("No pooled data available for summary")
            return
            
        try:
            std_id = self.variable_mappings['standardized']['id']
            std_dataset = self.variable_mappings['standardized']['dataset']
            std_anxiety = self.variable_mappings['standardized']['anxiety']
            std_mood = self.variable_mappings['standardized']['mood']
            std_gender = self.variable_mappings['standardized']['gender']
            std_location = self.variable_mappings['standardized']['location']
            std_age = self.variable_mappings['standardized']['age_group']
            
            self.logger.info("="*50)
            self.logger.info("POOLED DATASET SUMMARY")
            self.logger.info("="*50)
            
            # Overall counts
            total_obs = len(self.pooled_data)
            total_participants = self.pooled_data[std_id].nunique()
            
            self.logger.info(f"TOTAL OBSERVATIONS: {total_obs}")
            self.logger.info(f"TOTAL PARTICIPANTS: {total_participants}")
            
            # Dataset-specific counts
            datasets = self.pooled_data[std_dataset].unique()
            self.logger.info("\nOBSERVATIONS BY DATASET:")
            for dataset in datasets:
                ds_data = self.pooled_data[self.pooled_data[std_dataset] == dataset]
                ds_obs = len(ds_data)
                ds_participants = ds_data[std_id].nunique()
                ds_obs_pct = 100 * ds_obs / total_obs
                self.logger.info(f"  {dataset.upper()}: {ds_obs} obs ({ds_obs_pct:.1f}%), {ds_participants} participants")
            
            # Gender distribution
            self.logger.info("\nGENDER DISTRIBUTION:")
            genders = self.pooled_data[std_gender].value_counts()
            for gender, count in genders.items():
                gender_pct = 100 * count / total_obs
                self.logger.info(f"  {gender.upper()}: {count} obs ({gender_pct:.1f}%)")
            
            # Location distribution
            self.logger.info("\nLOCATION DISTRIBUTION:")
            locations = self.pooled_data[std_location].value_counts()
            for location, count in locations.items():
                location_pct = 100 * count / total_obs
                self.logger.info(f"  {location.upper()}: {count} obs ({location_pct:.1f}%)")
            
            # Age group distribution
            self.logger.info("\nAGE GROUP DISTRIBUTION:")
            age_groups = self.pooled_data[std_age].value_counts()
            for age, count in age_groups.items():
                age_pct = 100 * count / total_obs
                self.logger.info(f"  {age.upper()}: {count} obs ({age_pct:.1f}%)")
            
            # Data completeness
            self.logger.info("\nDATA COMPLETENESS:")
            frag_types = ['digital', 'mobility', 'overlap']
            for frag_type in frag_types:
                col = self.variable_mappings['standardized']['fragmentation'][frag_type]
                valid = self.pooled_data[col].notna().sum()
                valid_pct = 100 * valid / total_obs
                self.logger.info(f"  {frag_type.upper()} FRAGMENTATION: {valid}/{total_obs} ({valid_pct:.1f}%)")
            
            # Anxiety and mood statistics
            self.logger.info("\nANXIETY SCORES:")
            for dataset in datasets:
                ds_data = self.pooled_data[self.pooled_data[std_dataset] == dataset]
                anxiety_mean = ds_data[std_anxiety].mean()
                anxiety_std = ds_data[std_anxiety].std()
                self.logger.info(f"  {dataset.upper()}: mean={anxiety_mean:.3f}, sd={anxiety_std:.3f}")
            
            self.logger.info("\nMOOD SCORES:")
            for dataset in datasets:
                ds_data = self.pooled_data[self.pooled_data[std_dataset] == dataset]
                mood_mean = ds_data[std_mood].mean()
                mood_std = ds_data[std_mood].std()
                self.logger.info(f"  {dataset.upper()}: mean={mood_mean:.3f}, sd={mood_std:.3f}")
            
            self.logger.info("\nOutput file: " + str(self.output_dir / f"pooled_stai_data_{self.standardization_type}.csv"))
            self.logger.info("="*50)
            
        except Exception as e:
            self.logger.error(f"Error printing summary: {str(e)}")
            if self.debug:
                self.logger.exception("Detailed error:")

    def run_analysis(self):
        """Run the main analysis and save outputs."""
        # Merge datasets
        pooled_data = self.merge_datasets()
        
        if pooled_data is None or pooled_data.empty:
            self.logger.error("No pooled data available for analysis")
            return None
            
        # Save the pooled data
        output_file = self.output_dir / f"pooled_stai_data_{self.standardization_type}.csv"
        pooled_data.to_csv(output_file, index=False)
        self.logger.info(f"Saved pooled data to {output_file}")
        
        # Return the pooled data
        return pooled_data

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Pooled STAI Anxiety Analysis')
    parser.add_argument('--output_dir', type=str, help='Output directory for results')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    # Run analysis for both standardization types
    for std_type in ['participant', 'population']:
        print(f"\n\n{'='*80}\nRunning {std_type}-level standardization analysis\n{'='*80}\n")
        
        analysis = PooledSTAIAnalysis(
            output_dir=args.output_dir,
            debug=args.debug,
            standardization_type=std_type
        )
        
        pooled_data = analysis.run_analysis()
        
        if pooled_data is not None:
            print(f"Successfully created pooled dataset with {std_type}-level standardization")
            print(f"Output saved to {analysis.output_dir}/pooled_stai_data_{std_type}.csv")
        else:
            print(f"Failed to create pooled dataset with {std_type}-level standardization")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    # Set pandas to show more columns
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 1000)
    
    # Ignore certain warnings
    warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
    
    main()
