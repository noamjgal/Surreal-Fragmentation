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
        
        # Get script directory and set up data paths
        script_dir = Path(__file__).parent
        data_dir = script_dir / "data"
        
        # Set up data file paths
        if self.standardization_type == 'population':
            self.surreal_path = data_dir / 'ema_fragmentation_demographics_population.csv'
        else:  # participant level
            self.surreal_path = data_dir / 'ema_fragmentation_demographics_participant.csv'
            
        self.tlv_path = data_dir / 'combined_metrics.csv'
        
        # Set output directory relative to script location
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
                'date': 'date',  # Column containing date information
                'fragmentation': {
                    'digital': 'frag_digital_fragmentation_index',
                    'mobility': 'frag_mobility_fragmentation_index',
                    'overlap': 'frag_overlap_fragmentation_index',
                    'digital_home': 'frag_digital_home_fragmentation_index',
                    'digital_home_mobility_delta': 'frag_digital_home_mobility_delta'
                },
                'duration': {
                    'digital': 'frag_digital_total_duration',
                    'mobility': 'frag_mobility_total_duration',
                    'overlap': 'frag_overlap_total_duration',
                    'digital_home': 'frag_digital_home_total_duration',
                    'out_of_home': 'frag_out_of_home_duration',
                    'active_transport': 'frag_active_transport_duration',
                    'mechanized_transport': 'frag_automated_transport_duration',
                    'home': 'frag_home_duration'
                }
            },
            'tlv': {
                'id': 'user',
                'anxiety': 'STAI6_score',
                'happiness': 'HAPPY',
                'gender': 'Gender',
                'location': 'School',  # Assuming 'suburb' or other values
                'date': 'associated_data_date',  # Column containing date information 
                'fragmentation': {
                    'digital': 'digital_fragmentation_index',
                    'mobility': 'mobility_fragmentation_index',
                    'overlap': 'overlap_fragmentation_index',
                    'digital_home': 'digital_home_fragmentation_index',
                    'digital_home_mobility_delta': 'digital_home_mobility_delta'
                },
                'duration': {
                    'digital': 'digital_total_duration_minutes',
                    'mobility': 'moving_total_duration_minutes',
                    'overlap': 'overlap_total_duration_minutes',
                    'digital_home': 'digital_home_total_duration',
                    'active_transport': 'active_transport_total_duration_minutes',
                    'mechanized_transport': 'mechanized_transport_total_duration_minutes',
                    'home': 'home_total_duration_minutes'
                },
                'demographics': {
                    'gender': 'Gender',
                    'location': 'School',
                    'class': 'Class'
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
                'age': 'age',  # Added age field
                'class': 'class',  # Added class field for TLV
                'is_weekend': 'is_weekend',  # Added weekend status field
                'fragmentation': {
                    'digital': 'digital_fragmentation',
                    'mobility': 'mobility_fragmentation',
                    'overlap': 'overlap_fragmentation',
                    'digital_home': 'digital_home_fragmentation',
                    'digital_home_mobility_delta': 'digital_home_mobility_delta'
                },
                'duration': {
                    'digital': 'digital_total_duration',
                    'mobility': 'mobility_total_duration',
                    'overlap': 'overlap_total_duration',
                    'digital_home': 'digital_home_total_duration',
                    'active_transport': 'active_transport_duration',
                    'mechanized_transport': 'mechanized_transport_duration',
                    'home': 'home_duration',
                    'out_of_home': 'out_of_home_duration'
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
                            self.variable_mappings['surreal']['anxiety_raw']]
            for col in required_cols:
                if col not in df.columns:
                    self.logger.error(f"Required column '{col}' not found in SURREAL data")
                    return None
            
            # Extract participant ID
            df['participant_id'] = df[self.variable_mappings['surreal']['id']].astype(str)
            
            # Extract and restandardize anxiety scores
            anxiety_raw_col = self.variable_mappings['surreal']['anxiety_raw']
            df[self.variable_mappings['standardized']['anxiety_raw']] = df[anxiety_raw_col]
            
            # Restandardize anxiety scores instead of using pre-standardized values
            # This is necessary because filtering was done between previous standardization and now
            self.logger.info("Restandardizing SURREAL anxiety scores")
            if self.standardization_type == 'participant':
                # Participant-level standardization
                participant_groups = df.groupby('participant_id')[anxiety_raw_col]
                std_scores = []
                
                for participant_id, group in participant_groups:
                    if len(group) > 1 and group.std() > 0:
                        # Standardize within participant
                        std_scores.extend((group - group.mean()) / group.std())
                    else:
                        # If only one data point, set z-score to 0
                        std_scores.extend([0] * len(group))
                
                df[self.variable_mappings['standardized']['anxiety']] = std_scores
            else:
                # Population-level standardization
                df[self.variable_mappings['standardized']['anxiety']] = stats.zscore(
                    df[anxiety_raw_col], nan_policy='omit'
                )
            
            # Extract and restandardize depression score if available
            if self.variable_mappings['surreal']['depression_raw'] in df.columns:
                depression_raw_col = self.variable_mappings['surreal']['depression_raw']
                df[self.variable_mappings['standardized']['mood_raw']] = df[depression_raw_col]
                
                # Restandardize depression scores instead of using pre-standardized values
                self.logger.info("Restandardizing SURREAL depression scores")
                if self.standardization_type == 'participant':
                    # Participant-level standardization
                    participant_groups = df.groupby('participant_id')[depression_raw_col]
                    std_scores = []
                    
                    for participant_id, group in participant_groups:
                        if len(group) > 1 and group.std() > 0:
                            # Standardize within participant
                            std_scores.extend((group - group.mean()) / group.std())
                        else:
                            # If only one data point, set z-score to 0
                            std_scores.extend([0] * len(group))
                    
                    df[self.variable_mappings['standardized']['mood']] = std_scores
                else:
                    # Population-level standardization
                    df[self.variable_mappings['standardized']['mood']] = stats.zscore(
                        df[depression_raw_col], nan_policy='omit'
                    )
            
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
            
            # Add weekend status if date column is available
            if self.variable_mappings['surreal']['date'] in df.columns:
                date_col = self.variable_mappings['surreal']['date']
                try:
                    # Convert to datetime if not already
                    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                        df['date_parsed'] = pd.to_datetime(df[date_col], errors='coerce')
                    else:
                        df['date_parsed'] = df[date_col]
                    
                    # Extract day of week and determine weekend status (4=Friday, 5=Saturday)
                    df[self.variable_mappings['standardized']['is_weekend']] = df['date_parsed'].dt.dayofweek.isin([4, 5]).astype(int)
                    self.logger.info(f"Added weekend status based on {date_col}")
                    self.logger.info(f"Weekend observations: {df[self.variable_mappings['standardized']['is_weekend']].sum()} ({df[self.variable_mappings['standardized']['is_weekend']].mean():.1%})")
                except Exception as e:
                    self.logger.warning(f"Could not derive weekend status from {date_col}: {str(e)}")
                    df[self.variable_mappings['standardized']['is_weekend']] = np.nan
            else:
                self.logger.warning(f"Date column {self.variable_mappings['surreal']['date']} not found, cannot determine weekend status")
                df[self.variable_mappings['standardized']['is_weekend']] = np.nan
            
            # Extract fragmentation metrics if available
            for frag_type, col_name in self.variable_mappings['surreal']['fragmentation'].items():
                if col_name in df.columns:
                    std_col = self.variable_mappings['standardized']['fragmentation'][frag_type]
                    df[std_col] = df[col_name]
            
            # Extract duration metrics if available
            for duration_type, col_name in self.variable_mappings['surreal']['duration'].items():
                if col_name in df.columns:
                    std_col = self.variable_mappings['standardized']['duration'][duration_type]
                    df[std_col] = df[col_name]
            
            # Add dataset source identifier
            df[self.variable_mappings['standardized']['dataset']] = 'surreal'
            
            # Verify standardization worked correctly
            std_anxiety_mean = df[self.variable_mappings['standardized']['anxiety']].mean()
            std_anxiety_std = df[self.variable_mappings['standardized']['anxiety']].std()
            self.logger.info(f"Restandardized SURREAL anxiety: mean={std_anxiety_mean:.4f}, std={std_anxiety_std:.4f}")
            
            if self.variable_mappings['standardized']['mood'] in df.columns:
                std_mood_mean = df[self.variable_mappings['standardized']['mood']].mean()
                std_mood_std = df[self.variable_mappings['standardized']['mood']].std()
                self.logger.info(f"Restandardized SURREAL mood: mean={std_mood_mean:.4f}, std={std_mood_std:.4f}")
            
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
            
            # Print available columns for debugging
            self.logger.info("Available TLV columns:")
            for col in df.columns:
                self.logger.info(f"  {col}")
            
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
            
            # Add weekend status if date column is available
            if self.variable_mappings['tlv']['date'] in df.columns:
                date_col = self.variable_mappings['tlv']['date']
                try:
                    # Convert to datetime if not already
                    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                        df['date_parsed'] = pd.to_datetime(df[date_col], errors='coerce')
                    else:
                        df['date_parsed'] = df[date_col]
                    
                    # Extract day of week and determine weekend status (4=Friday, 5=Saturday)
                    df[self.variable_mappings['standardized']['is_weekend']] = df['date_parsed'].dt.dayofweek.isin([4, 5]).astype(int)
                    self.logger.info(f"Added weekend status based on {date_col}")
                    self.logger.info(f"Weekend observations: {df[self.variable_mappings['standardized']['is_weekend']].sum()} ({df[self.variable_mappings['standardized']['is_weekend']].mean():.1%})")
                    
                    # If 'is_weekend' column already exists in the original data, check for consistency
                    if 'is_weekend' in df.columns:
                        matches = (df['is_weekend'] == df[self.variable_mappings['standardized']['is_weekend']]).mean()
                        self.logger.info(f"Consistency check with existing 'is_weekend' column: {matches:.1%} match")
                except Exception as e:
                    self.logger.warning(f"Could not derive weekend status from {date_col}: {str(e)}")
                    
                    # If 'is_weekend' already exists in the dataset, use that instead
                    if 'is_weekend' in df.columns:
                        self.logger.info("Using existing 'is_weekend' column from dataset")
                        df[self.variable_mappings['standardized']['is_weekend']] = df['is_weekend']
                    else:
                        df[self.variable_mappings['standardized']['is_weekend']] = np.nan
            elif 'is_weekend' in df.columns:
                # If no date column but is_weekend exists, use the existing column
                self.logger.info("No date column found, using existing 'is_weekend' column")
                df[self.variable_mappings['standardized']['is_weekend']] = df['is_weekend']
            else:
                self.logger.warning(f"Date column {self.variable_mappings['tlv']['date']} not found, cannot determine weekend status")
                df[self.variable_mappings['standardized']['is_weekend']] = np.nan
            
            # Extract class information if available
            if 'Class' in df.columns:
                df[self.variable_mappings['standardized']['class']] = df['Class']
            
            # Extract fragmentation metrics if available
            for frag_type, col_name in self.variable_mappings['tlv']['fragmentation'].items():
                if col_name in df.columns:
                    std_col = self.variable_mappings['standardized']['fragmentation'][frag_type]
                    df[std_col] = df[col_name]
                    self.logger.info(f"Extracted {frag_type} fragmentation from {col_name}")
                else:
                    self.logger.warning(f"Missing fragmentation column: {col_name}")
            
            # Extract duration metrics if available
            for duration_type, col_name in self.variable_mappings['tlv']['duration'].items():
                if col_name in df.columns:
                    std_col = self.variable_mappings['standardized']['duration'][duration_type]
                    df[std_col] = df[col_name]
                    self.logger.info(f"Extracted {duration_type} duration from {col_name}")
                else:
                    self.logger.warning(f"Missing duration column: {col_name}")
                    
                    # Set missing out_of_home duration to NaN since it's not available in TLV
                    if duration_type == 'out_of_home':
                        df[std_col] = np.nan
            
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
        std_age_group = self.variable_mappings['standardized']['age_group']
        std_is_weekend = self.variable_mappings['standardized']['is_weekend']
        
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
        
        # Log demographic summaries at observation level
        if std_gender in df.columns:
            gender_counts = df[std_gender].value_counts()
            self.logger.info(f"  Gender distribution (observations):")
            for gender, count in gender_counts.items():
                self.logger.info(f"    {gender}: {count} ({count/n_observations:.1%})")
        
            # Add participant-level gender counts
            gender_participant_counts = df.groupby(std_gender)[std_id].nunique()
            self.logger.info(f"  Gender distribution (participants):")
            for gender, count in gender_participant_counts.items():
                self.logger.info(f"    {gender}: {count} ({count/n_participants:.1%})")
            
        if std_location in df.columns:
            location_counts = df[std_location].value_counts()
            self.logger.info(f"  Location distribution (observations):")
            for location, count in location_counts.items():
                self.logger.info(f"    {location}: {count} ({count/n_observations:.1%})")
            
            # Add participant-level location counts
            location_participant_counts = df.groupby(std_location)[std_id].nunique()
            self.logger.info(f"  Location distribution (participants):")
            for location, count in location_participant_counts.items():
                self.logger.info(f"    {location}: {count} ({count/n_participants:.1%})")
        
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

        # Log duration metrics if available
        for duration_type, std_col in self.variable_mappings['standardized']['duration'].items():
            if std_col in df.columns and df[std_col].notna().sum() > 0:
                self.logger.info(f"  {duration_type.capitalize()} duration (minutes):")
                self.logger.info(f"    Mean: {df[std_col].mean():.1f}")
                self.logger.info(f"    Std: {df[std_col].std():.1f}")
                self.logger.info(f"    Min: {df[std_col].min():.1f}")
                self.logger.info(f"    Max: {df[std_col].max():.1f}")
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
                self.variable_mappings['standardized']['age_group'],
                self.variable_mappings['standardized']['is_weekend']
            ]
            
            # Add fragmentation columns if available
            for std_col in self.variable_mappings['standardized']['fragmentation'].values():
                if (std_col in surreal_df.columns) or (std_col in tlv_df.columns):
                    if std_col not in std_columns:
                        std_columns.append(std_col)
            
            # Add duration columns if available
            for std_col in self.variable_mappings['standardized']['duration'].values():
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
            std_is_weekend = self.variable_mappings['standardized']['is_weekend']
            std_dataset = self.variable_mappings['standardized']['dataset']
            
            # Get duration column names
            duration_cols = [col for col in self.pooled_data.columns 
                            if col in self.variable_mappings['standardized']['duration'].values()]
            
            # Get fragmentation column names
            frag_cols = [col for col in self.pooled_data.columns
                        if col in self.variable_mappings['standardized']['fragmentation'].values()]
            
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
            
            # Check duration distributions 
            self.logger.info("Duration metrics distributions:")
            for col in duration_cols:
                valid_vals = self.pooled_data[col].dropna()
                if len(valid_vals) > 0:
                    col_name = col.replace('_', ' ').title()
                    self.logger.info(f"  {col_name}:")
                    self.logger.info(f"    Mean: {valid_vals.mean():.2f} minutes")
                    self.logger.info(f"    Median: {valid_vals.median():.2f} minutes")
                    self.logger.info(f"    Std: {valid_vals.std():.2f} minutes")
                    self.logger.info(f"    Min: {valid_vals.min():.2f} minutes")
                    self.logger.info(f"    Max: {valid_vals.max():.2f} minutes")
                    
                    # Check by dataset
                    for dataset in self.pooled_data[std_dataset].unique():
                        dataset_data = self.pooled_data[self.pooled_data[std_dataset] == dataset]
                        valid_ds_vals = dataset_data[col].dropna()
                        if len(valid_ds_vals) > 0:
                            self.logger.info(f"    {dataset.upper()}:")
                            self.logger.info(f"      Mean: {valid_ds_vals.mean():.2f} minutes")
                            self.logger.info(f"      Median: {valid_ds_vals.median():.2f} minutes")
                            self.logger.info(f"      Valid values: {len(valid_ds_vals)}")

            # Check for correlation between duration and fragmentation
            self.logger.info("Correlations between duration and fragmentation:")
            for frag_type in ['digital', 'mobility', 'overlap']:
                frag_col = self.variable_mappings['standardized']['fragmentation'][frag_type]
                dur_col = self.variable_mappings['standardized']['duration'][frag_type]
                
                if frag_col in self.pooled_data.columns and dur_col in self.pooled_data.columns:
                    valid_data = self.pooled_data[[frag_col, dur_col]].dropna()
                    if len(valid_data) > 5:
                        corr = valid_data.corr().iloc[0, 1]
                        self.logger.info(f"  {frag_type.title()}: r = {corr:.3f} (n={len(valid_data)})")
                        
                        # Check by dataset
                        for dataset in self.pooled_data[std_dataset].unique():
                            dataset_data = self.pooled_data[self.pooled_data[std_dataset] == dataset]
                            valid_ds_data = dataset_data[[frag_col, dur_col]].dropna()
                            if len(valid_ds_data) > 5:
                                ds_corr = valid_ds_data.corr().iloc[0, 1]
                                self.logger.info(f"    {dataset}: r = {ds_corr:.3f} (n={len(valid_ds_data)})")
            
            # Check for correlation between duration and anxiety
            self.logger.info("Correlations between duration and anxiety:")
            for dur_type in ['digital', 'mobility', 'overlap']:
                dur_col = self.variable_mappings['standardized']['duration'][dur_type]
                
                if dur_col in self.pooled_data.columns:
                    valid_data = self.pooled_data[[std_anxiety, dur_col]].dropna()
                    if len(valid_data) > 5:
                        corr = valid_data.corr().iloc[0, 1]
                        self.logger.info(f"  {dur_type.title()}: r = {corr:.3f} (n={len(valid_data)})")
                        
                        # Check by dataset
                        for dataset in self.pooled_data[std_dataset].unique():
                            dataset_data = self.pooled_data[self.pooled_data[std_dataset] == dataset]
                            valid_ds_data = dataset_data[[std_anxiety, dur_col]].dropna()
                            if len(valid_ds_data) > 5:
                                ds_corr = valid_ds_data.corr().iloc[0, 1]
                                self.logger.info(f"    {dataset}: r = {ds_corr:.3f} (n={len(valid_ds_data)})")
            
            # NEW: Add specific checks for the new digital home metrics
            digital_home_frag = self.variable_mappings['standardized']['fragmentation']['digital_home']
            digital_home_mobility_delta = self.variable_mappings['standardized']['fragmentation']['digital_home_mobility_delta']
            
            if digital_home_frag in self.pooled_data.columns:
                valid_home_frag = self.pooled_data[digital_home_frag].dropna()
                if len(valid_home_frag) > 0:
                    self.logger.info(f"\nDigital Home Fragmentation Statistics:")
                    self.logger.info(f"  Mean: {valid_home_frag.mean():.4f}")
                    self.logger.info(f"  Median: {valid_home_frag.median():.4f}")
                    self.logger.info(f"  Std: {valid_home_frag.std():.4f}")
                    self.logger.info(f"  Valid values: {len(valid_home_frag)} ({len(valid_home_frag)/len(self.pooled_data):.1%})")
                    
                    # Check by dataset
                    for dataset in self.pooled_data[std_dataset].unique():
                        dataset_data = self.pooled_data[self.pooled_data[std_dataset] == dataset]
                        valid_ds_vals = dataset_data[digital_home_frag].dropna()
                        if len(valid_ds_vals) > 0:
                            self.logger.info(f"    {dataset.upper()}:")
                            self.logger.info(f"      Mean: {valid_ds_vals.mean():.4f}")
                            self.logger.info(f"      Median: {valid_ds_vals.median():.4f}")
                            self.logger.info(f"      Valid values: {len(valid_ds_vals)} ({len(valid_ds_vals)/len(dataset_data):.1%})")
            
            if digital_home_mobility_delta in self.pooled_data.columns:
                valid_delta = self.pooled_data[digital_home_mobility_delta].dropna()
                if len(valid_delta) > 0:
                    self.logger.info(f"\nDigital Home-Mobility Delta Statistics:")
                    self.logger.info(f"  Mean: {valid_delta.mean():.4f}")
                    self.logger.info(f"  Median: {valid_delta.median():.4f}")
                    self.logger.info(f"  Std: {valid_delta.std():.4f}")
                    self.logger.info(f"  Positive values (home > mobility): {(valid_delta > 0).sum()} ({(valid_delta > 0).sum()/len(valid_delta):.1%})")
                    self.logger.info(f"  Negative values (mobility > home): {(valid_delta < 0).sum()} ({(valid_delta < 0).sum()/len(valid_delta):.1%})")
                    
                    # Check by dataset
                    for dataset in self.pooled_data[std_dataset].unique():
                        dataset_data = self.pooled_data[self.pooled_data[std_dataset] == dataset]
                        valid_ds_vals = dataset_data[digital_home_mobility_delta].dropna()
                        if len(valid_ds_vals) > 0:
                            self.logger.info(f"    {dataset.upper()}:")
                            self.logger.info(f"      Mean: {valid_ds_vals.mean():.4f}")
                            self.logger.info(f"      Median: {valid_ds_vals.median():.4f}")
                            self.logger.info(f"      Valid values: {len(valid_ds_vals)} ({len(valid_ds_vals)/len(dataset_data):.1%})")
            
            # NEW: Check relationship between digital home and overlap fragmentation
            if digital_home_frag in self.pooled_data.columns and 'overlap_fragmentation' in self.pooled_data.columns:
                valid_data = self.pooled_data[[digital_home_frag, 'overlap_fragmentation']].dropna()
                if len(valid_data) > 5:
                    corr = valid_data.corr().iloc[0, 1]
                    self.logger.info(f"\nCorrelation between digital home and overlap fragmentation: r = {corr:.3f} (n={len(valid_data)})")
                    
                    # Check by dataset
                    for dataset in self.pooled_data[std_dataset].unique():
                        dataset_data = self.pooled_data[self.pooled_data[std_dataset] == dataset]
                        valid_ds_data = dataset_data[[digital_home_frag, 'overlap_fragmentation']].dropna()
                        if len(valid_ds_data) > 5:
                            ds_corr = valid_ds_data.corr().iloc[0, 1]
                            self.logger.info(f"  {dataset}: r = {ds_corr:.3f} (n={len(valid_ds_data)})")
            
            # Continue with rest of quality checks...
            
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
            std_age_group = self.variable_mappings['standardized']['age_group']
            std_is_weekend = self.variable_mappings['standardized']['is_weekend']
            
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
            
            # Gender distribution - observations
            self.logger.info("\nGENDER DISTRIBUTION (OBSERVATIONS):")
            genders = self.pooled_data[std_gender].value_counts()
            for gender, count in genders.items():
                gender_pct = 100 * count / total_obs
                self.logger.info(f"  {gender.upper()}: {count} obs ({gender_pct:.1f}%)")
            
            # Gender distribution - participants
            self.logger.info("\nGENDER DISTRIBUTION (PARTICIPANTS):")
            gender_participants = self.pooled_data.groupby(std_gender)[std_id].nunique()
            for gender, count in gender_participants.items():
                gender_pct = 100 * count / total_participants
                self.logger.info(f"  {gender.upper()}: {count} participants ({gender_pct:.1f}%)")
            
            # Location distribution - observations
            self.logger.info("\nLOCATION DISTRIBUTION (OBSERVATIONS):")
            locations = self.pooled_data[std_location].value_counts()
            for location, count in locations.items():
                location_pct = 100 * count / total_obs
                self.logger.info(f"  {location.upper()}: {count} obs ({location_pct:.1f}%)")
            
            # Location distribution - participants
            self.logger.info("\nLOCATION DISTRIBUTION (PARTICIPANTS):")
            location_participants = self.pooled_data.groupby(std_location)[std_id].nunique()
            for location, count in location_participants.items():
                location_pct = 100 * count / total_participants
                self.logger.info(f"  {location.upper()}: {count} participants ({location_pct:.1f}%)")
            
            # Age group distribution
            self.logger.info("\nAGE GROUP DISTRIBUTION:")
            age_groups = self.pooled_data[std_age_group].value_counts()
            for age, count in age_groups.items():
                age_pct = 100 * count / total_obs
                self.logger.info(f"  {age.upper()}: {count} obs ({age_pct:.1f}%)")
            
            # Data completeness
            self.logger.info("\nDATA COMPLETENESS:")
            frag_types = ['digital', 'mobility', 'overlap', 'digital_home', 'digital_home_mobility_delta']
            for frag_type in frag_types:
                if frag_type in self.variable_mappings['standardized']['fragmentation']:
                    col = self.variable_mappings['standardized']['fragmentation'][frag_type]
                    if col in self.pooled_data.columns:
                        valid = self.pooled_data[col].notna().sum()
                        valid_pct = 100 * valid / total_obs
                        self.logger.info(f"  {frag_type.upper()} FRAGMENTATION: {valid}/{total_obs} ({valid_pct:.1f}%)")
            
            # Duration completeness
            self.logger.info("\nDURATION COMPLETENESS:")
            duration_types = ['digital', 'mobility', 'overlap', 'digital_home']
            for duration_type in duration_types:
                if duration_type in self.variable_mappings['standardized']['duration']:
                    col = self.variable_mappings['standardized']['duration'][duration_type]
                    if col in self.pooled_data.columns:
                        valid = self.pooled_data[col].notna().sum()
                        valid_pct = 100 * valid / total_obs
                        self.logger.info(f"  {duration_type.upper()} DURATION: {valid}/{total_obs} ({valid_pct:.1f}%)")
                        if valid > 0:
                            self.logger.info(f"    Mean: {self.pooled_data[col].mean():.1f} minutes")
                            self.logger.info(f"    Min: {self.pooled_data[col].min():.1f} minutes") 
                            self.logger.info(f"    Max: {self.pooled_data[col].max():.1f} minutes")
            
            # NEW: Add specific report for digital home metrics
            self.logger.info("\nDIGITAL HOME FRAGMENTATION METRICS:")
            digital_home_frag = self.variable_mappings['standardized']['fragmentation']['digital_home']
            digital_home_mobility_delta = self.variable_mappings['standardized']['fragmentation']['digital_home_mobility_delta']
            
            if digital_home_frag in self.pooled_data.columns:
                valid_vals = self.pooled_data[digital_home_frag].dropna()
                valid_pct = 100 * len(valid_vals) / total_obs
                self.logger.info(f"  Available data: {len(valid_vals)}/{total_obs} ({valid_pct:.1f}%)")
                if len(valid_vals) > 0:
                    self.logger.info(f"  Mean: {valid_vals.mean():.4f}")
                    self.logger.info(f"  Median: {valid_vals.median():.4f}")
            
            if digital_home_mobility_delta in self.pooled_data.columns:
                valid_vals = self.pooled_data[digital_home_mobility_delta].dropna()
                valid_pct = 100 * len(valid_vals) / total_obs
                self.logger.info(f"\nDIGITAL HOME-MOBILITY DELTA:")
                self.logger.info(f"  Available data: {len(valid_vals)}/{total_obs} ({valid_pct:.1f}%)")
                if len(valid_vals) > 0:
                    self.logger.info(f"  Mean: {valid_vals.mean():.4f}")
                    self.logger.info(f"  Median: {valid_vals.median():.4f}")
                    self.logger.info(f"  Positive values (home > mobility): {(valid_vals > 0).sum()} ({(valid_vals > 0).sum()/len(valid_vals):.1%})")
                    self.logger.info(f"  Negative values (mobility > home): {(valid_vals < 0).sum()} ({(valid_vals < 0).sum()/len(valid_vals):.1%})")
            
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
            
            self.logger.info("\nWEEKEND STATUS DISTRIBUTION:")
            if self.variable_mappings['standardized']['is_weekend'] in self.pooled_data.columns:
                weekend_counts = self.pooled_data[self.variable_mappings['standardized']['is_weekend']].value_counts()
                for status, count in weekend_counts.items():
                    status_label = "Weekend" if status == 1 else "Weekday"
                    status_pct = 100 * count / total_obs
                    self.logger.info(f"  {status_label}: {count} obs ({status_pct:.1f}%)")
                
                # Weekend distribution by dataset
                self.logger.info("\nWEEKEND STATUS BY DATASET:")
                for dataset in datasets:
                    ds_data = self.pooled_data[self.pooled_data[std_dataset] == dataset]
                    if len(ds_data) > 0 and not ds_data[self.variable_mappings['standardized']['is_weekend']].isna().all():
                        weekend_count = ds_data[self.variable_mappings['standardized']['is_weekend']].sum()
                        weekday_count = len(ds_data) - weekend_count
                        weekend_pct = 100 * weekend_count / len(ds_data) if len(ds_data) > 0 else 0
                        self.logger.info(f"  {dataset.upper()}:")
                        self.logger.info(f"    Weekend: {weekend_count} obs ({weekend_pct:.1f}%)")
                        self.logger.info(f"    Weekday: {weekday_count} obs ({100-weekend_pct:.1f}%)")
            else:
                self.logger.info("  No weekend status data available")
            
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