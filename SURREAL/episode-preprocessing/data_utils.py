#!/usr/bin/env python3
"""
Data Cleaning and Standardization Utilities

This module provides tools for standardizing participant IDs,
timestamps, and ensuring data consistency throughout the
SURREAL data processing pipeline.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Set, Tuple, Union, Optional
import re


class DataCleaner:
    """
    A class to standardize and clean data throughout the processing pipeline.
    """
    def __init__(self, logger=None):
        """Initialize with optional logger"""
        self.logger = logger or logging.getLogger(__name__)
        
    def standardize_participant_id(self, participant_id):
        """
        Universal standardization for participant IDs to ensure consistent matching.
        
        Args:
            participant_id: Original participant ID in any format
            
        Returns:
            Standardized participant ID as string
        """
        # Handle null/NaN values
        if pd.isna(participant_id):
            return ""
        
        # Convert to string and strip whitespace
        pid = str(participant_id).strip()
        
        # Remove common prefixes (case insensitive)
        prefixes = ['surreal', 'surreal_', 'surreal-', 'p_', 'p', 'pilot_']
        pid_lower = pid.lower()
        for prefix in prefixes:
            if pid_lower.startswith(prefix):
                pid = pid[len(prefix):]
                break
        
        # Remove 'p' suffix if present
        if pid.lower().endswith('p'):
            pid = pid[:-1]
        
        # Extract numeric part using regex
        numeric_part = re.search(r'(\d+)', pid)
        if numeric_part:
            # Get the numeric portion
            number = numeric_part.group(1)
            # Remove leading zeros
            number = number.lstrip('0')
            if not number:
                number = '0'
            return number
        
        # If no digits found, return cleaned original
        return ''.join(c for c in pid if c.isalnum())
    
    def standardize_dataframe_ids(self, df, id_column):
        """
        Apply standardization to ID column and add clean_id column.
        
        Args:
            df: DataFrame with ID column
            id_column: Name of ID column to standardize
            
        Returns:
            DataFrame with added 'participant_id_clean' column
        """
        if df is None or df.empty or id_column not in df.columns:
            self.logger.warning(f"Cannot standardize IDs: DataFrame is empty or missing column '{id_column}'")
            return df
            
        df_copy = df.copy()
        df_copy['participant_id_clean'] = df_copy[id_column].apply(self.standardize_participant_id)
        
        # Log the ID mapping for debugging
        id_mapping = df_copy[[id_column, 'participant_id_clean']].drop_duplicates()
        self.logger.info(f"ID standardization mapping ({len(id_mapping)} IDs):")
        for _, row in id_mapping.head(10).iterrows():
            self.logger.info(f"  {row[id_column]} -> {row['participant_id_clean']}")
        
        return df_copy
    
    def standardize_timestamps(self, df, datetime_columns):
        """
        Ensure all timestamps are standardized to naive datetime format.
        
        Args:
            df: DataFrame containing timestamp columns
            datetime_columns: List of column names containing datetime data
            
        Returns:
            DataFrame with standardized timestamps
        """
        if df is None or df.empty:
            return df
            
        df_copy = df.copy()
        
        for col in datetime_columns:
            if col in df_copy.columns:
                # First ensure it's datetime
                if not pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                    try:
                        df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
                        self.logger.info(f"Converted column '{col}' to datetime")
                    except Exception as e:
                        self.logger.warning(f"Could not convert column '{col}' to datetime: {str(e)}")
                        continue
                
                # Then make timezone naive if it has timezone info
                if not df_copy[col].empty and hasattr(df_copy[col].iloc[0], 'tz') and df_copy[col].iloc[0].tz is not None:
                    df_copy[col] = df_copy[col].dt.tz_localize(None)
                    self.logger.info(f"Removed timezone info from column '{col}'")
                    
                # Add date string column for each datetime
                date_col = f"{col}_date"
                df_copy[date_col] = df_copy[col].dt.date.astype(str)
                self.logger.info(f"Added date string column '{date_col}'")
        
        return df_copy
    
    def standardize_missing_values(self, df, numeric_columns=None):
        """
        Apply consistent missing value handling.
        
        Args:
            df: DataFrame to process
            numeric_columns: List of numeric columns to standardize, if None will detect automatically
        
        Returns:
            DataFrame with standardized missing values
        """
        if df is None or df.empty:
            return df
            
        df_copy = df.copy()
        
        # Auto-detect numeric columns if not specified
        if numeric_columns is None:
            numeric_columns = df_copy.select_dtypes(include=['number']).columns.tolist()
        
        # Handle numeric columns - replace None, "", etc. with NaN
        for col in numeric_columns:
            if col in df_copy.columns:
                try:
                    # Replace empty strings and None values
                    old_missing = df_copy[col].isna().sum()
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                    new_missing = df_copy[col].isna().sum()
                    
                    if new_missing > old_missing:
                        self.logger.info(f"Standardized '{col}': {new_missing - old_missing} additional missing values identified")
                    
                    # Set specific values to NaN (like 999, -999 etc. used as missing value indicators)
                    extreme_values = [999, -999, 9999, -9999]
                    for val in extreme_values:
                        mask = df_copy[col] == val
                        if mask.sum() > 0:
                            df_copy.loc[mask, col] = np.nan
                            self.logger.info(f"Converted {mask.sum()} instances of {val} to NaN in '{col}'")
                except Exception as e:
                    self.logger.warning(f"Error standardizing column '{col}': {str(e)}")
        
        return df_copy
    
    def merge_datasets(self, left_df, right_df, on=None, left_on=None, right_on=None, how='inner'):
        """
        Merge datasets with consistent logging and error handling.
        
        Args:
            left_df: Left DataFrame
            right_df: Right DataFrame
            on: Column(s) to join on
            left_on: Left DataFrame column(s)
            right_on: Right DataFrame column(s)
            how: Join method ('inner', 'left', 'right', 'outer')
            
        Returns:
            Merged DataFrame
        """
        if left_df is None or left_df.empty:
            self.logger.error("Left DataFrame is empty")
            return left_df
            
        if right_df is None or right_df.empty:
            self.logger.error("Right DataFrame is empty")
            return left_df
        
        # Log information about merge
        self.logger.info(f"Merging datasets: left={left_df.shape}, right={right_df.shape}, how='{how}'")
        
        # Extract participant IDs for debug logging
        if on is not None and isinstance(on, str) and on in left_df.columns and on in right_df.columns:
            left_participants = set(left_df[on].unique())
            right_participants = set(right_df[on].unique())
            common_participants = left_participants.intersection(right_participants)
            
            self.logger.info(f"Left dataset has {len(left_participants)} unique values in '{on}'")
            self.logger.info(f"Right dataset has {len(right_participants)} unique values in '{on}'")
            self.logger.info(f"Found {len(common_participants)} common values")
            
            if len(common_participants) == 0:
                self.logger.warning("No common values for joining - merge will produce empty result")
                # Debug sample values
                self.logger.info(f"Sample values in left: {sorted(list(left_participants))[:5]}")
                self.logger.info(f"Sample values in right: {sorted(list(right_participants))[:5]}")
        
        try:
            # Attempt merge
            merged_df = pd.merge(left_df, right_df, on=on, left_on=left_on, right_on=right_on, how=how)
            self.logger.info(f"Merged result: {merged_df.shape} rows")
            
            if merged_df.empty and how in ['inner', 'left']:
                self.logger.warning("Merge produced empty DataFrame - check join conditions")
            
            return merged_df
            
        except Exception as e:
            self.logger.error(f"Error merging datasets: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return left_df
    
    def validate_data(self, df, validation_rules=None):
        """
        Validate DataFrame according to specified rules.
        
        Args:
            df: DataFrame to validate
            validation_rules: Dictionary of validation rules
            
        Returns:
            Validated DataFrame
        """
        if df is None or df.empty:
            return df
            
        if validation_rules is None:
            validation_rules = {
                'required_columns': [],
                'numeric_ranges': {},  # e.g., {'age': (18, 100)}
                'allowed_values': {},  # e.g., {'gender_code': [0, 1]}
            }
        
        report = {
            'missing_columns': [],
            'out_of_range_values': {},
            'invalid_values': {},
            'rows_before': len(df),
            'rows_after': 0,
        }
        
        # Check required columns
        for col in validation_rules.get('required_columns', []):
            if col not in df.columns:
                report['missing_columns'].append(col)
        
        # If required columns are missing, return early
        if report['missing_columns']:
            self.logger.warning(f"Missing required columns: {report['missing_columns']}")
            return df
        
        valid_df = df.copy()
        
        # Check numeric ranges
        for col, (min_val, max_val) in validation_rules.get('numeric_ranges', {}).items():
            if col in valid_df.columns:
                invalid_mask = (valid_df[col] < min_val) | (valid_df[col] > max_val)
                invalid_count = invalid_mask.sum()
                
                if invalid_count > 0:
                    report['out_of_range_values'][col] = invalid_count
                    valid_df.loc[invalid_mask, col] = np.nan
                    self.logger.warning(f"Column '{col}': {invalid_count} values outside range [{min_val}, {max_val}]")
        
        # Check allowed values
        for col, allowed in validation_rules.get('allowed_values', {}).items():
            if col in valid_df.columns:
                invalid_mask = ~valid_df[col].isin(allowed)
                invalid_count = invalid_mask.sum()
                
                if invalid_count > 0:
                    report['invalid_values'][col] = invalid_count
                    valid_df.loc[invalid_mask, col] = np.nan
                    self.logger.warning(f"Column '{col}': {invalid_count} values not in allowed set {allowed}")
        
        report['rows_after'] = len(valid_df)
        
        # Log validation summary
        self.logger.info(f"Data validation complete: {len(valid_df)} rows validated")
        
        if report['out_of_range_values'] or report['invalid_values']:
            self.logger.warning("Validation found issues:")
            
            for col, count in report['out_of_range_values'].items():
                self.logger.warning(f"  Column '{col}': {count} out-of-range values")
        
            for col, count in report['invalid_values'].items():
                self.logger.warning(f"  Column '{col}': {count} invalid values")
        
        return valid_df

    def clean_and_standardize(self, df, id_column, datetime_columns=None, numeric_columns=None):
        """
        Apply complete standardization pipeline to a DataFrame.
        
        Args:
            df: DataFrame to clean
            id_column: Name of participant ID column
            datetime_columns: List of datetime columns (if None, will try to auto-detect)
            numeric_columns: List of numeric columns (if None, will use all numeric columns)
            
        Returns:
            Cleaned and standardized DataFrame
        """
        if df is None or df.empty:
            return df
            
        # Auto-detect datetime columns if not specified
        if datetime_columns is None:
            datetime_columns = [col for col in df.columns 
                              if any(term in col.lower() for term in ['time', 'date', 'timestamp', 'datetime'])]
        
        # 1. Standardize IDs
        self.logger.info(f"Standardizing participant IDs from column '{id_column}'")
        std_df = self.standardize_dataframe_ids(df, id_column)
        
        # 2. Standardize timestamps
        if datetime_columns:
            self.logger.info(f"Standardizing timestamps in columns: {datetime_columns}")
            std_df = self.standardize_timestamps(std_df, datetime_columns)
        
        # 3. Standardize missing values
        self.logger.info("Standardizing missing values")
        std_df = self.standardize_missing_values(std_df, numeric_columns)
        
        return std_df 