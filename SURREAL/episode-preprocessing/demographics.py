#!/usr/bin/env python3
"""
Demographic Data Merger

This script merges demographic data from the participant_info.xlsx file 
with the fragmentation metrics and EMA data.

Usage:
    python demographics.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
from datetime import datetime
import re

def setup_logging(output_dir):
    """Set up logging configuration"""
    log_dir = Path(output_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'demographic_merge_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def clean_participant_id(participant_id):
    """
    Standardize participant ID format for matching.
    
    Args:
        participant_id: Original participant ID
        
    Returns:
        Standardized participant ID
    """
    # Handle NaN values
    if pd.isna(participant_id):
        return ""
    
    # Convert to string
    pid = str(participant_id).strip()
    
    # Remove common prefixes
    for prefix in ['surreal', 'surreal_', 'surreal-', 'p_', 'p']:
        if pid.lower().startswith(prefix):
            pid = pid[len(prefix):]
    
    # Remove 'p' suffix if present
    if pid.lower().endswith('p'):
        pid = pid[:-1]
    
    # Extract numeric part
    numeric_part = re.search(r'(\d+)', pid)
    if numeric_part:
        # Get the numeric portion
        number = numeric_part.group(1)
        # Remove leading zeros
        number = number.lstrip('0')
        if not number:
            number = '0'
        return number
    
    return pid

def load_demographics(demographics_path):
    """
    Load and process demographic data.
    
    Args:
        demographics_path (str): Path to demographics file
        
    Returns:
        pd.DataFrame: Processed demographics
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading demographics from {demographics_path}")
    
    try:
        # First, load the file to inspect columns
        demographics = pd.read_excel(demographics_path)
        logger.info(f"Loaded demographics with shape: {demographics.shape}")
        logger.info(f"Columns in demographics file: {demographics.columns.tolist()}")
        
        # If the first row is sample data, and the first column contains participant IDs
        # Extract first row for inspection
        first_row = demographics.iloc[0].to_dict()
        logger.info(f"First row data: {first_row}")
        
        # Check if we need to rename columns
        # Assuming the first column is the participant ID
        first_col = demographics.columns[0]
        
        # Rename the first column to Participant_ID if needed
        if first_col != 'Participant_ID':
            logger.info(f"Renaming first column from '{first_col}' to 'Participant_ID'")
            demographics = demographics.rename(columns={first_col: 'Participant_ID'})
        
        # Handle column renames for other expected columns
        column_mapping = {
            'Start.day': 'start_date',
            'End.day': 'end_date',
            # Add other columns as needed
        }
        
        # Apply column renames where columns exist
        for old_name, new_name in column_mapping.items():
            if old_name in demographics.columns:
                demographics = demographics.rename(columns={old_name: new_name})
                
        # Log the updated columns
        logger.info(f"Updated columns: {demographics.columns.tolist()}")
        
        # Process age data
        if 'age' not in demographics.columns and 'DOB' in demographics.columns:
            logger.info("Calculating age from DOB")
            try:
                # Parse DOB - handle various formats (Aug-97, 1997-08-15, etc.)
                dob_series = demographics['DOB']
                
                # Try standard datetime conversion first
                demographics['DOB'] = pd.to_datetime(dob_series, errors='coerce')
                
                # If that fails for many values, try parsing custom formats
                if demographics['DOB'].isna().sum() > len(dob_series) * 0.5:
                    # Reset and try to parse month-year format (e.g., Aug-97)
                    month_year_pattern = r'([A-Za-z]{3})-(\d{2})'
                    
                    def parse_month_year(dob_str):
                        if pd.isna(dob_str):
                            return pd.NaT
                        
                        # Handle standard dates
                        try:
                            return pd.to_datetime(dob_str)
                        except:
                            pass
                        
                        # Try Month-YY format
                        match = re.match(month_year_pattern, str(dob_str))
                        if match:
                            month_str = match.group(1)
                            year_str = match.group(2)
                            # Parse month name to number
                            try:
                                month_num = datetime.strptime(month_str, '%b').month
                                # Assume 19xx for years > 50, 20xx for years < 50
                                year_full = 1900 + int(year_str) if int(year_str) > 50 else 2000 + int(year_str)
                                return pd.Timestamp(year=year_full, month=month_num, day=15)  # Use middle of month
                            except:
                                return pd.NaT
                        
                        return pd.NaT
                    
                    demographics['DOB'] = dob_series.apply(parse_month_year)
                
                # Calculate age as of reference year (2023)
                reference_year = 2023
                demographics['calculated_age'] = reference_year - demographics['DOB'].dt.year
                
                # Use calculated age or existing age column
                if 'age' in demographics.columns:
                    # Fill missing values in 'age' with calculated values
                    demographics['age'] = demographics['age'].fillna(demographics['calculated_age'])
                else:
                    demographics['age'] = demographics['calculated_age']
                
                # Drop temporary column
                if 'calculated_age' in demographics.columns:
                    demographics = demographics.drop(columns=['calculated_age'])
                    
            except Exception as e:
                logger.error(f"Error calculating age: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                # Create an empty age column if needed
                if 'age' not in demographics.columns:
                    demographics['age'] = np.nan
        
        # Process gender data
        if 'Gender' in demographics.columns:
            # Standardize gender coding
            demographics['gender_code'] = demographics['Gender'].str.upper().map({
                'M': 0, 'MALE': 0, 'MAN': 0, 'BOY': 0,
                'F': 1, 'FEMALE': 1, 'WOMAN': 1, 'GIRL': 1
            })
            
        # Add standardized ID column for matching
        demographics['participant_id_clean'] = demographics['Participant_ID'].apply(clean_participant_id)
        
        # Log examples of ID standardization
        id_samples = demographics[['Participant_ID', 'participant_id_clean']].head(10)
        logger.info(f"ID standardization examples:\n{id_samples}")
        
        # Log summary statistics
        logger.info(f"Processed demographics for {len(demographics)} participants")
        if 'age' in demographics.columns:
            age_mean = demographics['age'].mean()
            age_std = demographics['age'].std()
            logger.info(f"Age: mean={age_mean:.1f}, std={age_std:.1f}")
        
        if 'gender_code' in demographics.columns:
            gender_counts = demographics['Gender'].value_counts()
            logger.info(f"Gender distribution: {gender_counts.to_dict()}")
            
        return demographics
        
    except Exception as e:
        logger.error(f"Error loading demographics: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()  # Return empty DataFrame on error

def merge_demographics_with_data(data_path, demographics_df, output_path):
    """
    Merge demographics with fragmentation or EMA data.
    
    Args:
        data_path (str): Path to data file (CSV)
        demographics_df (pd.DataFrame): Demographics data
        output_path (str): Path to save merged data
        
    Returns:
        pd.DataFrame: Merged data
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Merging demographics with data from {data_path}")
    
    try:
        # Load the data
        data = pd.read_csv(data_path)
        logger.info(f"Loaded data with shape: {data.shape}")
        logger.info(f"Columns in data: {data.columns.tolist()}")
        
        # Show a sample of participant IDs from the data
        id_sample = []
        for col in data.columns:
            if 'participant' in col.lower() or 'id' in col.lower():
                id_sample.append((col, data[col].head(3).tolist()))
        logger.info(f"Participant ID columns and samples: {id_sample}")
        
        # Identify the participant ID column in the data
        id_columns = ['participant_id', 'participant_id_ema', 'participant_id_frag', 'Participant_ID', 'user', 'subject', 'id']
        found_col = False
        
        for col in id_columns:
            if col in data.columns:
                id_col = col
                logger.info(f"Using '{id_col}' as participant identifier")
                found_col = True
                break
        
        if not found_col:
            # Try to find any column with 'participant' or 'id' in the name
            for col in data.columns:
                if 'participant' in col.lower() or 'id' in col.lower():
                    id_col = col
                    logger.info(f"Using '{id_col}' as participant identifier (based on name matching)")
                    found_col = True
                    break
        
        if not found_col:
            logger.error(f"Could not identify participant ID column in data")
            return data  # Return original data
        
        # Create standardized ID for matching
        data['participant_id_clean'] = data[id_col].apply(clean_participant_id)
        
        # Log some examples of the ID cleaning for debugging
        sample_ids = data[id_col].head(10).tolist()
        cleaned_ids = data['participant_id_clean'].head(10).tolist()
        logger.info(f"ID cleaning examples: {list(zip(sample_ids, cleaned_ids))}")
        
        # Log ID distributions before merging
        logger.info(f"Data contains {data['participant_id_clean'].nunique()} unique participants")
        logger.info(f"Demographics contains {demographics_df['participant_id_clean'].nunique()} unique participants")
        
        # Find common participants
        data_participants = set(data['participant_id_clean'].unique())
        demo_participants = set(demographics_df['participant_id_clean'].unique())
        common_participants = data_participants.intersection(demo_participants)
        
        logger.info(f"Found {len(common_participants)} common participants")
        logger.info(f"Common participants: {sorted(list(common_participants))}")
        logger.info(f"Participants in data but not demographics: {sorted(list(data_participants - demo_participants))}")
        logger.info(f"Participants in demographics but not data: {sorted(list(demo_participants - data_participants))}")
        
        # Merge data
        merged_data = pd.merge(
            data,
            demographics_df,
            on='participant_id_clean',
            how='left',
            suffixes=('', '_demo')
        )
        
        logger.info(f"Merged data shape: {merged_data.shape}")
        
        # Check how many participants got demographic data
        has_demo = 0
        demo_check_col = 'age'
        if demo_check_col in merged_data.columns:
            has_demo = merged_data[demo_check_col].notna().sum()
            logger.info(f"Records with demographic data: {has_demo} of {len(merged_data)} ({has_demo/len(merged_data)*100:.1f}%)")
        else:
            logger.warning(f"Column '{demo_check_col}' not found in merged data. Cannot check demographic coverage.")
            
        # Drop duplicate ID columns and standardized ID
        drop_cols = [col for col in merged_data.columns if col.endswith('_demo') or col == 'participant_id_clean']
        merged_data = merged_data.drop(columns=drop_cols)
        
        # Save merged data
        merged_data.to_csv(output_path, index=False)
        logger.info(f"Saved merged data to {output_path}")
        
        return merged_data
        
    except Exception as e:
        logger.error(f"Error merging demographics: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()  # Return empty DataFrame on error

def main():
    """Main execution function"""
    # Get the SURREAL base directory
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find the SURREAL folder path
    # This assumes the script is run from within the SURREAL project structure
    surreal_path = None
    current_dir = script_dir
    while True:
        if os.path.basename(current_dir) == 'SURREAL' or os.path.exists(os.path.join(current_dir, 'data')) and os.path.exists(os.path.join(current_dir, 'processed')):
            surreal_path = current_dir
            break
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # Reached root directory
            break
        current_dir = parent_dir
    
    if not surreal_path:
        # Fallback to hardcoded path if we couldn't determine it automatically
        surreal_path = '/Users/noamgal/DSProjects/Fragmentation/SURREAL'
        print(f"Warning: Could not determine SURREAL path automatically. Using default: {surreal_path}")
    
    # Hardcode file paths relative to SURREAL folder
    demographics_path = os.path.join(surreal_path, 'data', 'raw', 'participant_info.xlsx')
    daily_ema_path = os.path.join(surreal_path, 'processed', 'daily_ema_fragmentation', 'ema_fragmentation_combined.csv')
    all_ema_path = os.path.join(surreal_path, 'processed', 'ema_fragmentation', 'ema_fragmentation_all.csv')
    output_dir = os.path.join(surreal_path, 'processed', 'merged_data')
    
    # Still allow command-line overrides for flexibility
    parser = argparse.ArgumentParser(description='Merge demographic data with fragmentation and EMA datasets')
    
    parser.add_argument('--demographics', type=str, default=demographics_path,
                        help='Path to demographics Excel file')
    
    parser.add_argument('--ema_data', type=str, 
                        default=daily_ema_path,
                        help='Path to daily EMA data CSV')
    
    parser.add_argument('--ema_all_data', type=str,
                        default=all_ema_path,
                        help='Path to all EMA data CSV (3 times daily)')
    
    parser.add_argument('--output_dir', type=str,
                        default=output_dir,
                        help='Directory to save output files')
    
    args = parser.parse_args()
    
    # Set up logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    logger = setup_logging(output_dir)
    
    logger.info("Starting demographic data merging process")
    
    # Load demographics
    demographics = load_demographics(args.demographics)
    
    if demographics.empty:
        logger.error("Failed to load demographics data - exiting")
        return
    
    # Merge with daily EMA data
    daily_ema_output_path = output_dir / 'ema_fragmentation_daily_demographics.csv'
    if Path(args.ema_data).exists():
        merged_ema = merge_demographics_with_data(args.ema_data, demographics, daily_ema_output_path)
        if not merged_ema.empty:
            logger.info("Successfully merged demographics with daily EMA data")
    else:
        logger.warning(f"Daily EMA data file not found: {args.ema_data}")
    
    # Merge with all EMA data (3 times daily)
    all_ema_output_path = output_dir / 'ema_fragmentation_window_demographics.csv'
    if Path(args.ema_all_data).exists():
        merged_all_ema = merge_demographics_with_data(args.ema_all_data, demographics, all_ema_output_path)
        if not merged_all_ema.empty:
            logger.info("Successfully merged demographics with all EMA data (3 times daily)")
    else:
        logger.warning(f"All EMA data file not found: {args.ema_all_data}")
    
    logger.info("Demographic data merging completed")

if __name__ == "__main__":
    main()