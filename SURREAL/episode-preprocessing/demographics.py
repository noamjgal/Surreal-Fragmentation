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
from data_utils import DataCleaner

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
        # Use DataCleaner
        data_cleaner = DataCleaner(logger)
        
        # Load the Excel file
        demographics = pd.read_excel(demographics_path)
        logger.info(f"Loaded {len(demographics)} demographic records")
        
        # Standardize participant IDs
        demographics = data_cleaner.standardize_dataframe_ids(demographics, 'Participant_ID')
        
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
        
        # Process age data using standardized numeric handling
        if 'DOB' in demographics.columns:
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
                    
                # Standardize age values
                demographics = data_cleaner.standardize_missing_values(
                    demographics, numeric_columns=['age']
                )
                
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
            
        # Apply final data validation
        validation_rules = {
            'numeric_ranges': {
                'age': (18, 100)  # typical age range
            }
        }
        demographics = data_cleaner.validate_data(demographics, validation_rules)
        
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
        # Use the DataCleaner
        data_cleaner = DataCleaner(logger)
        
        # Load the data
        data = pd.read_csv(data_path)
        logger.info(f"Loaded data with shape: {data.shape}")
        logger.info(f"Columns in data: {data.columns.tolist()}")
        
        # Log sample date values from both datasets for debugging
        date_columns_data = [col for col in data.columns if 'date' in col.lower()]
        if date_columns_data:
            logger.info(f"Date columns in data: {date_columns_data}")
            for col in date_columns_data[:2]:  # Show first 2 date columns
                logger.info(f"Sample dates from data['{col}']: {data[col].head(5).tolist()}")
        
        date_columns_demo = [col for col in demographics_df.columns if 'date' in col.lower()]
        if date_columns_demo:
            logger.info(f"Date columns in demographics: {date_columns_demo}")
            for col in date_columns_demo[:2]:  # Show first 2 date columns
                logger.info(f"Sample dates from demographics['{col}']: {demographics_df[col].head(5).tolist()}")
        
        # Show a sample of participant IDs from the data
        id_sample = []
        for col in data.columns:
            if 'participant' in col.lower() or 'id' in col.lower():
                id_sample.append((col, data[col].head(3).tolist()))
        logger.info(f"Participant ID columns and samples: {id_sample}")
        
        # Standardize IDs in data if not already done
        if 'participant_id_clean' not in data.columns:
            # Find appropriate ID column
            id_columns = ['participant_id', 'participant_id_ema', 'participant_id_frag', 
                          'Participant_ID', 'user', 'subject', 'id']
            id_col = next((col for col in id_columns if col in data.columns), None)
            
            if id_col:
                data = data_cleaner.standardize_dataframe_ids(data, id_col)
            else:
                logger.error(f"Could not identify participant ID column in data")
                return pd.DataFrame()
        
        # Find common participants
        data_participants = set(data['participant_id_clean'].unique())
        demo_participants = set(demographics_df['participant_id_clean'].unique())
        common_participants = data_participants.intersection(demo_participants)
        
        logger.info(f"Found {len(common_participants)} common participants")
        logger.info(f"Participants in data but not demographics: {sorted(list(data_participants - demo_participants))}")
        logger.info(f"Participants in demographics but not data: {sorted(list(demo_participants - data_participants))}")
        
        # MISSING VALUES REPORT - Demographics Dataset
        logger.info("\n" + "="*50)
        logger.info("DEMOGRAPHICS DATASET MISSING VALUES REPORT:")
        missing_demo = demographics_df.isna().sum()
        total_demo = len(demographics_df)
        missing_demo_pct = (missing_demo / total_demo * 100).round(1)
        
        # Create detailed report of missing values
        missing_report = []
        for col in demographics_df.columns:
            missing_count = demographics_df[col].isna().sum()
            missing_pct = (missing_count / total_demo * 100).round(1)
            if missing_count > 0:
                missing_report.append(f"{col}: {missing_count} missing ({missing_pct}%)")
            
        logger.info(f"Demographics total rows: {total_demo}")
        logger.info(f"Demographics missing value summary:")
        for report_line in missing_report:
            logger.info(f"  - {report_line}")
            
        # Log participants with missing key demographic variables
        key_demographics = ['age', 'gender', 'education', 'race', 'ethnicity']
        key_demographics = [col for col in key_demographics if col in demographics_df.columns]
        
        if key_demographics:
            missing_key_participants = {}
            for col in key_demographics:
                missing_participants = demographics_df[demographics_df[col].isna()]['participant_id_clean'].tolist()
                if missing_participants:
                    missing_key_participants[col] = missing_participants
            
            if missing_key_participants:
                logger.info("\nParticipants missing key demographic variables:")
                for col, participants in missing_key_participants.items():
                    logger.info(f"  - {col}: {len(participants)} participants missing this value")
                    logger.info(f"    IDs: {participants}")
        
        # Check if there are date columns to standardize for merging
        # This is the key addition from combine_metrics.py - handle date format inconsistencies
        date_pairs = []
        for date_col_data in date_columns_data:
            for date_col_demo in date_columns_demo:
                date_pairs.append((date_col_data, date_col_demo))
        
        # If there are potential date column pairs to match on
        if date_pairs and 'date' in data.columns:
            logger.info("Attempting to standardize date formats for better matching...")
            
            # Track date matches
            date_match_stats = {
                'total': 0,
                'direct_match': 0,
                'alt_format_match': 0,
                'day_offset_match': 0,
                'unmatched': 0
            }
            
            # Try to handle possible date format mismatches by creating standardized date columns
            def try_alternative_date_formats(date_str):
                if pd.isna(date_str):
                    return {
                        'std_date': None,
                        'alt_date': None,
                        'next_day': None,
                        'prev_day': None
                    }
                
                try:
                    # Parse original date
                    orig_date = pd.to_datetime(date_str)
                    
                    # Create alternative format by swapping day and month
                    alt_date = pd.Timestamp(year=orig_date.year, month=orig_date.day, day=orig_date.month)
                    
                    # Create day offsets for timezone handling
                    next_day = orig_date + pd.Timedelta(days=1)
                    prev_day = orig_date - pd.Timedelta(days=1)
                    
                    return {
                        'std_date': orig_date.strftime('%Y-%m-%d'),
                        'alt_date': alt_date.strftime('%Y-%m-%d'),
                        'next_day': next_day.strftime('%Y-%m-%d'),
                        'prev_day': prev_day.strftime('%Y-%m-%d')
                    }
                except:
                    return {
                        'std_date': None,
                        'alt_date': None,
                        'next_day': None,
                        'prev_day': None
                    }
            
            # Apply date format standardization to both datasets where needed
            for date_col in date_columns_data:
                if date_col in data.columns:
                    logger.info(f"Standardizing dates in data['{date_col}']")
                    date_formats = data[date_col].apply(try_alternative_date_formats)
                    data[f'{date_col}_std'] = date_formats.apply(lambda x: x['std_date'])
                    data[f'{date_col}_alt'] = date_formats.apply(lambda x: x['alt_date'])
            
            for date_col in date_columns_demo:
                if date_col in demographics_df.columns:
                    logger.info(f"Standardizing dates in demographics['{date_col}']")
                    date_formats = demographics_df[date_col].apply(try_alternative_date_formats)
                    demographics_df[f'{date_col}_std'] = date_formats.apply(lambda x: x['std_date'])
                    demographics_df[f'{date_col}_alt'] = date_formats.apply(lambda x: x['alt_date'])
        
        # Record original data completeness
        original_completeness = {}
        for col in data.columns:
            if col not in ['participant_id', 'participant_id_clean', 'date']:
                original_completeness[col] = data[col].notna().mean() * 100
        
        # Merge data with demographics
        merged_data = data_cleaner.merge_datasets(
            data, 
            demographics_df,
            on='participant_id_clean',
            how='left'
        )
        
        # MISSING VALUES REPORT - Merged Dataset
        logger.info("\n" + "="*50)
        logger.info("MERGED DATASET MISSING VALUES REPORT:")
        
        # Identify demographic columns in merged data
        demographic_cols = [col for col in merged_data.columns if col in demographics_df.columns 
                          and col not in ['participant_id_clean']]
        
        # Count rows missing any demographic information
        rows_missing_any_demographics = merged_data[demographic_cols].isna().any(axis=1).sum()
        rows_missing_all_demographics = merged_data[demographic_cols].isna().all(axis=1).sum()
        
        logger.info(f"Total records in merged data: {len(merged_data)}")
        logger.info(f"Records missing ANY demographic information: {rows_missing_any_demographics} ({rows_missing_any_demographics/len(merged_data)*100:.1f}%)")
        logger.info(f"Records missing ALL demographic information: {rows_missing_all_demographics} ({rows_missing_all_demographics/len(merged_data)*100:.1f}%)")
        
        # Log missing values by column in merged data
        logger.info("\nMissing values by column in merged data:")
        for col in demographic_cols:
            missing_count = merged_data[col].isna().sum()
            if missing_count > 0:
                logger.info(f"  - {col}: {missing_count} missing ({missing_count/len(merged_data)*100:.1f}%)")
                
        # Track demographic completeness by participant
        participant_completeness = {}
        for participant in merged_data['participant_id_clean'].unique():
            participant_data = merged_data[merged_data['participant_id_clean'] == participant]
            completeness = {}
            for col in demographic_cols:
                if col in participant_data.columns:
                    completeness[col] = participant_data[col].notna().mean() * 100
            participant_completeness[participant] = completeness
            
        # Find participants with partial demographic information
        participants_with_partial_demographics = []
        for participant, completeness in participant_completeness.items():
            if completeness and 0 < sum(completeness.values()) < len(completeness) * 100:
                participants_with_partial_demographics.append({
                    'participant': participant,
                    'completeness': completeness
                })
                
        if participants_with_partial_demographics:
            logger.info("\nParticipants with partial demographic information:")
            for participant_info in participants_with_partial_demographics:
                logger.info(f"  - {participant_info['participant']}:")
                for col, pct in participant_info['completeness'].items():
                    if pct < 100:
                        logger.info(f"      {col}: {pct:.1f}% complete")
        
        # Compare column completeness before and after merging
        logger.info("\nData completeness comparison (before vs. after merging):")
        for col in original_completeness:
            if col in merged_data.columns:
                before = original_completeness[col]
                after = merged_data[col].notna().mean() * 100
                diff = after - before
                logger.info(f"  - {col}: {before:.1f}% → {after:.1f}% ({diff:+.1f}%)")
                
        # Log merge statistics
        merge_stats = {
            'total_rows': len(data),
            'merged_rows': len(merged_data),
            'with_demographics': merged_data['age'].notna().sum() if 'age' in merged_data.columns else 0,
            'participants_in_data': len(data_participants),
            'participants_in_demographics': len(demo_participants),
            'common_participants': len(common_participants)
        }
        logger.info(f"\nMerge statistics: {merge_stats}")
        
        # Validate the merged data
        merged_data = data_cleaner.validate_data(merged_data)
        
        # Save merged data
        merged_data.to_csv(output_path, index=False)
        logger.info(f"Saved merged data to {output_path} with {len(merged_data)} rows")
        
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
    
    # Define input paths for all three normalization approaches
    input_file_paths = {
        "unstandardized": os.path.join(surreal_path, 'processed', 'daily_ema_fragmentation_unstd', 'combined_metrics_raw.csv'),
        "participant": os.path.join(surreal_path, 'processed', 'daily_ema_fragmentation', 'combined_metrics_participant_norm.csv'),
        "population": os.path.join(surreal_path, 'processed', 'daily_ema_fragmentation', 'combined_metrics_population_norm.csv')
    }
    
    # Also try backup locations if files don't exist
    backup_paths = {
        "unstandardized": "/Volumes/Extreme SSD/SURREAL-DataBackup/HUJI_data-main/processed/daily_ema_fragmentation_unstd/combined_metrics_raw.csv",
        "participant": "/Volumes/Extreme SSD/SURREAL-DataBackup/HUJI_data-main/processed/daily_ema_fragmentation/combined_metrics_participant_norm.csv",
        "population": "/Volumes/Extreme SSD/SURREAL-DataBackup/HUJI_data-main/processed/daily_ema_fragmentation/combined_metrics_population_norm.csv"
    }
    
    # Verify input files exist, use backup if needed
    for norm_type in input_file_paths:
        if not os.path.exists(input_file_paths[norm_type]) and os.path.exists(backup_paths[norm_type]):
            input_file_paths[norm_type] = backup_paths[norm_type]
    
    output_dir = os.path.join(surreal_path, 'processed', 'merged_data')
    
    # Still allow command-line overrides for flexibility
    parser = argparse.ArgumentParser(description='Merge demographic data with fragmentation and EMA datasets')
    
    parser.add_argument('--demographics', type=str, default=demographics_path,
                        help='Path to demographics Excel file')
    
    parser.add_argument('--ema_data', type=str, 
                        default=None,  # No default - will process all files if not specified
                        help='Path to specific EMA data CSV to process (optional)')
    
    parser.add_argument('--output_dir', type=str,
                        default=output_dir,
                        help='Directory to save output files')
    
    parser.add_argument('--types', type=str, default="all",
                        help='Types of normalization to process (comma-separated: unstandardized,participant,population or "all")')
    
    args = parser.parse_args()
    
    # Set up logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    logger = setup_logging(output_dir)
    
    logger.info("Starting demographic data merging process")
    
    # Load demographics only once
    demographics = load_demographics(args.demographics)
    
    if demographics.empty:
        logger.error("Failed to load demographics data - exiting")
        return
    
    # Determine which normalization types to process
    if args.types.lower() == "all":
        norm_types_to_process = list(input_file_paths.keys())
    else:
        norm_types_to_process = [t.strip() for t in args.types.split(",")]
    
    # Check if specific file was requested via command line
    if args.ema_data:
        # A specific file was provided via command line
        logger.info(f"Processing single file specified via command line: {args.ema_data}")
        
        if Path(args.ema_data).exists():
            # Try to determine the normalization type from the filename
            file_name = os.path.basename(args.ema_data)
            if "raw" in file_name:
                norm_type = "unstandardized"
            elif "participant" in file_name:
                norm_type = "participant"
            elif "population" in file_name:
                norm_type = "population"
            else:
                norm_type = "custom"
            
            output_filename = f"ema_fragmentation_demographics_{norm_type}.csv"
            output_path = output_dir / output_filename
            merged_data = merge_demographics_with_data(args.ema_data, demographics, output_path)
            
            if not merged_data.empty:
                logger.info(f"Successfully merged demographics with {norm_type} data file")
                
                # Print clear output path
                output_absolute_path = output_path.absolute()
                output_message = f"\n{'*'*80}\n{norm_type.upper()} OUTPUT WITH DEMOGRAPHICS SAVED TO:\n{output_absolute_path}\n{'*'*80}"
                logger.info(output_message)
                print(output_message)
        else:
            logger.warning(f"Specified data file not found: {args.ema_data}")
    else:
        # Process all specified normalization types
        successfully_processed = []
        
        for norm_type in norm_types_to_process:
            if norm_type not in input_file_paths:
                logger.warning(f"Unknown normalization type: {norm_type} - skipping")
                continue
                
            input_path = input_file_paths[norm_type]
            
            logger.info(f"\n{'='*40}\nProcessing {norm_type} data\n{'='*40}")
            
            if Path(input_path).exists():
                output_filename = f"ema_fragmentation_demographics_{norm_type}.csv"
                output_path = output_dir / output_filename
                
                merged_data = merge_demographics_with_data(input_path, demographics, output_path)
                
                if not merged_data.empty:
                    logger.info(f"Successfully merged demographics with {norm_type} data")
                    successfully_processed.append((norm_type, output_path))
                else:
                    logger.warning(f"Failed to merge demographics with {norm_type} data - output may be empty")
            else:
                logger.warning(f"{norm_type} data file not found: {input_path}")
        
        # Add a CLEAR summary about output files at the end
        logger.info(f"\n{'#'*80}\nOUTPUT SUMMARY:")
        
        if successfully_processed:
            for norm_type, file_path in successfully_processed:
                logger.info(f"- {norm_type.upper()} OUTPUT WITH DEMOGRAPHICS: {file_path.absolute()}")
                # Also print directly to terminal to ensure visibility
                print(f"\n{'*'*80}\n{norm_type.upper()} OUTPUT WITH DEMOGRAPHICS SAVED TO:\n{file_path.absolute()}\n{'*'*80}")
        else:
            no_output_msg = "- NO OUTPUT FILES CREATED - No successful merges completed"
            logger.info(no_output_msg)
            print(f"\n{'*'*80}\n{no_output_msg}\n{'*'*80}")
            
        logger.info(f"{'#'*80}\n")
    
    logger.info("Demographic data merging completed")

if __name__ == "__main__":
    main()