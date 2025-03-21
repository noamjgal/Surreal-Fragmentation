import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging
from datetime import datetime
from data_utils import DataCleaner  # Import the new DataCleaner class

import sys
import traceback

# Add parent directory to path to find config
sys.path.append(str(Path(__file__).parent.parent))
from config.paths import PROCESSED_DATA_DIR

def setup_logging():
    """Set up logging with proper error handling"""
    try:
        # Set up logging to file
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Include timestamp in filename to avoid overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"combine_metrics_report_{timestamp}.log"
        
        # Create a file handler
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Get the root logger and clear existing handlers
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Remove any existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
        # Add our handlers
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Test log
        root_logger.info("Logging system initialized successfully")
        
        return log_file
    except Exception as e:
        print(f"ERROR SETTING UP LOGGING: {e}")
        print(traceback.format_exc())
        sys.exit(1)

def load_normalized_ema_data(normalized_dir):
    """Load all normalized EMA data and create daily averages."""
    logging.info(f"Loading normalized EMA data from {normalized_dir}...")
    
    # Find all normalized participant files
    normalized_files = list(normalized_dir.glob("normalized_participant_*.csv"))
    logging.info(f"Found {len(normalized_files)} normalized participant files")
    
    all_daily_ema = []
    
    for file in normalized_files:
        try:
            # Extract participant ID
            participant_id = file.stem.replace('normalized_participant_', '')
            
            logging.info(f"Processing EMA data for participant {participant_id}")
            
            # Load data
            ema_data = pd.read_csv(file)
            
            # Skip if empty
            if ema_data.empty:
                logging.warning(f"Empty data for participant {participant_id}")
                continue
                
            # Convert datetime column
            if 'datetime' in ema_data.columns:
                ema_data['datetime'] = pd.to_datetime(ema_data['datetime'])
                ema_data['date'] = ema_data['datetime'].dt.date
            else:
                logging.warning(f"No datetime column found for participant {participant_id}")
                continue
            
            # Filter only for STAI and CES-D scales
            ema_data = ema_data[ema_data['Scale'].isin(['STAI-Y-A-6', 'CES-D-8'])].copy()

            # Calculate and log response counts per day statistics
            response_counts = ema_data.groupby(['Participant_ID', 'date', 'Scale']).size().reset_index(name='response_count')

            # Overall statistics across all participants
            all_responses_stats = response_counts.groupby('Scale')['response_count'].agg(['mean', 'min', 'max']).reset_index()
            logging.info(f"Overall daily response statistics:")
            for _, row in all_responses_stats.iterrows():
                logging.info(f"  Scale {row['Scale']}: Average: {row['mean']:.2f}, Min: {row['min']}, Max: {row['max']}")

            # Per-participant statistics
            for pid in response_counts['Participant_ID'].unique():
                part_responses = response_counts[response_counts['Participant_ID'] == pid]
                part_stats = part_responses.groupby('Scale')['response_count'].agg(['mean', 'min', 'max']).reset_index()
                logging.info(f"Participant {pid} daily response statistics:")
                for _, row in part_stats.iterrows():
                    logging.info(f"  Scale {row['Scale']}: Average: {row['mean']:.2f}, Min: {row['min']}, Max: {row['max']}")

            # Count days with multiple responses
            multiple_resp_days = response_counts[response_counts['response_count'] > 1]
            multiple_resp_count = len(multiple_resp_days)
            total_days = len(response_counts)
            logging.info(f"Days with multiple responses: {multiple_resp_count} out of {total_days} ({(multiple_resp_count/total_days*100):.1f}%)")

            # Distribution of response counts
            resp_dist = response_counts['response_count'].value_counts().sort_index()
            logging.info(f"Distribution of daily response counts:")
            for count, freq in resp_dist.items():
                logging.info(f"  {count} response(s): {freq} days ({(freq/total_days*100):.1f}%)")
            
            
            # Skip if no valid data after filtering
            if ema_data.empty:
                logging.warning(f"No STAI or CES-D data for participant {participant_id}")
                continue
            
            # Log the date range for this participant
            date_range = ema_data['date'].unique()
            logging.info(f"Participant {participant_id} has data for {len(date_range)} days: {date_range[0]} to {date_range[-1]}")
            
            # Create daily averages
            daily_data = []
            
            # Calculate daily averages for each scale
            for scale in ['STAI-Y-A-6', 'CES-D-8']:
                scale_data = ema_data[ema_data['Scale'] == scale].copy()
                
                if not scale_data.empty:
                    # Group by date and calculate mean z-scores
                    daily_scale = scale_data.groupby('date').agg({
                        'score_zstd': 'mean',
                        'score_reversed': 'mean',
                        'Participant_ID': 'first'  # Keep participant ID
                    }).reset_index()
                    
                    # Rename columns to include scale
                    daily_scale = daily_scale.rename(columns={
                        'score_zstd': f'{scale}_zstd',
                        'score_reversed': f'{scale}_raw'
                    })
                    
                    daily_data.append(daily_scale)
                    
                    logging.info(f"Processed {len(scale_data)} {scale} responses, creating {len(daily_scale)} daily averages")
            
            # Merge daily averages for different scales
            if daily_data:
                participant_daily = daily_data[0]
                for i in range(1, len(daily_data)):
                    participant_daily = pd.merge(
                        participant_daily, daily_data[i],
                        on=['date', 'Participant_ID'],
                        how='outer'
                    )
                
                # Format date as string to match fragmentation data
                participant_daily['date_str'] = participant_daily['date'].astype(str)
                
                # Add to all daily data
                all_daily_ema.append(participant_daily)
                
                logging.info(f"Created {len(participant_daily)} daily records for participant {participant_id}")
            else:
                logging.warning(f"No daily data created for participant {participant_id}")
                
        except Exception as e:
            logging.error(f"Error processing {file.name}: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
    
    # Combine all participants' daily data
    if all_daily_ema:
        combined_daily_ema = pd.concat(all_daily_ema, ignore_index=True)
        return combined_daily_ema
    else:
        logging.warning("No valid daily EMA data found")
        return None

def load_fragmentation_data(fragmentation_file):
    """Load fragmentation data."""
    logging.info(f"Loading fragmentation data from: {fragmentation_file}")
    
    try:
        # Check file modification time
        if fragmentation_file.exists():
            mod_time = datetime.fromtimestamp(fragmentation_file.stat().st_mtime)
            logging.info(f"Fragmentation file last modified: {mod_time}")
        else:
            logging.warning(f"Fragmentation file does not exist at: {fragmentation_file}")
            return None
            
        frag_data = pd.read_csv(fragmentation_file)
        logging.info(f"Loaded {len(frag_data)} fragmentation records")
        
        # Display sample rows to better understand the data
        logging.info("Sample of fragmentation data (first 3 rows):")
        for i, row in frag_data.head(3).iterrows():
            logging.info(f"  Row {i}: participant_id='{row['participant_id']}', date='{row['date']}'")
            
        return frag_data
    except Exception as e:
        logging.error(f"Error loading fragmentation data: {str(e)}")
        return None

def standardize_participant_ids(df, id_column='participant_id'):
    """
    Standardize participant IDs to handle format variations.
    This helps match IDs between different datasets.
    """
    if df is None or df.empty:
        return df
    
    def clean_id(participant_id):
        # Convert to string
        pid = str(participant_id).strip()
        
        # Handle NaN or empty values
        if pid.lower() == 'nan' or pid == '':
            return ''
        
        # Extract digits only if it contains digits, otherwise keep original
        digits_only = ''.join(c for c in pid if c.isdigit())
        if digits_only:
            # If the ID is just a number, pad to at least 3 digits for consistent matching
            if len(digits_only) <= 2 and digits_only.isdigit():
                return digits_only.zfill(3)
            return digits_only
        else:
            # Remove any non-alphanumeric characters
            return ''.join(c for c in pid if c.isalnum())
    
    df[f'{id_column}_clean'] = df[id_column].apply(clean_id)
    
    # Log the ID mapping for debugging
    id_mapping = df[[id_column, f'{id_column}_clean']].drop_duplicates()
    logging.info(f"ID standardization mapping ({len(id_mapping)} IDs):")
    for _, row in id_mapping.iterrows():
        logging.info(f"  {row[id_column]} -> {row[f'{id_column}_clean']}")
        
    return df

def merge_ema_and_fragmentation(ema_data, frag_data):
    """
    Merge EMA and fragmentation data based on participant ID and date.
    """
    if ema_data is None or frag_data is None:
        logging.error("Cannot merge: one or both datasets are missing")
        return None
    
    # Use the new DataCleaner class
    data_cleaner = DataCleaner(logging.getLogger())
    
    # Standardize participant IDs in both datasets
    ema_data = data_cleaner.standardize_dataframe_ids(ema_data, id_column='Participant_ID')
    frag_data = data_cleaner.standardize_dataframe_ids(frag_data, id_column='participant_id')
    
    # Log participant ID distributions for debugging
    ema_participants = sorted(ema_data['participant_id_clean'].unique())
    frag_participants = sorted(frag_data['participant_id_clean'].unique())
    
    logging.info(f"EMA data contains {len(ema_participants)} unique participants")
    logging.info(f"Fragmentation data contains {len(frag_participants)} unique participants")
    
    # Enhanced date handling - print sample dates from both datasets
    if 'date' in ema_data.columns:
        logging.info(f"EMA date sample (first 3): {ema_data['date'].head(3).tolist()}")
        logging.info(f"EMA date type: {type(ema_data['date'].iloc[0])}")
    elif 'datetime' in ema_data.columns:
        logging.info(f"EMA datetime sample (first 3): {ema_data['datetime'].head(3).tolist()}")
        logging.info(f"EMA datetime type: {type(ema_data['datetime'].iloc[0])}")
    
    if 'date' in frag_data.columns:
        logging.info(f"Fragmentation date sample (first 3): {frag_data['date'].head(3).tolist()}")
        logging.info(f"Fragmentation date type: {type(frag_data['date'].iloc[0])}")
    
    # Make copies to avoid modifying originals
    ema_for_merge = ema_data.copy()
    frag_for_merge = frag_data.copy()
    
    # Normalize date formats in both datasets
    # For EMA data - handle both date and datetime columns
    if 'datetime' in ema_for_merge.columns:
        # Convert datetime to pandas datetime and extract date part only
        ema_for_merge['datetime'] = pd.to_datetime(ema_for_merge['datetime'], errors='coerce')
        ema_for_merge['date_normalized'] = ema_for_merge['datetime'].dt.date
    elif 'date' in ema_for_merge.columns:
        # Handle potential string or datetime objects
        if isinstance(ema_for_merge['date'].iloc[0], str):
            ema_for_merge['date_normalized'] = pd.to_datetime(ema_for_merge['date'], errors='coerce').dt.date
        else:
            # Might already be date object
            ema_for_merge['date_normalized'] = ema_for_merge['date']
    
    # For fragmentation data
    if 'date' in frag_for_merge.columns:
        # Convert to pandas datetime and extract date
        frag_for_merge['date'] = pd.to_datetime(frag_for_merge['date'], errors='coerce')
        frag_for_merge['date_normalized'] = frag_for_merge['date'].dt.date
    
    # Convert dates to strings in a consistent format for joining
    ema_for_merge['date_str_normalized'] = ema_for_merge['date_normalized'].astype(str)
    frag_for_merge['date_str_normalized'] = frag_for_merge['date_normalized'].astype(str)
    
    # Log the normalized date samples
    logging.info(f"Normalized EMA dates sample: {ema_for_merge['date_str_normalized'].head(3).tolist()}")
    logging.info(f"Normalized frag dates sample: {frag_for_merge['date_str_normalized'].head(3).tolist()}")
    
    # Count unique dates in each dataset
    logging.info(f"EMA unique dates: {ema_for_merge['date_str_normalized'].nunique()}")
    logging.info(f"Fragmentation unique dates: {frag_for_merge['date_str_normalized'].nunique()}")
    
    # Find dates that appear in both datasets
    ema_dates = set(ema_for_merge['date_str_normalized'])
    frag_dates = set(frag_for_merge['date_str_normalized'])
    common_dates = ema_dates.intersection(frag_dates)
    logging.info(f"Common dates between datasets: {len(common_dates)}")
    
    # Print some common and missing dates for debugging
    if common_dates:
        logging.info(f"Sample common dates: {list(common_dates)[:5]}")
    
    missing_in_ema = frag_dates - ema_dates
    if missing_in_ema:
        logging.info(f"Sample dates in fragmentation but missing in EMA: {list(missing_in_ema)[:5]}")
    
    # Merge the datasets using normalized dates
    merged_data = data_cleaner.merge_datasets(
        ema_for_merge, 
        frag_for_merge,
        left_on=['participant_id_clean', 'date_str_normalized'],
        right_on=['participant_id_clean', 'date_str_normalized'],
        how='inner'
    )
    
    logging.info(f"Merged data: {len(merged_data)} records")
    
    if len(merged_data) == 0:
        logging.warning("No matching records found when merging datasets. Trying alternative approaches...")
        
        # Try a more lenient approach with fuzzy date matching
        # This adds one day tolerance to account for timezone issues
        logging.info("Attempting fuzzy date matching with one day tolerance...")
        
        matches = []
        
        # For each participant in fragmentation data
        for participant in frag_for_merge['participant_id_clean'].unique():
            frag_participant = frag_for_merge[frag_for_merge['participant_id_clean'] == participant]
            ema_participant = ema_for_merge[ema_for_merge['participant_id_clean'] == participant]
            
            if ema_participant.empty:
                continue
                
            # For each fragmentation record for this participant
            for _, frag_row in frag_participant.iterrows():
                frag_date = pd.to_datetime(frag_row['date_normalized'])
                
                # Check for exact match and +/- 1 day to handle timezone differences
                for day_offset in [0, -1, 1]:
                    fuzzy_date = (frag_date + pd.Timedelta(days=day_offset)).date()
                    
                    # Look for matches in EMA data
                    matching_ema = ema_participant[ema_participant['date_normalized'] == fuzzy_date]
                    
                    if not matching_ema.empty:
                        # For each matching EMA record
                        for _, ema_row in matching_ema.iterrows():
                            # Create a merged record
                            merged_row = {}
                            
                            # Add EMA columns
                            for col in ema_row.index:
                                merged_row[f'ema_{col}'] = ema_row[col]
                                
                            # Add fragmentation columns
                            for col in frag_row.index:
                                merged_row[f'frag_{col}'] = frag_row[col]
                                
                            # Add match metadata
                            merged_row['day_offset'] = day_offset
                            
                            matches.append(merged_row)
                        
                        # Break after finding a match for this fragmentation record
                        break
        
        if matches:
            alt_merged_data = pd.DataFrame(matches)
            logging.info(f"Fuzzy date matching found {len(alt_merged_data)} matches")
            
            # Rename columns to match expected format
            # Rename keys to match the standard format
            column_mapping = {
                'ema_Participant_ID': 'participant_id_ema',
                'frag_participant_id': 'participant_id_frag',
                'ema_date_normalized': 'date'
            }
            
            # Only rename columns that exist
            rename_cols = {k: v for k, v in column_mapping.items() if k in alt_merged_data.columns}
            alt_merged_data = alt_merged_data.rename(columns=rename_cols)
            
            merged_data = alt_merged_data
        else:
            logging.warning("No matches found with fuzzy date matching either")
            return None
    
    # Clean up redundant columns
    columns_to_drop = [
        'participant_id_clean', 
        'date_str_normalized',
        'date_normalized_x' if 'date_normalized_x' in merged_data.columns else None,
        'date_normalized_y' if 'date_normalized_y' in merged_data.columns else None,
        'date_str_x' if 'date_str_x' in merged_data.columns else None,
        'date_str_y' if 'date_str_y' in merged_data.columns else None
    ]
    
    columns_to_drop = [col for col in columns_to_drop if col is not None and col in merged_data.columns]
    merged_data = merged_data.drop(columns=columns_to_drop)
    
    # Rename columns for clarity
    column_renames = {
        'date_x': 'date' if 'date_x' in merged_data.columns else None,
        'Participant_ID': 'participant_id_ema' if 'Participant_ID' in merged_data.columns else None,
        'participant_id': 'participant_id_frag' if 'participant_id' in merged_data.columns else None
    }
    
    # Remove None values from the rename dict
    column_renames = {k: v for k, v in column_renames.items() if v is not None and k in merged_data.columns}
    if column_renames:
        merged_data = merged_data.rename(columns=column_renames)
    
    # Validate the merged data
    validation_rules = {
        'numeric_ranges': {
            'STAI-Y-A-6_zstd': (-5, 5) if 'STAI-Y-A-6_zstd' in merged_data.columns else None,
            'CES-D-8_zstd': (-5, 5) if 'CES-D-8_zstd' in merged_data.columns else None,
            'digital_fragmentation_index': (0, 1) if 'digital_fragmentation_index' in merged_data.columns else None,
            'mobility_fragmentation_index': (0, 1) if 'mobility_fragmentation_index' in merged_data.columns else None,
            'overlap_fragmentation_index': (0, 1) if 'overlap_fragmentation_index' in merged_data.columns else None
        }
    }
    
    # Remove None values from validation rules
    validation_rules['numeric_ranges'] = {k: v for k, v in validation_rules['numeric_ranges'].items() 
                                        if v is not None}
    
    merged_data = data_cleaner.validate_data(merged_data, validation_rules)
    
    return merged_data

def main():
    # Set up logging first thing
    log_file = setup_logging()
    
    # Print a prominent message about log file location both to log and terminal
    log_location_message = f"\n{'='*80}\nDETAILED REPORT WILL BE SAVED TO: {log_file.absolute()}\n{'='*80}\n"
    logging.info(log_location_message)
    # Also print directly to terminal to ensure visibility
    print(log_location_message)
    
    try:
        # Define paths - use standardized path for fragmentation
        frag_path = PROCESSED_DATA_DIR / 'fragmentation' / 'fragmentation_all_metrics.csv'
        
        # Update paths to match where EMA-Processing scripts actually output the files
        ema_processing_dir = Path("/Users/noamgal/DSProjects/Fragmentation/SURREAL/EMA-Processing")
        participant_norm_dir = ema_processing_dir / "output" / "normalized"
        population_norm_dir = ema_processing_dir / "output" / "normalized_population"
        
        # Create output directory within the standardized processed data directory
        output_dir = PROCESSED_DATA_DIR / 'daily_ema_fragmentation'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load fragmentation data (only need to do this once)
        frag_data = load_fragmentation_data(frag_path)
        
        if frag_data is None or frag_data.empty:
            logging.error("Failed to load fragmentation data. Check the fragmentation file path.")
            return
            
        # Process both normalization approaches
        for norm_type, norm_dir in [("participant", participant_norm_dir), ("population", population_norm_dir)]:
            logging.info(f"\n{'='*40}\nProcessing {norm_type}-level normalized EMA data\n{'='*40}")
            
            # Load EMA data for current normalization approach
            logging.info(f"Loading EMA data from {norm_dir}")
            daily_ema = load_normalized_ema_data(norm_dir)
            
            if daily_ema is None or daily_ema.empty:
                logging.error(f"Failed to load valid {norm_type}-normalized EMA data. Skipping this approach.")
                continue
                
            logging.info(f"Loaded {len(daily_ema)} daily {norm_type}-normalized EMA records")
            
            # Log column names to help debug
            logging.info(f"Fragmentation data columns: {list(frag_data.columns)}")
            
            # Log sample date formats from both datasets
            if 'date' in frag_data.columns:
                logging.info(f"Fragmentation data date samples: {frag_data['date'].head(5).tolist()}")
                logging.info(f"Fragmentation date type: {type(frag_data['date'].iloc[0])}")
            
            if 'date' in daily_ema.columns:
                logging.info(f"EMA data date samples: {daily_ema['date'].head(5).tolist()}")
                logging.info(f"EMA date type: {type(daily_ema['date'].iloc[0])}")
            elif 'date_str' in daily_ema.columns:
                logging.info(f"EMA data date_str samples: {daily_ema['date_str'].head(5).tolist()}")
                logging.info(f"EMA date_str type: {type(daily_ema['date_str'].iloc[0])}")
                
            # Clean and standardize the data before merging
            data_cleaner = DataCleaner()
            
            # Get unique participant IDs in EMA data
            ema_participants = set(daily_ema['Participant_ID'].unique())
            logging.info(f"Found {len(ema_participants)} unique participants in {norm_type}-normalized EMA data")
            
            # Track discard statistics
            discard_stats = {
                'total': len(frag_data),
                'discarded': 0,
                'no_participant': 0,
                'no_date': 0,
                'format_mismatch': 0,
                'no_participant_examples': [],
                'no_date_examples': [],
                'format_mismatch_examples': [],
                'ema_participants': sorted(list(ema_participants))
            }
            
            # Create a new dataframe to store matched records
            matched_records = []
            
            # First, standardize participant IDs in both datasets
            frag_data['cleaned_user_id'] = frag_data['participant_id'].apply(data_cleaner.standardize_participant_id)
            daily_ema['cleaned_participant_id'] = daily_ema['Participant_ID'].apply(data_cleaner.standardize_participant_id)
            
            # Create sets for efficient lookup
            ema_participants_clean = set(daily_ema['cleaned_participant_id'].unique())
            
            # Create a lookup table for EMA dates by participant
            ema_dates_by_participant = {}
            for participant in ema_participants_clean:
                participant_data = daily_ema[daily_ema['cleaned_participant_id'] == participant]
                ema_dates_by_participant[participant] = sorted(participant_data['date_str'].unique().tolist())
            
            # Process each fragmentation record
            logging.info(f"Processing fragmentation records for {norm_type}-normalized data...")
            
            for index, frag_row in frag_data.iterrows():
                # Get the standardized participant ID
                participant = frag_row['cleaned_user_id']
                
                # Check if participant exists in EMA data
                if participant not in ema_participants_clean:
                    discard_stats['no_participant'] += 1
                    discard_stats['discarded'] += 1
                    
                    # Store the full example
                    example = {
                        'user_id': frag_row['participant_id'],
                        'cleaned_user_id': participant,
                        'date': frag_row['date']
                    }
                    
                    # Add fragmentation indices if available
                    if 'digital_fragmentation_index' in frag_row:
                        example['digital_fragmentation_index'] = frag_row['digital_fragmentation_index']
                    if 'moving_fragmentation_index' in frag_row:
                        example['mobility_fragmentation_index'] = frag_row['moving_fragmentation_index']
                    
                    discard_stats['no_participant_examples'].append(example)
                    continue
                
                # Get the original date string from fragmentation data
                date_str = str(frag_row['date'])
                
                # Try direct match first
                matched = False
                if date_str in ema_dates_by_participant[participant]:
                    matched = True
                    match_type = "direct"
                    matching_date = date_str
                else:
                    # Try alternative date format (swap day and month)
                    try:
                        # Parse the date using pandas
                        orig_date = pd.to_datetime(date_str)
                        
                        # Create alternative format by swapping day and month
                        alt_date_str = f"{orig_date.year}-{orig_date.day:02d}-{orig_date.month:02d}"
                        
                        # Also try with dashes
                        alt_date_str2 = f"{orig_date.year}-{orig_date.day:02d}-{orig_date.month:02d}"
                        
                        # Log for debugging
                        logging.debug(f"Original date: {date_str}, Alternative: {alt_date_str}, Alternative 2: {alt_date_str2}")
                        
                        # Check if alternative format exists in EMA data
                        if alt_date_str in ema_dates_by_participant[participant]:
                            matched = True
                            match_type = "alt_format"
                            matching_date = alt_date_str
                        elif alt_date_str2 in ema_dates_by_participant[participant]:
                            matched = True
                            match_type = "alt_format2"
                            matching_date = alt_date_str2
                    except Exception as e:
                        logging.debug(f"Error parsing date {date_str}: {e}")
                
                if not matched:
                    discard_stats['no_date'] += 1
                    discard_stats['discarded'] += 1
                    
                    # Log the date format issue if parsing succeeded but matching failed
                    example = {
                        'user_id': frag_row['participant_id'],
                        'cleaned_user_id': participant,
                        'date': date_str,
                        'ema_available_dates': ema_dates_by_participant[participant]
                    }
                    
                    # Add fragmentation indices if available
                    if 'digital_fragmentation_index' in frag_row:
                        example['digital_fragmentation_index'] = frag_row['digital_fragmentation_index']
                    if 'moving_fragmentation_index' in frag_row:
                        example['mobility_fragmentation_index'] = frag_row['moving_fragmentation_index']
                    
                    discard_stats['no_date_examples'].append(example)
                    continue
                
                # If we reach here, we have a match
                # Find the matching EMA record
                ema_match = daily_ema[(daily_ema['cleaned_participant_id'] == participant) & 
                                     (daily_ema['date_str'] == matching_date)]
                
                if len(ema_match) == 0:
                    # This shouldn't happen based on our checks, but just in case
                    logging.warning(f"Logic error: No EMA match found for participant {participant} on {matching_date}")
                    continue
                    
                # Use the first matching record if multiple exist
                ema_match = ema_match.iloc[0]
                
                # Create a matched record
                matched_record = {
                    'user_id': frag_row['participant_id'],
                    'date': date_str,
                    'matching_date': matching_date,
                    'match_type': match_type,
                    'Participant_ID': ema_match['Participant_ID']
                }
                
                # Copy all columns from fragmentation data
                for col in frag_data.columns:
                    if col not in ['participant_id', 'date', 'cleaned_user_id']:
                        matched_record[f'frag_{col}'] = frag_row[col]
                
                # Copy relevant EMA data
                for col in daily_ema.columns:
                    if col not in ['date', 'date_str', 'Participant_ID', 'cleaned_participant_id']:
                        matched_record[f'ema_{col}'] = ema_match[col]
                
                matched_records.append(matched_record)
            
            # NEW: Track EMA data that doesn't have matching fragmentation data
            logging.info(f"\nChecking for EMA data without matching fragmentation records...")
            
            # Create a set of matched fragmentation data for efficient lookup
            matched_frag_keys = set()
            for record in matched_records:
                # Create a unique key from participant ID and date
                key = (record['Participant_ID'], record['matching_date'])
                matched_frag_keys.add(key)
            
            # Track EMA data without fragmentation matches
            ema_without_frag = {
                'total_ema_records': len(daily_ema),
                'missing_fragmentation': 0,
                'examples': []
            }
            
            # Create lookup table for fragmentation dates by participant
            frag_dates_by_participant = {}
            for index, row in frag_data.iterrows():
                participant = row['cleaned_user_id']
                date_str = str(row['date'])
                
                if participant not in frag_dates_by_participant:
                    frag_dates_by_participant[participant] = []
                
                frag_dates_by_participant[participant].append(date_str)
            
            # Check each EMA record
            for index, ema_row in daily_ema.iterrows():
                participant = ema_row['Participant_ID']
                cleaned_participant = ema_row['cleaned_participant_id']
                date_str = ema_row['date_str']
                
                # Create key to check if it was matched
                key = (participant, date_str)
                
                if key not in matched_frag_keys:
                    ema_without_frag['missing_fragmentation'] += 1
                    
                    # Create example with details
                    example = {
                        'participant_id': participant,
                        'cleaned_participant_id': cleaned_participant,
                        'date': date_str
                    }
                    
                    # Add EMA metrics if available
                    if 'STAI-Y-A-6_zstd' in ema_row:
                        example['STAI-Y-A-6_zstd'] = ema_row['STAI-Y-A-6_zstd']
                    if 'CES-D-8_zstd' in ema_row:
                        example['CES-D-8_zstd'] = ema_row['CES-D-8_zstd']
                    
                    # Add available fragmentation dates for this participant if any
                    if cleaned_participant in frag_dates_by_participant:
                        example['frag_available_dates'] = frag_dates_by_participant[cleaned_participant]
                    else:
                        example['frag_available_dates'] = []
                    
                    ema_without_frag['examples'].append(example)
            
            # Convert to dataframe
            if matched_records:
                combined_data = pd.DataFrame(matched_records)
                logging.info(f"Created {len(combined_data)} matched records for {norm_type}-normalized data")
                
                # Log match types
                match_type_counts = combined_data['match_type'].value_counts()
                logging.info(f"Match types: {match_type_counts.to_dict()}")
                
                # Save to CSV with normalization type in filename
                output_file = output_dir / f"combined_metrics_{norm_type}_norm.csv"
                combined_data.to_csv(output_file, index=False)
                logging.info(f"Saved {norm_type}-normalized combined data to {output_file}")
            else:
                logging.warning(f"No matches found for {norm_type}-normalized data! Could not create combined dataset.")
            
            # Log discard statistics
            logging.info(f"\n{'='*50}")
            logging.info(f"MATCH STATISTICS FOR {norm_type.upper()}-NORMALIZED DATA:")
            logging.info(f"Total fragmentation records: {discard_stats['total']}")
            logging.info(f"Successfully matched: {discard_stats['total'] - discard_stats['discarded']} ({100 - 100 * discard_stats['discarded'] / discard_stats['total']:.1f}%)")
            logging.info(f"Discarded: {discard_stats['discarded']} ({100 * discard_stats['discarded'] / discard_stats['total']:.1f}%)")
            
            logging.info("\nDISCARD REASONS:")
            if discard_stats['no_participant'] > 0:
                logging.info(f"- No matching participant: {discard_stats['no_participant']} ({100 * discard_stats['no_participant'] / discard_stats['total']:.1f}%)")
            if discard_stats['no_date'] > 0:
                logging.info(f"- No matching date: {discard_stats['no_date']} ({100 * discard_stats['no_date'] / discard_stats['total']:.1f}%)")
            
            # NEW: Log EMA without fragmentation statistics
            logging.info(f"\n{'='*50}")
            logging.info(f"EMA DATA WITHOUT MATCHING FRAGMENTATION ({norm_type.upper()}-NORMALIZED):")
            logging.info(f"Total EMA records: {ema_without_frag['total_ema_records']}")
            logging.info(f"Missing fragmentation data: {ema_without_frag['missing_fragmentation']} ({100 * ema_without_frag['missing_fragmentation'] / ema_without_frag['total_ema_records']:.1f}%)")
            
            # Group missing EMA data by participant for a more organized report
            by_participant = {}
            for example in ema_without_frag['examples']:
                pid = example['participant_id']
                if pid not in by_participant:
                    by_participant[pid] = []
                by_participant[pid].append(example)
            
            logging.info(f"\nDETAILED EMA WITHOUT FRAGMENTATION REPORT:")
            logging.info(f"Missing fragmentation data for {len(by_participant)} participants")
            
            # Sort by participant ID for consistent reporting
            for pid in sorted(by_participant.keys()):
                examples = by_participant[pid]
                logging.info(f"\nParticipant {pid} (Cleaned ID: {examples[0]['cleaned_participant_id']}):")
                logging.info(f"  Missing fragmentation for {len(examples)} EMA dates")
                
                # Sort examples by date
                sorted_examples = sorted(examples, key=lambda x: x['date'])
                
                # Log the first 10 examples, then summarize if more
                show_count = min(10, len(sorted_examples))
                for i, example in enumerate(sorted_examples[:show_count]):
                    logging.info(f"  {i+1}. Date: {example['date']}")
                    
                    # Show EMA metrics if available
                    metrics = []
                    if 'STAI-Y-A-6_zstd' in example:
                        metrics.append(f"STAI: {example['STAI-Y-A-6_zstd']:.2f}")
                    if 'CES-D-8_zstd' in example:
                        metrics.append(f"CES-D: {example['CES-D-8_zstd']:.2f}")
                    
                    if metrics:
                        logging.info(f"     EMA Metrics: {', '.join(metrics)}")
                
                # Indicate if there are more examples
                if len(sorted_examples) > show_count:
                    logging.info(f"     ... and {len(sorted_examples) - show_count} more dates")
                
                # Show available fragmentation dates for reference
                if 'frag_available_dates' in examples[0] and examples[0]['frag_available_dates']:
                    frag_dates = examples[0]['frag_available_dates']
                    show_frag_count = min(5, len(frag_dates))
                    
                    logging.info(f"  Available fragmentation dates for this participant ({len(frag_dates)} total):")
                    for i, date in enumerate(sorted(frag_dates)[:show_frag_count]):
                        logging.info(f"     {i+1}. {date}")
                    
                    if len(frag_dates) > show_frag_count:
                        logging.info(f"     ... and {len(frag_dates) - show_frag_count} more dates")
                else:
                    logging.info("  No fragmentation data available for this participant")
            
            # Log detailed examples
            if discard_stats['no_participant'] > 0 or discard_stats['no_date'] > 0:
                logging.info(f"\nDETAILED MISMATCH REPORT FOR {norm_type.upper()}-NORMALIZED DATA:")
                
                # Log participant mismatches
                if discard_stats['no_participant_examples']:
                    logging.info(f"\nALL PARTICIPANT MISMATCHES ({len(discard_stats['no_participant_examples'])} records):")
                    for i, example in enumerate(discard_stats['no_participant_examples']):
                        logging.info(f"  {i+1}. Original ID: {example['user_id']}, Cleaned ID: {example['cleaned_user_id']}, Date: {example['date']}")
                        if 'digital_fragmentation_index' in example:
                            logging.info(f"     Digital Fragmentation: {example['digital_fragmentation_index']}")
                        if 'mobility_fragmentation_index' in example:
                            logging.info(f"     Mobility Fragmentation: {example['mobility_fragmentation_index']}")
                
                # Log date mismatches
                if discard_stats['no_date_examples']:
                    logging.info(f"\nALL DATE MISMATCHES ({len(discard_stats['no_date_examples'])} records):")
                    for i, example in enumerate(discard_stats['no_date_examples']):
                        logging.info(f"  {i+1}. Participant: {example['user_id']} (Cleaned: {example['cleaned_user_id']}), Date: {example['date']}")
                        logging.info(f"     Available EMA dates for this participant: {example['ema_available_dates']}")
                        
                        if 'digital_fragmentation_index' in example:
                            logging.info(f"     Digital Fragmentation: {example['digital_fragmentation_index']}")
                        if 'mobility_fragmentation_index' in example:
                            logging.info(f"     Mobility Fragmentation: {example['mobility_fragmentation_index']}")
                
                # List all available EMA participants for reference
                logging.info("\nALL AVAILABLE EMA PARTICIPANTS:")
                for i, participant in enumerate(sorted(discard_stats['ema_participants'])):
                    logging.info(f"  {i+1}. {participant}")
        
        # Remind about log file location at the end
        end_message = f"\n{'='*80}\nCOMPLETE DETAILED REPORT SAVED TO: {log_file.absolute()}\n{'='*80}\n"
        logging.info(end_message)
        print(end_message)
        
        # Force flush all handlers
        for handler in logging.root.handlers:
            handler.flush()
            
    except Exception as e:
        # Log any unexpected errors
        logging.error(f"ERROR: {e}")
        logging.error(traceback.format_exc())
        print(f"ERROR: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()