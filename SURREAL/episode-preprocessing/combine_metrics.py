import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging
from datetime import datetime
from data_utils import DataCleaner  # Import the new DataCleaner class

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_normalized_ema_data(normalized_dir):
    """Load all normalized EMA data and create daily averages."""
    logging.info("Loading normalized EMA data...")
    
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
    
    # Ensure date formats match between datasets
    if 'datetime' in ema_data.columns:
        ema_data = data_cleaner.standardize_timestamps(ema_data, ['datetime'])
    
    # Create date string for joining if needed
    if 'date_str' not in ema_data.columns and 'date' in ema_data.columns:
        ema_data['date_str'] = ema_data['date'].astype(str)
    
    # Merge the datasets using clean IDs and dates
    merged_data = data_cleaner.merge_datasets(
        ema_data, 
        frag_data,
        left_on=['participant_id_clean', 'date_str'],
        right_on=['participant_id_clean', 'date'],
        how='inner'
    )
    
    logging.info(f"Merged data: {len(merged_data)} records")
    
    if len(merged_data) == 0:
        logging.warning("No matching records found when merging datasets. Trying alternative approaches...")
        
        # Try a more lenient approach - print sample IDs for debugging
        logging.info("Printing actual participant IDs to debug:")
        for i, ema_id in enumerate(ema_participants[:5]):
            logging.info(f"  EMA ID {i}: '{ema_id}'")
        for i, frag_id in enumerate(frag_participants[:5]):
            logging.info(f"  Frag ID {i}: '{frag_id}'")
        
        # Try direct ID match with date string conversion
        frag_data['date_str'] = frag_data['date'].astype(str)
        alt_merged_data = data_cleaner.merge_datasets(
            ema_data,
            frag_data,
            left_on=['participant_id_clean', 'date_str'],
            right_on=['participant_id_clean', 'date_str'],
            how='inner'
        )
        
        logging.info(f"Alternative merge produced {len(alt_merged_data)} records")
        
        if len(alt_merged_data) > 0:
            logging.info("Using alternative ID matching approach successfully!")
            merged_data = alt_merged_data
        else:
            logging.warning("No matches found with alternative ID approach either")
            return None
    
    # Clean up redundant columns
    columns_to_drop = [
        'participant_id_clean', 
        'date_str' if 'date_str' in merged_data.columns else None
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
    # Define paths - adjust these to match your directory structure
    # EMA data paths
    ema_output_dir = Path("/Users/noamgal/DSProjects/Fragmentation/SURREAL/EMA-Processing/output/normalized")
    
    # Fragmentation data paths
    fragmentation_file = Path("/Volumes/Extreme SSD/SURREAL-DataBackup/HUJI_data-main/processed/fragmentation/fragmentation_all_metrics.csv")
    
    # Output path
    output_dir = Path("/Users/noamgal/DSProjects/Fragmentation/SURREAL/processed/daily_ema_fragmentation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify input files exist
    if not ema_output_dir.exists():
        logging.error(f"EMA directory not found: {ema_output_dir}")
        return
    
    if not fragmentation_file.exists():
        logging.error(f"Fragmentation file not found: {fragmentation_file}")
        return
    
    logging.info(f"EMA directory: {ema_output_dir}")
    logging.info(f"Fragmentation file: {fragmentation_file}")
    
    # Load EMA daily averages
    daily_ema = load_normalized_ema_data(ema_output_dir)
    
    # Load fragmentation data
    frag_data = load_fragmentation_data(fragmentation_file)
    
    # If EMA data loading failed but we want to proceed just with fragmentation data
    if daily_ema is None:
        logging.warning("No EMA data found. Saving only fragmentation data...")
        if frag_data is not None:
            frag_output_file = output_dir / "fragmentation_only.csv"
            frag_data.to_csv(frag_output_file, index=False)
            logging.info(f"Fragmentation data saved to: {frag_output_file}")
            return
        else:
            logging.error("No data available to save")
            return
    
    # Merge datasets
    combined_data = merge_ema_and_fragmentation(daily_ema, frag_data)
    
    if combined_data is not None and not combined_data.empty:
        # Save combined data
        output_file = output_dir / "ema_fragmentation_combined.csv"
        combined_data.to_csv(output_file, index=False)
        logging.info(f"Combined data saved to: {output_file}")
        
        # Create a summary with key columns
        key_columns = [
            'date', 'participant_id_ema',
            'STAI-Y-A-6_zstd', 'STAI-Y-A-6_raw', 
            'CES-D-8_zstd', 'CES-D-8_raw',
            'digital_fragmentation_index', 'mobility_fragmentation_index', 'overlap_fragmentation_index',
            'digital_episode_count', 'mobility_episode_count', 'overlap_episode_count',
            'digital_total_duration', 'mobility_total_duration', 'overlap_total_duration'
        ]
        
        # Some columns might not exist if data is missing
        existing_columns = [col for col in key_columns if col in combined_data.columns]
        summary_data = combined_data[existing_columns].copy()
        
        # Save summary data
        summary_file = output_dir / "ema_fragmentation_summary.csv"
        summary_data.to_csv(summary_file, index=False)
        logging.info(f"Summary data saved to: {summary_file}")
        
        # Print basic stats
        logging.info("\nCombined Dataset Summary:")
        logging.info(f"Total records: {len(combined_data)}")
        logging.info(f"Unique participants: {combined_data['participant_id_ema'].nunique()}")
        logging.info(f"Date range: {combined_data['date'].min()} to {combined_data['date'].max()}")
        
        # Count records with different data types
        has_stai = (~combined_data['STAI-Y-A-6_zstd'].isna()).sum() if 'STAI-Y-A-6_zstd' in combined_data.columns else 0
        has_cesd = (~combined_data['CES-D-8_zstd'].isna()).sum() if 'CES-D-8_zstd' in combined_data.columns else 0
        has_digital = (~combined_data['digital_fragmentation_index'].isna()).sum() if 'digital_fragmentation_index' in combined_data.columns else 0
        has_mobility = (~combined_data['mobility_fragmentation_index'].isna()).sum() if 'mobility_fragmentation_index' in combined_data.columns else 0
        has_overlap = (~combined_data['overlap_fragmentation_index'].isna()).sum() if 'overlap_fragmentation_index' in combined_data.columns else 0
        
        logging.info(f"Records with STAI data: {has_stai} ({has_stai/len(combined_data)*100:.1f}%)")
        logging.info(f"Records with CES-D data: {has_cesd} ({has_cesd/len(combined_data)*100:.1f}%)")
        logging.info(f"Records with digital fragmentation: {has_digital} ({has_digital/len(combined_data)*100:.1f}%)")
        logging.info(f"Records with mobility fragmentation: {has_mobility} ({has_mobility/len(combined_data)*100:.1f}%)")
        logging.info(f"Records with overlap fragmentation: {has_overlap} ({has_overlap/len(combined_data)*100:.1f}%)")
    else:
        logging.warning("No combined data generated. Saving individual datasets...")
        
        # Save individual datasets
        if daily_ema is not None:
            ema_output_file = output_dir / "ema_daily_averages.csv"
            daily_ema.to_csv(ema_output_file, index=False)
            logging.info(f"EMA daily averages saved to: {ema_output_file}")
        
        if frag_data is not None:
            frag_output_file = output_dir / "fragmentation_metrics.csv"
            frag_data.to_csv(frag_output_file, index=False)
            logging.info(f"Fragmentation data saved to: {frag_output_file}")
        
        logging.error("Failed to create combined dataset. Check logs for details.")

if __name__ == "__main__":
    main()