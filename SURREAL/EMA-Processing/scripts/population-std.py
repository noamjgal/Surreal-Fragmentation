import pandas as pd
import numpy as np
from pathlib import Path
import logging
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define positive items that need to be reverse scored
POSITIVE_ITEMS = {
    'STAI-Y-A-6': ['CALM', 'PEACE', 'SATISFACTION'],
    'CES-D-8': ['HAPPY', 'ENJOYMENT_RECENT']
}

def convert_to_numeric(value):
    """
    Convert response values to numeric, handling different formats and validating ranges.
    """
    if pd.isna(value):
        return np.nan
    try:
        num_value = pd.to_numeric(value)
        return num_value
    except (ValueError, TypeError):
        try:
            numeric_str = ''.join(c for c in str(value) if c.isdigit() or c == '.')
            return float(numeric_str) if numeric_str else np.nan
        except (ValueError, TypeError):
            return np.nan

def reverse_score(row):
    """
    Reverse scores for positive items.
    STAI uses 1-4 scale, CES-D uses 1-5 scale.
    """
    if pd.isna(row['Response Key']):
        return np.nan
        
    if row['Scale'] in POSITIVE_ITEMS and row['Variable'] in POSITIVE_ITEMS[row['Scale']]:
        if row['Scale'] == 'STAI-Y-A-6':
            return 5 - row['Response Key']  # Reverse 1-4 scale
        elif row['Scale'] == 'CES-D-8':
            return 6 - row['Response Key']  # Reverse 1-5 scale
    return row['Response Key']

def main():
    project_root = str(Path(__file__).parent.parent)
    data_dir = Path(project_root) / "data" / "processed" / "participants"
    output_dir = Path(project_root) / "output" / "normalized_population"
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: First pass to collect all data for population statistics
    logging.info("First pass: Collecting data for population statistics")
    
    all_data = []
    participant_files = list(data_dir.glob("participant_*.csv"))
    
    for file in participant_files:
        try:
            # Read participant data
            participant_data = pd.read_csv(file)
            participant_id = file.stem.replace('participant_', '')
            participant_data['Participant_ID'] = participant_id
            
            # Convert date column
            participant_data['datetime'] = pd.to_datetime(participant_data['datetime'])
            
            # Filter only for STAI and CES-D responses
            scale_data = participant_data[participant_data['Scale'].isin(['STAI-Y-A-6', 'CES-D-8'])].copy()
            
            if len(scale_data) > 0:
                # Convert Response Key to numeric
                scale_data['Response Key'] = scale_data['Response Key'].apply(convert_to_numeric)
                
                # Add reversed scores
                scale_data['score_reversed'] = scale_data.apply(lambda row: reverse_score(row), axis=1)
                
                all_data.append(scale_data)
            
        except Exception as e:
            logging.error(f"Error processing {file.name} in first pass: {str(e)}")
    
    # Combine all data
    if not all_data:
        logging.error("No valid data found. Exiting.")
        return
        
    combined_data = pd.concat(all_data, ignore_index=True)
    logging.info(f"Combined data: {len(combined_data)} rows from {len(participant_files)} participants")
    
    # Step 2: Calculate population statistics for each scale
    population_stats = {}
    
    for scale in ['STAI-Y-A-6', 'CES-D-8']:
        scale_mask = combined_data['Scale'] == scale
        scale_data = combined_data.loc[scale_mask, 'score_reversed'].dropna()
        
        if len(scale_data) > 0:
            mean = scale_data.mean()
            std = scale_data.std()
            
            if std > 0:
                population_stats[scale] = {'mean': mean, 'std': std}
                logging.info(f"Population statistics for {scale}: mean={mean:.4f}, std={std:.4f}")
            else:
                logging.warning(f"Zero standard deviation for {scale} scale in population")
                population_stats[scale] = {'mean': mean, 'std': 1.0}  # Use 1.0 to avoid division by zero
    
    # Step 3: Second pass to apply population standardization
    logging.info("Second pass: Applying population-level standardization")
    
    scale_summaries = []
    variable_summaries = []
    
    for file in participant_files:
        try:
            # Read participant data
            participant_data = pd.read_csv(file)
            participant_data['datetime'] = pd.to_datetime(participant_data['datetime'])
            
            participant_id = file.stem.replace('participant_', '')
            participant_data['Participant_ID'] = participant_id
            logging.info(f"Processing {participant_id}")
            
            # Filter only for STAI and CES-D responses
            normalized_data = participant_data[participant_data['Scale'].isin(['STAI-Y-A-6', 'CES-D-8'])].copy()
            
            if len(normalized_data) > 0:
                # Convert Response Key to numeric
                normalized_data['Response Key'] = normalized_data['Response Key'].apply(convert_to_numeric)
                
                # Add reversed scores
                normalized_data['score_reversed'] = normalized_data.apply(lambda row: reverse_score(row), axis=1)
                
                # Initialize z-score column
                normalized_data['score_zstd'] = np.nan
                
                # Apply population-level standardization
                for scale in ['STAI-Y-A-6', 'CES-D-8']:
                    if scale in population_stats:
                        scale_mask = normalized_data['Scale'] == scale
                        if scale_mask.any():
                            mean = population_stats[scale]['mean']
                            std = population_stats[scale]['std']
                            
                            normalized_data.loc[scale_mask, 'score_zstd'] = (
                                (normalized_data.loc[scale_mask, 'score_reversed'] - mean) / std
                            ).where(normalized_data.loc[scale_mask, 'score_reversed'].notna())
                            
                # Save normalized data
                output_file = output_dir / f"normalized_{file.name}"
                normalized_data.to_csv(output_file, index=False)
                logging.info(f"Saved population-normalized data to: {output_file}")
                
                # Create summaries for this participant (if needed)
                # [Add your summary creation code here if desired]
                
            else:
                logging.warning(f"No STAI or CES-D data found for {participant_id}")
                
        except Exception as e:
            logging.error(f"Error processing {file.name} in second pass: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
    
    # Save population statistics for reference
    pd.DataFrame(population_stats).to_csv(output_dir / "population_statistics.csv")
    logging.info(f"Population statistics saved to: {output_dir / 'population_statistics.csv'}")
    
    logging.info("Population-level standardization complete")

if __name__ == "__main__":
    main()