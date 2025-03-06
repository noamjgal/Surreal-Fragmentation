# std.py
# standardize ema responses 
import pandas as pd
import numpy as np
from pathlib import Path
import logging

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
    STAI uses 1-4 scale, CES-D uses 1-5 scale.
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

def normalize_responses(participant_data):
    """
    Normalize CES and STAI responses at the participant level using z-scores.
    Performs separate standardization for each scale and variable.
    STAI uses 1-4 scale, CES-D uses 1-5 scale.
    """
    # Create a copy to avoid modifying the original
    data = participant_data.copy()
    
    # Filter only for STAI and CES-D responses
    data = data[data['Scale'].isin(['STAI-Y-A-6', 'CES-D-8'])].copy()
    
    if len(data) == 0:
        logging.warning("No STAI or CES-D responses found in data")
        return data
    
    # Convert Response Key to numeric and ensure it's working
    data['Response Key'] = data['Response Key'].apply(convert_to_numeric)
    
    # Check for any non-numeric values after conversion
    non_numeric = data['Response Key'].isna()
    if non_numeric.any():
        logging.warning(f"Found {non_numeric.sum()} non-numeric responses that will be excluded")
    
    # Validate response ranges
    stai_mask = data['Scale'] == 'STAI-Y-A-6'
    ces_mask = data['Scale'] == 'CES-D-8'
    
    invalid_stai = data[stai_mask & data['Response Key'].notna()]['Response Key'].apply(
        lambda x: x < 1 or x > 4
    )
    invalid_ces = data[ces_mask & data['Response Key'].notna()]['Response Key'].apply(
        lambda x: x < 1 or x > 5
    )
    
    if invalid_stai.any():
        logging.warning(f"Found {invalid_stai.sum()} STAI responses outside valid range (1-4)")
        data.loc[invalid_stai.index, 'Response Key'] = np.nan
        
    if invalid_ces.any():
        logging.warning(f"Found {invalid_ces.sum()} CES-D responses outside valid range (1-5)")
        data.loc[invalid_ces.index, 'Response Key'] = np.nan
    
    # Add reversed scores first
    data['score_reversed'] = data.apply(lambda row: reverse_score(row), axis=1)
    
    # Initialize columns
    data['score_zstd'] = np.nan
    
    # Process each scale as a whole
    for scale in ['STAI-Y-A-6', 'CES-D-8']:
        scale_mask = data['Scale'] == scale
        
        if scale_mask.any():
            # Work with reversed scores for all variables in this scale
            scale_data = data.loc[scale_mask, 'score_reversed']
            
            # Only process if we have valid numeric data
            valid_data = scale_data.dropna()
            
            if len(valid_data) > 0:
                # Z-score standardization across all variables in the scale
                mean = valid_data.mean()
                std = valid_data.std()
                
                if std > 0:
                    data.loc[scale_mask & scale_data.notna(), 'score_zstd'] = (
                        (scale_data - mean) / std
                    ).where(scale_data.notna())
                else:
                    logging.warning(f"Zero standard deviation for entire {scale} scale, setting z-scores to 0")
                    data.loc[scale_mask & scale_data.notna(), 'score_zstd'] = 0
            else:
                logging.warning(f"No valid data points for {scale}")
    
    return data

def create_ema_summary_by_scale(data):
    """
    Create summary statistics for each scale as a whole.
    """
    summary = []
    
    for scale in ['STAI-Y-A-6', 'CES-D-8']:
        # Get data for this scale
        scale_data = data[data['Scale'] == scale].copy()
        
        if len(scale_data) > 0:
            summary.append({
                'Participant_ID': scale_data['Participant_ID'].iloc[0],
                'Scale': scale,
                'N_Questions': len(scale_data),
                'N_Variables': scale_data['Variable'].nunique(),
                'N_Valid_Responses': scale_data['Response Key'].notna().sum(),
                'Raw_Mean': scale_data['Response Key'].mean(),
                'Raw_SD': scale_data['Response Key'].std(),
                'Reversed_Mean': scale_data['score_reversed'].mean(),
                'Reversed_SD': scale_data['score_reversed'].std(),
                'Zstd_Mean': scale_data['score_zstd'].mean(),
                'Zstd_SD': scale_data['score_zstd'].std(),
                'First_Response': scale_data['datetime'].min(),
                'Last_Response': scale_data['datetime'].max(),
                'N_Days': scale_data['datetime'].dt.date.nunique()
            })
    
    return pd.DataFrame(summary)

def create_ema_summary_by_variable(data):
    """
    Create summary statistics for each scale and variable combination.
    """
    summary = []
    
    for scale in ['STAI-Y-A-6', 'CES-D-8']:
        scale_data = data[data['Scale'] == scale].copy()
        
        for variable in scale_data['Variable'].unique():
            # Get data for this scale and variable
            var_data = scale_data[scale_data['Variable'] == variable].copy()
            
            if len(var_data) > 0:
                summary.append({
                    'Participant_ID': var_data['Participant_ID'].iloc[0],
                    'Scale': scale,
                    'Variable': variable,
                    'N_Responses': len(var_data),
                    'N_Valid_Responses': var_data['Response Key'].notna().sum(),
                    'Raw_Mean': var_data['Response Key'].mean(),
                    'Raw_SD': var_data['Response Key'].std(),
                    'Reversed_Mean': var_data['score_reversed'].mean(),
                    'Reversed_SD': var_data['score_reversed'].std(),
                    'Zstd_Mean': var_data['score_zstd'].mean(),
                    'Zstd_SD': var_data['score_zstd'].std(),
                    'First_Response': var_data['datetime'].min(),
                    'Last_Response': var_data['datetime'].max(),
                    'N_Days': var_data['datetime'].dt.date.nunique()
                })
    
    return pd.DataFrame(summary)

def main():
    project_root = str(Path(__file__).parent.parent)
    data_dir = Path(project_root) / "data" / "processed" / "participants"
    output_dir = Path(project_root) / "output" / "normalized"
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    scale_summaries = []
    variable_summaries = []
    
    # Process each participant's data
    for file in data_dir.glob("participant_*.csv"):
        try:
            # Read participant data
            participant_data = pd.read_csv(file)
            participant_data['datetime'] = pd.to_datetime(participant_data['datetime'])
            
            participant_id = file.stem.replace('participant_', '')
            participant_data['Participant_ID'] = participant_id
            logging.info(f"Processing {participant_id}")
            
            # Normalize responses
            normalized_data = normalize_responses(participant_data)
            
            # Save normalized data
            output_file = output_dir / f"normalized_{file.name}"
            normalized_data.to_csv(output_file, index=False)
            logging.info(f"Saved normalized data to: {output_file}")
            
            # Create summaries for this participant
            scale_summary = create_ema_summary_by_scale(normalized_data)
            variable_summary = create_ema_summary_by_variable(normalized_data)
            
            scale_summaries.append(scale_summary)
            variable_summaries.append(variable_summary)
            
            logging.info(f"Processed {participant_id}")
            
        except Exception as e:
            logging.error(f"Error processing {file.name}: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
    
    # Create and save overall summaries
    if scale_summaries and variable_summaries:
        # Combine and save scale-level summary
        overall_scale_summary = pd.concat(scale_summaries, ignore_index=True)
        overall_scale_summary.to_csv(output_dir / "overall_summary_by_scale.csv", index=False)
        
        # Combine and save variable-level summary
        overall_variable_summary = pd.concat(variable_summaries, ignore_index=True)
        overall_variable_summary.to_csv(output_dir / "overall_summary_by_variable.csv", index=False)
        
        logging.info("Created overall summaries")

if __name__ == "__main__":
    main()