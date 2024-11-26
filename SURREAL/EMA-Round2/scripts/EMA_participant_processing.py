import pandas as pd
import logging
from pathlib import Path
import json
import sys
import os
from datetime import datetime

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def setup_directory_structure():
    """Create all necessary directories."""
    base_dirs = ["data/raw", "data/processed/participants", "logs", "output"]
    for dir_path in base_dirs:
        os.makedirs(Path(project_root) / dir_path, exist_ok=True)

def print_data_overview(mappings_df, ema_data):
    """Print overview of input data structure."""
    logging.info("\n=== Input Data Overview ===")
    logging.info("\nColumns in Mappings DataFrame:")
    print(mappings_df.columns)
    logging.info("\nMappings DataFrame Sample (first 3 rows):")
    print(mappings_df[['Form', 'Variable', 'Scale', 'Points']].head(3))
    logging.info("\nUnique Scales:")
    print(mappings_df['Scale'].unique())
    
    logging.info("\nColumns in EMA Data:")
    print(ema_data.columns)
    logging.info("\nEMA Data Sample (first 3 rows):")
    print(ema_data[['Participant ID', 'Date form sent', 'Form name', 'Variable', 'Responses name']].head(3))
    logging.info(f"\nTotal number of participants: {len(ema_data['Participant ID'].unique())}")
    logging.info(f"Date range: {ema_data['Date form sent'].min()} to {ema_data['Date form sent'].max()}")

def process_participant_ema(participant_id, date, ema_time, ema_data, mappings_df):
    """Process a single EMA response for a participant."""
    participant_ema = ema_data[
        (ema_data['Participant ID'] == participant_id) &
        (pd.to_datetime(ema_data['Date form sent']).dt.date == date)
    ]
    
    # Create DataFrame for this EMA's responses
    ema_responses = []
    
    for _, question in mappings_df.iterrows():
        responses = participant_ema[participant_ema['Variable'] == question['Variable']]
        if not responses.empty:
            response = responses['Responses name'].iloc[0]
            try:
                hebrew_dict = json.loads(question['Hebrew_dict'].replace("'", '"'))
                numeric_value = hebrew_dict.get(response, None)
                
                ema_responses.append({
                    'Scale': question['Scale'],
                    'Variable': question['Variable'],
                    'Response': response,
                    'Numeric_Value': numeric_value,
                    'Points': question['Points'],
                    'Correct_Order': question['Correct_Order']
                })
            except Exception as e:
                logging.error(f"Error processing {question['Variable']}: {e}")
    
    return pd.DataFrame(ema_responses)

def main():
    setup_directory_structure()
    
    # Load data
    try:
        mappings_df = pd.read_excel(
            Path(project_root) / "data" / "raw" / "Corrected-Response-Mappings.xlsx",
            sheet_name="processed_response_mappings"
        )
        ema_data = pd.read_csv(
            Path(project_root) / "data" / "raw" / "comprehensive_ema_data_eng_updated.csv"
        )
        print_data_overview(mappings_df, ema_data)
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    error_count = 0
    success_count = 0
    
    # Process each participant
    for participant_id in ema_data['Participant ID'].unique():
        participant_data = ema_data[ema_data['Participant ID'] == participant_id]
        dates = pd.to_datetime(participant_data['Date form sent']).dt.date.unique()
        
        # Track the number of days and observations for each participant
        num_days = 0
        num_observations = 0
        observed_days = set()
        observed_observations = set()
        
        for date in dates:
            participant_dir = Path(project_root) / "data" / "processed" / "participants" / f"participant_{participant_id}" / str(date)
            os.makedirs(participant_dir, exist_ok=True)
            
            # Group by time of day
            day_data = participant_data[pd.to_datetime(participant_data['Date form sent']).dt.date == date]
            times = pd.to_datetime(day_data['Date form sent']).dt.time
            
            for time in times.unique():
                try:
                    # Create filename with timestamp
                    timestamp = datetime.combine(date, time).strftime('%Y%m%d_%H%M')
                    ema_num = times.unique().tolist().index(time) + 1
                    
                    results_df = process_participant_ema(
                        participant_id, date, time,
                        day_data[pd.to_datetime(day_data['Date form sent']).dt.time == time],
                        mappings_df
                    )
                    
                    # Save as CSV
                    output_path = participant_dir / f"EMA_{ema_num}_{timestamp}.csv"
                    results_df.to_csv(output_path, index=False)
                    success_count += 1
                    
                    # Track the number of observations for each day
                    num_observations += len(results_df)
                    observed_observations.update(results_df['Variable'])
                
                except Exception as e:
                    logging.error(f"Error processing {participant_id} on {date} at {time}: {e}")
                    error_count += 1
            
            # Track the number of days for each participant
            num_days += 1
            observed_days.add(date)
            
            # Create daily summary
            try:
                all_emas = pd.concat([pd.read_csv(f) for f in participant_dir.glob("EMA_*.csv")])
                summary = all_emas.groupby(['Scale', 'Variable']).agg({
                    'Numeric_Value': ['mean', 'std', 'count'],
                    'Points': 'first',
                    'Correct_Order': 'first'
                }).reset_index()
                
                summary.to_csv(participant_dir / "daily_summary.csv", index=False)
            except Exception as e:
                logging.error(f"Error creating daily summary for {participant_id} on {date}: {e}")
                error_count += 1

        # Log the percentage of participants that have all seven days
        if num_days == 7:
            logging.info(f"Participant {participant_id} has all seven days")
        
        # Log the percentage of days observed that have all 3 observations
        if num_observations == 3:
            logging.info(f"Participant {participant_id} has all three observations for {num_days} days")
        
        # Log the percentage of desired observations that are here
        desired_observations = len(mappings_df) * 3
        if num_observations == desired_observations:
            logging.info(f"Participant {participant_id} has all {desired_observations} observations for {num_days} days")
        else:
            logging.info(f"Participant {participant_id} has {num_observations} observations for {num_days} days")
            
    # Log the percentage of days observed that have all 3 observations
    logging.info(f"\n=== Processing Complete ===")
    logging.info(f"Successfully processed: {success_count} EMAs")
    logging.info(f"Errors encountered: {error_count}")
    
    # Print sample of output structure
    sample_participant = next(iter(ema_data['Participant ID'].unique()))
    sample_dir = next(iter(Path(project_root).glob(f"data/processed/participants/participant_{sample_participant}/*")))
    logging.info(f"\nSample output structure for {sample_participant}:")
    for file in sample_dir.glob("*.csv"):
        print(f"- {file.name}")
        df = pd.read_csv(file)
        print(df.head(2))
        print()

if __name__ == "__main__":
    main() 