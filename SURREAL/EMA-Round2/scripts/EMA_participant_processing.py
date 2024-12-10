import pandas as pd
from pathlib import Path
import logging
import sys
from datetime import datetime, timedelta
import os

# Setup
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def setup_directories():
    """Create basic directory structure."""
    base_dirs = ["data/processed/participants", "output"]
    for dir_path in base_dirs:
        os.makedirs(Path(project_root) / dir_path, exist_ok=True)

def process_participant_data(participant_id, participant_data, mappings_df):
    """Process all EMAs for a single participant into one CSV."""
    # Convert date strings to datetime
    participant_data = participant_data.copy()  # Avoid SettingWithCopyWarning
    participant_data['datetime'] = pd.to_datetime(participant_data['Date form sent'])
    
    # Merge with mappings
    merged_data = participant_data.merge(
        mappings_df,
        how='left',
        left_on=['Form name', 'Question name'],
        right_on=['Form', 'Question']
    )
    # Sort by datetime
    merged_data = merged_data.sort_values('datetime')

    merged_data = merged_data.drop(columns=['Trigger start date (configured)','Question_matched', 'English_dict', 'Hebrew_dict', 'English_Question_x', 'Variable_x', 'Trigger conditions ID', 'Trigger conditions name', 'Trigger conditions period', 'Number of reminders sent', 'Dates of reminders sent', 'Field type ID', 'Points', 'Start date of the trigger', 'End date of the trigger', 'Trigger ID', 'Trigger duration', 'Display duration of the form', 'Reordered'])
    merged_data = merged_data[merged_data['Form name'] != 'Consensus Sleep Diary-M']
    merged_data = merged_data.rename(columns={'Responses ID': 'Response Key', 'Responses name': 'Response Value'})
    
    # Save to CSV
    output_path = Path(project_root) / "data" / "processed" / "participants" / f"participant_{participant_id}.csv"
    merged_data.to_csv(output_path, index=False)
    
    return merged_data

def calculate_daily_summary(ema_data):
    """Calculate summary statistics by day for each participant."""
    # Extract day number from 'Trigger set name'
    ema_data = ema_data.copy()
    
    summary_data = []
    
    for participant_id in ema_data['Participant ID'].unique():
        participant_data = ema_data[ema_data['Participant ID'] == participant_id]
        
        # Count surveys for each day
        daily_counts = {'Participant ID': participant_id}
        for day in range(1, 8):  # Days 1-7
            day_data = participant_data[participant_data['Trigger set name'].str.contains(f'EMA day {day}', case=False)]
            # Count unique timestamps for that day
            unique_surveys = day_data['Date form sent'].nunique()
            daily_counts[f'Day_{day}'] = unique_surveys
        
        summary_data.append(daily_counts)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.set_index('Participant ID')
    
    # Add totals and averages
    summary_df['Total'] = summary_df.sum(axis=1)
    
    return summary_df

def calculate_scale_summary(ema_data, mappings_df):
    """Calculate the number of questions for each scale responded to each day."""
    # Merge with mappings first
    merged_data = ema_data.merge(
        mappings_df,
        how='left',
        left_on=['Form name', 'Question name'],
        right_on=['Form', 'Question']
    )
    
    # Get unique scales (excluding nan)
    scales = [scale for scale in merged_data['Scale'].unique() if pd.notna(scale)]
    
    for scale in scales:
        scale_data = merged_data[merged_data['Scale'] == scale]
        
        summary_data = []
        
        for participant_id in scale_data['Participant ID'].unique():
            participant_data = scale_data[scale_data['Participant ID'] == participant_id]
            
            # Count questions for each day
            daily_counts = {'Participant ID': participant_id}
            for day in range(1, 8):  # Days 1-7
                day_data = participant_data[participant_data['Trigger set name'].str.contains(f'EMA day {day}', case=False)]
                # Count unique questions for that day
                unique_questions = day_data['Question name'].nunique()
                daily_counts[f'Day_{day}'] = unique_questions
            
            summary_data.append(daily_counts)
        
        # Create summary DataFrame for the scale
        if summary_data:  # Only create DataFrame if there's data
            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.set_index('Participant ID')
            
            # Add totals
            summary_df['Total'] = summary_df.sum(axis=1)
            
            # Save to CSV
            output_path = Path(project_root) / "output" / f"{scale}_summary.csv"
            summary_df.to_csv(output_path)
            
            logging.info(f"Scale summary for {scale} saved to {output_path}")
            print(f"\nSummary for {scale}:")
            print(summary_df)

def main():
    setup_directories()
    
    # Load data
    try:
        mappings_df = pd.read_csv(Path(project_root) / "data" / "raw" / "processed_dictionaries.csv")
        ema_data = pd.read_csv(Path(project_root) / "data" / "raw" / "recoded_ema_data.csv")
        
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return
    
    # Process each participant
    for participant_id in ema_data['Participant ID'].unique():
        participant_data = ema_data[ema_data['Participant ID'] == participant_id]
        process_participant_data(participant_id, participant_data, mappings_df)
    
    # Calculate and save daily summary
    summary_df = calculate_daily_summary(ema_data)
    summary_df.to_csv(Path(project_root) / "output" / "daily_summary.csv")
    
    # Calculate and save scale summaries
    calculate_scale_summary(ema_data, mappings_df)
    
    logging.info(f"Processed {len(ema_data['Participant ID'].unique())} participants")
    logging.info("Daily summary saved to output/daily_summary.csv")
    
    # Display summary
    print("\nDaily Survey Completion Summary:")
    print(summary_df)

if __name__ == "__main__":
    main() 