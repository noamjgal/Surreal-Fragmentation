import pandas as pd
from pathlib import Path
import logging
import sys
from datetime import datetime, timedelta
import os


# Setup
project_root = str(Path(__file__).parent.parent)
OUTPUT_DIR = Path(project_root) / "output"
sys.path.append(project_root)

logger = logging.basicConfig(
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

    # Standardize procrastination variables
    mask = merged_data['Variable_y'].str.contains('PROCRASTINATION', na=False)
    merged_data.loc[mask, 'Variable_y'] = 'PROCRASTINATION'

    # List of columns to drop
    columns_to_drop = [
        'Trigger start date (configured)',
        'Question_matched',
        'English_dict_x',
        'Hebrew_dict_x',
        'Hebrew_dict_y',
        'English_Question_x',
        'Variable_x',
        'Trigger conditions ID',
        'Trigger conditions name',
        'Trigger conditions period',
        'Number of reminders sent',
        'Dates of reminders sent',
        'Field type ID',
        'Points',
        'Start date of the trigger',
        'End date of the trigger',
        'Trigger ID',
        'Trigger duration',
        'Display duration of the form',
        'Reordered',
        'Eng_dict',
        'Correct_Order',
        'Response_Counts'
    ]

    # Only drop columns that exist in the dataframe
    columns_to_drop = [col for col in columns_to_drop if col in merged_data.columns]
    merged_data = merged_data.drop(columns=columns_to_drop)
    
    # Rename columns to remove _y suffix
    merged_data = merged_data.rename(columns={
        'English_Question_y': 'English_Question',
        'Variable_y': 'Variable',
        'Hebrew_dict_processed': 'Hebrew_dict',
        'Eng_dict_processed': 'English_dict'
    })
    
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
        ema_data = pd.read_csv(Path(project_root) / "data" / "reordered" / "recoded_ema_data.csv")
        
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return
    
    # Drop test participants
    ema_data = ema_data[~ema_data['Participant ID'].str.contains('est', case=False)]
    logging.info(f"Dropped test participants. Remaining participants: {len(ema_data['Participant ID'].unique())}")
    
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

    # After creating all summaries, analyze STAI and CES-D responses
    stai_df = pd.read_csv(OUTPUT_DIR / "STAI-Y-A-6_summary.csv").set_index('Participant ID')
    ces_df = pd.read_csv(OUTPUT_DIR / "CES-D-8_summary.csv").set_index('Participant ID')

    # Drop test participants from these dataframes as well
    stai_df = stai_df[~stai_df.index.str.contains('est', case=False)]
    ces_df = ces_df[~ces_df.index.str.contains('est', case=False)]

    # Count days with sum ≥3 responses across both scales
    valid_days = []
    total_days = 0
    days_with_enough_responses = 0
    total_days_with_any_data = 0

    for day in range(1, 8):  # Days 1-7
        day_col = f'Day_{day}'
        stai_responses = stai_df[day_col]
        ces_responses = ces_df[day_col]
        
        # Count days where sum of both scales is ≥3 responses
        total_responses = stai_responses + ces_responses
        valid_day = total_responses >= 3
        days_with_responses = sum(valid_day)
        
        # Count days with any data
        days_with_any = (total_responses > 0).sum()
        total_days_with_any_data += days_with_any
        
        valid_days.append({
            'Day': day,
            'Days_with_3+_responses': days_with_responses,
            'Days_with_any_data': days_with_any,
            'Total_participants': len(stai_responses),
            'Percentage_of_all_days': (days_with_responses / len(stai_responses)) * 100,
            'Percentage_of_active_days': (days_with_responses / days_with_any * 100) if days_with_any > 0 else 0
        })
        
        total_days += len(stai_responses)
        days_with_enough_responses += days_with_responses

    # Create and save the report
    valid_days_df = pd.DataFrame(valid_days)
    valid_days_df.to_csv(OUTPUT_DIR / "valid_days_analysis.csv", index=False)
    
    overall_percentage = (days_with_enough_responses / total_days) * 100
    active_days_percentage = (days_with_enough_responses / total_days_with_any_data) * 100
    
    logging.info("\nDays with sum of 3+ responses across STAI and CES-D scales:")
    logging.info(valid_days_df.to_string(index=False))
    logging.info(f"\nOverall percentage of all possible days with 3+ total responses: {overall_percentage:.2f}%")
    logging.info(f"Percentage of days with any data that have 3+ total responses: {active_days_percentage:.2f}%")

    # Create boolean summary for days with 3+ total responses
    boolean_summary = pd.DataFrame(index=stai_df.index)
    
    for day in range(1, 8):  # Days 1-7
        day_col = f'Day_{day}'
        # Mark True if sum of responses is ≥3 for that day
        boolean_summary[day_col] = (stai_df[day_col] + ces_df[day_col]) >= 3
    
    # Add total count of valid days for each participant
    boolean_summary['Total_Valid_Days'] = boolean_summary.sum(axis=1)
    
    # Save the boolean summary
    boolean_summary.to_csv(OUTPUT_DIR / "valid_days_boolean_summary.csv")
    
    logging.info("\nBoolean Summary of Days with 3+ total responses (True = sum of both scales is 3+ responses):")
    logging.info(boolean_summary.to_string())
    
    # Calculate percentage of valid days per participant
    boolean_summary['Percentage_Valid_Days'] = (boolean_summary['Total_Valid_Days'] / 7) * 100
    
    logging.info("\nParticipant completion rates (days with 3+ total responses across both scales):")
    logging.info(boolean_summary['Percentage_Valid_Days'].describe().to_string())

if __name__ == "__main__":
    main() 
    