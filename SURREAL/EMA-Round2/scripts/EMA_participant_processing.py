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
    base_dirs = ["data/raw", "data/processed/participants", "logs"]
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
    
    ema_responses = []
    
    for _, response in participant_ema.iterrows():
        # Match based on Form name and Question name
        mapping = mappings_df[
            (mappings_df['Form'] == response['Form name']) &
            (mappings_df['Question'] == response['Question name'])
        ]
        
        if not mapping.empty:
            mapping = mapping.iloc[0]
            try:
                ema_responses.append({
                    'Scale': mapping['Scale'],
                    'Variable': mapping['Variable'],
                    'Question_Hebrew': mapping['Question'],
                    'Question_English': mapping['English_Question'],
                    'Response_Hebrew': response['Responses name'],
                    'Hebrew_dict': mapping['Hebrew_dict'],
                    'Eng_dict': mapping['Eng_dict'],
                    'Numeric_Value': json.loads(mapping['Hebrew_dict'].replace("'", '"')).get(response['Responses name'], None),
                    'Points': mapping['Points'],
                    'Correct_Order': mapping['Correct_Order']
                })
            except Exception as e:
                logging.error(f"Error processing response for {mapping['Variable']}: {e}")
    
    return pd.DataFrame(ema_responses)

def calculate_participant_statistics(ema_data):
    """Calculate aggregate statistics for all participants."""
    stats = {
        'total_participants': len(ema_data['Participant ID'].unique()),
        'total_emas': 0,
        'participants_with_full_week': 0,
        'participant_summaries': {},
        'daily_response_counts': {1: 0, 2: 0, 3: 0, 4: 0},  # Overall daily distribution
        'error_summary': {}
    }
    
    for participant_id in ema_data['Participant ID'].unique():
        participant_data = ema_data[ema_data['Participant ID'] == participant_id]
        dates = pd.to_datetime(participant_data['Date form sent']).dt.date.unique()
        
        # Count unique form completions per day
        daily_counts = participant_data.groupby([
            pd.to_datetime(participant_data['Date form sent']).dt.date,
            'Form name'
        ]).size().reset_index()
        
        # Count number of unique forms per day
        forms_per_day = daily_counts.groupby(daily_counts['Date form sent']).size()
        
        for count in forms_per_day:
            if count >= 4:
                stats['daily_response_counts'][4] += 1
            else:
                stats['daily_response_counts'][count] += 1
        
        # Calculate total unique EMAs
        unique_emas = len(participant_data.groupby([
            pd.to_datetime(participant_data['Date form sent']).dt.date,
            'Form name'
        ]))
        
        expected_emas = len(dates) * 3  # 3 unique forms per day
        completion_rate = unique_emas / expected_emas if expected_emas > 0 else 0
        
        stats['total_emas'] += unique_emas
        if len(dates) == 7:
            stats['participants_with_full_week'] += 1
            
        stats['participant_summaries'][participant_id] = {
            'days': len(dates),
            'emas': unique_emas,
            'expected_emas': expected_emas,
            'completion_rate': f"{completion_rate:.2%}",
            'complete_week': len(dates) == 7
        }
    
    return stats

def calculate_scale_averages(participant_dir):
    """Calculate averages for each scale from all EMAs."""
    all_emas = []
    for ema_file in participant_dir.glob("EMA_*.csv"):
        ema_df = pd.read_csv(ema_file)
        all_emas.append(ema_df)
    
    if all_emas:
        combined_df = pd.concat(all_emas)
        scale_averages = combined_df.groupby('Scale').agg({
            'Numeric_Value': ['mean', 'std', 'count'],
            'Points': 'first'
        }).reset_index()
        
        scale_averages.columns = ['Scale', 'Mean', 'Std', 'Count', 'Points']
        return scale_averages
    return None

def print_final_report(stats, error_dict):
    """Print final processing report."""
    logging.info("\n=== Processing Summary ===")
    logging.info(f"Total Participants: {stats['total_participants']}")
    logging.info(f"Total Unique EMAs Processed: {stats['total_emas']}")
    logging.info(f"Participants with Complete Week: {stats['participants_with_full_week']}")
    
    logging.info("\nDaily Response Distribution (based on unique form completions):")
    total_days = sum(stats['daily_response_counts'].values())
    for responses, count in stats['daily_response_counts'].items():
        percentage = (count / total_days * 100) if total_days > 0 else 0
        if responses == 4:
            logging.info(f"Days with {responses}+ forms: {count} ({percentage:.1f}%)")
        else:
            logging.info(f"Days with {responses} form(s): {count} ({percentage:.1f}%)")
    
    if error_dict:
        logging.info("\nProcessing Errors:")
        for error_type, count in error_dict.items():
            logging.info(f"- {error_type}: {count} occurrences")
    
    logging.info("\nParticipant Details:")
    df = pd.DataFrame.from_dict(stats['participant_summaries'], orient='index')
    print(df.sort_values(['complete_week', 'emas'], ascending=[False, False]))
    
    # Calculate overall completion rate
    total_expected = df['expected_emas'].sum()
    overall_completion = stats['total_emas'] / total_expected if total_expected > 0 else 0
    logging.info(f"\nOverall completion rate: {overall_completion:.2%}")

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

    stats = calculate_participant_statistics(ema_data)
    error_count = 0
    success_count = 0
    
    for participant_id in ema_data['Participant ID'].unique():
        participant_dir = Path(project_root) / "data" / "processed" / "participants"
        os.makedirs(participant_dir, exist_ok=True)
        
        all_responses = []
        
        # Process each participant's responses
        participant_data = ema_data[ema_data['Participant ID'] == participant_id]
        for _, row in participant_data.iterrows():
            date = pd.to_datetime(row['Date form sent']).date()
            time = pd.to_datetime(row['Date form sent']).time()
            
            try:
                results_df = process_participant_ema(
                    participant_id, date, time,
                    participant_data[pd.to_datetime(participant_data['Date form sent']).dt.time == time],
                    mappings_df
                )
                
                # Add datetime column
                results_df['datetime'] = pd.to_datetime(f"{date} {time}")
                all_responses.append(results_df)
                
            except Exception as e:
                logging.error(f"Error processing {participant_id} on {date} at {time}: {e}")
                error_count += 1
        
        if all_responses:
            # Combine all responses and sort by datetime
            combined_responses = pd.concat(all_responses).sort_values('datetime')
            output_path = participant_dir / f"participant_{participant_id}_responses.csv"
            combined_responses.to_csv(output_path, index=False)

    print_final_report(stats, {})
    
    # Print sample output structure
    logging.info("\nSample Output Structure:")
    sample_file = next(Path(project_root).glob("data/processed/participants/participant_*.csv"))
    if sample_file.exists():
        print(f"\n- {sample_file.name}")
        df = pd.read_csv(sample_file)
        print(df.head(2))

if __name__ == "__main__":
    main() 