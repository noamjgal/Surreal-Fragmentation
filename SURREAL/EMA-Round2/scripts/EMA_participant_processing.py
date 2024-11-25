import pandas as pd
import logging
from pathlib import Path
import json
import sys
import os

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

def process_participant_ema(participant_id, date, ema_num, ema_data, mappings_df):
    """Process a single EMA response for a participant."""
    # Filter data for this participant, date, and EMA
    participant_ema = ema_data[
        (ema_data['Participant ID'] == participant_id) &
        (pd.to_datetime(ema_data['Date form sent']).dt.date == date)
        # Add additional filtering for specific EMA number if available in your data
    ]
    
    # Process each scale
    scale_results = {}
    for scale in mappings_df['Scale'].unique():
        scale_questions = mappings_df[mappings_df['Scale'] == scale]
        scale_responses = {}
        
        for _, question in scale_questions.iterrows():
            variable = question['Variable']
            if variable in participant_ema.columns:
                response = participant_ema[variable].iloc[0]
                # Use mapping to interpret response
                scale_responses[variable] = {
                    'response': response,
                    'interpretation': interpret_response(response, question)
                }
        
        scale_results[scale] = scale_responses
    
    return scale_results

def interpret_response(response, question_mapping):
    """Interpret a response using the mapping information."""
    try:
        hebrew_dict = json.loads(question_mapping['Hebrew_dict'].replace("'", '"'))
        eng_dict = json.loads(question_mapping['Eng_dict'].replace("'", '"'))
        
        # Add interpretation logic based on the dictionaries and question properties
        # This will depend on your specific needs
        
        return {
            'numeric_value': hebrew_dict.get(response, None),
            'english_translation': eng_dict.get(response, None),
            'scale_points': question_mapping['Points'],
            'correct_order': question_mapping['Correct_Order']
        }
    except Exception as e:
        logging.error(f"Error interpreting response: {e}")
        return None

def create_daily_summary(participant_id, date, ema_results):
    """Create a summary of all EMAs for a day."""
    summary = {
        'participant_id': participant_id,
        'date': str(date),
        'ema_count': len(ema_results),
        'scale_summaries': {}
    }
    
    # Aggregate results across EMAs
    for ema_num, ema_data in ema_results.items():
        for scale, scale_data in ema_data.items():
            if scale not in summary['scale_summaries']:
                summary['scale_summaries'][scale] = []
            summary['scale_summaries'][scale].append(scale_data)
    
    return summary

def main():
    # Load data
    mappings_df = pd.read_excel(
        Path(project_root) / "data" / "raw" / "Corrected-Response-Mappings.xlsx",
        sheet_name="processed_response_mappings"
    )
    ema_data = pd.read_csv(
        Path(project_root) / "data" / "raw" / "comprehensive_ema_data_eng_updated.csv"
    )
    
    # Process each participant
    for participant_id in ema_data['Participant ID'].unique():
        participant_data = ema_data[ema_data['Participant ID'] == participant_id]
        dates = pd.to_datetime(participant_data['Date form sent']).dt.date.unique()
        
        for date in dates:
            ema_results = {}
            
            # Process each EMA
            for ema_num in range(1, 4):
                results = process_participant_ema(participant_id, date, ema_num, participant_data, mappings_df)
                ema_results[ema_num] = results
                
                # Save individual EMA results
                output_path = Path(project_root) / "output" / "participants" / f"participant_{participant_id}" / str(date) / f"EMA_{ema_num}.json"
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
            
            # Create and save daily summary
            summary = create_daily_summary(participant_id, date, ema_results)
            summary_path = Path(project_root) / "output" / "participants" / f"participant_{participant_id}" / str(date) / "daily_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main() 