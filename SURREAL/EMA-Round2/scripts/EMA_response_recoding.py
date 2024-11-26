import pandas as pd
import logging
from pathlib import Path
import json
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Add project root to Python path (from EMA_participant_processing.py)
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

def recode_traffic_v9(value):
    """
    Remap 5-point scale to 4-point scale for EMA V9 traffic question
    1->4, 2->3, 3->2, 4->2, 5->1
    """
    mapping = {'1': '4', '2': '3', '3': '2', '4': '2', '5': '1'}
    return mapping.get(str(value), value)

def recode_calm_v7(value):
    """
    Reorder responses for EMA V7 calm question
    1->1, 2->3, 3->2, 4->4
    """
    mapping = {'1': '1', '2': '3', '3': '2', '4': '4'}
    return mapping.get(str(value), value)

def reverse_scale(value, max_value=4):
    """
    Reverse numerical coding for FALSE items
    For 4-point scale: 1->4, 2->3, 3->2, 4->1
    For 5-point scale: 1->5, 2->4, 3->3, 4->2, 5->1
    """
    try:
        value = str(value)
        if not value.isdigit():
            return value
        value = int(value)
        return str(max_value - value + 1)
    except:
        return value

def process_response(response, question_metadata):
    """
    Process individual response based on question metadata
    """
    correct_order = question_metadata['correct_order']
    max_value = question_metadata.get('max_value', 4)
    
    if correct_order == 'RECODE':
        if 'traffic' in question_metadata['question_id'].lower():
            return recode_traffic_v9(response)
        elif 'calm' in question_metadata['question_id'].lower():
            return recode_calm_v7(response)
    elif correct_order == 'FALSE':
        return reverse_scale(response, max_value)
    
    return response

def main():
    # Raw data path (based on EMA_participant_processing.py structure)
    raw_data_path = Path(project_root) / "data/raw/participants"
    
    # Metadata path (we'll create this)
    metadata_path = Path(project_root) / "data/metadata/question_metadata.json"
    
    # Process each participant's data
    for participant_file in raw_data_path.glob("*.csv"):
        logging.info(f"Processing {participant_file.name}")
        df = pd.read_csv(participant_file)
        
        # Load question metadata
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Process each response
        for question_id in metadata:
            if question_id in df.columns:
                df[question_id] = df[question_id].apply(
                    lambda x: process_response(x, metadata[question_id])
                )
        
        # Save processed data
        output_path = Path(project_root) / "data/processed/participants" / f"recoded_{participant_file.name}"
        df.to_csv(output_path, index=False)
        
        # Print sample of recoded data for verification
        logging.info(f"\nSample of recoded data for {participant_file.name}:")
        print(df.head())

if __name__ == "__main__":
    main() 