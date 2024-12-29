import pandas as pd
import logging
from pathlib import Path
import sys

# Setup
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_participant_data(participant_id):
    """Load processed data for a single participant."""
    file_path = Path(project_root) / "data" / "processed" / "participants" / f"participant_{participant_id}.csv"
    return pd.read_csv(file_path)

def analyze_stai_responses(data):
    """Analyze STAI responses and print unique values."""
    # Filter for STAI questions
    stai_data = data[data['Scale'] == 'STAI'].copy()
    
    if stai_data.empty:
        logging.warning("No STAI data found")
        return
    
    # Print unique questions and their responses
    for variable in stai_data['Variable'].unique():
        responses = stai_data[stai_data['Variable'] == variable]
        logging.info(f"\nVariable: {variable}")
        logging.info("Unique Response Keys (numerical):")
        logging.info(responses['Response Key'].unique())
        logging.info("Unique Response Values (text):")
        logging.info(responses['Response Value'].unique())

def main():
    # Load mapping data to identify reverse-scored items
    mappings_path = Path(project_root) / "data" / "raw" / "processed_dictionaries.csv"
    mappings_df = pd.read_csv(mappings_path)
    
    # Get list of all participant files
    participant_dir = Path(project_root) / "data" / "processed" / "participants"
    participant_files = list(participant_dir.glob("participant_*.csv"))
    
    # Process first participant as an example
    if participant_files:
        participant_id = participant_files[0].stem.split('_')[1]
        data = load_participant_data(participant_id)
        
        logging.info(f"Analyzing STAI responses for participant {participant_id}")
        analyze_stai_responses(data)
    else:
        logging.error("No participant data files found")

if __name__ == "__main__":
    main() 