import pandas as pd
import logging
from pathlib import Path
import os
import sys

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

# Create necessary directories
os.makedirs(f'{project_root}/logs', exist_ok=True)
os.makedirs(f'{project_root}/output/participants', exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{project_root}/logs/validation.log'),
        logging.StreamHandler()
    ]
)

def load_and_validate_data():
    """Load and perform initial validation of the data."""
    # Load the corrected mappings from the specific sheet
    mappings_path = Path(project_root) / "data" / "raw" / "Corrected-Response-Mappings.xlsx"
    ema_data_path = Path(project_root) / "data" / "raw" / "comprehensive_ema_data_eng_updated.csv"
    
    logging.info(f"Loading mappings from: {mappings_path}")
    logging.info(f"Loading EMA data from: {ema_data_path}")
    
    mappings_df = pd.read_excel(
        mappings_path, 
        sheet_name="processed_response_mappings"
    )
    ema_data = pd.read_csv(ema_data_path)
    
    # Get unique scales
    unique_scales = mappings_df['Scale'].unique()
    logging.info("\nUnique scales found:")
    for scale in unique_scales:
        logging.info(f"- {scale}")
        # Count questions per scale
        question_count = len(mappings_df[mappings_df['Scale'] == scale])
        logging.info(f"  Number of questions: {question_count}")
    
    # Get unique participants and dates
    participants = ema_data['Participant ID'].unique()
    logging.info(f"\nNumber of unique participants: {len(participants)}")
    
    return mappings_df, ema_data

def setup_participant_directories(ema_data):
    """Create directory structure for participant-level analysis."""
    output_dir = Path(project_root) / "output" / "participants"
    
    # Get unique participants and dates
    participants = ema_data['Participant ID'].unique()
    
    for participant in participants:
        # Create participant directory
        participant_dir = output_dir / f"participant_{participant}"
        os.makedirs(participant_dir, exist_ok=True)
        
        # Get dates for this participant
        participant_data = ema_data[ema_data['Participant ID'] == participant]
        dates = pd.to_datetime(participant_data['Date form sent']).dt.date.unique()
        
        for date in dates:
            # Create date directory
            date_dir = participant_dir / str(date)
            os.makedirs(date_dir, exist_ok=True)
            
            # Create placeholder files for each EMA and daily summary
            for ema_num in range(1, 4):
                (date_dir / f"EMA_{ema_num}.json").touch()
            (date_dir / "daily_summary.json").touch()
            
    logging.info(f"Created directory structure for {len(participants)} participants")

def main():
    mappings_df, ema_data = load_and_validate_data()
    setup_participant_directories(ema_data)
    
if __name__ == "__main__":
    main()
