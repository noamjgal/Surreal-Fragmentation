import pandas as pd
import logging
from pathlib import Path
import json
import sys
import argparse

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
    1->1, 3->2, 2->3, 4->4
    """
    mapping = {'1': '1', '3': '2', '2': '3', '4': '4'}
    return mapping.get(str(value), value)

def reverse_scale(value, max_value):
    """
    Reverse numerical coding for FALSE items
    """
    try:
        value = int(value)
        return str(max_value - value + 1)
    except (ValueError, TypeError):
        return value

def process_response(value, metadata):
    """
    Process response based on metadata
    """
    if pd.isna(value):
        return value
        
    correct_order = metadata.get('correct_order', '')
    max_value = metadata.get('max_value', 4)
    
    if correct_order == 'RECODE':
        if metadata.get('Variable') == 'TRAFFIC' and metadata.get('Form') == 'EMA V9':
            return recode_traffic_v9(value)
        elif metadata.get('Variable') == 'CALM' and metadata.get('Form') == 'EMA V7':
            return recode_calm_v7(value)
    elif correct_order == 'FALSE':
        return reverse_scale(value, max_value)
    
    return value

def main():
    parser = argparse.ArgumentParser(description="Process EMA response mappings")
    parser.add_argument("response_eng_path", help="Path to the response mapping English CSV file")
    parser.add_argument("comprehensive_data_path", help="Path to the comprehensive EMA data CSV file")
    parser.add_argument("output_dir", help="Path to the output directory")
    args = parser.parse_args()

    # Load raw data
    raw_data_path = Path(args.comprehensive_data_path)
    if not raw_data_path.exists():
        logging.error(f"Raw data file not found: {raw_data_path}")
        return
        
    df = pd.read_csv(raw_data_path)
    
    # Load response mappings
    response_mappings_path = Path(args.response_eng_path)
    if not response_mappings_path.exists():
        logging.error(f"Response mappings file not found: {response_mappings_path}")
        return
        
    response_mappings = pd.read_csv(response_mappings_path)
    
    # Create metadata dictionary from response mappings
    metadata = {}
    for _, row in response_mappings.iterrows():
        metadata[row['Question']] = {
            'Form': row['Form'],
            'Variable': row['Variable'],
            'correct_order': row.get('correct_order', ''),
            'max_value': row.get('max_value', 4)
        }
    
    # Process each response
    for question in metadata:
        if question in df.columns:
            df[question] = df[question].apply(
                lambda x: process_response(x, metadata[question])
            )
    
    # Save processed data
    output_path = Path(args.output_dir) / "processed_comprehensive_ema_data.csv"
    df.to_csv(output_path, index=False)
    
    # Print sample of recoded data for verification
    logging.info("\nSample of recoded data:")
    print(df.head())
    
    # Verify specific recodings
    logging.info("\nVerifying Traffic V9 recoding:")
    traffic_v9 = df[df['Form'] == 'EMA V9']['TRAFFIC']
    print(traffic_v9.value_counts())
    
    logging.info("\nVerifying Calm V7 recoding:")
    calm_v7 = df[df['Form'] == 'EMA V7']['CALM']
    print(calm_v7.value_counts())

if __name__ == "__main__":
    main() 