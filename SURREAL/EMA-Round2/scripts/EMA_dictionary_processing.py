import pandas as pd
import logging
from pathlib import Path
import json
import sys
import re

# Set up logging configuration at the start of the file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)
MAPPING_FILE_PATH = project_root + '/data/raw/Corrected-Response-Mappings.xlsx'

def process_dictionary(dict_string, question_metadata):
    """Process dictionary string with detailed logging for sorting and reversing."""
    if not dict_string or pd.isna(dict_string):
        return None
    
    try:
        correct_order = str(question_metadata.get('Correct_Order', '')).upper()
        question_id = str(question_metadata.get('Variable', '')).lower()
        points = int(question_metadata.get('Points', 4))
        
        dict_string = dict_string.replace("'", '"').strip()
        if not dict_string.startswith('{'): 
            return None
            
        dict_data = json.loads(dict_string)
        dict_data = {k: str(v) for k, v in dict_data.items()}
        
        # Store original for logging
        original_dict = dict_data.copy()
        
        # Process based on correct_order flag
        if correct_order == 'RECODE':
            if 'traffic' in question_id:
                mapping = {'1': '4', '2': '3', '3': '2', '4': '2', '5': '1'} if points == 5 else {'1': '4', '2': '3', '3': '2', '4': '1'}
                dict_data = {k: mapping[v] for k, v in dict_data.items()}
                process_dictionary.recode_examples.append({
                    'type': 'traffic',
                    'question_id': question_id,
                    'original': original_dict,
                    'processed': dict(sorted(dict_data.items(), key=lambda x: int(x[1])))
                })
            elif 'calm' in question_id:
                mapping = {'1': '3', '2': '1', '3': '2', '4': '4', '5': '5'} if points == 5 else {'1': '3', '2': '1', '3': '2', '4': '4'}
                dict_data = {k: mapping[v] for k, v in dict_data.items()}
                process_dictionary.recode_examples.append({
                    'type': 'calm',
                    'question_id': question_id,
                    'original': original_dict,
                    'processed': dict(sorted(dict_data.items(), key=lambda x: int(x[1])))
                })
        
        elif correct_order == 'FALSE':
            max_val = max(int(v) for v in dict_data.values())
            reverse_mapping = {str(i): str(max_val - i + 1) for i in range(1, max_val + 1)}
            dict_data = {k: reverse_mapping[v] for k, v in dict_data.items()}
            process_dictionary.reverse_examples.append({
                'question_id': question_id,
                'original': original_dict,
                'processed': dict(sorted(dict_data.items(), key=lambda x: int(x[1])))
            })
        
        # Always sort the dictionary
        sorted_dict = dict(sorted(dict_data.items(), key=lambda x: int(x[1])))
        process_dictionary.sort_examples.append({
            'question_id': question_id,
            'original': original_dict,
            'processed': sorted_dict
        })
        
        return sorted_dict
        
    except Exception as e:
        logger.error(f"Error processing dictionary for {question_id}: {e}")
        logger.error(f"Problematic dictionary string: {dict_string}")
        return None

# Initialize storage for examples
process_dictionary.recode_examples = []
process_dictionary.reverse_examples = []
process_dictionary.sort_examples = []

def main():
    mapping_data = pd.read_excel(MAPPING_FILE_PATH)
    logger.info(f"Loaded mapping data from {MAPPING_FILE_PATH}")
    logger.info(f"Mapping data shape: {mapping_data.shape}")
    logger.info(f"Mapping data columns: {mapping_data.columns}")
    
    unique_orders = mapping_data['Correct_Order'].unique()
    logger.info(f"\nUnique values in Correct_Order column: {unique_orders}")
    
    # Process each type separately
    for order_type in unique_orders:
        subset = mapping_data[mapping_data['Correct_Order'] == order_type]
        logger.info(f"\nProcessing {order_type} type entries: {len(subset)} items")
        
        for _, row in subset.iterrows():
            for dict_col in ['Hebrew_dict', 'Eng_dict']:
                if pd.notna(row[dict_col]):
                    process_dictionary(row[dict_col], row)
    
    # Log examples after processing
    logger.info("\n=== RECODE Examples ===")
    for i, example in enumerate(process_dictionary.recode_examples[:5], 1):
        logger.info(f"\nExample {i} ({example['type']}) - {example['question_id']}:")
        logger.info(f"Original: {example['original']}")
        logger.info(f"Processed: {example['processed']}")
    
    logger.info("\n=== REVERSE Examples ===")
    for i, example in enumerate(process_dictionary.reverse_examples[:5], 1):
        logger.info(f"\nExample {i} - {example['question_id']}:")
        logger.info(f"Original: {example['original']}")
        logger.info(f"Processed: {example['processed']}")
    
    logger.info("\n=== SORT Examples ===")
    for i, example in enumerate(process_dictionary.sort_examples[:5], 1):
        logger.info(f"\nExample {i} - {example['question_id']}:")
        logger.info(f"Original: {example['original']}")
        logger.info(f"Processed: {example['processed']}")

if __name__ == "__main__":
    main() 



