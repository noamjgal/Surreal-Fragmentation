import pandas as pd
import logging
from pathlib import Path
import json
import sys
import re
import os

# Set up logging configuration at the start of the file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)
MAPPING_FILE_PATH = Path(project_root) / "data" / "raw" / "Corrected-Response-Mappings.xlsx"
OUTPUT_DIR = Path(project_root) / "data"
PROCESSED_DICT_PATH = OUTPUT_DIR / "raw" / "processed_dictionaries.csv"

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
        
        # Add special handling for long procrastination
        if 'LONG_PROCRASTINATION' in str(question_metadata.get('Variable', '')):
            # Map to same scale as regular procrastination
            standard_proc_dict = {
                "בכלל לא": "1",
                "לעתים רחוקות": "2",
                "לפעמים": "3",
                "בדרך כלל": "4",
                "כל הזמן": "5"
            }
            # Map the long procrastination responses to the standard scale
            long_proc_mapping = {
                "מעט": "2",  # Map "A little" to "Rarely"
                "במידה מסוימת": "3",  # Map "To a certain extent" to "Sometimes"
                "במידה מתונה": "3",  # Map "To a moderate extent" to "Sometimes"
                "במידה בינונית": "4",  # Map "Moderately" to "Usually"
                "מאוד": "5"  # Map "Very" to "All the time"
            }
            dict_data = long_proc_mapping
        
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
            # Get the points scale (4 or 5)
            points = int(question_metadata.get('Points', 4))
            
            # Create reverse mapping based on the full scale (4 or 5 points)
            reverse_mapping = {str(i): str(points - i + 1) for i in range(1, points + 1)}
            
            # Apply the mapping to the dictionary values
            dict_data = {k: reverse_mapping[v] for k, v in dict_data.items()}
            
            process_dictionary.reverse_examples.append({
                'question_id': question_id,
                'points': points,
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
									
def save_processed_data(mapping_data):
    """Save processed dictionaries to output directory."""
    # Create output directory if it doesn't exist
    PROCESSED_DICT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for saving
    results = []
    for _, row in mapping_data.iterrows():
        result = row.to_dict()
        
        # Process Hebrew dictionary if exists
        if pd.notna(row['Hebrew_dict']):
            processed_hebrew = process_dictionary(row['Hebrew_dict'], row)
            if processed_hebrew:
                processed_hebrew = dict(sorted(processed_hebrew.items(), key=lambda x: int(x[1])))
                result['Hebrew_dict_processed'] = json.dumps(processed_hebrew, ensure_ascii=False)
            else:
                result['Hebrew_dict_processed'] = None
            
        # Process English dictionary if exists
        if pd.notna(row['Eng_dict']):
            processed_eng = process_dictionary(row['Eng_dict'], row)
            if processed_eng:
                processed_eng = dict(sorted(processed_eng.items(), key=lambda x: int(x[1])))
                result['Eng_dict_processed'] = json.dumps(processed_eng)
            else:
                result['Eng_dict_processed'] = None
            
        results.append(result)
    
    # Convert to DataFrame and save
    output_df = pd.DataFrame(results)
    output_df.to_csv(PROCESSED_DICT_PATH, index=False)
    logger.info(f"\nSaved processed dictionaries to: {PROCESSED_DICT_PATH}")
    
    # Save examples to text file
    examples_path = OUTPUT_DIR / "raw" / "processing_examples.txt"
    with open(examples_path, 'w', encoding='utf-8') as f:
        f.write("=== RECODE Examples ===\n")
        for i, example in enumerate(process_dictionary.recode_examples[:5], 1):
            f.write(f"\nExample {i} ({example['type']}) - {example['question_id']}:\n")
            f.write(f"Original: {example['original']}\n")
            f.write(f"Processed: {example['processed']}\n")
        
        f.write("\n=== REVERSE Examples ===\n")
        for i, example in enumerate(process_dictionary.reverse_examples[:5], 1):
            f.write(f"\nExample {i} - {example['question_id']}:\n")
            f.write(f"Original: {example['original']}\n")
            f.write(f"Processed: {example['processed']}\n")
        
        f.write("\n=== SORT Examples ===\n")
        for i, example in enumerate(process_dictionary.sort_examples[:5], 1):
            f.write(f"\nExample {i} - {example['question_id']}:\n")
            f.write(f"Original: {example['original']}\n")
            f.write(f"Processed: {example['processed']}\n")
    
    logger.info(f"Saved processing examples to: {examples_path}")

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
    
    # Log examples and save results
    save_processed_data(mapping_data)

if __name__ == "__main__":
    main() 



