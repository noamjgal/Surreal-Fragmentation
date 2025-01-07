import pandas as pd
import logging
from pathlib import Path
import json
import sys
import os

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

# Input
EMA_DATA_PATH = Path(project_root) / "data" / "raw" / "comprehensive_ema_data_var.csv"
DICT_PATH = Path(project_root) / "data" / "raw" / "processed_dictionaries_merged.csv"
MAPPINGS_PATH = Path(project_root) / "data" / "raw" / "Corrected-Response-Mappings.xlsx"
# Output
RECODED_PATH = Path(project_root) / "data" / "reordered" / "recoded_ema_data.csv"

def print_recode_example(original, recoded, question_id, recode_type):
    """Helper function to print recoding examples"""
    logging.info(f"\n{recode_type} Recoding Example for {question_id}:")
    logging.info(f"Original value: {original} -> Recoded value: {recoded}")

def recode_traffic_v9(value):
    """
    Remap 5-point scale to 4-point scale for EMA V9 traffic question
    1->4, 2->3, 3->2, 4->2, 5->1
    """
    mapping = {'1': '4', '2': '3', '3': '2', '4': '2', '5': '1'}
    original = value
    result = mapping.get(str(value), value)
    return result

def recode_calm_v7(value):
    """
    Reorder responses for EMA V7 calm question
    1->3, 2->1, 3->2, 4->4
    """
    mapping = {'1': '3', '2': '1', '3': '2', '4': '4'}
    original = value
    result = mapping.get(str(value), value)
    return result

def reverse_scale(value, max_value=4):
    """
    Reverse numerical coding for FALSE items
    For 4-point scale: 1->4, 2->3, 3->2, 4->1
    For 5-point scale: 1->5, 2->4, 3->3, 4->2, 5->1
    """
    original = value
    try:
        value = str(value)
        if not value.isdigit():
            return value
        value = int(value)
        result = str(max_value - value + 1)
        return result
    except:
        return value

def process_response(response, question_metadata):
    """Process individual response based on question metadata"""
    question_id = question_metadata['question_id']
    correct_order = question_metadata['correct_order']
    max_value = question_metadata.get('max_value', 4)
    
    # Track if we've printed examples for this question
    if not hasattr(process_response, 'printed_examples'):
        process_response.printed_examples = set()

    # Only print one example per question_id
    if question_id not in process_response.printed_examples:
        process_response.printed_examples.add(question_id)
        
        if correct_order == 'RECODE':
            if 'traffic' in question_id.lower():
                recoded_value = recode_traffic_v9(response)
                print_recode_example(response, recoded_value, question_id, "Traffic")
                return recoded_value
            elif 'calm' in question_id.lower():
                recoded_value = recode_calm_v7(response)
                print_recode_example(response, recoded_value, question_id, "Calm")
                return recoded_value
        elif correct_order == 'FALSE':
            recoded_value = reverse_scale(response, max_value)
            print_recode_example(response, recoded_value, question_id, "Reverse")
            return recoded_value
    
    # Normal processing without printing for subsequent values
    if correct_order == 'RECODE':
        if 'traffic' in question_id.lower():
            return recode_traffic_v9(response)
        elif 'calm' in question_id.lower():
            return recode_calm_v7(response)
    elif correct_order == 'FALSE':
        try:
            if str(response).isdigit():
                return str(max_value - int(response) + 1)
        except:
            pass
    return response

def load_data(file_path):
    """Helper function to load data files"""
    try:
        if file_path.suffix == '.xlsx':
            return pd.read_excel(file_path)
        elif file_path.suffix == '.csv':
            return pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return None

def reverse_dictionary(d, max_value):
    """Reverse the keys and values in a dictionary based on max_value."""
    logging.debug(f"Reversing dictionary: {d} with max_value: {max_value}")
    reversed_dict = {str(max_value + 1 - int(v)): k for k, v in d.items()}
    logging.debug(f"Reversed dictionary: {reversed_dict}")
    print('REVERSED DICTIONARY:', reversed_dict)
    return reversed_dict

def parse_and_reverse_dict(dict_str, max_value):
    """Parse a dictionary string and reverse its values."""
    if not dict_str or dict_str.strip() == "/":
        logging.warning("Skipping empty or invalid dictionary string.")
        return dict_str
    
    logging.debug(f"Beginning dict string: {dict_str}")
    
    try:
        # Attempt to parse the dictionary string
        parsed_dict = json.loads(dict_str.replace("'", '"'))
        logging.debug(f"Parsed dictionary: {parsed_dict}")
        
        # Reverse the values
        reversed_dict = {k: str(max_value + 1 - int(v)) for k, v in parsed_dict.items()}
        logging.debug(f"Reversed dictionary: {reversed_dict}")
        
        return json.dumps(reversed_dict, ensure_ascii=False)
    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding error: {e} - Original string: {dict_str}")
        return dict_str  # Return the original string if parsing fails
    except ValueError as e:
        logging.error(f"Value error during reversal: {e} - Original string: {dict_str}")
        return dict_str

def process_reverse_coding(row, ema_data, recoded_ema):
    """Process reverse coding for a single row of mapping data."""
    max_value = row['Points']
    form = row['Form']
    variable = row['Variable']
    
    logging.info(f"\n{'='*50}")
    logging.info(f"Processing reverse coding for Form: {form}, Variable: {variable}")
    mask = (ema_data['Form name'] == form) & (ema_data['Variable'] == variable)
    
    if not mask.any():
        logging.warning(f"No matching data found for Form: {form}, Variable: {variable}")
        return 0, 0  # Return both changes and unchanged
    
    # Print value distribution before recoding
    original_values = recoded_ema.loc[mask, 'Responses ID'].copy()
    value_counts_before = original_values.value_counts().sort_index()
    logging.info("\nValue distribution BEFORE recoding:")
    logging.info(value_counts_before)
    
    # Reverse the numerical values
    recoded_ema.loc[mask, 'Responses ID'] = recoded_ema.loc[mask, 'Responses ID'].apply(
        lambda x: str(max_value + 1 - int(x)) if str(x).isdigit() else x
    )
    
    # Print value distribution after recoding
    recoded_values = recoded_ema.loc[mask, 'Responses ID']
    value_counts_after = recoded_values.value_counts().sort_index()
    logging.info("\nValue distribution AFTER recoding:")
    logging.info(value_counts_after)
    
    changes = (original_values != recoded_values).sum()
    unchanged = (original_values == recoded_values).sum()
    logging.info(f"\nNumber of values changed: {changes}")
    logging.info(f"Number of values unchanged: {unchanged}")
    return changes, unchanged

def main():
    # Set up detailed logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join('SURREAL', 'EMA-Processing', 'data', 'reordered')
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Created output directory: {output_dir}")
    
    
    # Load the data with explicit data validation
    try:
        mappings_df = pd.read_excel(MAPPINGS_PATH, sheet_name="processed_response_mappings")
        ema_data = pd.read_csv(EMA_DATA_PATH)
        processed_dicts = pd.read_csv(DICT_PATH)
        
        # Add debugging information
        logging.info("\nData Loading Summary:")
        logging.info(f"EMA Data Shape: {ema_data.shape}")
        logging.info(f"EMA Data Columns: {ema_data.columns.tolist()}")
        logging.info(f"Form name unique values: {ema_data['Form name'].unique().tolist()}")
        logging.info(f"Variable unique values: {ema_data['Variable'].unique().tolist()}")
        
        # Check for data type issues
        logging.info("\nColumn Data Types:")
        logging.info(f"Form name dtype: {ema_data['Form name'].dtype}")
        logging.info(f"Variable dtype: {ema_data['Variable'].dtype}")
        
        # Convert columns to string type if needed
        ema_data['Form name'] = ema_data['Form name'].astype(str)
        ema_data['Variable'] = ema_data['Variable'].astype(str)
        
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise
    
    # Create a copy for recoding
    recoded_ema = ema_data.copy()
    recoding_counts = {
        'traffic': {'changed': 0, 'unchanged': 0},
        'calm': {'changed': 0, 'unchanged': 0},
        'reverse': {'changed': 0, 'unchanged': 0}
    }
    
    # Update dictionaries in recoded_ema with processed versions
    for _, row in processed_dicts.iterrows():
        form = row['Form']
        variable = row['Variable']
        mask = (recoded_ema['Form name'] == form) & (recoded_ema['Variable'] == variable)
        
        if pd.notna(row.get('Hebrew_dict_processed')):
            recoded_ema.loc[mask, 'Hebrew_dict'] = row['Hebrew_dict_processed']
        if pd.notna(row.get('Eng_dict_processed')):
            recoded_ema.loc[mask, 'English_dict'] = row['Eng_dict_processed']
    # Process all possible recodings
    for order_type in mappings_df['Correct_Order'].unique():
        subset = mappings_df[mappings_df['Correct_Order'] == order_type]
        logging.info(f"\nProcessing {order_type} type entries: {len(subset)} items")
        
        for _, row in subset.iterrows():
            if 'F' in str(order_type):  # Handle reversals
                changes, unchanged = process_reverse_coding(row, ema_data, recoded_ema)
                recoding_counts['reverse']['changed'] += changes
                recoding_counts['reverse']['unchanged'] += unchanged
            elif order_type == 'RECODE':
                if 'TRAFFIC' in str(row['Variable']):
                    mask = (ema_data['Form name'] == row['Form']) & (ema_data['Variable'] == 'TRAFFIC')
                    if mask.any():
                        original_values = recoded_ema.loc[mask, 'Responses ID'].copy()
                        traffic_mapping = {'1': '4', '2': '3', '3': '2', '4': '2', '5': '1'}
                        recoded_ema.loc[mask, 'Responses ID'] = recoded_ema.loc[mask, 'Responses ID'].map(traffic_mapping)
                        changes = (original_values != recoded_ema.loc[mask, 'Responses ID']).sum()
                        unchanged = (original_values == recoded_ema.loc[mask, 'Responses ID']).sum()
                        recoding_counts['traffic']['changed'] += changes
                        recoding_counts['traffic']['unchanged'] += unchanged
                elif 'CALM' in str(row['Variable']):
                    mask = (ema_data['Form name'] == row['Form']) & (ema_data['Variable'] == 'CALM')
                    if mask.any():
                        original_values = recoded_ema.loc[mask, 'Responses ID'].copy()
                        calm_mapping = {'1': '3', '2': '1', '3': '2', '4': '4'}
                        recoded_ema.loc[mask, 'Responses ID'] = recoded_ema.loc[mask, 'Responses ID'].map(calm_mapping)
                        changes = (original_values != recoded_ema.loc[mask, 'Responses ID']).sum()
                        unchanged = (original_values == recoded_ema.loc[mask, 'Responses ID']).sum()
                        recoding_counts['calm']['changed'] += changes
                        recoding_counts['calm']['unchanged'] += unchanged
    
    first = len(recoded_ema) 
    
    # Update all procrastination variables to simply "PROCRASTINATION"
    mask = recoded_ema['Variable'].str.contains('PROCRASTINATION', na=False)
    recoded_ema.loc[mask, 'Variable'] = 'PROCRASTINATION'
    
    # Map long procrastination responses to standard scale
    long_proc_mapping = {
        'מעט': '2',  # A little -> Rarely
        'במידה מסוימת': '3',  # To a certain extent -> Sometimes
        'במידה מתונה': '3',  # To a moderate extent -> Sometimes
        'במידה בינונית': '4',  # Moderately -> Usually
        'מאוד': '5'  # Very -> All the time
    }
    
    # Apply mapping for V2 and V7 forms
    mask = ((recoded_ema['Form name'].str.contains('V2') | recoded_ema['Form name'].str.contains('V7')) & 
            (recoded_ema['Variable'] == 'PROCRASTINATION'))
    recoded_ema.loc[mask, 'Responses ID'] = recoded_ema.loc[mask, 'Responses name'].map(long_proc_mapping)
    
    after = len(recoded_ema)
    
    # Save the recoded responses

    recoded_ema.to_csv(RECODED_PATH, index=False)
    
    # Print summary
    logging.info('\nRecoding Summary:')
    for recode_type, counts in recoding_counts.items():
        logging.info(f"{recode_type.capitalize()} recodings:")
        logging.info(f"  Changed: {counts['changed']}")
        logging.info(f"  Unchanged: {counts['unchanged']}")
        logging.info(f"  Total processed: {counts['changed'] + counts['unchanged']}")
    
    logging.info(f"\nProcessed data saved to: {RECODED_PATH}")
    print(f"Dropped {first - after} rows")
    print(recoded_ema['Variable'].unique())

if __name__ == "__main__":
    main()
