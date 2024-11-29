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
        return 0
    
    # Print sample rows before recoding
    logging.info("\nSample rows BEFORE recoding:")
    sample_rows = recoded_ema[mask].head()
    logging.info("\n" + sample_rows[['Form name', 'Variable', 'Responses ID']].to_string())
    
    original_values = recoded_ema.loc[mask, 'Responses ID'].copy()
    value_counts_before = original_values.value_counts().sort_index()
    logging.info("\nValue distribution BEFORE recoding:")
    logging.info(value_counts_before)
    
    # Reverse the numerical values
    recoded_ema.loc[mask, 'Responses ID'] = recoded_ema.loc[mask, 'Responses ID'].apply(
        lambda x: str(max_value + 1 - int(x)) if str(x).isdigit() else x
    )
    
    # Print sample rows after recoding
    logging.info("\nSame rows AFTER recoding:")
    logging.info("\n" + recoded_ema[mask].head()[['Form name', 'Variable', 'Responses ID']].to_string())
    
    recoded_values = recoded_ema.loc[mask, 'Responses ID']
    value_counts_after = recoded_values.value_counts().sort_index()
    logging.info("\nValue distribution AFTER recoding:")
    logging.info(value_counts_after)
    
    changes = (original_values != recoded_values).sum()
    logging.info(f"\nNumber of values changed: {changes}")
    return changes

def main():
    # Set up detailed logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Create output directory
    output_dir = os.path.join('SURREAL', 'EMA-Round2', 'data', 'reordered')
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Created output directory: {output_dir}")
    
    # Define all file paths
    data_dir = Path(project_root) / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    
    # Input files
    mappings_path = raw_dir / "Corrected-Response-Mappings.xlsx"
    ema_data_path = raw_dir / "comprehensive_ema_data_eng_updated.csv"
    
    # Load the mapping data
    mappings_df = pd.read_excel(
        mappings_path,
        sheet_name="processed_response_mappings"
    )
    
    # Load the EMA data
    ema_data = pd.read_csv(ema_data_path)
    print('Forms and Variables in mapping data:')
    print(mappings_df[['Form','Variable']].head())
    print('Forms and Variables in EMA data:')
    print(ema_data[['Form name','Variable']].head())
    
    logging.info("\nLoaded mapping data:")
    logging.info(mappings_df.head())
    logging.info("\nLoaded EMA data:")
    logging.info(ema_data['Question name'].head())
    # Create metadata dictionary from mappings_df
    metadata = {}
    for _, row in mappings_df.iterrows():
        metadata[row['Question']] = {
            'question_id': row['Question'],
            'correct_order': str(row.get('Correct_Order', 'TRUE')).upper(),  # Ensure string comparison
            'max_value': row.get('Points', 4)
        }
    print('test metadata')
    print(metadata)
    
    # Process the comprehensive EMA data
    logging.info("\nProcessing EMA data...")
    recoded_ema = ema_data.copy()
    recoding_counts = {'traffic': 0, 'calm': 0, 'reverse': 0}
    
    # Handle TRAFFIC recoding (EMA V9)
    traffic_mask = (ema_data['Form name'] == 'EMA V9') & (ema_data['Variable'] == 'TRAFFIC')
    if traffic_mask.any():
        logging.info("\n" + "="*50)
        logging.info("Processing TRAFFIC recoding (EMA V9)")
        
        # Print sample rows before recoding
        logging.info("\nSample rows BEFORE recoding:")
        logging.info("\n" + ema_data[traffic_mask].head()[['Form name', 'Variable', 'Responses ID']].to_string())
        
        original_values = ema_data.loc[traffic_mask, 'Responses ID'].copy()
        value_counts_before = original_values.value_counts().sort_index()
        logging.info("\nValue distribution BEFORE recoding:")
        logging.info(value_counts_before)
        
        # Perform recoding
        traffic_mapping = {'1': '4', '2': '3', '3': '3', '4': '2', '5': '1'}
        recoded_ema.loc[traffic_mask, 'Responses ID'] = recoded_ema.loc[traffic_mask, 'Responses ID'].map(traffic_mapping)
        
        # Print sample rows after recoding
        logging.info("\nSame rows AFTER recoding:")
        logging.info("\n" + recoded_ema[traffic_mask].head()[['Form name', 'Variable', 'Responses ID']].to_string())
        
        value_counts_after = recoded_ema.loc[traffic_mask, 'Responses ID'].value_counts().sort_index()
        logging.info("\nValue distribution AFTER recoding:")
        logging.info(value_counts_after)
        
        recoding_counts['traffic'] = traffic_mask.sum()
    
    # Handle CALM recoding (EMA V7)
    calm_mask = (ema_data['Form name'] == 'EMA V7') & (ema_data['Variable'] == 'CALM')
    if calm_mask.any():
        logging.info("\n" + "="*50)
        logging.info("Processing CALM recoding (EMA V7)")
        
        # Print sample rows before recoding
        logging.info("\nSample rows BEFORE recoding:")
        logging.info("\n" + ema_data[calm_mask].head()[['Form name', 'Variable', 'Responses ID']].to_string())
        
        original_values = ema_data.loc[calm_mask, 'Responses ID'].copy()
        value_counts_before = original_values.value_counts().sort_index()
        logging.info("\nValue distribution BEFORE recoding:")
        logging.info(value_counts_before)
        
        # Perform recoding
        calm_mapping = {'1': '2', '2': '3', '3': '1', '4': '4'}
        recoded_ema.loc[calm_mask, 'Responses ID'] = recoded_ema.loc[calm_mask, 'Responses ID'].map(calm_mapping)
        
        # Print sample rows after recoding
        logging.info("\nSame rows AFTER recoding:")
        logging.info("\n" + recoded_ema[calm_mask].head()[['Form name', 'Variable', 'Responses ID']].to_string())
        
        value_counts_after = recoded_ema.loc[calm_mask, 'Responses ID'].value_counts().sort_index()
        logging.info("\nValue distribution AFTER recoding:")
        logging.info(value_counts_after)
        
        recoding_counts['calm'] = calm_mask.sum()

    # Handle reverse scale recoding
    logging.info("\nProcessing reverse scale recodings...")
    total_recoded_questions = 0
    
    for idx, row in mappings_df.iterrows():
        if 'F' in str(row['Correct_Order']):
            changes = process_reverse_coding(row, ema_data, recoded_ema)
            recoding_counts['reverse'] += changes
            total_recoded_questions += 1
            mappings_df.at[idx, 'Correct_Order'] = 'REORDERED'
    
    # Save the recoded responses
    recoded_ema.to_csv(os.path.join(output_dir, 'recoded_ema_data.csv'), index=False)
    
    # Save the mapping data without dictionary updates
    mappings_df.to_csv(os.path.join(output_dir, 'response_mapping_no_dicts.csv'), index=False)
    
    logging.info('\nRecoding Summary:')
    logging.info(f"Traffic recodings: {recoding_counts['traffic']}")
    logging.info(f"Calm recodings: {recoding_counts['calm']}")
    logging.info(f"Reverse scale recodings: {recoding_counts['reverse']}")

if __name__ == "__main__":
    main()
