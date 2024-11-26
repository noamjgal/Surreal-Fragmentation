import pandas as pd
import re
import argparse
import logging
from fuzzywuzzy import fuzz, process
import ast
import numpy as np
import json
from utils import translate_hebrew

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {e}")
        return None

def clean_and_process_responses(responses):
    result = {}
    if pd.isna(responses):
        return result
    
    pairs = responses.split(';')
    for pair in pairs:
        parts = pair.split(':', 1)
        if len(parts) == 2:
            key, value = parts
            result[key.strip()] = value.strip()
        else:
            logging.warning(f"Warning: Invalid pair format: {pair}")
    
    return result

def fuzzy_match(x, choices, scorer=fuzz.token_sort_ratio):
    return process.extractOne(x, choices, scorer=scorer)[0]

def fuzzy_match_question(question, choices, threshold=80):
    matches = process.extractBests(question, choices, scorer=fuzz.token_sort_ratio, score_cutoff=threshold)
    if matches:
        return matches[0][0]
    return None

def process_response_mappings(response_eng_path, comprehensive_data_path):
    response_eng_df = load_data(response_eng_path)
    comprehensive_df = load_data(comprehensive_data_path)
    if response_eng_df is None or comprehensive_df is None:
        return None

    response_eng_df = response_eng_df.rename(columns={'Responses.1': 'Responses_English'})
    response_eng_df = response_eng_df[response_eng_df['Form'].str.startswith('EMA', na=False)]

    logging.info(f"Number of rows after filtering: {len(response_eng_df)}")

    response_eng_df['Hebrew_dict'] = response_eng_df['Responses'].apply(clean_and_process_responses)
    response_eng_df['English_dict'] = response_eng_df['Responses_English'].apply(clean_and_process_responses)

    # Add English_Question column
    question_to_english = comprehensive_df.set_index('Question name')['English_Question'].to_dict()
    response_eng_df['English_Question'] = response_eng_df['Question'].map(question_to_english)

    # Add Variable column
    question_to_variable = comprehensive_df.set_index('Question name')['Variable'].to_dict()
    response_eng_df['Variable'] = response_eng_df['Question'].apply(
        lambda q: map_question_to_variable(q) or question_to_variable.get(q)
    )

    # Add Count column (will be filled later)
    response_eng_df['Response_Counts'] = '{}'

    logging.info("\nResponses with no keys:")
    for index, row in response_eng_df.iterrows():
        if len(row['Hebrew_dict']) == 0 and not pd.isna(row['Responses']):
            logging.info(f"Hebrew - Question: {row['Question']}, Response: {row['Responses']}")
        if len(row['English_dict']) == 0 and not pd.isna(row['Responses_English']):
            logging.info(f"English - Question: {row['Question']}, Response: {row['Responses_English']}")

    logging.info("Response mapping DataFrame columns:")
    logging.info(response_eng_df.columns.tolist())
    
    logging.info("\nFirst few rows of response_eng_df:")
    logging.info(response_eng_df.head().to_string())

    # Reorder columns
    columns_order = ['Form', 'Question', 'English_Question', 'Variable', 'Responses', 'Responses_English', 'Hebrew_dict', 'English_dict', 'Response_Counts']
    response_eng_df = response_eng_df[columns_order]

    # Add special handling for EFFORT responses
    effort_mapping = {
        'במידה מתונה': '2',
        'במידה מסוימת': '2',
        'לעתים רחוקות': '1',
        'לפעמים': '2',
        'בדרך כלל': '3',
        'כל הזמן': '4',
        'ככלל לא': '1',
        'בכלל לא': '1'
    }

    # Add EFFORT questions if they don't exist
    effort_questions = comprehensive_df[comprehensive_df['Question name'].str.contains('מאמץ', na=False)]
    for _, row in effort_questions.iterrows():
        question = row['Question name']
        form = row['Form name']
        if question not in response_eng_df['Question'].values:
            effort_row = pd.DataFrame({
                'Question': [question],
                'Form': [form],
                'Variable': ['EFFORT'],
                'Responses': ['במידה מתונה:2;לעתים רחוקות:1;לפעמים:2;בדרך כלל:3;כל הזמן:4;ככלל לא:1'],
                'Responses_English': ['Moderately:2;Rarely:1;Sometimes:2;Usually:3;All the time:4;Not at all:1'],
                'Hebrew_dict': [str(effort_mapping)],
                'English_dict': [str({k: v for k, v in effort_mapping.items() if not any(c in k for c in 'אבגדהוזחטיכלמנסעפצקרשת')})],
                'English_Question': ['How much effort did your actions require?'],
                'Response_Counts': ['{}']
            })
            response_eng_df = pd.concat([response_eng_df, effort_row], ignore_index=True)

    return response_eng_df

def safe_len(x):
    if isinstance(x, dict):
        return len(x)
    elif pd.isna(x):
        return 0
    else:
        return len(str(x))

def safe_dict_convert(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return {}
    return {}

def fuzzy_match_question_and_responses(row, response_dict_df, question_threshold=80, response_threshold=70):

    best_match = None
    best_score = 0
    
    for _, ref_row in response_dict_df.iterrows():
        question_score = fuzz.token_sort_ratio(row['Question name'], ref_row['Question'])
        
        if question_score >= question_threshold:
            response_score = max(
                fuzz.token_set_ratio(str(row['Responses name']), ' '.join(ref_row['Hebrew_dict'].keys())),
                fuzz.token_set_ratio(str(row['Responses name']), ' '.join(ref_row['English_dict'].keys()))
            )
            
            combined_score = (question_score + response_score) / 2
            
            if combined_score > best_score and response_score >= response_threshold:
                best_score = combined_score
                best_match = ref_row
    
    return best_match

def process_comprehensive_data(comprehensive_data_path, response_dict_df):
    comprehensive_data_df = load_data(comprehensive_data_path)
    if comprehensive_data_df is None:
        return None

    # Get list of forms for matching
    forms = response_dict_df['Form'].unique().tolist()
    logging.info("\nForm list for matching:")
    logging.info(forms)
    
    logging.info("\nSample of Form name values from comprehensive_data_df:")
    logging.info(comprehensive_data_df['Form name'].head())

    # Initialize new columns
    comprehensive_data_df['Hebrew_dict'] = None
    comprehensive_data_df['English_dict'] = None
    comprehensive_data_df['Question_matched'] = None

    def match_and_fill(row):
        best_match = fuzzy_match_question_and_responses(row, response_dict_df)
        if best_match is not None:
            return pd.Series({
                'Question_matched': best_match['Question'],
                'Hebrew_dict': best_match['Hebrew_dict'],
                'English_dict': best_match['English_dict']
            })
        return pd.Series({
            'Question_matched': None, 
            'Hebrew_dict': {}, 
            'English_dict': {}
        })

    # Apply matching to each row
    matched_data = comprehensive_data_df.apply(match_and_fill, axis=1)
    
    # Update the columns with matched data
    comprehensive_data_df['Question_matched'] = matched_data['Question_matched']
    comprehensive_data_df['Hebrew_dict'] = matched_data['Hebrew_dict']
    comprehensive_data_df['English_dict'] = matched_data['English_dict']

    return comprehensive_data_df

def fuzzy_match(value, choices):
    try:
        logging.debug(f"Fuzzy matching value: {value}")
        logging.debug(f"Against choices: {choices}")
        
        if not isinstance(choices, (list, np.ndarray)):
            logging.error(f"Choices is not a list or array. Type: {type(choices)}")
            return value
            
        if value in choices:
            return value
        
        matches = process.extract(str(value), [str(c) for c in choices], limit=1)
        logging.debug(f"Fuzzy match result: {matches}")
        
        if matches and matches[0][1] >= 80:
            return matches[0][0]
        return value
    except Exception as e:
        logging.error(f"Error in fuzzy matching: {e}")
        logging.error(f"Value: {value}, Type: {type(value)}")
        logging.error(f"Choices: {choices}, Type: {type(choices)}")
        return value

def calculate_response_counts(comprehensive_data_df, response_dict_df):
    # Group by Form, Variable and calculate counts separately
    grouped = comprehensive_data_df.groupby(['Form name', 'Variable'])
    
    # Create a dictionary to store counts
    response_counts = {}
    
    # First create mapping of responses to their numerical codes
    response_mappings = {}
    for _, row in response_dict_df.iterrows():
        form = row['Form']
        variable = row['Variable']
        
        # Handle the Hebrew dictionary string
        if isinstance(row['Hebrew_dict'], str):
            hebrew_dict_str = row['Hebrew_dict'].replace("'", '"')
        else:
            hebrew_dict_str = str(row['Hebrew_dict']).replace("'", '"')
            
        try:
            hebrew_dict = json.loads(hebrew_dict_str)
        except json.JSONDecodeError:
            try:
                # Clean up the string more thoroughly
                hebrew_dict_str = hebrew_dict_str.replace('{', '{"').replace(': ', '": "').replace(', ', '", "').replace('}', '"}')
                hebrew_dict = json.loads(hebrew_dict_str)
            except:
                logging.error(f"Failed to parse Hebrew dict: {hebrew_dict_str}")
                # Instead of continuing, initialize an empty dict
                hebrew_dict = {}
        
        response_mappings[(form, variable)] = hebrew_dict
        
    # Add debug logging before grouping
    logging.info("\nDEBUG - Before processing:")
    logging.info(f"Total rows in comprehensive_data_df: {len(comprehensive_data_df)}")
    logging.info(f"Columns in comprehensive_data_df: {comprehensive_data_df.columns.tolist()}")
    logging.info("\nUnique Form names:")
    logging.info(comprehensive_data_df['Form name'].unique())
    
    # Debug EMA V4 specifically
    ema_v4_rows = comprehensive_data_df[comprehensive_data_df['Form name'] == 'EMA V4']
    logging.info(f"\nEMA V4 rows found: {len(ema_v4_rows)}")
    if not ema_v4_rows.empty:
        logging.info("\nSample of EMA V4 rows:")
        logging.info(ema_v4_rows[['Form name', 'Question name', 'Responses name']].head())
        
        # Check for effort questions
        effort_rows = ema_v4_rows[ema_v4_rows['Question name'].str.contains('מאמץ', na=False)]
        logging.info(f"\nEMA V4 effort questions found: {len(effort_rows)}")
        if not effort_rows.empty:
            logging.info("\nEffort questions and responses:")
            logging.info(effort_rows[['Question name', 'Responses name']].to_string())
    
    # Count responses using numerical codes
    for (form, variable), group in grouped:
        counts = {}
        
        # Special handling for EFFORT variable
        if variable == 'EFFORT':
            counts = {}
            logging.info(f"\nProcessing EFFORT responses for {form}")
            
            # Find responses for effort questions
            effort_mask = (comprehensive_data_df['Form name'] == form) & \
                         (comprehensive_data_df['Question name'].str.contains('מאמץ', na=False))
            
            # Get both response IDs and names for debugging
            responses = comprehensive_data_df[effort_mask][['Responses ID', 'Responses name']]
            
            logging.info(f"Found {len(responses)} responses")
            logging.info("Sample responses:")
            logging.info(responses.head().to_string())
            
            # Count the Response IDs directly
            for _, row in responses.iterrows():
                response_id = str(row['Responses ID'])
                if response_id and response_id.isdigit():
                    counts[response_id] = counts.get(response_id, 0) + 1
                    logging.info(f"Counted response ID: {response_id} (Text: {row['Responses name']})")
            
            if counts:
                logging.info(f"Final counts for {form} - EFFORT: {counts}")
            else:
                logging.error(f"No counts found for {form} - EFFORT")
                logging.info("All responses found:")
                logging.info(responses.to_string())
            
            response_counts[(form, variable)] = counts
            continue
            
        # Regular handling for other variables
        if (form, variable) not in response_mappings:
            logging.error(f"No mapping found for {form} - {variable}")
            continue
        
        mapping = response_mappings[(form, variable)]
        if not mapping:
            logging.error(f"Empty mapping for {form} - {variable}")
            continue
            
        for response in group['Responses name']:
            if response not in mapping:
                logging.warning(f"Response not found in mapping: {response}")
                continue
                
            code = mapping[response]
            counts[code] = counts.get(code, 0) + 1
        
        response_counts[(form, variable)] = counts
    
    # Update the response_dict_df with the new counts
    for idx, row in response_dict_df.iterrows():
        form = row['Form']
        variable = row['Variable']
        if (form, variable) in response_counts:
            response_dict_df.at[idx, 'Response_Counts'] = json.dumps(response_counts[(form, variable)])
        else:
            logging.error(f"No counts found for {form} - {variable}")
    
    return response_dict_df

def response_exists_in_mapping(response, form, variable, response_dict_df):
    """Check if a response exists in the mapping dictionary."""
    matching_row = response_dict_df[
        (response_dict_df['Form'] == form) & 
        (response_dict_df['Variable'] == variable)
    ]
    
    if matching_row.empty:
        return False
        
    hebrew_dict = matching_row.iloc[0]['Hebrew_dict']
    if isinstance(hebrew_dict, dict) and response in hebrew_dict:
        return True
    return False

def process_responses(form, variable, responses_df, response_dict_df):
    # Only log when there are issues or important summaries
    response_counts = responses_df[responses_df['Variable'] == variable]['Responses name'].value_counts()
    
    if response_counts.empty:
        logging.warning(f"No responses found for {form} - {variable}")
        return {}
    
    # Check for unmapped responses
    unmapped = []
    for response in response_counts.index:
        if not response_exists_in_mapping(response, form, variable, response_dict_df):
            unmapped.append(response)
    
    # Only log if there are unmapped responses
    if unmapped:
        logging.warning(f"{form} - {variable}: Found {len(unmapped)} unmapped responses")
        logging.debug(f"Unmapped responses: {', '.join(unmapped)}")  # Debug level for details
        
    return dict(response_counts)

def map_question_to_variable(question_text):
    """Map Hebrew questions to their corresponding variable names"""
    question_mapping = {
        'כמה זמן במהלך השעות האחרונות הרגשת שכל פעולה דורשת ממך מאמץ רב?': 'EFFORT',
        # Add other mappings as needed
    }
    
    # Special case for EFFORT question
    if 'כל פעולה דורשת ממך מאמץ' in question_text:
        return 'EFFORT'
        
    return question_mapping.get(question_text, None)

def main(response_eng_path, comprehensive_data_path, output_dir):
    response_dict_df = process_response_mappings(response_eng_path, comprehensive_data_path)
    if response_dict_df is None:
        return

    comprehensive_data_df = process_comprehensive_data(comprehensive_data_path, response_dict_df)
    if comprehensive_data_df is None:
        return
    
    # Calculate response counts
    response_dict_df = calculate_response_counts(comprehensive_data_df, response_dict_df)

    processed_response_mappings_path = f"{output_dir}/processed_response_mappings.csv"
    response_dict_df.to_csv(processed_response_mappings_path, index=False)
    logging.info(f"Processed response mappings saved to: {processed_response_mappings_path}")

    logging.info("\nSummary of problematic matches:")
    empty_hebrew = comprehensive_data_df[comprehensive_data_df['Hebrew_dict'].apply(lambda x: safe_len(x) == 0)]
    logging.info(f"Total rows with empty or NaN Hebrew dictionary: {len(empty_hebrew)}")
    
    empty_english = comprehensive_data_df[comprehensive_data_df['English_dict'].apply(lambda x: safe_len(x) == 0)]
    logging.info(f"Total rows with empty or NaN English dictionary: {len(empty_english)}")

    if not empty_hebrew.empty:
        logging.info("\nSample of questions with empty Hebrew dictionary:")
        for _, row in empty_hebrew[['Question name', 'Responses name', 'Question_matched']].head(5).iterrows():
            logging.info(f"Question: {row['Question name']} | Response: {row['Responses name']} | Matched to: {row['Question_matched']}")

    if not empty_english.empty:
        logging.info("\nSample of questions with empty English dictionary:")
        for _, row in empty_english[['Question name', 'Responses name', 'Question_matched']].head(5).iterrows():
            logging.info(f"Question: {row['Question name']} | Response: {row['Responses name']} | Matched to: {row['Question_matched']}")

    output_file = f"{output_dir}/comprehensive_ema_data_eng_updated.csv"
    comprehensive_data_df.to_csv(output_file, index=False)
    logging.info(f"\nUpdated data saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process EMA response mappings and comprehensive data")
    parser.add_argument("response_eng_path", help="Path to the response mapping English CSV file")
    parser.add_argument("comprehensive_data_path", help="Path to the comprehensive EMA data CSV file")
    parser.add_argument("output_dir", help="Path to the output directory")
    args = parser.parse_args()

    main(args.response_eng_path, args.comprehensive_data_path, args.output_dir)
