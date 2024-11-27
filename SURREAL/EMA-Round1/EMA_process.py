import pandas as pd
import re
import argparse
import logging
from fuzzywuzzy import fuzz, process
import ast
import numpy as np
import json
from utils import translate_hebrew
import math

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
    if response_eng_df is None:
        return None
        
    # Load comprehensive data for EFFORT questions
    comprehensive_df = load_data(comprehensive_data_path)
    if comprehensive_df is None:
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

    # Standardize EFFORT response mappings
    effort_mapping = {
        # Hebrew responses
        'במידה מתונה': '2',
        'במידה מסוימת': '2',
        'לעתים רחוקות': '1',
        'ככלל לא': '1',
        'בכלל לא': '1',
        'לפעמים': '2',
        'בדרך כלל': '3',
        'כל הזמן': '4',
        # Additional variations
        'מסכים': '3',
        'לא מסכים': '2',
        'בהחלט לא מסכים': '1',
        'אני לא מסכים': '1'
    }

    def update_response_counts(row):
        if pd.isna(row['Response_Counts']) or row['Response_Counts'] == '{}':
            row['Response_Counts'] = '{}'
        
        try:
            counts = eval(row['Response_Counts'])
            if not isinstance(counts, dict):
                counts = {}
        except:
            counts = {}
            
        return str(counts)

    # Process EFFORT questions specifically
    effort_questions = comprehensive_df[
        (comprehensive_df['Variable'] == 'EFFORT') & 
        (comprehensive_df['Form name'].str.contains('EMA V', na=False))
    ]

    for form_name, form_responses in effort_questions.groupby('Form name'):
        logging.info(f"\nProcessing EFFORT responses for {form_name}")
        logging.info(f"Found {len(form_responses)} responses")
        logging.info("Sample responses:")
        logging.info(form_responses[['Responses ID', 'Responses name']].head())
        
        logging.info(f"\nDEBUG - EFFORT Response Details for {form_name}:")
        
        # Create response mappings
        effort_mapping = {
            'במידה מתונה': '2',
            'לעתים רחוקות': '1',
            'בכלל לא': '1',
            'לפעמים': '2',
            'בדרך כלל': '3',
            'כל הזמן': '4'
        }
        
        effort_mapping_eng = {
            'Moderately': '2',
            'Rarely': '1',
            'Not at all': '1',
            'Sometimes': '2',
            'Usually': '3',
            'All the time': '4'
        }
        
        logging.info("\nResponse mappings for this form:")
        logging.info(f"Mappings: {effort_mapping}")
        
        # Count responses
        response_counts = form_responses['Responses name'].value_counts().to_dict()
        logging.info("\nUnique responses found:")
        logging.info(f"Responses: {form_responses['Responses name'].unique()}")
        logging.info("\nResponse ID distribution:")
        logging.info(f"ID counts: {form_responses['Responses ID'].value_counts()}")
        
        # Create form dictionary
        form_dict = {
            'Form': form_name,
            'Question': form_responses['Question name'].iloc[0],
            'Variable': 'EFFORT',
            'Responses': ';'.join([f"{k}:{v}" for k, v in effort_mapping.items()]),
            'Responses_English': ';'.join([f"{k}:{v}" for k, v in effort_mapping_eng.items()]),
            'Hebrew_dict': effort_mapping,  # Store as dictionary, not string
            'English_dict': effort_mapping_eng,  # Store as dictionary, not string
            'English_Question': form_responses['English_Question'].iloc[0] if 'English_Question' in form_responses.columns else '',
            'Response_Counts': response_counts  # Store as dictionary, not string
        }
        
        # Debug logging for response counting
        logging.info("\nCounting process:")
        counts = {}
        for _, row in form_responses.iterrows():
            response_id = row['Responses ID']
            response_text = row['Responses name']
            logging.info(f"Processing - ID: {response_id}, Text: {response_text}")
            if response_id not in counts:
                counts[response_id] = 0
            counts[response_id] += 1
            logging.info(f"Updated count for ID {response_id}: {counts[response_id]}")
        
        # Update or append to response_eng_df
        mask = (response_eng_df['Form'] == form_name) & (response_eng_df['Variable'] == 'EFFORT')
        if mask.any():
            for key, value in form_dict.items():
                response_eng_df.loc[mask, key] = value
        else:
            response_eng_df = pd.concat([response_eng_df, pd.DataFrame([form_dict])], ignore_index=True)

    # Apply the counts update
    response_eng_df['Response_Counts'] = response_eng_df.apply(update_response_counts, axis=1)

    # Ensure dictionaries are properly formatted
    response_eng_df['Hebrew_dict'] = response_eng_df['Hebrew_dict'].apply(
        lambda x: {} if isinstance(x, float) and math.isnan(x) else 
                 eval(x) if isinstance(x, str) else x
    )
    
    response_eng_df['English_dict'] = response_eng_df['English_dict'].apply(
        lambda x: {} if isinstance(x, float) and math.isnan(x) else 
                 eval(x) if isinstance(x, str) else x
    )
    
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

def fuzzy_match_question_and_responses(row, response_dict_df):
    # Calculate similarity scores for each potential match
    scores = []
    for _, ref_row in response_dict_df.iterrows():
        try:
            # Convert string representation of dict to actual dict if needed
            hebrew_dict = ref_row['Hebrew_dict']
            if isinstance(hebrew_dict, str):
                hebrew_dict = eval(hebrew_dict)
            elif isinstance(hebrew_dict, float) and math.isnan(hebrew_dict):
                hebrew_dict = {}
                
            english_dict = ref_row['English_dict']
            if isinstance(english_dict, str):
                english_dict = eval(english_dict)
            elif isinstance(english_dict, float) and math.isnan(english_dict):
                english_dict = {}
            
            # Calculate similarity scores
            question_score = fuzz.token_set_ratio(str(row['Question name']), str(ref_row['Question']))
            response_score = fuzz.token_set_ratio(
                str(row['Responses name']), 
                ' '.join(hebrew_dict.keys()) if hebrew_dict else ''
            )
            
            form_match = 100 if row['Form name'] == ref_row['Form'] else 0
            
            # Combined score with form match having high weight
            combined_score = (question_score * 0.4 + response_score * 0.3 + form_match * 0.3)
            scores.append((combined_score, ref_row))
            
        except Exception as e:
            logging.warning(f"Error processing row: {e}")
            scores.append((0, ref_row))
    
    # Find best match
    if not scores:
        return None
        
    best_score, best_match = max(scores, key=lambda x: x[0])
    
    # Only return match if score is above threshold
    if best_score > 60:  # You can adjust this threshold
        return best_match
    return None

def process_comprehensive_data(comprehensive_data_path, response_dict_df):
    comprehensive_data_df = load_data(comprehensive_data_path)
    if comprehensive_data_df is None:
        return None

    # Standardize form names in comprehensive data
    comprehensive_data_df['Form name'] = comprehensive_data_df['Form name'].apply(
        lambda x: re.sub(r'EMA[_\s]*V?(\d+)', r'EMA V\1', str(x)) if pd.notnull(x) else x
    )
    
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
    print('jump here for new debug data')
    # Debug V4 EFFORT processing specifically
    v4_effort_data = comprehensive_data_df[
        (comprehensive_data_df['Form name'] == 'EMA V4') & 
        (comprehensive_data_df['Question name'].str.contains('מאמץ', na=False))
    ]
    
    logging.info("\nDEBUG - V4 EFFORT Processing:")
    logging.info(f"Number of V4 EFFORT rows found: {len(v4_effort_data)}")
    
    # Check if we're finding the mapping
    v4_effort_mapping = response_dict_df[
        (response_dict_df['Form'] == 'EMA V4') & 
        (response_dict_df['Variable'] == 'EFFORT')
    ]
    logging.info(f"V4 EFFORT mapping rows: {len(v4_effort_mapping)}")
    if len(v4_effort_mapping) > 0:
        logging.info("Sample mapping:")
        logging.info(v4_effort_mapping[['Question', 'Responses', 'Hebrew_dict']].head())
    
    # Check response matching
    for _, row in v4_effort_data.iterrows():
        logging.info(f"\nProcessing response: {row['Responses name']}")
        matching_map = response_dict_df[
            (response_dict_df['Form'] == 'EMA V4') & 
            (response_dict_df['Question'] == row['Question name'])
        ]
        logging.info(f"Found {len(matching_map)} matching mapping rows")

    # Add validation for EFFORT questions
    effort_mask = comprehensive_data_df['Question name'].apply(
        lambda x: any(keyword in str(x) for keyword in [
            'כל פעולה דורשת ממך מאמץ',
            'דורשת ממך מאמץ רב',
            'הרגשת שכל פעולה דורשת'
        ])
    )
    
    # Force update Variable column for EFFORT questions
    comprehensive_data_df.loc[effort_mask, 'Variable'] = 'EFFORT'
    
    # Debug logging for EFFORT questions
    effort_rows = comprehensive_data_df[effort_mask]
    logging.info(f"\nFound {len(effort_rows)} EFFORT questions")
    logging.info("\nEFFORT questions detected:")
    for _, row in effort_rows.iterrows():
        logging.info(f"Form: {row['Form name']}, Question: {row['Question name']}, Variable: {row['Variable']}")

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
            
            # Add debug logging for EFFORT responses in EMA V4
            if variable == 'EFFORT':
                logging.info(f"\nDEBUG - EFFORT Response Details for {form}:")
                effort_data = comprehensive_data_df[effort_mask]
                
                # Debug response mappings
                logging.info("\nResponse mappings for this form:")
                form_mappings = response_mappings.get((form, variable), {})
                logging.info(f"Mappings: {form_mappings}")
                
                # Debug actual responses
                logging.info("\nUnique responses found:")
                unique_responses = effort_data['Responses name'].unique()
                logging.info(f"Responses: {unique_responses}")
                
                # Debug Response IDs
                logging.info("\nResponse ID distribution:")
                id_counts = effort_data['Responses ID'].value_counts()
                logging.info(f"ID counts: {id_counts}")
                
                # Debug the counting process
                logging.info("\nCounting process:")
                for _, row in effort_data.iterrows():
                    response_id = str(row['Responses ID'])
                    response_text = row['Responses name']
                    logging.info(f"Processing - ID: {response_id}, Text: {response_text}")
                    if response_id and response_id.isdigit():
                        counts[response_id] = counts.get(response_id, 0) + 1
                        logging.info(f"Updated count for ID {response_id}: {counts[response_id]}")
                    else:
                        logging.warning(f"Invalid response ID: {response_id}")
            
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
    """Check if a response exists in the mapping dictionary using fuzzy matching."""
    matching_row = response_dict_df[
        (response_dict_df['Form'] == form) & 
        (response_dict_df['Variable'] == variable)
    ]
    
    if matching_row.empty:
        return False
        
    hebrew_dict = matching_row.iloc[0]['Hebrew_dict']
    if not isinstance(hebrew_dict, dict):
        try:
            hebrew_dict = ast.literal_eval(hebrew_dict)
        except:
            return False
            
    # Use fuzzy matching for response comparison
    if any(fuzz.ratio(response, key) > 90 for key in hebrew_dict.keys()):
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
    # First check for exact EFFORT question match
    effort_questions = [
        'כמה זמן במהלך השעות האחרונות הרגשת שכל פעולה דורשת ממך מאמץ רב?',
        # Add other variations of the effort question if they exist
    ]
    
    if any(question_text == q for q in effort_questions):
        return 'EFFORT'
    
    # Then check for partial matches containing key phrases
    effort_keywords = [
        'כל פעולה דורשת ממך מאמץ',
        'דורשת ממך מאמץ רב',
        'הרגשת שכל פעולה דורשת'
    ]
    
    if any(keyword in question_text for keyword in effort_keywords):
        return 'EFFORT'
        
    # Other mappings can go here
    question_mapping = {
        # Add other specific mappings as needed
    }
    
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
