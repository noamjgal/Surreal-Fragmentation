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
    response_eng_df['Variable'] = response_eng_df['Question'].map(question_to_variable)

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
    logging.info(f"\nMatching for question: {row['Question name']}")
    logging.info(f"Available columns in row: {row.index.tolist()}")
    logging.info(f"Available columns in response_dict_df: {response_dict_df.columns.tolist()}")

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
    
    if best_match is None or best_score < 90:  # Log only if no match or low confidence match
        logging.warning(f"Potential mismatch for '{row['Question name']}' with response '{row['Responses name']}'")
        if best_match is not None:
            logging.warning(f"Best match: '{best_match['Question']}' with score {best_score}")
            logging.warning(f"Matched Hebrew responses: {best_match['Hebrew_dict']}")
            logging.warning(f"Matched English responses: {best_match['English_dict']}")
        logging.warning("---")
    
    return best_match

def process_comprehensive_data(comprehensive_data_path, response_dict_df):
    comprehensive_data_df = load_data(comprehensive_data_path)
    if comprehensive_data_df is None:
        return None

    comprehensive_data_df = comprehensive_data_df[comprehensive_data_df['Form name'].str.startswith('EMA', na=False)]

    comprehensive_data_df['Form_matched'] = comprehensive_data_df['Form name'].apply(lambda x: fuzzy_match(x, response_dict_df['Form']))
    
    logging.info("Comprehensive data DataFrame columns:")
    logging.info(comprehensive_data_df.columns.tolist())
    
    logging.info("\nFirst few rows of comprehensive_data_df:")
    logging.info(comprehensive_data_df.head().to_string())

    logging.info("\nResponse dict DataFrame columns:")
    logging.info(response_dict_df.columns.tolist())
    
    logging.info("\nFirst few rows of response_dict_df:")
    logging.info(response_dict_df.head().to_string())

    def match_and_fill(row):
        best_match = fuzzy_match_question_and_responses(row, response_dict_df)
        if best_match is not None:
            return pd.Series({
                'Question_matched': best_match['Question'],
                'Hebrew_dict': best_match['Hebrew_dict'],
                'English_dict': best_match['English_dict']
            })
        return pd.Series({'Question_matched': None, 'Hebrew_dict': None, 'English_dict': None})

    matched_data = comprehensive_data_df.apply(match_and_fill, axis=1)
    comprehensive_data_df = pd.concat([comprehensive_data_df, matched_data], axis=1)

    return comprehensive_data_df

def calculate_response_counts(comprehensive_data_df, response_dict_df):
    for _, row in response_dict_df.iterrows():
        question = row['Question']
        hebrew_dict = row['Hebrew_dict']
        
        question_data = comprehensive_data_df[comprehensive_data_df['Question_matched'] == question]
        
        response_counts = question_data['Responses name'].value_counts().to_dict()
        
        # Map the response counts to the numerical keys and translate if necessary
        mapped_counts = {}
        for k, v in response_counts.items():
            hebrew_key = next((key for key, value in hebrew_dict.items() if value == k), k)
            english_key = translate_hebrew(hebrew_key)
            mapped_counts[english_key] = v
        
        response_dict_df.loc[response_dict_df['Question'] == question, 'Response_Counts'] = json.dumps(mapped_counts)
    
    return response_dict_df

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
