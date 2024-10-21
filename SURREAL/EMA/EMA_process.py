import pandas as pd
import re
import argparse
import logging
from fuzzywuzzy import fuzz, process
import ast
import numpy as np

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

def process_response_mappings(response_eng_path):
    response_eng_df = load_data(response_eng_path)
    if response_eng_df is None:
        return None

    response_eng_df = response_eng_df.rename(columns={'Responses.1': 'Responses_English'})
    response_eng_df = response_eng_df[response_eng_df['Form'].str.startswith('EMA', na=False)]

    logging.info(f"Number of rows after filtering: {len(response_eng_df)}")

    response_eng_df['Hebrew_dict'] = response_eng_df['Responses'].apply(clean_and_process_responses)
    response_eng_df['English_dict'] = response_eng_df['Responses_English'].apply(clean_and_process_responses)

    logging.info("\nResponses with no keys:")
    for index, row in response_eng_df.iterrows():
        if len(row['Hebrew_dict']) == 0 and not pd.isna(row['Responses']):
            logging.info(f"Hebrew - Question: {row['Question']}, Response: {row['Responses']}")
        if len(row['English_dict']) == 0 and not pd.isna(row['Responses_English']):
            logging.info(f"English - Question: {row['Question']}, Response: {row['Responses_English']}")

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
                fuzz.token_set_ratio(str(row['Responses name']), str(ref_row['Responses'])),
                fuzz.token_set_ratio(str(row['Responses name']), ' '.join(ref_row['Hebrew_dict'].keys()))
            )
            
            combined_score = (question_score + response_score) / 2
            
            if combined_score > best_score and response_score >= response_threshold:
                best_score = combined_score
                best_match = ref_row
    
    if best_match is None or best_score < 90:  # Log only if no match or low confidence match
        logging.warning(f"Potential mismatch for '{row['Question name']}' with response '{row['Responses name']}'")
        if best_match:
            logging.warning(f"Best match: '{best_match['Question']}' with score {best_score}")
            logging.warning(f"Matched response: '{best_match['Responses']}'")
        logging.warning("---")
    
    return best_match

def process_comprehensive_data(comprehensive_data_path, response_dict_df):
    comprehensive_data_df = load_data(comprehensive_data_path)
    if comprehensive_data_df is None:
        return None

    comprehensive_data_df = comprehensive_data_df[comprehensive_data_df['Form name'].str.startswith('EMA', na=False)]

    comprehensive_data_df['Form_matched'] = comprehensive_data_df['Form name'].apply(lambda x: fuzzy_match(x, response_dict_df['Form']))
    
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

    logging.info("\nQuestions with empty Hebrew dictionary after matching:")
    empty_hebrew = comprehensive_data_df[comprehensive_data_df['Hebrew_dict'].apply(lambda x: safe_len(x) == 0)]
    for _, row in empty_hebrew[['Question name', 'Responses name', 'Question_matched', 'Hebrew_dict', 'English_dict']].drop_duplicates().iterrows():
        logging.info(f"Original Question: {row['Question name']}")
        logging.info(f"Original Response: {row['Responses name']}")
        logging.info(f"Matched to: {row['Question_matched']}")
        logging.info(f"Hebrew dict: {row['Hebrew_dict']}")
        logging.info(f"English dict: {row['English_dict']}")
        logging.info("---")

    return comprehensive_data_df

def main(response_eng_path, comprehensive_data_path, output_dir):
    response_dict_df = process_response_mappings(response_eng_path)
    if response_dict_df is None:
        return

    processed_response_mappings_path = f"{output_dir}/processed_response_mappings.csv"
    response_dict_df.to_csv(processed_response_mappings_path, index=False)
    logging.info(f"Processed response mappings saved to: {processed_response_mappings_path}")

    comprehensive_data_df = process_comprehensive_data(comprehensive_data_path, response_dict_df)
    if comprehensive_data_df is None:
        return

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
