import pandas as pd
import re
import argparse
import logging
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import ast

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

def safe_dict_convert(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return {}
    return {}

def process_comprehensive_data(comprehensive_data_path, response_dict_df):
    comprehensive_data_df = load_data(comprehensive_data_path)
    if comprehensive_data_df is None:
        return None

    comprehensive_data_df = comprehensive_data_df[comprehensive_data_df['Form name'].str.startswith('EMA', na=False)]

    comprehensive_data_df['Form_matched'] = comprehensive_data_df['Form name'].apply(lambda x: fuzzy_match(x, response_dict_df['Form']))
    comprehensive_data_df['Question_matched'] = comprehensive_data_df['Question name'].apply(lambda x: fuzzy_match(x, response_dict_df['Question']))

    merged_df = pd.merge(comprehensive_data_df, 
                         response_dict_df[['Form', 'Question', 'Hebrew_dict', 'English_dict']], 
                         left_on=['Form_matched', 'Question_matched'], 
                         right_on=['Form', 'Question'], 
                         how='left')

    logging.info("Sample of Hebrew_dict before conversion:")
    logging.info(merged_df['Hebrew_dict'].head())

    merged_df['Hebrew_dict'] = merged_df['Hebrew_dict'].fillna({}).apply(safe_dict_convert)
    merged_df['English_dict'] = merged_df['English_dict'].fillna({}).apply(safe_dict_convert)

    logging.info("Sample of Hebrew_dict after conversion:")
    logging.info(merged_df['Hebrew_dict'].head())

    columns_to_drop = ['Form_matched', 'Question_matched', 'Form', 'Question']
    merged_df = merged_df.drop(columns=columns_to_drop)

    return merged_df

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

    logging.info("\nExamples where Hebrew dictionary is empty:")
    empty_hebrew = comprehensive_data_df[comprehensive_data_df['Hebrew_dict'].apply(lambda x: len(x) == 0)]
    logging.info(empty_hebrew[['Question name', 'Responses name']].head(40))

    logging.info("\nExamples where English dictionary is empty:")
    empty_english = comprehensive_data_df[comprehensive_data_df['English_dict'].apply(lambda x: len(x) == 0)]
    logging.info(empty_english[['Question name', 'Responses name']].head(40))

    logging.info(f"\nTotal rows with empty Hebrew dictionary: {len(empty_hebrew)}")
    logging.info(f"Total rows with empty English dictionary: {len(empty_english)}")

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
