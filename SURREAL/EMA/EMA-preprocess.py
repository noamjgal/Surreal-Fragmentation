import pandas as pd
import numpy as np
import os
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    try:
        return pd.read_csv(file_path, sep='|', quotechar='"')
    except Exception as e:
        logging.error(f"Error loading file: {e}")
        return None

def get_questions_for_form(df, form_name):
    form_data = df[df['Form name'] == form_name]
    return form_data['Question name'].unique()

def analyze_responses(df, form_name):
    form_data = df[df['Form name'] == form_name]
    response_analysis = {}
    for question in form_data['Question name'].unique():
        question_data = form_data[form_data['Question name'] == question]
        field_type = question_data['Field type name'].iloc[0]
        unique_responses = question_data['Responses name'].unique()
        response_analysis[question] = {
            'field_type': field_type,
            'unique_responses': unique_responses
        }
    return response_analysis

def main(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(input_file)
    if df is None:
        return

    form_names = df['Form name'].unique()

    for form_name in form_names:
        logging.info(f"\nAnalyzing form: {form_name}")
        questions = get_questions_for_form(df, form_name)
        
        response_analysis = analyze_responses(df, form_name)
        
        logging.info(f"Number of questions in {form_name}: {len(questions)}")
        
        with open(os.path.join(output_dir, f'{form_name.replace(" ", "_")}_survey_structure.txt'), 'w', encoding='utf-8') as f:
            for question in questions:
                analysis = response_analysis[question]
                f.write(f"Question: {question}\n")
                f.write(f"Field type: {analysis['field_type']}\n")
                f.write(f"Unique responses: {', '.join(analysis['unique_responses'])}\n\n")
                
                logging.info(f"\nQuestion: {question}")
                logging.info(f"Field type: {analysis['field_type']}")
                logging.info(f"Number of unique responses: {len(analysis['unique_responses'])}")
                if analysis['field_type'] == 'Slider':
                    numeric_responses = [float(r) for r in analysis['unique_responses'] if r.replace('.', '').isdigit()]
                    if numeric_responses:
                        logging.info(f"Range of numeric responses: {min(numeric_responses)} to {max(numeric_responses)}")

    comprehensive_df = df.copy()
    comprehensive_df.to_csv(os.path.join(output_dir, 'comprehensive_ema_data.csv'), index=False)

    logging.info("\nProcessing complete. Files created in: %s", output_dir)
    logging.info("1. comprehensive_ema_data.csv")
    logging.info("2. [Form_Name]_survey_structure.txt for each form")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess EMA data")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("output_dir", help="Path to the output directory")
    args = parser.parse_args()

    main(args.input_file, args.output_dir)
