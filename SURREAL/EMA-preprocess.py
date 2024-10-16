import pandas as pd
import numpy as np
import re
import os

ema_file_path = "/Users/noamgal/Downloads/Research-Projects/SURREAL/EMA-Surreal/ema-responses.csv"
output_dir = "/Users/noamgal/Downloads/Research-Projects/SURREAL/EMA-Surreal/preprocessed"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the CSV file with the correct separator
df = pd.read_csv(ema_file_path, sep='|', quotechar='"')

# Function to get unique questions for a given form
def get_questions_for_form(df, form_name):
    form_data = df[df['Form name'] == form_name]
    return form_data['Question name'].unique()

# Get unique form names
form_names = df['Form name'].unique()

# Function to create variable names
def create_variable_name(question):
    # Remove any non-alphanumeric characters and replace spaces with underscores
    variable = re.sub(r'[^\w\s]', '', question)
    variable = variable.replace(' ', '_')
    # Ensure the variable name starts with a letter
    if not variable[0].isalpha():
        variable = 'q_' + variable
    return variable.upper()

# Dictionary to store question mappings
question_mappings = {}

# Function to analyze responses for a given form
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

# Create question mappings, analyze responses, and store original questions
for form_name in form_names:
    print(f"\nAnalyzing form: {form_name}")
    questions = get_questions_for_form(df, form_name)
    question_mappings[form_name] = {q: create_variable_name(q) for q in questions}
    
    # Analyze responses for this form
    response_analysis = analyze_responses(df, form_name)
    
    print(f"Number of questions in {form_name}: {len(questions)}")
    
    # Save survey structure to a text file
    with open(os.path.join(output_dir, f'{form_name.replace(" ", "_")}_survey_structure.txt'), 'w', encoding='utf-8') as f:
        for question in questions:
            analysis = response_analysis[question]
            f.write(f"Question: {question}\n")
            f.write(f"Variable: {question_mappings[form_name][question]}\n")
            f.write(f"Field type: {analysis['field_type']}\n")
            f.write(f"Unique responses: {', '.join(analysis['unique_responses'])}\n\n")
            
            # Print summary for each question
            print(f"\nQuestion: {question}")
            print(f"Field type: {analysis['field_type']}")
            print(f"Number of unique responses: {len(analysis['unique_responses'])}")
            if analysis['field_type'] == 'Slider':
                numeric_responses = [float(r) for r in analysis['unique_responses'] if r.replace('.', '').isdigit()]
                if numeric_responses:
                    print(f"Range of numeric responses: {min(numeric_responses)} to {max(numeric_responses)}")

# Create comprehensive CSV
comprehensive_df = df.copy()
comprehensive_df.to_csv(os.path.join(output_dir, 'comprehensive_ema_data.csv'), index=False)

print("\nProcessing complete. Files created in:", output_dir)
print("1. comprehensive_ema_data.csv")
print("2. [Form_Name]_survey_structure.txt for each form")