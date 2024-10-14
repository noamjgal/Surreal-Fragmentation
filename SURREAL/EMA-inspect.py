import pandas as pd
import numpy as np

ema_file_path = "/Users/noamgal/Downloads/Research-Projects/SURREAL/EMA-Surreal/ema-responses.csv"

# Load the CSV file with the correct separator
df = pd.read_csv(ema_file_path, sep='|', quotechar='"')

# Function to get unique questions for a given form
def get_questions_for_form(df, form_name):
    form_data = df[df['Form name'] == form_name]
    return form_data['Question name'].unique()

# Get unique form names
form_names = df['Form name'].unique()

# Print questions for each form
for form_name in form_names:
    print(f"\nQuestions for {form_name}:")
    questions = get_questions_for_form(df, form_name)
    for i, question in enumerate(questions, 1):
        print(f"{i}. {question}")

# Print summary of forms and question counts
print("\nSummary of forms and question counts:")
for form_name in form_names:
    question_count = len(get_questions_for_form(df, form_name))
    print(f"{form_name}: {question_count} questions")
