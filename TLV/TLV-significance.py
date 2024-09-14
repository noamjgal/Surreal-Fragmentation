import pandas as pd
import numpy as np
from scipy import stats
import os

# Define input and output directories
input_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/fragmentation'
survey_file = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/Survey/End_of_the_day_questionnaire.xlsx'
output_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/analysis_results'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the fragmentation results
mobility_frag = pd.read_csv(os.path.join(input_dir, 'mobility_episodes_fragmentation_summary.csv'))
digital_frag = pd.read_csv(os.path.join(input_dir, 'digital_episodes_fragmentation_summary.csv'))

# Load the survey responses
survey_responses = pd.read_excel(survey_file)

# Print information about input data
print("Mobility Fragmentation Data:")
print(mobility_frag.info())
print("\nSample of mobility_frag['date']:")
print(mobility_frag['date'].head())

print("\nDigital Fragmentation Data:")
print(digital_frag.info())
print("\nSample of digital_frag['date']:")
print(digital_frag['date'].head())

print("\nSurvey Responses Data:")
print(survey_responses.info())
print("\nSample of survey_responses['StartDate']:")
print(survey_responses['StartDate'].head())

# Data preprocessing
def parse_frag_date(date_series):
    # Assuming the date is just the day of the month
    return date_series

def parse_survey_date(date_series):
    return pd.to_datetime(date_series).dt.day

mobility_frag['date'] = parse_frag_date(mobility_frag['date'])
digital_frag['date'] = parse_frag_date(digital_frag['date'])
survey_responses['date'] = parse_survey_date(survey_responses['StartDate'])

# Ensure participant_id is treated as string in all dataframes
mobility_frag['participant_id'] = mobility_frag['participant_id'].astype(str)
digital_frag['participant_id'] = digital_frag['participant_id'].astype(str)
survey_responses['Participant_ID'] = survey_responses['Participant_ID'].astype(str)

# Merge the datasets
merged_data = pd.merge(mobility_frag, digital_frag, on=['participant_id', 'date'], suffixes=('_mobility', '_digital'))
merged_data = pd.merge(merged_data, survey_responses, left_on=['participant_id', 'date'], right_on=['Participant_ID', 'date'], how='inner')

print("\nMerged Data:")
print(merged_data.info())
print("\nSample of merged data:")
print(merged_data.head())

# Ensure emotional scores are numeric
emotion_columns = ['PEACE', 'TENSE', 'IRRITATION', 'RELAXATION', 'SATISFACTION', 'WORRY', 'HAPPY']
for col in emotion_columns:
    merged_data[col] = pd.to_numeric(merged_data[col], errors='coerce')

# Define fragmentation indices
frag_columns = [
    'Moving_fragmentation_index',
    'Moving_AID_mean',
    'Digital_fragmentation_index',
    'Digital_AID_mean'
]

# Function to perform t-test and calculate effect size
def perform_ttest(high_group, low_group, emotion):
    t_stat, p_value = stats.ttest_ind(high_group[emotion].dropna(), low_group[emotion].dropna())
    effect_size = (high_group[emotion].mean() - low_group[emotion].mean()) / np.sqrt((high_group[emotion].std()**2 + low_group[emotion].std()**2) / 2)
    return t_stat, p_value, effect_size

# Perform t-tests for each fragmentation index
results = []
for frag_index in frag_columns:
    print(f"\nAnalyzing {frag_index}:")
    median = merged_data[frag_index].median()
    high_frag = merged_data[merged_data[frag_index] > median]
    low_frag = merged_data[merged_data[frag_index] <= median]
    
    print(f"  Median: {median}")
    print(f"  High group size: {len(high_frag)}")
    print(f"  Low group size: {len(low_frag)}")
    
    for emotion in emotion_columns:
        t_stat, p_value, effect_size = perform_ttest(high_frag, low_frag, emotion)
        results.append({
            'Fragmentation Index': frag_index,
            'Emotion': emotion,
            't-statistic': t_stat,
            'p-value': p_value,
            'Effect Size': effect_size,
            'High Group Mean': high_frag[emotion].mean(),
            'Low Group Mean': low_frag[emotion].mean(),
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })
        print(f"  {emotion}: p-value = {p_value:.4f}, effect size = {effect_size:.4f}, significant: {'Yes' if p_value < 0.05 else 'No'}")

# Convert results to DataFrame and sort by p-value
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('p-value')

# Save full results to CSV
full_results_path = os.path.join(output_dir, 'full_significance_results.csv')
results_df.to_csv(full_results_path, index=False)
print(f"\nFull significance results saved as '{full_results_path}'")

# Print all results sorted by p-value
print("\nAll Results Sorted by p-value:")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print(results_df.to_string(index=False))

# Print significant results (p < 0.05)
significant_results = results_df[results_df['p-value'] < 0.05]
print("\nSignificant Results (p < 0.05):")
print(significant_results.to_string(index=False))

print(f"\nAll analysis results have been saved in the directory: {output_dir}")