import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import scipy.stats as stats
import os
import warnings
from tqdm import tqdm

file_path = "pooled/processed/pooled_stai_data_population.csv"

df = pd.read_csv(file_path)

print("Original columns:")
print(df.columns)
print(f"Original dataset shape: {df.shape}")

# Define a mapping of current column names to more human-readable names
column_mapping = {
    'participant_id': 'Participant ID',
    'dataset_source': 'Dataset Source',
    'anxiety_score_std': 'Anxiety (Z)',
    'anxiety_score_raw': 'Anxiety (Raw)',
    'mood_score_std': 'Depressed Mood (Z)',
    'mood_score_raw': 'Depressed Mood (Raw)',
    'gender_standardized': 'Gender',
    'location_type': 'Location Type',
    'age_group': 'Age Group',
    'is_weekend': 'Weekend Status',
    'digital_fragmentation': 'Digital Fragmentation',
    'mobility_fragmentation': 'Mobility Fragmentation',
    'overlap_fragmentation': 'Digital Mobile Fragmentation',
    'digital_home_fragmentation': 'Digital Home Fragmentation',
    'digital_home_mobility_delta': 'Digital Home Mobility Delta',
    'digital_total_duration': 'Digital Duration',
    'mobility_total_duration': 'Mobile Duration',
    'overlap_total_duration': 'Digital Mobile Duration',
    'digital_home_total_duration': 'Digital Home Duration',
    'active_transport_duration': 'Active Transport Duration',
    'mechanized_transport_duration': 'Mechanized Transport Duration',
    'home_duration': 'Home Duration',
    'out_of_home_duration': 'Out of Home Duration'
}

# Rename the columns
df.rename(columns=column_mapping, inplace=True)

print("\nRenamed columns:")
print(df.columns)

# Drop all spatial behavior related columns
spatial_columns = [
    'Mobility Fragmentation', 
    'Digital Mobile Fragmentation',
    'Digital Home Fragmentation', 
    'Digital Home Mobility Delta',
    'Mobile Duration', 
    'Digital Mobile Duration',
    'Digital Home Duration', 
    'Active Transport Duration',
    'Mechanized Transport Duration', 
    'Home Duration',
    'Out of Home Duration'
]

# Drop the spatial behavior columns
df.drop(columns=spatial_columns, errors='ignore', inplace=True)

# Save the updated DataFrame to a new file
output_file_path = file_path.replace('.csv', '_clean_digital.csv')
df.to_csv(output_file_path, index=False)
print(f"\nSaved cleaned data to: {output_file_path}")

# Print summary statistics for the key variables
print("\nSummary statistics for key variables after cleaning:")
summary_vars = ['Anxiety (Z)', 'Depressed Mood (Z)', 'Digital Fragmentation', 'Digital Duration']
print(df[summary_vars].describe())
print(df.columns)