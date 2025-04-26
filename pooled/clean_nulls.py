import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import scipy.stats as stats
import os
import warnings
from tqdm import tqdm

file_path = "processed/pooled_stai_data_population.csv"

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
    'out_of_home_duration': 'Out of Home Duration',
    'mobility_episode_count': 'Mobility Episode Count'
}

# Rename the columns
df.rename(columns=column_mapping, inplace=True)

print("\nRenamed columns:")
print(df.columns)

# Replace NaN values with 0
df.fillna(0, inplace=True)

# Recalculate Digital Home Mobility Delta after NaN replacements
df['Digital Home Mobility Delta'] = df['Digital Mobile Fragmentation'] - df['Digital Home Fragmentation']

# Drop observations where home duration is 0
zero_home_count = len(df[df['Home Duration'] == 0])
print(f"\nDropping {zero_home_count} observations where Home Duration is 0")
df = df[df['Home Duration'] > 0]
print(f"Dataset shape after dropping zero home duration: {df.shape}")

# Drop the Out of Home Duration column
df.drop(columns=['Out of Home Duration'], inplace=True)

# Save the updated DataFrame to a new file
output_file_path = file_path.replace('.csv', '_cleaned.csv')
df.to_csv(output_file_path, index=False)
print(f"\nSaved cleaned data to: {output_file_path}")

# Print summary statistics for the key variables
print("\nSummary statistics for key variables after cleaning:")
summary_vars = ['Anxiety (Z)', 'Depressed Mood (Z)', 'Digital Fragmentation', 
                'Mobility Fragmentation', 'Digital Mobile Fragmentation', 
                'Home Duration']
print(df[summary_vars].describe())