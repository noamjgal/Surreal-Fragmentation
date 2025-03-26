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

print(df.head())

# Save the updated DataFrame to a new file
output_file_path = file_path.replace('.csv', '_renamed.csv')
df.to_csv(output_file_path, index=False)
print(f"\nSaved renamed data to: {output_file_path}")

'''
columns:
Index(['participant_id', 'dataset_source', 'anxiety_score_std',
       'anxiety_score_raw', 'mood_score_std', 'mood_score_raw',
       'gender_standardized', 'location_type', 'age_group',
       'is_weekend', 'digital_fragmentation', 'mobility_fragmentation',
       'overlap_fragmentation', 'digital_home_fragmentation',
       'digital_home_mobility_delta', 'digital_total_duration',
       'mobility_total_duration', 'overlap_total_duration',
       'digital_home_total_duration', 'active_transport_duration',
       'mechanized_transport_duration', 'home_duration',
       'out_of_home_duration'],
      dtype='object')
'''