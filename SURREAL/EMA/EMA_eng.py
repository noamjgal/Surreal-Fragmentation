import pandas as pd

response_dict_path = '/Users/noamgal/Downloads/Research-Projects/SURREAL/EMA-Surreal/processed_response_mappings.csv'
response_dict_df = pd.read_csv(response_dict_path)
comprehensive_data_path = '/Users/noamgal/Downloads/Research-Projects/SURREAL/EMA-Surreal/comprehensive_ema_data_eng.csv'
comprehensive_data_df = pd.read_csv(comprehensive_data_path)

# Drop all rows where the 'Form' column does not begin with 'EMA'
comprehensive_data_df = comprehensive_data_df[comprehensive_data_df['Form name'].str.startswith('EMA', na=False)]

print(response_dict_df.head())
print(response_dict_df.columns)
print(response_dict_df['English_dict'])
print(response_dict_df['Hebrew_dict'])

print(comprehensive_data_df.head())
print(comprehensive_data_df.columns)

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# Function for fuzzy matching
def fuzzy_match(x, choices, scorer=fuzz.token_sort_ratio):
    return process.extractOne(x, choices, scorer=scorer)[0]

# Fuzzy match Form and Question
comprehensive_data_df['Form_matched'] = comprehensive_data_df['Form name'].apply(lambda x: fuzzy_match(x, response_dict_df['Form']))
comprehensive_data_df['Question_matched'] = comprehensive_data_df['Question name'].apply(lambda x: fuzzy_match(x, response_dict_df['Question']))

# Merge the dataframes
merged_df = pd.merge(comprehensive_data_df, 
                     response_dict_df[['Form', 'Question', 'Hebrew_dict', 'English_dict']], 
                     left_on=['Form_matched', 'Question_matched'], 
                     right_on=['Form', 'Question'], 
                     how='left')

# Add Hebrew and English dict columns
merged_df['Hebrew_dict'] = merged_df['Hebrew_dict'].fillna('{}').apply(eval)
merged_df['English_dict'] = merged_df['English_dict'].fillna('{}').apply(eval)

# Clean up unnecessary columns
columns_to_drop = ['Form_matched', 'Question_matched', 'Form', 'Question']
merged_df = merged_df.drop(columns=columns_to_drop)

# Update comprehensive_data_df
comprehensive_data_df = merged_df

print(comprehensive_data_df[['Question name', 'Responses name', 'Hebrew_dict', 'English_dict']].head())

# Print examples where dictionary is empty
print("\nExamples where Hebrew dictionary is empty:")
empty_hebrew = comprehensive_data_df[comprehensive_data_df['Hebrew_dict'].apply(lambda x: len(x) == 0)]
print(empty_hebrew[['Question name', 'Responses name']].head(40))

print("\nExamples where English dictionary is empty:")
empty_english = comprehensive_data_df[comprehensive_data_df['English_dict'].apply(lambda x: len(x) == 0)]
print(empty_english[['Question name', 'Responses name']].head(40))

print(f"\nTotal rows with empty Hebrew dictionary: {len(empty_hebrew)}")
print(f"Total rows with empty English dictionary: {len(empty_english)}")

# Save the updated comprehensive data
output_file = '/Users/noamgal/Downloads/Research-Projects/SURREAL/EMA-Surreal/comprehensive_ema_data_eng_updated.csv'
comprehensive_data_df.to_csv(output_file, index=False)
print(f"\nUpdated data saved to: {output_file}")
