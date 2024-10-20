import pandas as pd
import re

response_eng_path = '/Users/noamgal/Downloads/Research-Projects/SURREAL/EMA-Surreal/response_mapping_english_add.csv'

response_eng_df = pd.read_csv(response_eng_path)

# Rename 'Responses.1' column to 'Responses_English'
response_eng_df = response_eng_df.rename(columns={'Responses.1': 'Responses_English'})

# Drop all rows where the 'Form' column does not begin with 'EMA'
response_eng_df = response_eng_df[response_eng_df['Form'].str.startswith('EMA', na=False)]

# Print the number of remaining rows
print(f"\nNumber of rows after filtering: {len(response_eng_df)}")

# Display the first few rows of the filtered DataFrame
print("\nFirst few rows of the filtered DataFrame:")
print(response_eng_df.head())

def clean_and_process_responses(responses):
    result = {}
    if pd.isna(responses):
        return result
    
    # Split the responses by semicolon
    pairs = responses.split(';')
    for pair in pairs:
        # Split each pair by colon, but only split on the first occurrence
        parts = pair.split(':', 1)
        if len(parts) == 2:
            key, value = parts
            result[key.strip()] = value.strip()
        else:
            print(f"Warning: Invalid pair format: {pair}")
    
    return result

# Apply the function to create Hebrew and English dictionaries
response_eng_df['Hebrew_dict'] = response_eng_df['Responses'].apply(clean_and_process_responses)
response_eng_df['English_dict'] = response_eng_df['Responses_English'].apply(clean_and_process_responses)

print("\nSample processed response mappings:")
print(response_eng_df[['Question', 'Hebrew_dict', 'English_dict']].head())

# Print out responses with no keys
print("\nResponses with no keys:")
for index, row in response_eng_df.iterrows():
    if len(row['Hebrew_dict']) == 0 and not pd.isna(row['Responses']):
        print(f"Hebrew - Question: {row['Question']}, Response: {row['Responses']}")
    if len(row['English_dict']) == 0 and not pd.isna(row['Responses_English']):
        print(f"English - Question: {row['Question']}, Response: {row['Responses_English']}")

# Save the processed response mappings
output_file = '/Users/noamgal/Downloads/Research-Projects/SURREAL/EMA-Surreal/processed_response_mappings.csv'
response_eng_df.to_csv(output_file, index=False)
print(f"\nProcessed response mappings saved to: {output_file}")
