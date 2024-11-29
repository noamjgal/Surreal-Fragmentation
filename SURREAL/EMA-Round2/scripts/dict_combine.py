import pandas as pd
import sys
import ast
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

def normalize_dict(dict_str):
    """Convert dictionary string to normalized form (lowercase keys)"""
    try:
        d = ast.literal_eval(dict_str)
        return {k.lower(): v for k, v in d.items()}
    except:
        return {}

def is_subset_dict(dict1_str, dict2_str):
    """Check if dict1 is a subset of dict2 (case-insensitive)"""
    try:
        dict1 = normalize_dict(dict1_str)
        dict2 = normalize_dict(dict2_str)
        return all(k in dict2 and dict2[k] == v for k, v in dict1.items())
    except:
        return False

def dict_similarity(dict1_str, dict2_str):
    """Return number of shared key-value pairs between two dictionaries (case-insensitive)"""
    try:
        dict1 = normalize_dict(dict1_str)
        dict2 = normalize_dict(dict2_str)
        shared_items = {k: v for k, v in dict1.items() if k in dict2 and dict2[k] == v}
        return len(shared_items)
    except:
        return 0

def should_merge(dict1_str, dict2_str):
    """Check if dictionaries should be merged based on subset or similarity"""
    return (is_subset_dict(dict1_str, dict2_str) or 
            is_subset_dict(dict2_str, dict1_str))

def merge_dicts(dict1_str, dict2_str):
    """Merge two dictionaries, keeping all unique entries"""
    dict1 = ast.literal_eval(dict1_str)
    dict2 = ast.literal_eval(dict2_str)
    
    # Create a mapping of lowercase keys to original keys
    dict1_lower = {k.lower(): k for k in dict1.keys()}
    dict2_lower = {k.lower(): k for k in dict2.keys()}
    
    # Merge preserving the first encountered case for each key
    merged = {}
    all_lower_keys = set(dict1_lower.keys()) | set(dict2_lower.keys())
    
    for lower_key in all_lower_keys:
        # Determine which original key to use
        original_key = dict1_lower.get(lower_key) or dict2_lower.get(lower_key)
        # Get the value from either dict1 or dict2
        value = dict1.get(dict1_lower.get(lower_key)) or dict2.get(dict2_lower.get(lower_key))
        merged[original_key] = value
    
    return str(merged)

def process_variable_dicts(dict_list):
    """Process dictionaries for a variable, merging based on subset or similarity"""
    if not dict_list:
        return []
    
    result = [dict_list[0]]
    
    for i in range(1, len(dict_list)):
        should_merge_with = None
        
        for j, existing_dict in enumerate(result):
            if should_merge(dict_list[i], existing_dict):
                should_merge_with = j
                break
        
        if should_merge_with is not None:
            result[should_merge_with] = merge_dicts(dict_list[i], result[should_merge_with])
        else:
            result.append(dict_list[i])
    
    return result

# Read the CSV
mappings_df = pd.read_csv(Path(project_root) / "data" / "raw" / "processed_dictionaries.csv")

# Process each variable
new_rows = []
for variable in mappings_df['Variable'].unique():
    variable_df = mappings_df[mappings_df['Variable'] == variable]
    dict_list = variable_df['Eng_dict_processed'].tolist()
    
    # Get merged dictionaries
    merged_dicts = process_variable_dicts(dict_list)
    
    # Create new rows for the merged dictionaries
    for merged_dict in merged_dicts:
        template_row = variable_df.iloc[0].copy()
        template_row['Eng_dict_processed'] = merged_dict
        new_rows.append(template_row)

# Create new dataframe from processed rows
result_df = pd.DataFrame(new_rows)

# Save the updated dataframe
output_path = Path(project_root) / "data" / "raw" / "processed_dictionaries_merged.csv"
result_df.to_csv(output_path, index=False)

# Print results
print("\nProcessed dictionaries for each variable:")
for variable in result_df['Variable'].unique():
    print(f"\n{variable} dict set:")
    for dict_str in result_df[result_df['Variable'] == variable]['Eng_dict_processed']:
        print(dict_str)
