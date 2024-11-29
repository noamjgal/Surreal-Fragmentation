import pandas as pd
import sys
import ast
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

def normalize_key(key):
    """Normalize key to Title Case for first word, lowercase for rest, and remove trailing spaces"""
    words = key.strip().split()
    if not words:
        return key
    return (words[0].capitalize() + ' ' + ' '.join(w.lower() for w in words[1:])).strip()

def normalize_hebrew_key(key):
    """Normalize Hebrew key by only removing trailing spaces"""
    return key.strip()

def normalize_dict_keys(dict_str, is_hebrew=False):
    """Convert dictionary keys to normalized format"""
    try:
        d = ast.literal_eval(dict_str)
        if is_hebrew:
            return str({normalize_hebrew_key(k): v for k, v in d.items()})
        return str({normalize_key(k): v for k, v in d.items()})
    except:
        return dict_str

def dict_similarity(dict1_str, dict2_str):
    """Return True if dictionaries should be merged based on similarity or subset relationship"""
    try:
        dict1 = ast.literal_eval(normalize_dict_keys(dict1_str))
        dict2 = ast.literal_eval(normalize_dict_keys(dict2_str))
        
        # Check if one is a subset of the other
        items1 = set((k, v) for k, v in dict1.items())
        items2 = set((k, v) for k, v in dict2.items())
        
        # Return True if either:
        # 1. They share 3 or more items
        # 2. One dictionary is a complete subset of the other
        shared_items = items1.intersection(items2)
        return len(shared_items) >= 3 or items1.issubset(items2) or items2.issubset(items1)
    except:
        return False

def merge_similar_dicts(dict1_str, dict2_str):
    """Merge two dictionaries, keeping all unique entries"""
    dict1 = ast.literal_eval(normalize_dict_keys(dict1_str))
    dict2 = ast.literal_eval(normalize_dict_keys(dict2_str))
    merged = dict1.copy()
    merged.update(dict2)
    return str(merged)

def sort_dict_by_values(dict_str):
    """Sort dictionary by numerical values"""
    try:
        d = ast.literal_eval(dict_str)
        # Sort by numerical value
        sorted_items = sorted(d.items(), key=lambda x: int(x[1]))
        return str(dict(sorted_items))
    except:
        return dict_str

def process_variable_dicts(dict_list):
    """
    Process a list of dictionaries for a variable:
    - Normalize all dictionary keys
    - Sort by numerical values
    - Merge dictionaries that are similar or subsets
    - Keep unique dictionaries
    """
    if not dict_list:
        return []
    
    # First normalize all dictionaries in the list and sort their items
    normalized_list = [sort_dict_by_values(normalize_dict_keys(d)) for d in dict_list]
    
    # Sort by length (descending) to prioritize merging into larger dictionaries
    normalized_list.sort(key=lambda x: len(ast.literal_eval(x)), reverse=True)
    
    result = [normalized_list[0]]
    
    for i in range(1, len(normalized_list)):
        should_merge = False
        merge_index = None
        
        for j, existing_dict in enumerate(result):
            if dict_similarity(normalized_list[i], existing_dict):
                should_merge = True
                merge_index = j
                break
        
        if should_merge:
            merged = merge_similar_dicts(normalized_list[i], result[merge_index])
            result[merge_index] = sort_dict_by_values(merged)
        else:
            result.append(normalized_list[i])
    
    return result

# Read the CSV
mappings_df = pd.read_csv(Path(project_root) / "data" / "raw" / "processed_dictionaries.csv")

# Process each variable
processed_rows = []
for variable in mappings_df['Variable'].unique():
    variable_df = mappings_df[mappings_df['Variable'] == variable]
    eng_dict_list = variable_df['Eng_dict_processed'].tolist()
    heb_dict_list = variable_df['Hebrew_dict_processed'].tolist()
    
    # Get merged dictionaries for both languages
    merged_eng_dicts = process_variable_dicts(eng_dict_list)
    merged_heb_dicts = process_variable_dicts([d if isinstance(d, str) else str(d) for d in heb_dict_list])
    
    # Keep all original rows
    for idx, row in variable_df.iterrows():
        new_row = row.copy()
        
        # Update the dictionaries with the most comprehensive version if available
        if merged_eng_dicts:
            new_row['Eng_dict_processed'] = merged_eng_dicts[0]  # Use the most comprehensive dict
        if merged_heb_dicts:
            new_row['Hebrew_dict_processed'] = merged_heb_dicts[0]  # Use the most comprehensive dict
            
        processed_rows.append(new_row)

# Create new dataframe from processed rows
result_df = pd.DataFrame(processed_rows)

# Save the updated dataframe
output_path = Path(project_root) / "data" / "reordered" / "processed_dictionaries_merged.csv"
result_df.to_csv(output_path, index=False)

# Print results
print("\nProcessed dictionaries for each variable:")
for variable in result_df['Variable'].unique():
    print(f"\n{variable} dict set:")
    for dict_str in result_df[result_df['Variable'] == variable]['Eng_dict_processed']:
        print(dict_str)
