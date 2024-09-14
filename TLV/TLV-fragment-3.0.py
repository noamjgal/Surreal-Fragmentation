import pandas as pd
import numpy as np
import os

def calculate_fragmentation_index(episodes_df, column):
    fragmentation_indices = {}

    categories = episodes_df[column].unique()

    for category in categories:
        category_episodes = episodes_df[episodes_df[column] == category]
        S = len(category_episodes)
        T = category_episodes['duration'].sum()

        if S > 1 and T > 0:
            normalized_durations = category_episodes['duration'] / T
            sum_squared = sum(normalized_durations ** 2)
            index = (1 - sum_squared) / (1 - (1 / S))
        else:
            index = np.nan  # Not enough data to calculate fragmentation

        fragmentation_indices[f"{category}_index"] = index

    return fragmentation_indices

def calculate_aid(episodes_df, column):
    aid_values = {}

    categories = episodes_df[column].unique()

    for category in categories:
        category_episodes = episodes_df[episodes_df[column] == category].sort_values('start_time')
        if len(category_episodes) > 1:
            inter_episode_durations = (category_episodes['start_time'].iloc[1:] - category_episodes['end_time'].iloc[:-1]).dt.total_seconds() / 60
            aid = abs(inter_episode_durations.mean())  # Use absolute value to ensure positive AID
        else:
            aid = np.nan

        aid_values[f"{category}_AID"] = aid

    return aid_values

def process_episode_summary(file_path, episode_type):
    print(f"Processing file: {file_path}")
    df = pd.read_csv(file_path)
    
    print("Original columns:")
    print(df.columns)
    print("\nFirst few rows of the original data:")
    print(df.head())
    
    # Rename unnamed columns
    df.columns = ['participant_id', 'date', 'Unnamed: 2'] + list(df.columns[3:])
    
    print("\nColumns after renaming:")
    print(df.columns)
    print("\nFirst few rows after renaming columns:")
    print(df.head())
    
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    df['date'] = pd.to_datetime(df['date'])

    results = []

    # Determine the correct column name based on episode type
    if episode_type == 'mobility_episodes':
        column = 'movement_type'
    elif episode_type == 'indoor_outdoor_episodes':
        column = 'indoor_outdoor'
    elif episode_type == 'digital_episodes':
        column = 'digital_use'
    else:
        print(f"Error: Unknown episode type '{episode_type}'")
        return pd.DataFrame()

    print(f"\nColumn name for this episode type: {column}")
    
    if column not in df.columns:
        print(f"Error: Column '{column}' not found in the DataFrame.")
        print("Available columns:", df.columns)
        return pd.DataFrame()

    for (participant_id, date), group in df.groupby(['participant_id', 'date']):
        print(f"\nProcessing participant {participant_id} on {date}")
        result = {
            'participant_id': participant_id,
            'date': date,
            'total_episodes': len(group),
            'total_duration': group['duration'].sum(),
            'avg_episode_length': group['duration'].mean(),
        }

        print(f"Unique categories in {column}: {group[column].unique()}")
        for category in group[column].unique():
            category_episodes = group[group[column] == category]
            result[f'{category}_episodes'] = len(category_episodes)
            result[f'{category}_duration'] = category_episodes['duration'].sum()
            result[f'{category}_avg_length'] = category_episodes['duration'].mean()

        indices = calculate_fragmentation_index(group, column)
        aid_values = calculate_aid(group, column)
        result.update(indices)
        result.update(aid_values)

        results.append(result)

    return pd.DataFrame(results)

def print_summary_statistics(df, episode_type):
    print(f"\nSummary Statistics for {episode_type}:")
    print(f"Total participants: {df['participant_id'].nunique()}")
    print(f"Total days: {len(df)}")
    print(f"Average episodes per day: {df['total_episodes'].mean():.2f}")
    print(f"Average episode length: {df['avg_episode_length'].mean():.2f} minutes")
    print(f"Average total duration per day: {df['total_duration'].mean():.2f} minutes")

    # Print fragmentation indices
    index_columns = [col for col in df.columns if col.endswith('_index')]
    print("\nAverage Fragmentation Indices:")
    for col in index_columns:
        print(f"{col}: {df[col].mean():.4f}")

    # Print AIDs
    aid_columns = [col for col in df.columns if col.endswith('_AID')]
    print("\nAverage Interval Durations (AID):")
    for col in aid_columns:
        print(f"{col}: {df[col].mean():.2f} minutes")

def main(input_dir, output_dir):
    episode_types = ['mobility_episodes', 'indoor_outdoor_episodes', 'digital_episodes']
    
    all_results = {}
    
    for episode_type in episode_types:
        input_file = os.path.join(input_dir, f'{episode_type}_summary.csv')
        if os.path.exists(input_file):
            print(f"\nProcessing {episode_type}...")
            results = process_episode_summary(input_file, episode_type)
            if not results.empty:
                all_results[episode_type] = results
                
                # Save individual results
                output_file = os.path.join(output_dir, f'{episode_type}_fragmentation_summary.csv')
                results.to_csv(output_file, index=False)
                print(f"Saved fragmentation summary for {episode_type} to {output_file}")

                # Print summary statistics
                print_summary_statistics(results, episode_type)
            else:
                print(f"No valid results for {episode_type}")
        else:
            print(f"Warning: {input_file} not found. Skipping {episode_type} analysis.")

    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results.values(), axis=1)
        combined_results = combined_results.loc[:,~combined_results.columns.duplicated()]  # Remove duplicate columns
        combined_output_file = os.path.join(output_dir, 'combined_fragmentation_summary.csv')
        combined_results.to_csv(combined_output_file, index=False)
        print(f"Saved combined fragmentation summary to {combined_output_file}")
    else:
        print("No results to combine. Please check if the input files exist and contain valid data.")

if __name__ == "__main__":
    input_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/'
    output_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/'
    main(input_dir, output_dir)