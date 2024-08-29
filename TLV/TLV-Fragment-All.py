import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from scipy.signal import medfilt

def smooth_data(data, window_size=5):
    return medfilt(data, kernel_size=window_size)

def classify_movement(speed, stationary_threshold=0.5, mobile_threshold=2.0):
    if pd.isna(speed):
        return 'Unknown'
    elif speed < stationary_threshold:
        return 'Stationary'
    elif speed >= mobile_threshold:
        return 'Mobile'
    else:
        return 'Transition'

def detect_episodes(df, movement_window=5, indoor_outdoor_window=3, digital_window=0.25, min_episode_duration=10, min_speed_change=1.0):
    df['smoothed_speed'] = smooth_data(df['speed'].fillna(0))
    df['movement_type'] = df['smoothed_speed'].apply(classify_movement)

    episodes = []
    current_episode = {
        'start_time': df['Timestamp'].iloc[0],
        'movement_type': df['movement_type'].iloc[0],
        'indoor_outdoor': 'Indoor' if df['indoors'].iloc[0] == 'True' else 'Outdoor',
        'digital_use': 'Digital Usage' if df['isapp'].iloc[0] else 'Zero-Digital Usage'
    }

    for i in range(1, len(df)):
        if (df['movement_type'].iloc[i] != current_episode['movement_type'] or
            df['indoors'].iloc[i] != (current_episode['indoor_outdoor'] == 'Indoor') or
            df['isapp'].iloc[i] != (current_episode['digital_use'] == 'Digital Usage')):

            if abs(df['smoothed_speed'].iloc[i] - df['smoothed_speed'].iloc[i-1]) >= min_speed_change:
                current_episode['end_time'] = df['Timestamp'].iloc[i-1]
                current_episode['duration'] = (current_episode['end_time'] - current_episode['start_time']).total_seconds() / 60

                if ((current_episode['movement_type'] != 'Unknown' and current_episode['duration'] >= min_episode_duration) or
                    (current_episode['digital_use'] == 'Digital Usage' and current_episode['duration'] >= digital_window)):
                    episodes.append(current_episode)

                current_episode = {
                    'start_time': df['Timestamp'].iloc[i],
                    'movement_type': df['movement_type'].iloc[i],
                    'indoor_outdoor': 'Indoor' if df['indoors'].iloc[i] == 'True' else 'Outdoor',
                    'digital_use': 'Digital Usage' if df['isapp'].iloc[i] else 'Zero-Digital Usage'
                }

    # Add the last episode
    current_episode['end_time'] = df['Timestamp'].iloc[-1]
    current_episode['duration'] = (current_episode['end_time'] - current_episode['start_time']).total_seconds() / 60

    if ((current_episode['movement_type'] != 'Unknown' and current_episode['duration'] >= min_episode_duration) or
        (current_episode['digital_use'] == 'Digital Usage' and current_episode['duration'] >= digital_window)):
        episodes.append(current_episode)

    return pd.DataFrame(episodes)

def calculate_fragmentation_index(episodes_df, column):
    fragmentation_indices = {}

    categories = {
        'movement_type': ['Stationary', 'Mobile', 'Transition'],
        'indoor_outdoor': ['Indoor', 'Outdoor'],
        'digital_use': ['Digital Usage', 'Zero-Digital Usage']
    }

    for category in categories.get(column, []):
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

    categories = {
        'movement_type': ['Stationary', 'Mobile', 'Transition'],
        'indoor_outdoor': ['Indoor', 'Outdoor'],
        'digital_use': ['Digital Usage', 'Zero-Digital Usage']
    }

    for category in categories.get(column, []):
        category_episodes = episodes_df[episodes_df[column] == category].sort_values('start_time')
        if len(category_episodes) > 1:
            inter_episode_durations = (category_episodes['start_time'].iloc[1:] - category_episodes['end_time'].iloc[:-1]).dt.total_seconds() / 60
            aid = abs(inter_episode_durations.mean())  # Use absolute value to ensure positive AID
        else:
            aid = np.nan

        aid_values[f"{category}_AID"] = aid

    return aid_values

def check_data_quality(result):
    issues = []
    if result['total_episodes'] < 2:
        issues.append("Too few episodes")
    if result['total_duration'] < 360:  # Less than 6 hours
        issues.append("Short duration")
    if result['Stationary_duration'] / result['total_duration'] > 0.99 or result['Mobile_duration'] / result['total_duration'] > 0.99:
        issues.append("Extreme movement imbalance")

    # Check for outlier fragmentation indices
    index_columns = [col for col in result.keys() if col.endswith('ary_index' or 'ile_index')]
    for col in index_columns:
        if result[col] < 0.05 or result[col] > 0.95:
            issues.append(f"Outlier {col}")

    return issues

def analyze_participant_day(file_path):
    try:
        participant_df = pd.read_csv(file_path)
        participant_df['Timestamp'] = pd.to_datetime(participant_df['Timestamp'])
        participant_df['speed'] = pd.to_numeric(participant_df['speed'], errors='coerce')
        participant_df['isapp'] = participant_df['isapp'].astype(bool)
        participant_df['indoors'] = participant_df['indoors'].astype(str)
        participant_df = participant_df.sort_values('Timestamp').reset_index(drop=True)

        if participant_df.empty or 'speed' not in participant_df.columns:
            return None, f"Empty DataFrame or missing 'speed' column in {file_path}"

        episodes_df = detect_episodes(participant_df)

        if episodes_df.empty:
            return None, f"No episodes detected in {file_path}"

        result = {
            'participant_id': participant_df['user'].iloc[0],
            'date': participant_df['Timestamp'].dt.date.iloc[0],
            'total_episodes': len(episodes_df),
            'total_duration': episodes_df['duration'].sum(),
        }

        for column in ['movement_type', 'indoor_outdoor', 'digital_use']:
            if column in episodes_df.columns:
                for category in episodes_df[column].unique():
                    category_episodes = episodes_df[episodes_df[column] == category]
                    result[f'{category}_episodes'] = len(category_episodes)
                    result[f'{category}_duration'] = category_episodes['duration'].sum()

                indices = calculate_fragmentation_index(episodes_df, column)
                aid_values = calculate_aid(episodes_df, column)
                result.update(indices)
                result.update(aid_values)

        # Handle missing durations
        for category in ['Stationary', 'Mobile', 'Indoor', 'Outdoor', 'Digital Usage', 'Zero-Digital Usage']:
            if f'{category}_duration' not in result:
                result[f'{category}_duration'] = 0

        result['data_quality_issues'] = check_data_quality(result)
        return result, None

    except Exception as e:
        return None, f"Error processing file {os.path.basename(file_path)}: {str(e)}"
    
def main(input_dir, output_dir):
    all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv')]
    print(f"Found {len(all_files)} CSV files in the input directory")

    all_results = []
    problematic_days = []
    error_messages = []

    for file in tqdm(all_files, desc="Processing files", ncols=70):
        result, error = analyze_participant_day(file)
        if error:
            error_messages.append(error)
        elif result is not None:
            if result['data_quality_issues']:
                problematic_days.append(result)
            else:
                all_results.append(result)

    if all_results:
        summary_df = pd.DataFrame(all_results)
        
        # Reorder columns
        column_order = ['participant_id', 'date', 'total_episodes', 'total_duration',
                        'Stationary_episodes', 'Stationary_duration', 'Mobile_episodes', 'Mobile_duration',
                        'Transition_episodes', 'Transition_duration',
                        'Indoor_episodes', 'Indoor_duration', 'Outdoor_episodes', 'Outdoor_duration',
                        'Digital Usage_episodes', 'Digital Usage_duration', 'Zero-Digital Usage_episodes', 'Zero-Digital Usage_duration',
                        'Stationary_index', 'Mobile_index', 'Transition_index',
                        'Indoor_index', 'Outdoor_index',
                        'Digital Usage_index', 'Zero-Digital Usage_index',
                        'Stationary_AID', 'Mobile_AID', 'Transition_AID',
                        'Indoor_AID', 'Outdoor_AID',
                        'Digital Usage_AID', 'Zero-Digital Usage_AID']
        
        summary_df = summary_df.reindex(columns=[col for col in column_order if col in summary_df.columns])
        
        summary_df.to_csv(os.path.join(output_dir, 'fragmentation_daily_summary.csv'), index=False)
        print(f"\nDaily summary saved to CSV. Generated {len(all_results)} valid results out of {len(all_files)} files.")

        print("\nDays with potential issues (discarded from analysis):")
        for day in problematic_days:
            print(f"Participant {day['participant_id']} on {day['date']}: {', '.join(day['data_quality_issues'])}")

        problematic_percentage = (len(problematic_days) / len(all_files)) * 100
        print(f"\n{problematic_percentage:.2f}% of days have potential data quality issues and were discarded.")
    else:
        print("No valid results were generated. Please check the detailed error messages below.")

    print("\nSummary of all processed data:")
    print(f"Total files processed: {len(all_files)}")
    print(f"Valid results: {len(all_results)}")
    print(f"Problematic days: {len(problematic_days)}")

    if problematic_days:
        issue_counts = {}
        for day in problematic_days:
            for issue in day['data_quality_issues']:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        print("\nBreakdown of data quality issues:")
        for issue, count in issue_counts.items():
            print(f"{issue}: {count}")

    if error_messages:
        print("\nErrors encountered during processing:")
        for error in error_messages:
            print(error)

if __name__ == "__main__":
    input_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/preprocessed_data'
    output_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/'
    main(input_dir, output_dir)
