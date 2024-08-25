import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from ruptures import Pelt
from scipy import stats
import time
import multiprocessing

def limit_cpu():
    max_processes = max(1, multiprocessing.cpu_count() // 2)
    print(f"Limiting to {max_processes} concurrent processes")
    return max_processes

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds to run.")
        return result
    return wrapper

def classify_movement(speed):
    if pd.isna(speed):
        return 'Unknown'
    elif speed < 1.5:
        return 'Stationary'
    elif speed < 7:
        return 'Active Transport'
    else:
        return 'Mechanized Transport'

@timer
def detect_changepoints(data, column, min_size=5, jump=5, pen=1):
    print(f"Detecting changepoints for column: {column}")
    model = Pelt(model="rbf", jump=jump, min_size=min_size).fit(data[column].values.reshape(-1, 1))
    change_points = model.predict(pen=pen)
    print(f"Found {len(change_points)} changepoints")
    return change_points

@timer
def create_episodes(df, change_points):
    print("Creating episodes...")
    episodes = []
    for i in range(len(change_points) - 1):
        start_idx = change_points[i]
        end_idx = change_points[i + 1]
        
        episode = df.iloc[start_idx:end_idx]
        
        episode_summary = {
            'start_time': episode['Timestamp'].iloc[0],
            'end_time': episode['Timestamp'].iloc[-1],
            'duration': (episode['Timestamp'].iloc[-1] - episode['Timestamp'].iloc[0]).total_seconds() / 60,
            'movement_type': stats.mode(episode['movement_type'])[0][0],
            'indoor_outdoor': stats.mode(episode['indoors'])[0][0],
            'digital_use': 'Yes' if (episode['isapp'] == 1).any() else 'No',
            'avg_speed': episode['speed'].mean()
        }
        
        episodes.append(episode_summary)
    
    print(f"Created {len(episodes)} episodes")
    return pd.DataFrame(episodes)

@timer
def calculate_fragmentation_index(episodes_df, column):
    print(f"Calculating fragmentation index for column: {column}")
    total_duration = episodes_df['duration'].sum()
    episode_counts = episodes_df[column].value_counts()
    fragmentation_indices = {}
    
    for category in episode_counts.index:
        category_episodes = episodes_df[episodes_df[column] == category]
        S = len(category_episodes)
        T = category_episodes['duration'].sum()
        
        if S > 1:
            index = (1 - sum((category_episodes['duration'] / T) ** 2)) / (1 - (1 / S))
        else:
            index = 0
        
        fragmentation_indices[f"{category}_index"] = index
    
    return fragmentation_indices

def preprocess_data(df):
    print("Starting preprocessing...")
    df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
    
    df.loc[:, 'time'] = df['time'].astype(str)
    df.loc[:, 'Timestamp'] = df.apply(lambda row: pd.Timestamp.combine(row['date'].date(), pd.to_datetime(row['time']).time()), axis=1)
    df = df.dropna(subset=['Timestamp'])
    df.loc[:, 'movement_type'] = df['speed'].apply(classify_movement)
    
    print(f"Preprocessed data shape: {df.shape}")
    return df

@timer
def analyze_participant_day(participant_day_df):
    try:
        print(f"Analyzing data for {participant_day_df['Timestamp'].dt.date.iloc[0]}")
        
        change_points = detect_changepoints(participant_day_df, 'speed')
        episodes_df = create_episodes(participant_day_df, change_points)

        print("Calculating fragmentation indices...")
        fragmentation_indices_movement = calculate_fragmentation_index(episodes_df, 'movement_type')
        fragmentation_indices_io = calculate_fragmentation_index(episodes_df, 'indoor_outdoor')
        fragmentation_indices_digital = calculate_fragmentation_index(episodes_df, 'digital_use')

        all_indices = {**fragmentation_indices_movement, **fragmentation_indices_io, **fragmentation_indices_digital}

        print("Calculating modes...")
        modes = {
            'movement_mode': stats.mode(participant_day_df['movement_type'])[0][0],
            'indoor_outdoor_mode': stats.mode(participant_day_df['indoors'])[0][0],
            'digital_use_mode': 'Yes' if stats.mode(participant_day_df['isapp'])[0][0] == 1 else 'No'
        }

        return episodes_df, all_indices, modes

    except Exception as e:
        print(f"Error processing participant day: {str(e)}")
        return None, None, None

@timer
def load_data(file_path):
    print(f"Loading data from {file_path}")
    return pd.read_excel(file_path, sheet_name='gpsappS_8')

def process_participant_day(args):
    participant_id, day_df, episode_dir, date = args
    print(f"\nProcessing participant {participant_id} for date {date}")
    
    day_df = preprocess_data(day_df)
    episodes_df, fragmentation_indices, modes = analyze_participant_day(day_df)
    
    if episodes_df is not None and fragmentation_indices is not None and modes is not None:
        result = {
            'participant_id': participant_id,
            'date': date,
            **fragmentation_indices,
            **modes
        }
        
        # Save episode details for each day
        episodes_df.to_csv(os.path.join(episode_dir, f'participant_{participant_id}_day_{date}_episodes.csv'), index=False)
        return result
    else:
        return None

if __name__ == "__main__":
    max_processes = limit_cpu()
    
    print("Loading data...")
    tlv_path = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/gpsappS_9.1_excel.xlsx'
    tlv_df = load_data(tlv_path)
    print("Data loaded successfully.")
    print(f"Shape of the DataFrame: {tlv_df.shape}")
    
    output_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/'
    episode_dir = os.path.join(output_dir, 'fragment-episodes')
    os.makedirs(episode_dir, exist_ok=True)
    
    print("\nStarting analysis of all participants...")
    
    # Group data by user and date
    grouped = tlv_df.groupby(['user', 'date'])
    
    # Prepare arguments for multiprocessing
    args_list = [
        (name[0], group, episode_dir, name[1]) 
        for name, group in grouped
    ]
    
    with multiprocessing.Pool(processes=max_processes) as pool:
        all_results = pool.map(process_participant_day, args_list)
    
    # Filter out None results
    all_results = [result for result in all_results if result is not None]
    
    print("\nCreating summary DataFrame...")
    summary_df = pd.DataFrame(all_results)

    if not summary_df.empty:
        summary_df.to_csv(os.path.name(output_dir, 'fragmentation_summary.csv'), index=False)
        print("Summary saved to CSV.")

        print("\nGenerating descriptive statistics...")
        print(summary_df.describe())

        print("\nCreating visualizations...")
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=summary_df[[col for col in summary_df.columns if col.endswith('_index')]])
        plt.title('Distribution of Fragmentation Indices')
        plt.ylabel('Fragmentation Index')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fragmentation_indices_distribution.png'))
        plt.close()

        print("Visualization saved as 'fragmentation_indices_distribution.png' in the output directory.")
    else:
        print("No valid results were generated. Please check your data and error messages.")

    print("\nScript execution completed.")