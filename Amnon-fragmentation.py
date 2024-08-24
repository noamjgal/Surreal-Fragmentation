import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from ruptures import Pelt
from scipy import stats
import time

# Add timing function
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds to run.")
        return result
    return wrapper

# Load the data
print("Loading data...")
tlv_path = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/gpsappS_9.1_excel.xlsx'
tlv_df = pd.read_excel(tlv_path, sheet_name='gpsappS_8')
print("Data loaded successfully.")
print(f"Shape of the DataFrame: {tlv_df.shape}")
print("\nColumn names:")
print(tlv_df.columns)

# Create output directories
output_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/'
episode_dir = os.path.join(output_dir, 'fragment-episodes')
os.makedirs(episode_dir, exist_ok=True)


# Add this code after loading the data and before the preprocessing step
print("\nInspecting 'date' and 'time' columns:")
print(tlv_df[['date', 'time']].dtypes)
print("\nSample data:")
print(tlv_df[['date', 'time']].head())

# Check for any null values
print("\nNull values:")
print(tlv_df[['date', 'time']].isnull().sum())

# Check unique values in each column
print("\nUnique values in 'date' column:")
print(tlv_df['date'].nunique())
print("\nUnique values in 'time' column:")
print(tlv_df['time'].nunique())

# Display a few unique values from each column
print("\nSample unique values in 'date' column:")
print(tlv_df['date'].unique()[:5])
print("\nSample unique values in 'time' column:")
print(tlv_df['time'].unique()[:5])

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
    
    # Convert 'time' column to string if it's not already
    df['time'] = df['time'].astype(str)
    
    # Combine date and time
    df['Timestamp'] = df.apply(lambda row: pd.Timestamp.combine(row['date'].date(), pd.to_datetime(row['time']).time()), axis=1)
    
    # Drop rows with invalid timestamps
    df = df.dropna(subset=['Timestamp'])
    
    # Classify movement
    df['movement_type'] = df['speed'].apply(classify_movement)
    
    print(f"Preprocessed data shape: {df.shape}")
    return df

# Update the analyze_participant function
@timer
def analyze_participant(participant_df):
    try:
        print("Preprocessing data...")
        participant_df = preprocess_data(participant_df)
        
        change_points = detect_changepoints(participant_df, 'speed')
        episodes_df = create_episodes(participant_df, change_points)

        print("Calculating fragmentation indices...")
        fragmentation_indices_movement = calculate_fragmentation_index(episodes_df, 'movement_type')
        fragmentation_indices_io = calculate_fragmentation_index(episodes_df, 'indoor_outdoor')
        fragmentation_indices_digital = calculate_fragmentation_index(episodes_df, 'digital_use')

        all_indices = {**fragmentation_indices_movement, **fragmentation_indices_io, **fragmentation_indices_digital}

        print("Calculating modes...")
        modes = {
            'movement_mode': stats.mode(participant_df['movement_type'])[0][0],
            'indoor_outdoor_mode': stats.mode(participant_df['indoors'])[0][0],
            'digital_use_mode': 'Yes' if stats.mode(participant_df['isapp'])[0][0] == 1 else 'No'
        }

        return episodes_df, all_indices, modes

    except Exception as e:
        print(f"Error processing participant: {str(e)}")
        return None, None, None
    
# Analyze all participants
print("\nStarting analysis of all participants...")
all_participants = tlv_df['user'].unique()
all_results = []

total_participants = len(all_participants)
for i, participant_id in enumerate(all_participants, 1):
    print(f"\nProcessing participant {participant_id} ({i}/{total_participants})")
    start_time = time.time()
    participant_df = tlv_df[tlv_df['user'] == participant_id]
    
    episodes_df, fragmentation_indices, modes = analyze_participant(participant_df)
    
    if episodes_df is not None and fragmentation_indices is not None and modes is not None:
        result = {'participant_id': participant_id, **fragmentation_indices, **modes}
        all_results.append(result)
        
        # Save episode details
        episodes_df.to_csv(os.path.join(episode_dir, f'participant_{participant_id}_episodes.csv'), index=False)
    
    end_time = time.time()
    print(f"Participant {participant_id} processed in {end_time - start_time:.2f} seconds")

# Create summary DataFrame
print("\nCreating summary DataFrame...")
summary_df = pd.DataFrame(all_results)

if not summary_df.empty:
    summary_df.to_csv(os.path.join(output_dir, 'fragmentation_summary.csv'), index=False)
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