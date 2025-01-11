import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

# Global counters for tracking fragmentation calculation statistics
fragmentation_stats = {
    'digital': {'success': 0, 'insufficient_episodes': 0, 'zero_duration': 0, 'failed_days': []},
    'moving': {'success': 0, 'insufficient_episodes': 0, 'zero_duration': 0, 'failed_days': []},
    'digital_during_mobility': {'success': 0, 'insufficient_episodes': 0, 'zero_duration': 0, 'failed_days': []}
}

def extract_info_from_filename(filename):
    """Extract participant ID and date from filename."""
    parts = filename.split('_')
    date_str = parts[-1].split('.')[0]  # Remove the .csv extension
    participant_id = parts[-2]
    date = datetime.strptime(date_str, "%Y-%m-%d").date()
    return participant_id, date

def calculate_mobility_metrics(df):
    """Calculate basic mobility metrics from episodes dataframe."""
    total_duration = df['duration'].sum()
    avg_duration = df['duration'].mean()
    episode_count = len(df)
    return total_duration, avg_duration, episode_count

def calculate_aid(episodes_df):
    """Calculate Average Inter-episode Duration (AID) and related statistics.
    Returns AID (mean) and additional statistics about the inter-episode durations."""
    if len(episodes_df) > 1:
        inter_episode_durations = np.abs((episodes_df['start_time'].iloc[1:] - episodes_df['end_time'].iloc[:-1]).dt.total_seconds() / 60)
        
        if len(inter_episode_durations) > 0:
            aid = np.mean(inter_episode_durations)  # This is the AID
            # Additional statistics about inter-episode durations
            duration_stats = {
                'inter_episode_std': np.std(inter_episode_durations),
                'inter_episode_min': np.min(inter_episode_durations),
                'inter_episode_max': np.max(inter_episode_durations)
            }
        else:
            aid = np.nan
            duration_stats = {
                'inter_episode_std': np.nan,
                'inter_episode_min': np.nan,
                'inter_episode_max': np.nan
            }
    else:
        aid = np.nan
        duration_stats = {
            'inter_episode_std': np.nan,
            'inter_episode_min': np.nan,
            'inter_episode_max': np.nan,
        }
    
    return aid, duration_stats

def calculate_fragmentation_index(episodes_df, frag_type, min_episodes=3, date=None, participant_id=None):
    """
    Calculate fragmentation index with improved logging.
    
    Args:
        episodes_df: DataFrame containing episode data
        frag_type: String indicating type of fragmentation ('digital', 'moving', or 'digital_during_mobility')
        min_episodes: Minimum number of episodes required
    """
    S = len(episodes_df)
    T = episodes_df['duration'].sum()
    
    if S < min_episodes:
        print(f"{frag_type.capitalize()} fragmentation: Insufficient episodes ({S} < {min_episodes})")
        fragmentation_stats[frag_type]['insufficient_episodes'] += 1
        if date and participant_id:
            fragmentation_stats[frag_type]['failed_days'].append({
                'date': date,
                'participant_id': participant_id,
                'reason': 'insufficient_episodes',
                'episodes': S,
                'total_duration': T
            })
        return np.nan
    
    if T <= 0:
        print(f"{frag_type.capitalize()} fragmentation: Zero total duration")
        fragmentation_stats[frag_type]['zero_duration'] += 1
        if date and participant_id:
            fragmentation_stats[frag_type]['failed_days'].append({
                'date': date,
                'participant_id': participant_id,
                'reason': 'zero_duration',
                'episodes': S,
                'total_duration': T
            })
        return np.nan
        
    normalized_durations = episodes_df['duration'] / T
    sum_squared = sum(normalized_durations ** 2)
    index = (1 - sum_squared) / (1 - (1 / S))
    
    if index > 0.9999:
        print(f"{frag_type.capitalize()} fragmentation warning: Very high index ({index:.4f})")
        print(f"  Number of episodes: {S}")
        print(f"  Total duration: {T:.2f}")
        print(f"  Normalized durations: {[f'{d:.4f}' for d in normalized_durations]}")
        print(f"  Sum of squared normalized durations: {sum_squared:.4f}")
    
    fragmentation_stats[frag_type]['success'] += 1
    return index

def calculate_digital_frag_during_mobility(digital_df, moving_df):
    """Calculate fragmentation of digital use during mobility periods with improved logging."""
    if digital_df.empty or moving_df.empty:
        print("Digital during mobility fragmentation: Empty dataframes")
        fragmentation_stats['digital_during_mobility']['insufficient_episodes'] += 1
        return np.nan
        
    mobility_periods = []
    for _, move in moving_df.iterrows():
        mobility_periods.append((move['start_time'], move['end_time']))
    
    mobile_digital_episodes = []
    for _, digital in digital_df.iterrows():
        digital_start = digital['start_time']
        digital_end = digital['end_time']
        
        for mob_start, mob_end in mobility_periods:
            if (digital_start < mob_end) and (digital_end > mob_start):
                episode_end = max(digital_end, mob_end)
                duration = (episode_end - digital_start).total_seconds() / 60
                
                mobile_digital_episodes.append({
                    'start_time': digital_start,
                    'end_time': episode_end,
                    'duration': duration
                })
                break
    
    if mobile_digital_episodes:
        episodes_df = pd.DataFrame(mobile_digital_episodes)
        return calculate_fragmentation_index(episodes_df, 'digital_during_mobility')
    
    print("Digital during mobility fragmentation: No overlapping episodes found")
    fragmentation_stats['digital_during_mobility']['insufficient_episodes'] += 1
    return np.nan

def print_summary_statistics(df):
    """Print summary statistics and create visualization histograms."""
    print("\nSummary Statistics:")
    print(f"Total participants: {df['participant_id'].nunique()}")
    print(f"Total days: {len(df)}")
    
    for col in df.columns:
        if col not in ['participant_id', 'date']:
            print(f"Average {col}: {df[col].mean():.4f}")

    # Create histograms
    for col in ['digital_fragmentation_index', 'moving_fragmentation_index', 'digital_frag_during_mobility']:
        plt.figure(figsize=(10, 6))
        plt.hist(df[col].dropna(), bins=50)
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.savefig(f'{col}_histogram.png')
        plt.close()

def analyze_failed_days():
    """Analyze and summarize statistics about failed calculations."""
    print("\nDetailed Analysis of Failed Days:")
    print("-" * 50)
    
    for frag_type in fragmentation_stats:
        failed_days = fragmentation_stats[frag_type]['failed_days']
        if not failed_days:
            continue
            
        print(f"\n{frag_type.replace('_', ' ').title()} Failures:")
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(failed_days)
        
        # Analyze by participant
        participant_counts = df['participant_id'].value_counts()
        print("\nParticipants with failures:")
        print(f"Total participants with failures: {len(participant_counts)}")
        print("\nTop 5 participants by number of failures:")
        for pid, count in participant_counts.head().items():
            print(f"  Participant {pid}: {count} failed days")
            
        # Analyze by reason
        reason_counts = df['reason'].value_counts()
        print("\nFailure reasons:")
        for reason, count in reason_counts.items():
            print(f"  {reason}: {count} days")
            
        # Statistics for failed days
        if 'episodes' in df.columns:
            print("\nEpisode statistics for failed days:")
            print(f"  Average episodes: {df['episodes'].mean():.2f}")
            print(f"  Min episodes: {df['episodes'].min()}")
            print(f"  Max episodes: {df['episodes'].max()}")
        
        if 'total_duration' in df.columns:
            print("\nDuration statistics for failed days:")
            print(f"  Average duration: {df['total_duration'].mean():.2f} minutes")
            print(f"  Min duration: {df['total_duration'].min():.2f} minutes")
            print(f"  Max duration: {df['total_duration'].max():.2f} minutes")

def print_fragmentation_summary():
    """Print summary of fragmentation calculation statistics."""
    print("\nFragmentation Calculation Summary:")
    print("-" * 50)
    
    for frag_type in fragmentation_stats:
        stats = fragmentation_stats[frag_type]
        # Only sum the numeric statistics, excluding the failed_days list
        total = stats['success'] + stats['insufficient_episodes'] + stats['zero_duration']
        
        print(f"\n{frag_type.replace('_', ' ').title()} Fragmentation:")
        print(f"  Total calculations attempted: {total}")
        print(f"  Successful calculations: {stats['success']} ({stats['success']/total*100:.1f}%)")
        print(f"  Failed due to insufficient episodes: {stats['insufficient_episodes']} ({stats['insufficient_episodes']/total*100:.1f}%)")
        print(f"  Failed due to zero duration: {stats['zero_duration']} ({stats['zero_duration']/total*100:.1f}%)")

def process_episode_summary(digital_file_path, moving_file_path, print_sample=False):
    """Process episode data and calculate fragmentation metrics."""
    try:
        digital_df = pd.read_csv(digital_file_path)
        moving_df = pd.read_csv(moving_file_path)
        
        for df in [digital_df, moving_df]:
            df['start_time'] = pd.to_datetime(df['start_time'])
            df['end_time'] = pd.to_datetime(df['end_time'])
        
        total_time = (digital_df['end_time'] - digital_df['start_time']).dt.total_seconds().sum() / 60

        if print_sample:
            print(f"\nSample data for {os.path.basename(digital_file_path)}:")
            print(digital_df.head())
            print(f"\nSample data for {os.path.basename(moving_file_path)}:")
            print(moving_df.head())

        participant_id, date = extract_info_from_filename(os.path.basename(digital_file_path))

        digital_frag_index = calculate_fragmentation_index(digital_df, 'digital', date=date, participant_id=participant_id)
        moving_frag_index = calculate_fragmentation_index(moving_df, 'moving', date=date, participant_id=participant_id)
        digital_frag_during_mobility = calculate_digital_frag_during_mobility(digital_df, moving_df)

        total_duration_mobility, avg_duration_mobility, count_mobility = calculate_mobility_metrics(moving_df)

        result = {
            'participant_id': participant_id,
            'date': date,
            'total_time_on_device': total_time,
            'digital_fragmentation_index': digital_frag_index,
            'moving_fragmentation_index': moving_frag_index,
            'digital_frag_during_mobility': digital_frag_during_mobility,
            'total_duration_mobility': total_duration_mobility,
            'avg_duration_mobility': avg_duration_mobility,
            'count_mobility': count_mobility
        }

        for episode_type, df in [('digital', digital_df), ('moving', moving_df)]:
            aid, duration_stats = calculate_aid(df)
            result[f'{episode_type}_AID'] = aid
            for stat_name, stat_value in duration_stats.items():
                result[f'{episode_type}_{stat_name}'] = stat_value

        return pd.DataFrame([result])
    except Exception as e:
        print(f"Error processing files {digital_file_path} and {moving_file_path}: {str(e)}")
        return None

def main(input_dir, output_dir):
    """Main function to process all episode files and generate summary."""
    digital_files = sorted([f for f in os.listdir(input_dir) if f.startswith('digital_episodes_') and f.endswith('.csv')])
    moving_files = sorted([f for f in os.listdir(input_dir) if f.startswith('moving_episodes_') and f.endswith('.csv')])
    
    if len(digital_files) != len(moving_files):
        print("Warning: Mismatch in the number of digital and moving episode files.")
    
    all_results = []
    for i, (digital_file, moving_file) in enumerate(tqdm(zip(digital_files, moving_files), desc="Processing episodes")):
        digital_path = os.path.join(input_dir, digital_file)
        moving_path = os.path.join(input_dir, moving_file)
        results = process_episode_summary(digital_path, moving_path, print_sample=(i==0))
        if results is not None:
            all_results.append(results)
    
    combined_results = pd.concat(all_results, ignore_index=True)
    output_file = os.path.join(output_dir, 'fragmentation_summary.csv')
    combined_results.to_csv(output_file, index=False)
    print(f"\nSaved fragmentation summary to {output_file}")
    
    print_summary_statistics(combined_results)
    print_fragmentation_summary()
    analyze_failed_days()

if __name__ == "__main__":
    input_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/episodes'
    output_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/fragmentation'
    main(input_dir, output_dir)