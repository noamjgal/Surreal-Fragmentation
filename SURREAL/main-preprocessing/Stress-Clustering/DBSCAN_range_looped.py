import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import os
import glob

def detect_mos_clusters_dbscan_range(df, time_col='time_iso', mos_score_col='MOS_Score', min_score=0.5, max_score=1.0, eps=5, min_samples=3):
    """
    Detect clusters of moments of stress (MOS) in the given dataframe using DBSCAN based on a range of MOS_Score values.
    
    Args:
        df (pandas.DataFrame): The input dataframe.
        time_col (str): The name of the column containing the timestamp.
        mos_score_col (str): The name of the column containing the MOS score.
        min_score (float): The minimum MOS score value to consider for clustering.
        max_score (float): The maximum MOS score value to consider for clustering.
        eps (int): Max number of minutes between two samples for them to be considered as in the same neighborhood.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        
    Returns:
        pandas.DataFrame: A dataframe containing the start and end times of each detected cluster along with various statistics.
    """
    # Convert the time column to datetime
    df[time_col] = pd.to_datetime(df[time_col])
    
    # Convert the time column to a numeric representation (minutes since the start of the study)
    df['time_numeric'] = (df[time_col] - df[time_col].min()).dt.total_seconds() // 60
    
    # Filter only the MOS entries where MOS_Score is within the specified range
    df_mos = df[(df[mos_score_col] >= min_score) & (df[mos_score_col] <= max_score)].copy()
    
    if df_mos.empty:
        print(f"No data points found in the MOS score range {min_score} to {max_score}")
        return pd.DataFrame()

    # Apply DBSCAN clustering to the filtered MOS observations
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(df_mos[['time_numeric']])
    
    # Add cluster labels to the dataframe
    df_mos['cluster'] = clustering.labels_
    
    # Identify clusters (excluding noise)
    clusters = df_mos[df_mos['cluster'] != -1].groupby('cluster')
    
    # Prepare the results
    cluster_df = []
    for cluster_id, cluster_data in clusters:
        start_time = cluster_data[time_col].min()
        end_time = cluster_data[time_col].max()
        
        # Get all data points within the cluster time range
        cluster_period = df[(df[time_col] >= start_time) & (df[time_col] <= end_time)]
        
        cluster_df.append({
            'cluster_id': f"stress_{cluster_id}",
            'start_time': start_time,
            'end_time': end_time,
            'num_observations': len(cluster_period),
            'num_stress_observations': len(cluster_period[(cluster_period[mos_score_col] >= min_score) & (cluster_period[mos_score_col] <= max_score)]),
            'num_non_stress_observations': len(cluster_period[(cluster_period[mos_score_col] < min_score) | (cluster_period[mos_score_col] > max_score)]),
            'percentage_stress': (len(cluster_period[(cluster_period[mos_score_col] >= min_score) & (cluster_period[mos_score_col] <= max_score)]) / len(cluster_period)) * 100
        })
    
    return pd.DataFrame(cluster_df)

def detect_cold_spots_filtered(df, time_col='time_iso', mos_score_col='MOS_Score', min_duration=5):
    """
    Detect periods of no stress (cold spots) in the given dataframe.
    
    Args:
        df (pandas.DataFrame): The input dataframe.
        time_col (str): The name of the column containing the timestamp.
        mos_score_col (str): The name of the column containing the MOS score.
        min_duration (int): The minimum duration (in minutes) for a period to be considered a cold spot.
        
    Returns:
        pandas.DataFrame: A dataframe containing the start and end times of each detected cold spot along with various statistics.
    """
    # Convert the time column to datetime and sort
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(by=time_col)
    
    # Create a mask for rows where MOS_Score is 0
    mask = df[mos_score_col] == 0
    
    # Create groups of consecutive 0s
    groups = mask.ne(mask.shift()).cumsum()[mask]
    
    # Group the data
    grouped = df[mask].groupby(groups)
    
    # Filter groups that are at least min_duration minutes long
    cold_spots = grouped.filter(lambda x: (x[time_col].max() - x[time_col].min()).total_seconds() >= min_duration * 60)
    
    # Create the result dataframe
    result = cold_spots.groupby(groups).agg({
        time_col: ['min', 'max'],
        mos_score_col: 'count'
    }).reset_index(drop=True)
    
    result.columns = ['start_time', 'end_time', 'num_observations']
    result['num_stress_observations'] = 0
    result['num_non_stress_observations'] = result['num_observations']
    result['percentage_stress'] = 0
    
    # Add cluster ID
    result['cluster_id'] = [f"no_stress_{i}" for i in range(len(result))]
    
    # Reorder columns
    result = result[['cluster_id', 'start_time', 'end_time', 'num_observations', 
                     'num_stress_observations', 'num_non_stress_observations', 'percentage_stress']]
    
    return result

def remove_overlaps_and_short_clusters(no_stress_df, stress_df, min_duration=5):
    """
    Remove overlapping periods from no_stress clusters and filter out short clusters.
    
    Args:
        no_stress_df (pandas.DataFrame): The dataframe containing no-stress clusters.
        stress_df (pandas.DataFrame): The dataframe containing stress clusters.
        min_duration (int): The minimum duration (in minutes) for a cluster to be retained.
        
    Returns:
        pandas.DataFrame: The updated no_stress dataframe with overlaps removed and short clusters filtered out.
    """
    # Convert start_time and end_time to datetime if they're not already
    for df in [no_stress_df, stress_df]:
        for col in ['start_time', 'end_time']:
            df[col] = pd.to_datetime(df[col])

    # Remove overlapping periods from no_stress clusters
    for _, stress_row in stress_df.iterrows():
        no_stress_df = no_stress_df[~((no_stress_df['start_time'] < stress_row['end_time']) & 
                                      (no_stress_df['end_time'] > stress_row['start_time']))]

    # Recalculate durations and remove short clusters
    no_stress_df['duration'] = (no_stress_df['end_time'] - no_stress_df['start_time']).dt.total_seconds() / 60
    no_stress_df = no_stress_df[no_stress_df['duration'] >= min_duration]

    return no_stress_df.drop('duration', axis=1)

def check_for_overlaps(no_stress_df, stress_df):
    """
    Check for any remaining overlaps between no-stress and stress clusters.
    
    Args:
        no_stress_df (pandas.DataFrame): The dataframe containing no-stress clusters.
        stress_df (pandas.DataFrame): The dataframe containing stress clusters.
        
    Returns:
        bool: True if no overlaps are found, False otherwise.
    """
    for _, stress_row in stress_df.iterrows():
        for _, no_stress_row in no_stress_df.iterrows():
            if (stress_row['start_time'] <= no_stress_row['end_time'] and 
                no_stress_row['start_time'] <= stress_row['end_time']):
                print(f"Overlap detected:")
                print(f"Stress cluster: {stress_row['start_time']} to {stress_row['end_time']}")
                print(f"No-stress cluster: {no_stress_row['start_time']} to {no_stress_row['end_time']}")
                return False
    return True

def report_percentage(df, cluster_df):
    """
    Report the percentage of all observations that are in a cluster.
    
    Args:
        df (pandas.DataFrame): The original dataframe.
        cluster_df (pandas.DataFrame): The dataframe containing cluster information.
    """
    total_observations = len(df)
    observations_in_clusters = cluster_df['num_observations'].sum()
    percentage_in_clusters = (observations_in_clusters / total_observations) * 100
    print(f"Percentage of all observations that are in a cluster: {percentage_in_clusters:.2f}%")

# Main execution
directory = "/Users/noamgal/Downloads/Research-Projects/SURREAL/HUJI_data-main/Stress-Inputs"
cluster_file = "/Users/noamgal/Downloads/Research-Projects/SURREAL/HUJI_data-main/Stress-Outputs"
error = []

for participant in os.listdir(directory):
    if participant.startswith('.'):
        continue 
    pID = participant.split("_")[0] 

    try:
        print(f"Processing participant: {pID}")
        file = glob.glob(os.path.join(directory, f"{pID}*"))
        if not file:
            raise FileNotFoundError(f"No file found for participant {pID}")
        path = file[0]
        print(f"File path: {path}")
        df = pd.read_csv(path)
        print('Data read successfully')
        df = df.head(100000)
        print(f"MOS_Score value counts:\n{df['MOS_Score'].value_counts()}")

        # Generate all clusters
        no_stress_cluster_df = detect_cold_spots_filtered(df)
        regular_stress_cluster_df = detect_mos_clusters_dbscan_range(df, min_score=0.5, max_score=1.0, eps=5, min_samples=3)
        high_stress_cluster_df = detect_mos_clusters_dbscan_range(df, min_score=1.5, max_score=2, eps=5, min_samples=3)

        # Combine stress clusters
        stress_cluster_df = pd.concat([regular_stress_cluster_df, high_stress_cluster_df], ignore_index=True)

        # Remove overlaps and short clusters
        no_stress_cluster_df = remove_overlaps_and_short_clusters(no_stress_cluster_df, stress_cluster_df)

        # Check for overlaps after processing
        if check_for_overlaps(no_stress_cluster_df, stress_cluster_df):
            print("No overlaps detected after processing.")
        else:
            print("WARNING: Overlaps still present after processing!")

        # Report percentages and save results
        report_percentage(df, no_stress_cluster_df)
        no_stress_cluster_df.to_csv(os.path.join(cluster_file, f'{pID}_no_stress_clusters.csv'), index=False)
        print(f'Success for no_stress filtering for pID: {pID}')

        report_percentage(df, regular_stress_cluster_df)
        regular_stress_cluster_df.to_csv(os.path.join(cluster_file, f'{pID}_regular_stress_clusters.csv'), index=False)
        print(f'Success for regular_stress for pID: {pID}')

        report_percentage(df, high_stress_cluster_df)
        high_stress_cluster_df.to_csv(os.path.join(cluster_file, f'{pID}_high_stress_clusters.csv'), index=False)
        print(f'Success for high_stress for pID: {pID}')

        print(f"Processing completed for participant: {pID}\n")

    except Exception as e:
        print(f'Error processing participant {pID}: {str(e)}')    
        error.append(pID)    

print("There were errors with the following participants:")
print(error)
