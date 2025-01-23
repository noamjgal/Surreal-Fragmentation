#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 16:05:29 2024

@author: noamgal
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import os
import glob




# defines function to detect MOS clusters using DBSCAN
def detect_mos_clusters_dbscan(df, time_col='time_iso', mos_col='detectedMOS', eps=5, min_samples=3):
    """
    Detect clusters of moments of stress (MOS) in the given dataframe using DBSCAN.
    
    Args:
        df (pandas.DataFrame): The input dataframe.
        time_col (str): The name of the column containing the timestamp.
        mos_col (str): The name of the column containing the MOS indicator.
        eps (int): Max number of minutes between two samples for them to be considered as in the same neighborhood.
        Note that no other clusters can form under within the neighborhood of a core point.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        
    Returns:
        pandas.DataFrame: A dataframe containing the start and end times of each detected cluster along with the number of observations, number of stress observations, number of non-stress observations, and percentage of real stress observations in each cluster.
    """
    # Convert the time column to datetime
    df[time_col] = pd.to_datetime(df[time_col])
    
    # Convert the time column to a numeric representation (minutes since the start of the study)
    df['time_numeric'] = (df[time_col] - df[time_col].min()).dt.total_seconds() // 60
    
    # Filter only the MOS entries where detectedMOS is 1
    df_mos = df[df[mos_col] == 1].copy()
    
    # Apply DBSCAN clustering to positive MOS observations
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(df_mos[['time_numeric']])
    
    # Add cluster labels to the dataframe
    df_mos.loc[:, 'cluster'] = clustering.labels_
    
    # Identify clusters (excluding noise)
    clusters = df_mos[df_mos['cluster'] != -1].groupby('cluster')
    
    # Prepare the results by adding the start and end times for each cluster to a dataframe
    cluster_df = []
    for cluster_id, cluster_data in clusters:
        start_time = cluster_data[time_col].min()
        end_time = cluster_data[time_col].max()
        cluster_df.append({'start_time': start_time, 'end_time': end_time, 'cluster_id': cluster_id})
    # Convert to dataframe
    cluster_df = pd.DataFrame(cluster_df)
    
    # Include the MOS=0 entries that fall within the clusters' start and end times
    def assign_to_cluster(row):
        for index, cluster in cluster_df.iterrows():
            if cluster['start_time'] <= row[time_col] <= cluster['end_time']:
                return cluster['cluster_id']
        return -1  # Noise   
    df['cluster'] = df.apply(assign_to_cluster, axis=1)
    
    # Calculate the number of observations, stress observations, and non-stress observations in each cluster
    for cluster_id in cluster_df['cluster_id']:
        num_observations = df[df['cluster'] == cluster_id].shape[0]
        num_stress = df[(df['cluster'] == cluster_id) & (df[mos_col] == 1)].shape[0]
        num_non_stress = num_observations - num_stress
        percentage_real_stress = (num_stress / num_observations) * 100
        cluster_df.loc[cluster_df['cluster_id'] == cluster_id, 'num_observations'] = num_observations
        cluster_df.loc[cluster_df['cluster_id'] == cluster_id, 'num_stress_observations'] = num_stress
        cluster_df.loc[cluster_df['cluster_id'] == cluster_id, 'num_non_stress_observations'] = num_non_stress
        cluster_df.loc[cluster_df['cluster_id'] == cluster_id, 'percentage_real_stress'] = percentage_real_stress

    return cluster_df

# runs function for every stress file in the input folder
# adds output for each participant to an outputs file

# file name of directory of participant data on your computer
directory = "/Users/noamgal/Downloads/Research-Projects/SURREAL/HUJI_data-main/Stress-Inputs"
# file name of outputs file for cluster-data on your computer
cluster_file = "/Users/noamgal/Downloads/Research-Projects/SURREAL/HUJI_data-main/Stress-Outputs"  
# error list
error = []


for participant in os.listdir(directory): # loop through all the participant folders
    # skips file that start with a . in order to avoid the hidden files such as .DS_Store
    if participant.startswith('.'):
        continue 
    pID = participant.split("_")[0] 

    try:
        print(pID)
        # access data file
        file = glob.glob(directory + "/" + pID + '*', recursive = True)
        path = str(file[0])
        print(path)
        df = pd.read_csv(path)
        print('read')
        
        # Call clustering function with default parameters and print output
        cluster_df = detect_mos_clusters_dbscan(df, time_col='time_iso', mos_col='detectedMOS', eps=5, min_samples=3)
        
        # Print the percentage of all observations that are in a cluster
        total_observations = df.shape[0]
        observations_in_clusters = cluster_df['num_observations'].sum()
        percentage_in_clusters = (observations_in_clusters / total_observations) * 100
        print(f"Percentage of all observations that are in a cluster: {percentage_in_clusters:.2f}%")

        # Print the percentage of all MOS positive observations in a cluster
        stress_in = 100*(np.sum(cluster_df['num_stress_observations'])/df[df['detectedMOS'] == 1].shape[0])                            
        print(f"Percentage of MOS observations in a cluster: {stress_in:.2f}%")
        cluster_df.to_csv(f'{cluster_file}/{pID}MOS_clusters')
        print(f'success for pID: {pID}')
    except Exception as e:
        print(f'error with {e}')    
        error.append(pID)    

print("There were errors with the following participants:")
print(error)


                                                  
