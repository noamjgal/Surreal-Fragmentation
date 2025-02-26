import pandas as pd
import os
import glob
import numpy as np
from scipy import stats

# file name of directory of participant data on your computer
directory = "participant-file-loc"
# file name of processed data on your computer
processed = "processed-target-file-loc" 
  
error = []


def calculate_distance(lat1, lon1, lat2, lon2):
    return ((lat1 - lat2)**2 + (lon1 - lon2)**2)**0.5 * 111000  # Approximate meters

def detect_outliers(df, speed_threshold=100, std_dev_threshold=5):
    """
    Detect outliers in GPS data.
    
    :param df: DataFrame containing 'LATITUDE', 'LONGITUDE', and 'UTC DATE TIME' columns
    :param speed_threshold: Maximum allowed speed in m/s (default 100 m/s, about 360 km/h)
    :param std_dev_threshold: Number of standard deviations for outlier detection
    :return: DataFrame with an additional 'is_outlier' column
    """
    
    # Convert UTC DATE TIME to datetime if it's not already
    df[' UTC DATE TIME'] = pd.to_datetime(df[' UTC DATE TIME'])
    
    # Sort by time
    df = df.sort_values(' UTC DATE TIME')
    
    # Calculate time difference
    df['time_diff'] = df[' UTC DATE TIME'].diff().dt.total_seconds()
    
    # Calculate distance
    df['distance'] = calculate_distance(df[' LATITUDE'], df[' LONGITUDE'], 
                                        df[' LATITUDE'].shift(), df[' LONGITUDE'].shift())
    
    # Calculate speed
    df['speed'] = df['distance'] / df['time_diff']
    
    # Method 1: Speed-based outlier detection
    speed_outliers = df['speed'] > speed_threshold
    
    # Method 2: Standard deviation-based outlier detection
    lat_outliers = np.abs(stats.zscore(df[' LATITUDE'])) > std_dev_threshold
    lon_outliers = np.abs(stats.zscore(df[' LONGITUDE'])) > std_dev_threshold
    
    # Combine outlier detection methods
    df['is_outlier'] = speed_outliers | lat_outliers | lon_outliers
    
    return df

for participant in os.listdir(directory): # loop through all the participant folders
    # skips file that start with a . in order to avoid the hidden files such as .DS_Store
    if participant.startswith('.'):
        continue
    
    try:
        pID = participant.split("_")[1]
        # access data file
        file = glob.glob(directory + "/*" + pID + "/1 - QStarz BT-Q1000XT/*educed*.csv", recursive = True)
        print(file)
        print(pID)
        df = pd.read_csv(file[0])
        # combine time and date columns
        df[" UTC DATE TIME"] = df[" UTC DATE"] + df[" UTC TIME"]
        df[" LOCAL DATE TIME"] = df[" LOCAL DATE"] + df[" LOCAL TIME"]
    
        # add column with participant id
        df.insert(0, "Participant ID", participant)
    
        # remove redundant columns
        del df[" UTC DATE"]
        del df[" UTC TIME"]
        del df[" LOCAL DATE"]
        del df[" LOCAL TIME"]
    
        # Detect outliers
        df = detect_outliers(df)
        
        # Remove outliers
        df_clean = df[~df['is_outlier']]
        
        # Save the cleaned data
        df_clean.to_csv(processed + "/"+ pID + "_1_Qstarz_processed" + ".csv", index=False)
        
        # Sves a separate file with only the outliers for review
        df_outliers = df[df['is_outlier']]
        df_outliers.to_csv(processed + "/"+ pID + "_1_Qstarz_outliers" + ".csv", index=False)
        
    except Exception as e:
        error.append(f"{pID}: {str(e)}")
        
print("There were errors with the following participants:")
print(error)
        
print("There were errors with the following participants:")
print(error)
