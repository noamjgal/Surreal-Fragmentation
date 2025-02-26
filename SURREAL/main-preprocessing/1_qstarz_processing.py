import pandas as pd
import os
import glob
import numpy as np
from scipy import stats
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='qstarz_processing.log',
    filemode='w'
)

# file name of directory of participant data on your computer
directory = "/Volumes/Extreme SSD/SURREAL-DataBackup/HUJI_data-main/raw/Participants"
# file name of processed data on your computer
processed = "/Volumes/Extreme SSD/SURREAL-DataBackup/HUJI_data-main/processed/qstarz" 
  
# Create processed directory if it doesn't exist
os.makedirs(processed, exist_ok=True)

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
    df[' UTC DATE TIME'] = pd.to_datetime(df[' UTC DATE TIME'], errors='coerce')
    
    # Drop rows with invalid datetime
    invalid_dates = df[' UTC DATE TIME'].isna()
    if invalid_dates.any():
        logging.warning(f"Dropped {invalid_dates.sum()} rows with invalid date/time")
        df = df[~invalid_dates]
    
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
    # Only calculate if enough data points and no constant values
    lat_outliers = pd.Series(False, index=df.index)
    lon_outliers = pd.Series(False, index=df.index)
    
    if len(df) > 5 and df[' LATITUDE'].std() > 0:
        lat_outliers = np.abs(stats.zscore(df[' LATITUDE'])) > std_dev_threshold
    
    if len(df) > 5 and df[' LONGITUDE'].std() > 0:
        lon_outliers = np.abs(stats.zscore(df[' LONGITUDE'])) > std_dev_threshold
    
    # Combine outlier detection methods
    df['is_outlier'] = speed_outliers | lat_outliers | lon_outliers
    
    return df

for participant in os.listdir(directory): # loop through all the participant folders
    # skips file that start with a . in order to avoid the hidden files such as .DS_Store
    if participant.startswith('.'):
        continue
    
    try:
        # Extract participant ID
        pID = participant.split("_")[1]
        logging.info(f"Processing participant {pID}")
        
        # Construct glob pattern and search for matching files
        glob_pattern = directory + "/*" + pID + "/1 - QStarz BT-Q1000XT/*educed*.csv"
        file = glob.glob(glob_pattern, recursive=True)
        
        if not file:
            error_msg = f"No files found matching pattern: {glob_pattern}"
            error.append(f"{pID}: {error_msg}")
            logging.error(error_msg)
            continue
        
        # Show which file we're processing
        logging.info(f"Found file: {file[0]}")
        print(f"Processing {pID}: {file[0]}")
        
        # Read CSV file
        try:
            df = pd.read_csv(file[0])
            logging.info(f"Successfully read CSV with {len(df)} rows")
        except Exception as e:
            error_msg = f"Error reading CSV: {str(e)}"
            error.append(f"{pID}: {error_msg}")
            logging.error(error_msg)
            continue
            
        # Check for expected columns
        expected_cols = [" UTC DATE", " UTC TIME", " LOCAL DATE", " LOCAL TIME", " LATITUDE", " LONGITUDE"]
        missing_cols = [col for col in expected_cols if col not in df.columns]
        
        if missing_cols:
            error_msg = f"Missing expected columns: {missing_cols}. Available columns: {df.columns.tolist()}"
            error.append(f"{pID}: {error_msg}")
            logging.error(error_msg)
            continue
            
        # Combine time and date columns
        try:
            df[" UTC DATE TIME"] = df[" UTC DATE"] + df[" UTC TIME"]
            df[" LOCAL DATE TIME"] = df[" LOCAL DATE"] + df[" LOCAL TIME"]
        
            # add column with participant id
            df.insert(0, "Participant ID", participant)
        
            # remove redundant columns
            del df[" UTC DATE"]
            del df[" UTC TIME"]
            del df[" LOCAL DATE"]
            del df[" LOCAL TIME"]
        except Exception as e:
            error_msg = f"Error processing date/time columns: {str(e)}"
            error.append(f"{pID}: {error_msg}")
            logging.error(error_msg)
            continue
    
        # Detect outliers
        try:
            df = detect_outliers(df)
            logging.info(f"Outlier detection complete. Found {df['is_outlier'].sum()} outliers")
            
            # Remove outliers
            df_clean = df[~df['is_outlier']]
            
            # Save the cleaned data
            clean_file = processed + "/"+ pID + "_1_Qstarz_processed" + ".csv"
            df_clean.to_csv(clean_file, index=False)
            logging.info(f"Saved clean data to {clean_file}")
            
            # Saves a separate file with only the outliers for review
            outlier_file = processed + "/"+ pID + "_1_Qstarz_outliers" + ".csv"
            df_outliers = df[df['is_outlier']]
            df_outliers.to_csv(outlier_file, index=False)
            logging.info(f"Saved outliers to {outlier_file}")
            
        except Exception as e:
            error_msg = f"Error in outlier detection or saving: {str(e)}"
            error.append(f"{pID}: {error_msg}")
            logging.error(error_msg)
            continue
            
    except Exception as e:
        if 'pID' in locals():
            error.append(f"{pID}: {str(e)}")
            logging.error(f"Error processing participant {pID}: {str(e)}")
        else:
            error.append(f"{participant}: {str(e)}")
            logging.error(f"Error processing participant {participant}: {str(e)}")
        
print("\nProcessing complete. There were errors with the following participants:")
for err in error:
    print(f"- {err}")

# Save error log to file
with open(processed + "/error_log.txt", "w") as f:
    f.write("Errors encountered during processing:\n")
    for err in error:
        f.write(f"- {err}\n")
