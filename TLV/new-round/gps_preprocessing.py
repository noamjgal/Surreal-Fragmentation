import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

# Define paths
RAW_GPS_PATH = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/Survey/gpsappS_9.1_excel.xlsx'
OUTPUT_DIR = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/preprocessed_summaries'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_gps_summary():
    """
    Create a daily summary of GPS data that includes:
    - First reading of the day
    - Last reading of the day
    - Number of readings
    - Time gaps in data
    - Basic quality metrics
    """
    print("Loading raw GPS data...")
    raw_gps = pd.read_excel(RAW_GPS_PATH, sheet_name='gpsappS_8')
    
    # Convert dates and ensure consistent types
    raw_gps['date'] = pd.to_datetime(raw_gps['date']).dt.date
    raw_gps['user'] = raw_gps['user'].astype(str)
    raw_gps['Timestamp'] = pd.to_datetime(raw_gps['Timestamp'])
    
    print("Creating daily summaries...")
    daily_summaries = []
    
    # Group by user and date
    for (user, date), day_data in tqdm(raw_gps.groupby(['user', 'date'])):
        day_summary = {
            'user': user,
            'date': date,
            'first_reading': day_data['Timestamp'].min(),
            'last_reading': day_data['Timestamp'].max(),
            'total_readings': len(day_data),
            'has_morning_data': any(day_data['Timestamp'].dt.hour < 12),
            'has_evening_data': any(day_data['Timestamp'].dt.hour >= 17),
            'max_gap_minutes': day_data['Timestamp'].diff().max().total_seconds() / 60 if len(day_data) > 1 else np.nan
        }
        daily_summaries.append(day_summary)
    
    gps_summary_df = pd.DataFrame(daily_summaries)
    
    # Save the summary
    summary_path = os.path.join(OUTPUT_DIR, 'gps_daily_summary.csv')
    gps_summary_df.to_csv(summary_path, index=False)
    print(f"GPS summary saved to: {summary_path}")
    
    return gps_summary_df

if __name__ == "__main__":
    gps_summary_df = create_gps_summary() 