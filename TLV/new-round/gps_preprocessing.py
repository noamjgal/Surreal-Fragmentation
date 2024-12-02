import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

# Define paths
RAW_GPS_PATH = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/Survey/gpsappS_9.1_excel.xlsx'
OUTPUT_DIR = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/preprocessed_summaries'
PARTICIPANT_INFO_PATH = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/participant_info.csv'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_gps_summary():
    """
    Create a daily summary of GPS data that includes demographic information
    and enhanced quality metrics
    """
    print("Loading raw GPS data...")
    raw_gps = pd.read_excel(RAW_GPS_PATH, sheet_name='gpsappS_8')
    
    # Load participant info for demographics
    participant_info = pd.read_csv(PARTICIPANT_INFO_PATH)
    
    # Convert dates and ensure consistent types
    raw_gps['date'] = pd.to_datetime(raw_gps['date']).dt.date
    raw_gps['user'] = raw_gps['user'].astype(str)
    raw_gps['Timestamp'] = pd.to_datetime(raw_gps['Timestamp'])
    participant_info['user'] = participant_info['user'].astype(str)
    
    # Merge demographic information
    raw_gps = pd.merge(
        raw_gps,
        participant_info[['user', 'school_n', 'sex']],
        on='user',
        how='left'
    )
    
    print("Creating daily summaries...")
    daily_summaries = []
    
    # Group by user and date
    for (user, date), day_data in tqdm(raw_gps.groupby(['user', 'date'])):
        # Get demographic info for this user
        user_demo = day_data[['school_n', 'sex']].iloc[0]
        
        day_summary = {
            'user': user,
            'date': date,
            'school_type': user_demo['school_n'],
            'gender': user_demo['sex'],
            'first_reading': day_data['Timestamp'].min(),
            'last_reading': day_data['Timestamp'].max(),
            'total_readings': len(day_data),
            'has_morning_data': any(day_data['Timestamp'].dt.hour < 12),
            'has_evening_data': any(day_data['Timestamp'].dt.hour >= 17),
            'max_gap_minutes': day_data['Timestamp'].diff().max().total_seconds() / 60 if len(day_data) > 1 else np.nan,
            'coverage_hours': (day_data['Timestamp'].max() - day_data['Timestamp'].min()).total_seconds() / 3600
        }
        daily_summaries.append(day_summary)
    
    gps_summary_df = pd.DataFrame(daily_summaries)
    
    # Add quality metrics
    gps_summary_df['data_quality'] = np.where(
        (gps_summary_df['has_morning_data']) & 
        (gps_summary_df['has_evening_data']) & 
        (gps_summary_df['coverage_hours'] >= 8),
        'good',
        'partial'
    )
    
    # Save the summary with demographic breakdowns
    summary_path = os.path.join(OUTPUT_DIR, 'gps_daily_summary.csv')
    gps_summary_df.to_csv(summary_path, index=False)
    
    # Create demographic summary
    demo_summary = gps_summary_df.groupby(['school_type', 'gender']).agg({
        'user': 'nunique',
        'date': 'count',
        'data_quality': lambda x: (x == 'good').mean()
    }).round(3)
    demo_summary.columns = ['unique_participants', 'total_days', 'good_quality_ratio']
    demo_summary.to_csv(os.path.join(OUTPUT_DIR, 'demographic_summary.csv'))
    
    print(f"GPS summary saved to: {summary_path}")
    return gps_summary_df

if __name__ == "__main__":
    gps_summary_df = create_gps_summary() 