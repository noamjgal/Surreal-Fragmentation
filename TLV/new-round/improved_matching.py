import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths
GPS_SUMMARY_PATH = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/preprocessed_summaries/gps_daily_summary.csv'
SURVEY_PATH = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/Survey/End_of_the_day_questionnaire.xlsx'
PARTICIPANT_INFO_PATH = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/participant_info.csv'
OUTPUT_DIR = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/data_coverage'

def analyze_matching():
    # Load data
    print("Loading data...")
    gps_summary = pd.read_csv(GPS_SUMMARY_PATH)
    survey = pd.read_excel(SURVEY_PATH)
    participant_info = pd.read_csv(PARTICIPANT_INFO_PATH)
    
    # Convert dates
    gps_summary['date'] = pd.to_datetime(gps_summary['date']).dt.date
    survey['date'] = pd.to_datetime(survey['StartDate']).dt.date
    
    # Ensure consistent ID types
    gps_summary['user'] = gps_summary['user'].astype(str)
    survey['Participant_ID'] = survey['Participant_ID'].astype(str)
    participant_info['user'] = participant_info['user'].astype(str)
    
    # Initial merge
    merged = pd.merge(
        gps_summary,
        survey,
        left_on=['user', 'date'],
        right_on=['Participant_ID', 'date'],
        how='outer',
        indicator=True
    )
    
    # Analyze unmatched survey days
    unmatched_survey = merged[merged['_merge'] == 'right_only']
    
    print("\nAnalyzing unmatched survey days...")
    for idx, row in unmatched_survey.iterrows():
        # Check for GPS data within ±1 day
        nearby_gps = gps_summary[
            (gps_summary['user'] == row['Participant_ID']) & 
            (abs((gps_summary['date'] - row['date']).dt.days) <= 1)
        ]
        
        if not nearby_gps.empty:
            print(f"\nParticipant {row['Participant_ID']} on {row['date']}:")
            print("Nearby GPS data:")
            print(nearby_gps[['date', 'total_readings', 'has_morning_data', 'has_evening_data']])
    
    # Add demographic information
    merged_with_demo = pd.merge(
        merged,
        participant_info[['user', 'school_n', 'sex']],
        left_on='user',
        right_on='user',
        how='left'
    )
    
    print("\nCoverage Analysis:")
    print(f"Total GPS days: {len(gps_summary)}")
    print(f"Total survey days: {len(survey)}")
    print(f"Matched days: {len(merged[merged['_merge'] == 'both'])}")
    print(f"GPS only days: {len(merged[merged['_merge'] == 'left_only'])}")
    print(f"Survey only days: {len(merged[merged['_merge'] == 'right_only'])}")
    
    # Analyze data quality of matches
    matches = merged[merged['_merge'] == 'both']
    print("\nQuality of matched days:")
    print("Morning data available:", matches['has_morning_data'].mean() * 100, "%")
    print("Evening data available:", matches['has_evening_data'].mean() * 100, "%")
    print("Average readings per day:", matches['total_readings'].mean())
    
    # Save detailed results
    output_path = os.path.join(OUTPUT_DIR, 'detailed_matching_analysis.csv')
    merged_with_demo.to_csv(output_path, index=False)
    
    return merged_with_demo

if __name__ == "__main__":
    results = analyze_matching() 