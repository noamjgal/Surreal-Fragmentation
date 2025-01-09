import pandas as pd
from datetime import timedelta

# Define paths
GPS_SUMMARY_PATH = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/preprocessed_summaries/gps_daily_summary.csv'
SURVEY_PATH = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/Survey/End_of_the_day_questionnaire.xlsx'
PARTICIPANT_INFO_PATH = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/participant_info.csv'
OUTPUT_DIR = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/data_coverage'

def process_and_match_data():
    # Load data
    gps_summary = pd.read_csv(GPS_SUMMARY_PATH)
    survey = pd.read_excel(SURVEY_PATH)
    participant_info = pd.read_csv(PARTICIPANT_INFO_PATH)
    
    # Convert dates with explicit format handling
    gps_summary['date'] = pd.to_datetime(gps_summary['date']).dt.strftime('%Y-%m-%d')
    survey['timestamp'] = pd.to_datetime(survey['StartDate'])
    
    # Process survey dates
    survey['hour'] = survey['timestamp'].dt.hour
    survey['date'] = survey['timestamp'].dt.strftime('%Y-%m-%d')
    survey['adjusted_date'] = survey.apply(
        lambda row: (row['timestamp'] - timedelta(days=1)).strftime('%Y-%m-%d')
        if row['hour'] <= 7 
        else row['date'],
        axis=1
    )
    
    # Add debugging for adjusted dates
    print("\nAdjusted Dates Summary:")
    adjusted_records = survey[survey['date'] != survey['adjusted_date']]
    print(f"\nNumber of adjusted records: {len(adjusted_records)}")
    print("\nSample of adjusted records:")
    print(adjusted_records[['timestamp', 'hour', 'date', 'adjusted_date']].head())
    
    # Original debugging
    print("\nDate Format Verification:")
    print("GPS date example:", gps_summary['date'].iloc[0], "type:", type(gps_summary['date'].iloc[0]))
    print("Survey adjusted date example:", survey['adjusted_date'].iloc[0], "type:", type(survey['adjusted_date'].iloc[0]))

    # Ensure consistent ID types
    gps_summary['user'] = gps_summary['user'].astype(str)
    survey['Participant_ID'] = survey['Participant_ID'].astype(str)
    participant_info['user'] = participant_info['user'].astype(str)
    
    # Merge using string dates
    merged = pd.merge(
        gps_summary,
        survey,
        left_on=['user', 'date'],
        right_on=['Participant_ID', 'adjusted_date'],
        how='outer',
        indicator=True
    )
    
    # Add demographic information
    final_data = pd.merge(
        merged,
        participant_info[['user', 'school_n', 'sex']],
        left_on='user',
        right_on='user',
        how='left'
    )
    
    # Basic statistics
    stats = {
        'total_gps_days': len(gps_summary),
        'total_survey_days': len(survey),
        'matched_days': len(merged[merged['_merge'] == 'both']),
        'gps_only_days': len(merged[merged['_merge'] == 'left_only']),
        'survey_only_days': len(merged[merged['_merge'] == 'right_only'])
    }
    
    return final_data, stats

if __name__ == "__main__":
    matched_data, matching_stats = process_and_match_data()
    print("\nMatching Statistics:")
    for key, value in matching_stats.items():
        print(f"{key}: {value}")