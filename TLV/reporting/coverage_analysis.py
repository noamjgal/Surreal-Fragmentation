import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths
RAW_GPS_PATH = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/Survey/gpsappS_9.1_excel.xlsx'
SURVEY_PATH = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/Survey/End_of_the_day_questionnaire.xlsx'
PARTICIPANT_INFO_PATH = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/participant_info.csv'
OUTPUT_DIR = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/data_coverage'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_data_coverage():
    # Load raw data
    print("Loading raw GPS data...")
    raw_gps = pd.read_excel(RAW_GPS_PATH, sheet_name='gpsappS_8')
    
    print("Loading survey data...")
    survey = pd.read_excel(SURVEY_PATH)
    
    print("Loading participant info...")
    participant_info = pd.read_csv(PARTICIPANT_INFO_PATH)
    
    # Print initial date formats
    print("\nInitial Date Formats:")
    print(f"GPS date type: {raw_gps['date'].dtype}")
    print(f"Survey StartDate type: {survey['StartDate'].dtype}")
    
    # Convert dates using the same approach as significance-combined.py
    raw_gps['date'] = pd.to_datetime(raw_gps['date']).dt.date
    survey['date'] = pd.to_datetime(survey['StartDate']).dt.date
    
    # Print sample of dates after conversion
    print("\nSample dates after conversion:")
    print("GPS dates:", raw_gps['date'].head().tolist())
    print("Survey dates:", survey['date'].head().tolist())
    
    # Get unique participant-days with consistent ID handling
    raw_gps['user'] = raw_gps['user'].astype(str)
    survey['Participant_ID'] = survey['Participant_ID'].astype(str)
    participant_info['user'] = participant_info['user'].astype(str)
    
    gps_days = raw_gps.groupby(['user', 'date']).size().reset_index(name='gps_records')
    survey_days = survey.groupby(['Participant_ID', 'date']).size().reset_index(name='survey_responses')
    
    print("\nBasic Data Summary:")
    print(f"Total GPS records: {len(raw_gps):,}")
    print(f"GPS date range: {raw_gps['date'].min()} to {raw_gps['date'].max()}")
    print(f"Unique participants in GPS data: {raw_gps['user'].nunique()}")
    print(f"Total participant-days in GPS data: {len(gps_days)}")
    
    print(f"\nTotal survey responses: {len(survey):,}")
    print(f"Survey date range: {survey['date'].min()} to {survey['date'].max()}")
    print(f"Unique participants in survey: {survey['Participant_ID'].nunique()}")
    print(f"Total participant-days in survey: {len(survey_days)}")
    
    # Merge datasets
    merged_days = pd.merge(
        gps_days, 
        survey_days,
        left_on=['user', 'date'],
        right_on=['Participant_ID', 'date'],
        how='outer',
        indicator=True
    )
    
    # Add demographic information
    merged_with_demo = pd.merge(
        merged_days,
        participant_info[['user', 'school_n', 'sex']],
        left_on='user',
        right_on='user',
        how='left'
    )
    
    print("\nData Coverage Analysis:")
    print(f"Days with both GPS and survey data: {len(merged_days[merged_days['_merge'] == 'both']):,}")
    print(f"Days with only GPS data: {len(merged_days[merged_days['_merge'] == 'left_only']):,}")
    print(f"Days with only survey data: {len(merged_days[merged_days['_merge'] == 'right_only']):,}")
    
    print("\nParticipant Demographics:")
    print("\nBy School Type:")
    school_counts = participant_info['school_n'].value_counts()
    for school, count in school_counts.items():
        print(f"{school}: {count} participants")
    
    print("\nBy Gender:")
    gender_counts = participant_info['sex'].value_counts()
    for gender, count in gender_counts.items():
        print(f"{gender}: {count} participants")
    
    # Detailed overlap analysis
    print("\nDetailed Overlap Analysis:")
    overlap_by_participant = merged_days[merged_days['_merge'] == 'both']['user'].value_counts()
    print(f"\nParticipants with overlapping data: {len(overlap_by_participant)}")
    print("\nNumber of overlapping days per participant:")
    print(overlap_by_participant.describe())
    
    # Get participants with overlapping data
    participants_with_overlap = merged_days[merged_days['_merge'] == 'both']['user'].unique()
    
    # Get demographic details for these participants
    overlap_demographics = participant_info[participant_info['user'].isin(participants_with_overlap)]
    
    print("\nDemographics for Participants with Overlapping Data:")
    print("\nBy School Type:")
    school_counts = overlap_demographics['school_n'].value_counts()
    for school, count in school_counts.items():
        print(f"{school}: {count} participants")
    
    print("\nBy Gender:")
    gender_counts = overlap_demographics['sex'].value_counts()
    for gender, count in gender_counts.items():
        print(f"{gender}: {count} participants")
    
    print("\nBy Class:")
    if 'Class' in overlap_demographics.columns:
        class_counts = overlap_demographics['Class'].value_counts()
        for class_num, count in class_counts.items():
            print(f"Class {class_num}: {count} participants")
    else:
        print("Class information not available in participant_info file")
    
    # Enhanced demographic analysis
    def create_demographic_summary(data, group_cols, metric_cols):
        summary = data.groupby(group_cols)[metric_cols].agg({
            'gps_records': ['count', 'mean'],
            'survey_responses': ['count', 'mean']
        }).round(2)
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        return summary
    
    # Create demographic summaries
    school_summary = create_demographic_summary(
        merged_with_demo,
        'school_n',
        ['gps_records', 'survey_responses']
    )
    
    gender_summary = create_demographic_summary(
        merged_with_demo,
        'sex',
        ['gps_records', 'survey_responses']
    )
    
    combined_demo_summary = create_demographic_summary(
        merged_with_demo,
        ['school_n', 'sex'],
        ['gps_records', 'survey_responses']
    )
    
    # Calculate coverage metrics by demographics
    coverage_metrics = merged_with_demo.groupby(['school_n', 'sex']).agg({
        '_merge': lambda x: (x == 'both').mean(),
        'gps_records': 'mean',
        'survey_responses': 'mean'
    }).round(3)
    coverage_metrics.columns = ['overlap_ratio', 'avg_gps_records', 'avg_survey_responses']
    
    # Save detailed demographic summaries
    school_summary.to_csv(os.path.join(OUTPUT_DIR, 'school_coverage_summary.csv'))
    gender_summary.to_csv(os.path.join(OUTPUT_DIR, 'gender_coverage_summary.csv'))
    combined_demo_summary.to_csv(os.path.join(OUTPUT_DIR, 'combined_demographic_summary.csv'))
    coverage_metrics.to_csv(os.path.join(OUTPUT_DIR, 'coverage_metrics_by_demographics.csv'))
    
    # Create visualizations with demographic breakdowns
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=merged_with_demo, x='school_n', y='gps_records', hue='sex')
    plt.title('GPS Records Distribution by School Type and Gender')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'gps_records_by_demographics.png'))
    plt.close()
    
    # Create overlap visualization by demographics
    plt.figure(figsize=(12, 6))
    overlap_by_demo = merged_with_demo[merged_with_demo['_merge'] == 'both'].groupby(['school_n', 'sex']).size().unstack()
    overlap_by_demo.plot(kind='bar', stacked=True)
    plt.title('Overlapping Days by School Type and Gender')
    plt.xlabel('School Type')
    plt.ylabel('Number of Days')
    plt.legend(title='Gender')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'overlap_by_demographics.png'))
    plt.close()
    
    # Save detailed results
    results = {
        'merged_days': merged_days,
        'overlap_by_participant': overlap_by_participant,
        'gps_days': gps_days,
        'survey_days': survey_days
    }
    
    # Create visualizations
    plt.figure(figsize=(10, 6))
    sns.histplot(data=overlap_by_participant, bins=20)
    plt.title('Distribution of Overlapping Days per Participant')
    plt.xlabel('Number of Days')
    plt.ylabel('Number of Participants')
    plt.savefig(os.path.join(OUTPUT_DIR, 'overlap_distribution.png'))
    plt.close()
    
    return results

if __name__ == "__main__":
    results = analyze_data_coverage() 