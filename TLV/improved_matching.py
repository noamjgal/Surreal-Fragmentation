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
    
    # Convert dates with more lenient parsing
    def safe_date_parse(date_str):
        try:
            # Try multiple formats
            for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', 
                       '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d']:
                try:
                    return pd.to_datetime(date_str, format=fmt)
                except:
                    continue
            # If specific formats fail, let pandas try to figure it out
            return pd.to_datetime(date_str)
        except:
            print(f"Warning: Could not parse date: {date_str}")
            return None

    # Load and convert dates
    print("Converting dates...")
    gps_summary['date'] = gps_summary['date'].apply(safe_date_parse)
    survey['timestamp'] = pd.to_datetime(survey['StartDate'])
    
    # Debug print some sample dates
    print("\nSample date conversions:")
    print("GPS dates sample:")
    print(gps_summary[['user', 'date']].head())
    print("\nSurvey dates sample:")
    print(survey[['Participant_ID', 'StartDate', 'timestamp']].head())
    
    # Extract hour and normalize dates
    survey['hour'] = survey['timestamp'].dt.hour
    survey['date'] = survey['timestamp'].dt.normalize()
    
    # Apply the matching rules with more explicit date handling
    def adjust_date(row):
        if row['hour'] <= 5:  # Midnight through 5 AM
            # Convert to timestamp, subtract one day, then back to date
            return (row['timestamp'] - timedelta(days=1)).normalize()
        return row['date']
    
    survey['adjusted_date'] = survey.apply(adjust_date, axis=1)
    
    # Debug print to verify adjustments
    print("\nVerifying early morning adjustments:")
    early_morning = survey[survey['hour'] <= 5]
    if not early_morning.empty:
        print(early_morning[['Participant_ID', 'timestamp', 'date', 'adjusted_date', 'hour']].to_string())
    
    # Sample of all adjustments
    print("\nSample adjustments (showing where dates changed):")
    changed_dates = survey[survey['date'] != survey['adjusted_date']]
    print(changed_dates[['Participant_ID', 'timestamp', 'date', 'adjusted_date', 'hour']].head(10))
    
    # Ensure consistent ID types for all dataframes
    gps_summary['user'] = gps_summary['user'].astype(str)
    survey['Participant_ID'] = survey['Participant_ID'].astype(str)
    participant_info['user'] = participant_info['user'].astype(str)
    
    # Extract class information from survey data
    survey['Class'] = survey['Class'].astype(str)
    
    # First merge (GPS and Survey)
    merged = pd.merge(
        gps_summary,
        survey,
        left_on=['user', 'date'],
        right_on=['Participant_ID', 'adjusted_date'],
        how='outer',
        indicator=True
    )
    
    # Add demographic information
    merged_with_demo = pd.merge(
        merged,
        participant_info[['user', 'school_n', 'sex']],
        left_on='user',  # Use 'user' instead of Participant_ID
        right_on='user',
        how='left'
    )
    
    # Print merge diagnostics
    print("\nMerge with demographics results:")
    print(merged_with_demo['_merge'].value_counts())
    
    # Print merge results immediately after merge
    print("\nImmediate merge results:")
    print(merged['_merge'].value_counts())
    
    # Print unique participants before merge
    print("\nUnique participants before merge:")
    print(f"GPS participants: {sorted(gps_summary['user'].unique())}")
    print(f"Survey participants: {sorted(survey['Participant_ID'].unique())}")
    
    # First pass merge to identify already matched days
    initial_merge = pd.merge(
        gps_summary,
        survey,
        left_on=['user', 'date'],
        right_on=['Participant_ID', 'date'],
        how='outer',
        indicator=True
    )
    
    # Store already matched days
    matched_days = initial_merge[initial_merge['_merge'] == 'both'][['Participant_ID', 'date']].copy()
    
    # Print adjustment summary
    adjustments = survey[survey['date'] != survey['adjusted_date']]
    print("\nDate Adjustments Made:")
    print(f"Total adjustments: {len(adjustments)}")
    print("\nMidnight/Early morning adjustments (midnight to 5 AM):")
    print(adjustments[adjustments['hour'] <= 5][['Participant_ID', 'date', 'adjusted_date', 'hour']].to_string())
    
    # We'll keep track of late night submissions but won't adjust them
    late_night = survey[survey['hour'] >= 22]
    if not late_night.empty:
        print("\nLate night submissions (after 10 PM) - NOT adjusted:")
        print(late_night[['Participant_ID', 'date', 'hour']].to_string())
    
    # Analyze unmatched survey days
    unmatched_survey = merged[merged['_merge'] == 'right_only']
    
    print("\nAnalyzing unmatched survey days...")
    print("\nFor each survey-only day, showing closest GPS data date:")
    print("Format: Participant ID | Survey Date | Closest GPS Date | Days Difference | Survey Hour")
    print("-" * 80)
    
    potential_matches = []
    
    for idx, row in unmatched_survey.iterrows():
        # Get all GPS dates for this participant
        participant_gps = gps_summary[gps_summary['user'] == row['Participant_ID']]
        
        if not participant_gps.empty:
            # Calculate time difference for all GPS dates
            time_diffs = abs((participant_gps['date'] - row['adjusted_date']).dt.total_seconds() / 86400)
            closest_idx = time_diffs.idxmin()
            closest_gps = participant_gps.loc[closest_idx]
            days_diff = (closest_gps['date'] - row['adjusted_date']).total_seconds() / 86400
            
            print(f"{row['Participant_ID']:>13} | {row['adjusted_date'].strftime('%Y-%m-%d'):>10} | "
                  f"{closest_gps['date'].strftime('%Y-%m-%d'):>10} | {days_diff:>14.1f} | "
                  f"{row['hour']:>11}")
            
            # Continue with potential matches collection as before...
            if abs(days_diff) <= 2:  # Within 48 hours
                potential_matches.append({
                    'participant_id': row['Participant_ID'],
                    'survey_date': row['adjusted_date'],
                    'survey_timestamp': row['timestamp'],
                    'survey_hour': row['hour'],
                    'gps_date': closest_gps['date'],
                    'days_difference': days_diff,
                    'gps_readings': closest_gps['total_readings'],
                    'has_morning': closest_gps['has_morning_data'],
                    'has_evening': closest_gps['has_evening_data']
                })
    
    if potential_matches:
        potential_matches_df = pd.DataFrame(potential_matches)
        
        print("\nDetailed Analysis of Potential Matches:")
        print(f"\nTotal unmatched survey responses with nearby GPS data: {len(potential_matches_df['survey_date'].unique())}")
        
        print("\nBreakdown of survey submission times:")
        hour_dist = potential_matches_df['survey_hour'].value_counts().sort_index()
        print(hour_dist)
        
        print("\nBreakdown of day differences:")
        diff_dist = potential_matches_df['days_difference'].value_counts().sort_index()
        print(diff_dist)
        
        # Analyze late night submissions (after 10 PM)
        late_night = potential_matches_df[potential_matches_df['survey_hour'] >= 22]
        if not late_night.empty:
            print("\nLate night submissions (after 10 PM):")
            print(f"Count: {len(late_night)}")
            print("\nBreakdown by participant:")
            print(late_night['participant_id'].value_counts())
        
        # Analyze early morning submissions (before 6 AM)
        early_morning = potential_matches_df[potential_matches_df['survey_hour'] < 6]
        if not early_morning.empty:
            print("\nEarly morning submissions (before 6 AM):")
            print(f"Count: {len(early_morning)}")
            print("\nBreakdown by participant:")
            print(early_morning['participant_id'].value_counts())
        
        # Save detailed analysis
        output_path = os.path.join(OUTPUT_DIR, 'unmatched_detailed_analysis.csv')
        potential_matches_df.to_csv(output_path, index=False)
        
        # Create recommendations for matching
        print("\nRecommended matches:")
        for _, group in potential_matches_df.groupby(['participant_id', 'survey_date']):
            best_match = group.iloc[np.abs(group['days_difference']).argmin()]
            if (best_match['survey_hour'] >= 22 and best_match['days_difference'] < 0) or \
               (best_match['survey_hour'] <= 6 and best_match['days_difference'] > 0):
                print(f"Participant {best_match['participant_id']}: "
                      f"Survey on {best_match['survey_timestamp']} should probably be matched with "
                      f"GPS data from {best_match['gps_date']}")
    
    # After the merge analysis, add demographic summary
    matched_participants = merged[merged['_merge'] == 'both']['user'].unique()
    
    print("\nDemographic information for matched participants:")
    print("Format: ID | School | Gender | Class | Number of Matches")
    print("-" * 75)
    
    participant_matches = merged[merged['_merge'] == 'both']['user'].value_counts()
    
    for participant_id in sorted(matched_participants):
        demo = participant_info[participant_info['user'] == participant_id].iloc[0]
        participant_class = survey[survey['Participant_ID'] == participant_id]['Class'].iloc[0] if not survey[survey['Participant_ID'] == participant_id].empty else 'N/A'
        matches = participant_matches.get(participant_id, 0)
        print(f"ID: {participant_id:>3} | School: {demo['school_n']:>2} | "
              f"Gender: {demo['sex']:>6} | Class: {participant_class:>5} | Matches: {matches:>3}")
    
    # Print summary statistics with class information
    print("\nSummary by school and class:")
    school_class_summary = pd.DataFrame({
        'participant_id': matched_participants
    }).merge(participant_info[['user', 'school_n', 'sex']], 
            left_on='participant_id', 
            right_on='user'
    ).merge(survey[['Participant_ID', 'Class']].drop_duplicates(),
            left_on='participant_id',
            right_on='Participant_ID',
            how='left'
    ).groupby(['school_n', 'Class']).agg({
        'user': 'count',
        'sex': lambda x: x.value_counts().to_dict()
    }).rename(columns={'user': 'count'})
    
    print(school_class_summary)
    
    # Add class distribution visualization
    plt.figure(figsize=(12, 6))
    class_dist = survey.groupby(['Class', 'Participant_ID']).size().reset_index()
    class_dist = class_dist.groupby('Class').size()
    class_dist.plot(kind='bar')
    plt.title('Distribution of Participants by Class')
    plt.xlabel('Class')
    plt.ylabel('Number of Participants')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'class_distribution.png'))
    plt.close()
    
    # Add class information to the detailed analysis
    if potential_matches:
        potential_matches_df['Class'] = potential_matches_df['participant_id'].map(
            survey.drop_duplicates('Participant_ID').set_index('Participant_ID')['Class']
        )
        
        print("\nBreakdown of unmatched surveys by class:")
        print(potential_matches_df.groupby('Class')['participant_id'].nunique())
    
    # Include class in the coverage analysis
    print("\nCoverage Analysis by Class:")
    
    # First ensure we have the Class information in our merged data
    class_info = survey[['Participant_ID', 'Class']].drop_duplicates()
    
    # Perform the merge with more explicit parameters and handle duplicate columns
    coverage_analysis = merged_with_demo.merge(
        class_info,
        left_on='user',
        right_on='Participant_ID',
        how='left',
        suffixes=('_drop', '')  # Keep the right version of duplicate columns
    )
    
    # Debug prints
    print("\nDebug - Available columns after merge:")
    print(coverage_analysis.columns.tolist())
    print("\nDebug - Sample of merged data:")
    print(coverage_analysis[['user', 'Participant_ID', 'Class']].head())
    
    # Only proceed with groupby if Class column exists
    if 'Class' in coverage_analysis.columns:
        class_coverage = coverage_analysis.groupby('Class').agg({
            'user': 'nunique',
            '_merge': lambda x: (x == 'both').sum() / len(x)
        }).rename(columns={
            'user': 'participants',
            '_merge': 'match_rate'
        })
        print("\nClass Coverage Analysis:")
        print(class_coverage)
        
        # Save class-specific analysis
        class_analysis_path = os.path.join(OUTPUT_DIR, 'class_analysis.csv')
        class_coverage.to_csv(class_analysis_path)
    else:
        print("\nError: Class column not found after merge!")
        print("Available columns:", coverage_analysis.columns.tolist())
    
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
    
    # Analyze matches by participant
    matches_by_participant = merged[merged['_merge'] == 'both']['user'].value_counts()
    print("\nMatches by participant:")
    print(matches_by_participant.describe())
    
    # Save detailed results
    output_path = os.path.join(OUTPUT_DIR, 'detailed_matching_analysis.csv')
    merged_with_demo.to_csv(output_path, index=False)
    
    return merged_with_demo, potential_matches_df if potential_matches else None

if __name__ == "__main__":
    results, potential_matches = analyze_matching()