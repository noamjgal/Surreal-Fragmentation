import pandas as pd
import numpy as np
from scipy import stats
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Define input and output directories
input_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/fragmentation'
survey_file = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/Survey/End_of_the_day_questionnaire.xlsx'
output_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/analysis_results'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def load_and_preprocess_data():
    # Load the fragmentation results
    moving_frag = pd.read_csv(os.path.join(input_dir, 'moving_fragmentation_summary.csv'))
    digital_frag = pd.read_csv(os.path.join(input_dir, 'digital_fragmentation_summary.csv'))

    # Load the survey responses
    survey_responses = pd.read_excel(survey_file)

    # Ensure date columns are in datetime format
    moving_frag['date'] = pd.to_datetime(moving_frag['date']).dt.date
    digital_frag['date'] = pd.to_datetime(digital_frag['date']).dt.date
    survey_responses['date'] = pd.to_datetime(survey_responses['StartDate']).dt.date

    # Ensure participant_id is treated as string in all dataframes
    moving_frag['participant_id'] = moving_frag['participant_id'].astype(str)
    digital_frag['participant_id'] = digital_frag['participant_id'].astype(str)
    survey_responses['Participant_ID'] = survey_responses['Participant_ID'].astype(str)

    # Merge the datasets
    merged_data = pd.merge(moving_frag, digital_frag, on=['participant_id', 'date'], suffixes=('_moving', '_digital'))
    merged_data = pd.merge(merged_data, survey_responses, left_on=['participant_id', 'date'], right_on=['Participant_ID', 'date'], how='inner')

    return merged_data

def calculate_stai6_score(df):
    # STAI-6 items
    stai6_items = ['TENSE', 'RELAXATION', 'WORRY', 'PEACE', 'IRRITATION', 'SATISFACTION']
    
    # Reverse score the positive items
    for item in ['RELAXATION', 'PEACE', 'SATISFACTION']:
        df[f'{item}_rev'] = 5 - df[item]
    
    # Calculate STAI-6 score
    df['STAI6_score'] = df[['TENSE', 'WORRY', 'IRRITATION', 'RELAXATION_rev', 'PEACE_rev', 'SATISFACTION_rev']].mean(axis=1) * 20/6
    
    return df

def perform_ttest(high_group, low_group, metric):
    t_stat, p_value = stats.ttest_ind(high_group[metric].dropna(), low_group[metric].dropna())
    effect_size = (high_group[metric].mean() - low_group[metric].mean()) / np.sqrt((high_group[metric].std()**2 + low_group[metric].std()**2) / 2)
    return t_stat, p_value, effect_size

def analyze_fragmentation(merged_data):
    merged_data = calculate_stai6_score(merged_data)

    # Define fragmentation indices
    frag_columns = [
        'fragmentation_index_moving',
        'AID_mean_moving',
        'fragmentation_index_digital',
        'AID_mean_digital'
    ]

    # Define metrics to analyze
    metrics = ['TENSE', 'RELAXATION', 'WORRY', 'PEACE', 'IRRITATION', 'SATISFACTION', 'STAI6_score', 'HAPPY']

    results = []
    for frag_index in frag_columns:
        print(f"\nAnalyzing {frag_index}:")
        median = merged_data[frag_index].median()
        high_frag = merged_data[merged_data[frag_index] > median]
        low_frag = merged_data[merged_data[frag_index] <= median]
        
        print(f"  Median: {median}")
        print(f"  High group size: {len(high_frag)}")
        print(f"  Low group size: {len(low_frag)}")
        
        for metric in metrics:
            t_stat, p_value, effect_size = perform_ttest(high_frag, low_frag, metric)
            results.append({
                'Fragmentation Index': frag_index,
                'Metric': metric,
                't-statistic': t_stat,
                'p-value': p_value,
                'Effect Size': effect_size,
                'High Group Mean': high_frag[metric].mean(),
                'Low Group Mean': low_frag[metric].mean(),
                'Significant': 'Yes' if p_value < 0.05 else 'No'
            })
            print(f"  {metric}: p-value = {p_value:.4f}, effect size = {effect_size:.4f}, significant: {'Yes' if p_value < 0.05 else 'No'}")

    return pd.DataFrame(results)

def visualize_results(results_df):
    plt.figure(figsize=(15, 10))
    sns.barplot(x='Fragmentation Index', y='Effect Size', hue='Metric', data=results_df)
    plt.title('Effect Sizes of Fragmentation Indices on Emotional Metrics')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'effect_sizes_barplot.png'))
    plt.close()

def main():
    merged_data = load_and_preprocess_data()
    results_df = analyze_fragmentation(merged_data)

    # Save full results to CSV
    full_results_path = os.path.join(output_dir, 'comprehensive_significance_results.csv')
    results_df.to_csv(full_results_path, index=False)
    print(f"\nFull significance results saved as '{full_results_path}'")

    # Print all results
    print("\nAll Results:")
    print(results_df.to_string(index=False))

    # Visualize results
    visualize_results(results_df)

    print(f"\nAll analysis results have been saved in the directory: {output_dir}")

if __name__ == "__main__":
    main()