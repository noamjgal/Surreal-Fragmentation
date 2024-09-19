import pandas as pd
import numpy as np
from scipy import stats
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Define input and output directories
input_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/fragmentation'
survey_file = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/Survey/End_of_the_day_questionnaire.xlsx'
participant_info_file = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/participant_info.csv'
output_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/analysis_results'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def load_and_preprocess_data():
    # Load the fragmentation results
    moving_frag = pd.read_csv(os.path.join(input_dir, 'moving_fragmentation_summary.csv'))
    digital_frag = pd.read_csv(os.path.join(input_dir, 'digital_fragmentation_summary.csv'))

    # Load the survey responses
    survey_responses = pd.read_excel(survey_file)

    # Load participant info
    participant_info = pd.read_csv(participant_info_file)

    # Ensure date columns are in datetime format
    moving_frag['date'] = pd.to_datetime(moving_frag['date']).dt.date
    digital_frag['date'] = pd.to_datetime(digital_frag['date']).dt.date
    survey_responses['date'] = pd.to_datetime(survey_responses['StartDate']).dt.date

    # Ensure participant_id is treated as string in all dataframes
    moving_frag['participant_id'] = moving_frag['participant_id'].astype(str)
    digital_frag['participant_id'] = digital_frag['participant_id'].astype(str)
    survey_responses['Participant_ID'] = survey_responses['Participant_ID'].astype(str)
    participant_info['user'] = participant_info['user'].astype(str)

    # Merge the datasets
    merged_data = pd.merge(moving_frag, digital_frag, on=['participant_id', 'date'], suffixes=('_moving', '_digital'))
    merged_data = pd.merge(merged_data, survey_responses, left_on=['participant_id', 'date'], right_on=['Participant_ID', 'date'], how='inner')
    merged_data = pd.merge(merged_data, participant_info, left_on='participant_id', right_on='user', how='left')

    print("Data types after merging:")
    print(merged_data.dtypes)
    print("\nUnique values in 'sex' column:")
    print(merged_data['sex'].value_counts())

    return merged_data

def preprocess_data(merged_data):
    # Handle missing values
    numeric_columns = merged_data.select_dtypes(include=[np.number]).columns
    categorical_columns = merged_data.select_dtypes(include=['object', 'category']).columns

    for col in numeric_columns:
        merged_data[col] = merged_data[col].fillna(merged_data[col].median())
    
    for col in categorical_columns:
        # Check if the column contains datetime objects
        if merged_data[col].dtype == 'object' and pd.api.types.is_datetime64_any_dtype(merged_data[col]):
            merged_data[col] = merged_data[col].fillna(merged_data[col].mode()[0])
        else:
            merged_data[col] = merged_data[col].fillna(merged_data[col].mode()[0])
    
    # Reverse score positive emotions
    for item in ['RELAXATION', 'PEACE', 'SATISFACTION']:
        merged_data[item] = 5 - merged_data[item]
    
    # Calculate STAI6 score
    stai6_items = ['TENSE', 'RELAXATION', 'WORRY', 'PEACE', 'IRRITATION', 'SATISFACTION']
    merged_data['STAI6_score'] = merged_data[stai6_items].mean(axis=1) * 20/6
    
    # Drop rows where STAI-6 items or fragmentation indices are still missing
    frag_indices = ['fragmentation_index_moving', 'fragmentation_index_digital']
    merged_data_clean = merged_data.dropna(subset=stai6_items + frag_indices)
    
    print(f"\nOriginal data shape: {merged_data.shape}")
    print(f"Clean data shape: {merged_data_clean.shape}")
    
    return merged_data_clean

def perform_ttest(group1, group2, metric):
    if len(group1) < 2 or len(group2) < 2:
        print(f"Warning: Not enough samples for t-test on {metric}. Group sizes: {len(group1)}, {len(group2)}")
        return np.nan, np.nan, np.nan
    
    t_stat, p_value = stats.ttest_ind(group1[metric].dropna(), group2[metric].dropna())
    effect_size = (group1[metric].mean() - group2[metric].mean()) / np.sqrt((group1[metric].std()**2 + group2[metric].std()**2) / 2)
    return t_stat, p_value, effect_size

def analyze_fragmentation_differences(merged_data):
    results = []
    for frag_index in ['fragmentation_index_moving', 'fragmentation_index_digital']:
        # Gender analysis
        male = merged_data[merged_data['sex'] == 'Male']
        female = merged_data[merged_data['sex'] == 'Female']
        if len(male) > 0 and len(female) > 0:
            t_stat, p_value, effect_size = perform_ttest(male, female, frag_index)
            results.append({
                'Analysis': 'Gender',
                'Fragmentation Index': frag_index,
                'Group1': 'Male',
                'Group2': 'Female',
                't-statistic': t_stat,
                'p-value': p_value,
                'Effect Size': effect_size,
                'Group1 Mean': male[frag_index].mean(),
                'Group2 Mean': female[frag_index].mean(),
                'Group1 Size': len(male),
                'Group2 Size': len(female)
            })
        else:
            print(f"Warning: Not enough data for gender analysis on {frag_index}")

        # School analysis
        schools = merged_data['school_n'].unique()
        for i in range(len(schools)):
            for j in range(i+1, len(schools)):
                school1 = merged_data[merged_data['school_n'] == schools[i]]
                school2 = merged_data[merged_data['school_n'] == schools[j]]
                if len(school1) > 0 and len(school2) > 0:
                    t_stat, p_value, effect_size = perform_ttest(school1, school2, frag_index)
                    results.append({
                        'Analysis': 'School',
                        'Fragmentation Index': frag_index,
                        'Group1': f'School {schools[i]}',
                        'Group2': f'School {schools[j]}',
                        't-statistic': t_stat,
                        'p-value': p_value,
                        'Effect Size': effect_size,
                        'Group1 Mean': school1[frag_index].mean(),
                        'Group2 Mean': school2[frag_index].mean(),
                        'Group1 Size': len(school1),
                        'Group2 Size': len(school2)
                    })
                else:
                    print(f"Warning: Not enough data for school analysis between School {schools[i]} and School {schools[j]} on {frag_index}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('p-value')
    return results_df

def perform_regression_analysis(merged_data):
    frag_indices = ['fragmentation_index_moving', 'fragmentation_index_digital']
    metrics = ['TENSE', 'RELAXATION', 'WORRY', 'PEACE', 'IRRITATION', 'SATISFACTION', 'STAI6_score', 'HAPPY']
    
    results = []
    for frag_index in frag_indices:
        for metric in metrics:
            X = sm.add_constant(merged_data[frag_index])
            y = merged_data[metric]
            model = sm.OLS(y, X).fit()
            
            results.append({
                'Fragmentation Index': frag_index,
                'Metric': metric,
                'Coefficient': model.params[frag_index],
                'p-value': model.pvalues[frag_index],
                'R-squared': model.rsquared,
                'CI_Lower': model.conf_int().loc[frag_index, 0],
                'CI_Upper': model.conf_int().loc[frag_index, 1]
            })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('p-value')
    return results_df

def perform_multilevel_analysis(merged_data):
    # Calculate participant-level mean fragmentation indices
    merged_data['frag_moving_mean'] = merged_data.groupby('participant_id')['fragmentation_index_moving'].transform('mean')
    merged_data['frag_digital_mean'] = merged_data.groupby('participant_id')['fragmentation_index_digital'].transform('mean')
    
    # Calculate daily deviations from participant mean
    merged_data['frag_moving_dev'] = merged_data['fragmentation_index_moving'] - merged_data['frag_moving_mean']
    merged_data['frag_digital_dev'] = merged_data['fragmentation_index_digital'] - merged_data['frag_digital_mean']
    
    metrics = ['TENSE', 'RELAXATION', 'WORRY', 'PEACE', 'IRRITATION', 'SATISFACTION', 'STAI6_score', 'HAPPY']
    
    results = []
    for metric in metrics:
        formula = f"{metric} ~ frag_moving_mean + frag_digital_mean + frag_moving_dev + frag_digital_dev"
        model = smf.mixedlm(formula, data=merged_data, groups='participant_id')
        
        try:
            fit = model.fit()
            for var in ['frag_moving_mean', 'frag_digital_mean', 'frag_moving_dev', 'frag_digital_dev']:
                results.append({
                    'Outcome': metric,
                    'Predictor': var,
                    'Coefficient': fit.params[var],
                    'p-value': fit.pvalues[var],
                    'CI_Lower': fit.conf_int().loc[var, 0],
                    'CI_Upper': fit.conf_int().loc[var, 1]
                })
        except Exception as e:
            print(f"Error fitting model for {metric}: {str(e)}")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('p-value')
    return results_df

def visualize_fragmentation_differences(results_df):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Analysis', y='Effect Size', hue='Fragmentation Index', data=results_df)
    plt.title('Effect Sizes of Fragmentation Differences Between Groups')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fragmentation_differences_barplot.png'))
    plt.close()

def visualize_fragmentation_impact(results_dfs, threshold):
    plt.figure(figsize=(14, 7))
    
    # Prepare data for plotting
    plot_data = []
    for frag_index, df in results_dfs.items():
        df['Fragmentation Index'] = frag_index
        plot_data.append(df)
    
    plot_df = pd.concat(plot_data)
    
    # Create the plot
    sns.barplot(x='Metric', y='Effect Size', hue='Fragmentation Index', data=plot_df)
    plt.title(f'Effect Sizes of Fragmentation Impact on Emotional Metrics (Threshold: {threshold})')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fragmentation_impact_barplot_{threshold}.png'))
    plt.close()

def analyze_fragmentation_impact(merged_data, threshold='median'):
    frag_columns = ['fragmentation_index_moving', 'fragmentation_index_digital']
    metrics = ['TENSE', 'RELAXATION', 'WORRY', 'PEACE', 'IRRITATION', 'SATISFACTION', 'STAI6_score', 'HAPPY']

    results = {frag_index: [] for frag_index in frag_columns}
    
    for frag_index in frag_columns:
        if threshold == 'median':
            split_value = merged_data[frag_index].median()
        elif threshold == '25th':
            split_value = merged_data[frag_index].quantile(0.25)
        
        high_frag = merged_data[merged_data[frag_index] > split_value]
        low_frag = merged_data[merged_data[frag_index] <= split_value]
        
        for metric in metrics:
            t_stat, p_value, effect_size = perform_ttest(high_frag, low_frag, metric)
            results[frag_index].append({
                'Metric': metric,
                'Threshold': threshold,
                't-statistic': t_stat,
                'p-value': p_value,
                'Effect Size': effect_size,
                'High Group Mean': high_frag[metric].mean(),
                'Low Group Mean': low_frag[metric].mean(),
                'High Group Count': len(high_frag),
                'Low Group Count': len(low_frag)
            })

    results_dfs = {frag_index: pd.DataFrame(results[frag_index]).sort_values('p-value') 
                   for frag_index in frag_columns}
    return results_dfs

def main():
    merged_data = load_and_preprocess_data()
    merged_data = preprocess_data(merged_data)
    
    if len(merged_data) == 0:
        print("Error: No data left after preprocessing. Please check your data and preprocessing steps.")
        return
    
    # Print IQR statistics for fragmentation indices
    frag_indices = ['fragmentation_index_moving', 'fragmentation_index_digital']
    for index in frag_indices:
        q1 = merged_data[index].quantile(0.25)
        median = merged_data[index].median()
        q3 = merged_data[index].quantile(0.75)
        iqr = q3 - q1
        print(f"\nIQR statistics for {index}:")
        print(f"25th percentile: {q1:.4f}")
        print(f"Median: {median:.4f}")
        print(f"75th percentile: {q3:.4f}")
        print(f"IQR: {iqr:.4f}")
    
    # Analyze fragmentation differences between populations
    frag_diff_results = analyze_fragmentation_differences(merged_data)
    if not frag_diff_results.empty:
        frag_diff_results.to_csv(os.path.join(output_dir, 'fragmentation_differences.csv'), index=False)
        visualize_fragmentation_differences(frag_diff_results)
    else:
        print("Warning: No results for fragmentation differences analysis.")

    # Analyze fragmentation impact using median split
    frag_impact_median = analyze_fragmentation_impact(merged_data, threshold='median')
    for frag_index, df in frag_impact_median.items():
        df.to_csv(os.path.join(output_dir, f'fragmentation_impact_median_{frag_index}.csv'), index=False)
    visualize_fragmentation_impact(frag_impact_median, 'median')

    # Analyze fragmentation impact using 25th percentile split
    frag_impact_25th = analyze_fragmentation_impact(merged_data, threshold='25th')
    for frag_index, df in frag_impact_25th.items():
        df.to_csv(os.path.join(output_dir, f'fragmentation_impact_25th_{frag_index}.csv'), index=False)
    visualize_fragmentation_impact(frag_impact_25th, '25th')

    # Print group sizes for median and 25th percentile splits
    for index in frag_indices:
        median = merged_data[index].median()
        q1 = merged_data[index].quantile(0.25)
        print(f"\nGroup sizes for {index}:")
        print(f"Median split:")
        print(f"  High group (> {median:.4f}): {sum(merged_data[index] > median)}")
        print(f"  Low group (<= {median:.4f}): {sum(merged_data[index] <= median)}")
        print(f"25th percentile split:")
        print(f"  High group (> {q1:.4f}): {sum(merged_data[index] > q1)}")
        print(f"  Low group (<= {q1:.4f}): {sum(merged_data[index] <= q1)}")

    # Perform regression analysis
    regression_results = perform_regression_analysis(merged_data)
    regression_results.to_csv(os.path.join(output_dir, 'regression_analysis.csv'), index=False)

    # Perform multilevel analysis
    multilevel_results = perform_multilevel_analysis(merged_data)
    multilevel_results.to_csv(os.path.join(output_dir, 'multilevel_analysis.csv'), index=False)

    print(f"All analysis results have been saved in the directory: {output_dir}")
if __name__ == "__main__":
    main()