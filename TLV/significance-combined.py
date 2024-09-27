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
    frag_summary = pd.read_csv(os.path.join(input_dir, 'fragmentation_summary.csv'))

    # Load the survey responses
    survey_responses = pd.read_excel(survey_file)

    # Load participant info
    participant_info = pd.read_csv(participant_info_file)

    # Ensure date columns are in datetime format
    frag_summary['date'] = pd.to_datetime(frag_summary['date']).dt.date
    survey_responses['date'] = pd.to_datetime(survey_responses['StartDate']).dt.date

    # Ensure participant_id is treated as string in all dataframes
    frag_summary['participant_id'] = frag_summary['participant_id'].astype(str)
    survey_responses['Participant_ID'] = survey_responses['Participant_ID'].astype(str)
    participant_info['user'] = participant_info['user'].astype(str)

    # Merge the datasets
    merged_data = pd.merge(frag_summary, survey_responses, left_on=['participant_id', 'date'], right_on=['Participant_ID', 'date'], how='inner')
    merged_data = pd.merge(merged_data, participant_info, left_on='participant_id', right_on='user', how='left')

    # Map Hebrew gender labels to binary
    merged_data['Gender_binary'] = merged_data['Gender'].map({'נקבה': 0, 'זכר': 1})

    # Map school locations
    merged_data['School_location'] = merged_data['School'].map({'suburb': 0, 'city_center': 1})

    return merged_data

def preprocess_data(merged_data):
    # Handle missing values
    numeric_columns = merged_data.select_dtypes(include=[np.number]).columns
    categorical_columns = merged_data.select_dtypes(include=['object', 'category']).columns

    for col in numeric_columns:
        merged_data[col] = merged_data[col].fillna(merged_data[col].median())
    
    for col in categorical_columns:
        if merged_data[col].dtype == 'object' and pd.api.types.is_datetime64_any_dtype(merged_data[col]):
            merged_data[col] = merged_data[col].fillna(merged_data[col].mode()[0])
        else:
            merged_data[col] = merged_data[col].fillna(merged_data[col].mode()[0])
    
    # Create temporary reversed columns for STAI6 calculation
    for item in ['RELAXATION', 'PEACE', 'SATISFACTION']:
        merged_data[f'{item}_reversed'] = 5 - merged_data[item]
    
    # Calculate STAI6 score using the reversed columns
    stai6_items = ['TENSE', 'RELAXATION_reversed', 'WORRY', 'PEACE_reversed', 'IRRITATION', 'SATISFACTION_reversed']
    merged_data['STAI6_score'] = merged_data[stai6_items].mean(axis=1) * 20/6
    
    # Remove temporary reversed columns
    merged_data = merged_data.drop(columns=[f'{item}_reversed' for item in ['RELAXATION', 'PEACE', 'SATISFACTION']])
    
    # Drop rows where STAI-6 items or fragmentation indices are still missing
    frag_indices = ['digital_fragmentation_index', 'moving_fragmentation_index', 'digital_frag_during_mobility']
    original_stai6_items = ['TENSE', 'RELAXATION', 'WORRY', 'PEACE', 'IRRITATION', 'SATISFACTION']
    merged_data_clean = merged_data.dropna(subset=original_stai6_items + frag_indices)
    
    return merged_data_clean

def analyze_digital_frag_mobility_relationship(merged_data):
    digital_frag_metrics = ['digital_fragmentation_index', 'digital_frag_during_mobility']
    mobility_metrics = ['total_duration_mobility', 'avg_duration_mobility', 'count_mobility']
    
    results = []
    for frag_metric in digital_frag_metrics:
        for mobility_metric in mobility_metrics:
            digital_frag = merged_data[frag_metric]
            mobility_data = merged_data[mobility_metric]

            # Perform Pearson correlation
            corr, p_value = stats.pearsonr(digital_frag, mobility_data)
            
            # Perform linear regression
            X = sm.add_constant(digital_frag)
            y = mobility_data
            model = sm.OLS(y, X).fit()
            
            # Perform two-population t-test based on 25th percentile split
            split_value = digital_frag.quantile(0.25)
            high_frag = merged_data[merged_data[frag_metric] > split_value][mobility_metric]
            low_frag = merged_data[merged_data[frag_metric] <= split_value][mobility_metric]
            t_stat, t_p_value = stats.ttest_ind(high_frag, low_frag)
            
            results.append({
                'Fragmentation Metric': frag_metric,
                'Mobility Metric': mobility_metric,
                'Pearson Correlation': corr,
                'Pearson p-value': p_value,
                'Regression Coefficient': model.params[frag_metric],
                'Regression p-value': model.pvalues[frag_metric],
                'R-squared': model.rsquared,
                'T-test statistic': t_stat,
                'T-test p-value': t_p_value,
                'High Frag Mean': high_frag.mean(),
                'Low Frag Mean': low_frag.mean(),
                'High Frag Count': len(high_frag),
                'Low Frag Count': len(low_frag)
            })
    
    return pd.DataFrame(results)


def analyze_population_differences(merged_data):
    frag_indices = ['digital_fragmentation_index', 'moving_fragmentation_index', 'digital_frag_during_mobility']
    population_factors = ['Gender_binary', 'Class', 'School_location']
    
    results = []
    for factor in population_factors:
        for frag_index in frag_indices:
            unique_values = sorted(merged_data[factor].unique())
            
            if factor == 'School_location':
                group0 = merged_data[merged_data[factor] == unique_values[0]][frag_index]
                group1 = merged_data[merged_data[factor] == unique_values[1]][frag_index]
                
                t_stat, p_value = stats.ttest_ind(group0, group1, equal_var=False)
                effect_size = (group0.mean() - group1.mean()) / np.sqrt((group0.std()**2 + group1.std()**2) / 2)
                
                results.append({
                    'Factor': factor,
                    'Fragmentation Index': frag_index,
                    'Group1': 'suburban',
                    'Group2': 'city-center',
                    'Group1 Mean': group0.mean(),
                    'Group2 Mean': group1.mean(),
                    'Group1 Size': len(group0),
                    'Group2 Size': len(group1),
                    't-statistic': t_stat,
                    'p-value': p_value,
                    'Effect Size': effect_size,
                    'Group1 Std': group0.std(),
                    'Group2 Std': group1.std()
                })
            elif factor == 'Gender_binary':
                group0 = merged_data[merged_data[factor] == 0][frag_index]
                group1 = merged_data[merged_data[factor] == 1][frag_index]
                
                t_stat, p_value = stats.ttest_ind(group0, group1, equal_var=False)
                effect_size = (group0.mean() - group1.mean()) / np.sqrt((group0.std()**2 + group1.std()**2) / 2)
                
                results.append({
                    'Factor': factor,
                    'Fragmentation Index': frag_index,
                    'Group1': 'female',
                    'Group2': 'male',
                    'Group1 Mean': group0.mean(),
                    'Group2 Mean': group1.mean(),
                    'Group1 Size': len(group0),
                    'Group2 Size': len(group1),
                    't-statistic': t_stat,
                    'p-value': p_value,
                    'Effect Size': effect_size,
                    'Group1 Std': group0.std(),
                    'Group2 Std': group1.std()
                })
            else:
                for i in range(len(unique_values)):
                    for j in range(i+1, len(unique_values)):
                        group1 = merged_data[merged_data[factor] == unique_values[i]][frag_index]
                        group2 = merged_data[merged_data[factor] == unique_values[j]][frag_index]
                        
                        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
                        effect_size = (group1.mean() - group2.mean()) / np.sqrt((group1.std()**2 + group2.std()**2) / 2)
                        
                        results.append({
                            'Factor': factor,
                            'Fragmentation Index': frag_index,
                            'Group1': unique_values[i],
                            'Group2': unique_values[j],
                            'Group1 Mean': group1.mean(),
                            'Group2 Mean': group2.mean(),
                            'Group1 Size': len(group1),
                            'Group2 Size': len(group2),
                            't-statistic': t_stat,
                            'p-value': p_value,
                            'Effect Size': effect_size,
                            'Group1 Std': group1.std(),
                            'Group2 Std': group2.std()
                        })
    
    results_df = pd.DataFrame(results)
    results_df['p-value'] = results_df['p-value'].apply(lambda x: 0 if x < 0.000001 else round(x, 6))
    results_df = results_df.sort_values('p-value')
    
    return results_df


def analyze_fragmentation_relationships(merged_data):
    frag_indices = ['digital_fragmentation_index', 'moving_fragmentation_index', 'digital_frag_during_mobility']
    
    results = []
    for i, index1 in enumerate(frag_indices):
        for j, index2 in enumerate(frag_indices[i+1:], start=i+1):
            # Perform Pearson correlation
            corr, p_value = stats.pearsonr(merged_data[index1], merged_data[index2])
            
            # Perform linear regression
            X = sm.add_constant(merged_data[index1])
            y = merged_data[index2]
            model = sm.OLS(y, X).fit()
            
            results.append({
                'Index1': index1,
                'Index2': index2,
                'Pearson Correlation': corr,
                'Pearson p-value': p_value,
                'Regression Coefficient': model.params[index1],
                'Regression p-value': model.pvalues[index1],
                'R-squared': model.rsquared
            })
    
    return pd.DataFrame(results)

def analyze_frag_emotional_relationship(merged_data):
    frag_indices = ['digital_fragmentation_index', 'moving_fragmentation_index', 'digital_frag_during_mobility']
    emotional_outcomes = ['TENSE', 'RELAXATION', 'WORRY', 'PEACE', 'IRRITATION', 'SATISFACTION', 'STAI6_score', 'HAPPY']
    
    results = []
    for frag_index in frag_indices:
        for outcome in emotional_outcomes:
            # Perform Pearson correlation
            corr, p_value = stats.pearsonr(merged_data[frag_index], merged_data[outcome])
            
            # Perform linear regression
            X = sm.add_constant(merged_data[frag_index])
            y = merged_data[outcome]
            model = sm.OLS(y, X).fit()
            
            # Perform two-population t-test based on median split
            median = merged_data[frag_index].median()
            high_frag = merged_data[merged_data[frag_index] > median][outcome]
            low_frag = merged_data[merged_data[frag_index] <= median][outcome]
            t_stat, t_p_value = stats.ttest_ind(high_frag, low_frag)
            
            results.append({
                'Fragmentation Index': frag_index,
                'Emotional Outcome': outcome,
                'Pearson Correlation': corr,
                'Pearson p-value': p_value,
                'Regression Coefficient': model.params[frag_index],
                'Regression p-value': model.pvalues[frag_index],
                'R-squared': model.rsquared,
                'T-test statistic': t_stat,
                'T-test p-value': t_p_value,
                'High Frag Mean': high_frag.mean(),
                'Low Frag Mean': low_frag.mean()
            })
    
    return pd.DataFrame(results)

def perform_multilevel_analysis(merged_data, threshold='median'):
    frag_types = ['digital_fragmentation_index', 'moving_fragmentation_index', 'digital_frag_during_mobility']
    
    for frag_type in frag_types:
        if threshold == 'median':
            merged_data[f'high_{frag_type}'] = merged_data[frag_type] > merged_data[frag_type].median()
        elif threshold == '25th':
            merged_data[f'high_{frag_type}'] = merged_data[frag_type] > merged_data[frag_type].quantile(0.25)
    
    metrics = ['TENSE', 'RELAXATION', 'WORRY', 'PEACE', 'IRRITATION', 'SATISFACTION', 'STAI6_score', 'HAPPY']
    
    results = []
    for metric in metrics:
        for frag_type in frag_types:
            formula = f"{metric} ~ high_{frag_type} + C(Gender_binary) + C(Class) + C(School_location)"
            model = smf.mixedlm(formula, data=merged_data, groups='participant_id')
            
            try:
                fit = model.fit()
                for var in [f'high_{frag_type}', 'C(Gender_binary)[T.1]', 'C(Class)[T.2]', 'C(Class)[T.3]', 'C(School_location)[T.1]']:
                    if var in fit.params:
                        results.append({
                            'Threshold': threshold,
                            'Fragmentation Type': frag_type,
                            'Outcome': metric,
                            'Predictor': var,
                            'Coefficient': fit.params[var],
                            'p-value': fit.pvalues[var],
                            'CI_Lower': fit.conf_int().loc[var, 0],
                            'CI_Upper': fit.conf_int().loc[var, 1]
                        })
            except Exception as e:
                print(f"Error fitting model for {metric} with {frag_type}: {str(e)}")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('p-value')
    return results_df

def main():
    merged_data = load_and_preprocess_data()
    merged_data = preprocess_data(merged_data)
    
    if len(merged_data) == 0:
        print("Error: No data left after preprocessing. Please check your data and preprocessing steps.")
        return
    
    # Print IQR statistics for fragmentation indices
    frag_indices = ['digital_fragmentation_index', 'moving_fragmentation_index', 'digital_frag_during_mobility']
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
    
    # Analyze relationship between digital fragmentation and mobility metrics
    digital_frag_mobility_results = analyze_digital_frag_mobility_relationship(merged_data)
    digital_frag_mobility_results.to_csv(os.path.join(output_dir, 'digital_frag_mobility_relationship.csv'), index=False)

    # Analyze relationships between fragmentation indices
    frag_relationships_results = analyze_fragmentation_relationships(merged_data)
    frag_relationships_results.to_csv(os.path.join(output_dir, 'fragmentation_relationships.csv'), index=False)

    # Analyze relationships between fragmentation indices and emotional outcomes
    frag_emotional_results = analyze_frag_emotional_relationship(merged_data)
    frag_emotional_results.to_csv(os.path.join(output_dir, 'fragmentation_emotional_relationship.csv'), index=False)

    # Perform multilevel analysis for median split
    multilevel_results_median = perform_multilevel_analysis(merged_data, threshold='median')
    multilevel_results_median.to_csv(os.path.join(output_dir, 'multilevel_analysis_median.csv'), index=False)

    # Perform multilevel analysis for 25th percentile split
    multilevel_results_25th = perform_multilevel_analysis(merged_data, threshold='25th')
    multilevel_results_25th.to_csv(os.path.join(output_dir, 'multilevel_analysis_25th.csv'), index=False)

    # Analyze population differences
    population_diff_results = analyze_population_differences(merged_data)
    population_diff_results.to_csv(os.path.join(output_dir, 'population_differences.csv'), index=False)

    print(f"All analysis results have been saved in the directory: {output_dir}")

if __name__ == "__main__":
    main()