import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from math import pi
import sys

# Create output folder
output_folder = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/mobile-analysis-results'
os.makedirs(output_folder, exist_ok=True)
print(f"Output folder created: {output_folder}")

# Function to get full path for output files
def get_output_path(filename):
    return os.path.join(output_folder, filename)

# Save console output to a file
class OutputCapture:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.file = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()

def round_and_abs_aid(df):
    # Round all numeric columns to 2 decimal points
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].round(2)
    
    # Take absolute value of AID columns
    aid_columns = [col for col in df.columns if 'AID' in col]
    df[aid_columns] = df[aid_columns].abs()
    
    return df

def create_all_ttest_results(high_stationary, low_stationary, high_mobile, low_mobile):
    emotion_columns = ['PEACE', 'TENSE', 'IRRITATION', 'RELAXATION', 'SATISFACTION', 'WORRY', 'HAPPY']
    results = []

    for emotion in emotion_columns:
        # Stationary fragmentation
        high_data = high_stationary[emotion].dropna()
        low_data = low_stationary[emotion].dropna()
        t_stat, p_value = stats.ttest_ind(high_data, low_data)
        high_mean = high_data.mean()
        low_mean = low_data.mean()
        results.append({
            'Fragmentation': 'Stationary_index',
            'Emotion': emotion,
            'High_Mean': f'{high_mean:.2f}',
            'Low_Mean': f'{low_mean:.2f}',
            'T_Statistic': f'{t_stat:.3f}',
            'P_value': f'{p_value:.4f}',
            'High_Count': len(high_data),
            'Low_Count': len(low_data)
        })

        # Mobile fragmentation
        high_data = high_mobile[emotion].dropna()
        low_data = low_mobile[emotion].dropna()
        t_stat, p_value = stats.ttest_ind(high_data, low_data)
        high_mean = high_data.mean()
        low_mean = low_data.mean()
        results.append({
            'Fragmentation': 'Mobile_index',
            'Emotion': emotion,
            'High_Mean': f'{high_mean:.2f}',
            'Low_Mean': f'{low_mean:.2f}',
            'T_Statistic': f'{t_stat:.3f}',
            'P_value': f'{p_value:.4f}',
            'High_Count': len(high_data),
            'Low_Count': len(low_data)
        })

    return pd.DataFrame(results)

def create_comprehensive_analysis_table(correlation_matrix, p_values, high_stationary, low_stationary, high_mobile, low_mobile, threshold=0.05):
    frag_columns = ['Stationary_index', 'Mobile_index', 'Stationary_AID', 'Mobile_AID']
    emotion_columns = ['PEACE', 'TENSE', 'IRRITATION', 'RELAXATION', 'SATISFACTION', 'WORRY', 'HAPPY']

    results = []

    # T-test analysis
    for emotion in emotion_columns:
        # Stationary fragmentation
        high_data = high_stationary[emotion].dropna()
        low_data = low_stationary[emotion].dropna()
        t_stat, p_value = stats.ttest_ind(high_data, low_data)
        if p_value < threshold:
            high_mean = high_data.mean()
            low_mean = low_data.mean()
            results.append({
                'Analysis': 'T-test (Stationary)',
                'Fragmentation': 'Stationary_index',
                'Emotion': emotion,
                'Statistic': f'{t_stat:.3f}',
                'P-value': f'{p_value:.4f}',
                'Direction': f'High: {high_mean:.2f}, Low: {low_mean:.2f}'
            })

        # Mobile fragmentation
        high_data = high_mobile[emotion].dropna()
        low_data = low_mobile[emotion].dropna()
        t_stat, p_value = stats.ttest_ind(high_data, low_data)
        if p_value < threshold:
            high_mean = high_data.mean()
            low_mean = low_data.mean()
            results.append({
                'Analysis': 'T-test (Mobile)',
                'Fragmentation': 'Mobile_index',
                'Emotion': emotion,
                'Statistic': f'{t_stat:.3f}',
                'P-value': f'{p_value:.4f}',
                'Direction': f'High: {high_mean:.2f}, Low: {low_mean:.2f}'
            })

    return pd.DataFrame(results)



def calculate_descriptive_statistics(data, column):
    return {
        'Count': len(data),
        'Mean': round(data[column].mean(), 2),
        'Median': round(data[column].median(), 2),
        'Std Dev': round(data[column].std(), 2),
        'Min': round(data[column].min(), 2),
        'Max': round(data[column].max(), 2)
    }

def create_population_statistics(merged_data):
    frag_columns = ['Stationary_index', 'Mobile_index', 'Stationary_AID', 'Mobile_AID']
    results = []

    for col in frag_columns:
        median_value = merged_data[col].median()
        high_data = merged_data[merged_data[col] > median_value]
        low_data = merged_data[merged_data[col] <= median_value]

        high_stats = calculate_descriptive_statistics(high_data, col)
        low_stats = calculate_descriptive_statistics(low_data, col)

        # Calculate total mobile duration for high and low groups
        high_mobile_duration = high_data['mobile_duration'].sum()
        low_mobile_duration = low_data['mobile_duration'].sum()

        results.append({
            'Fragmentation': col,
            'Median Cut-off': round(median_value, 2),
            'High Population Count': high_stats['Count'],
            'Low Population Count': low_stats['Count'],
            'High Mean': high_stats['Mean'],
            'Low Mean': low_stats['Mean'],
            'High Median': high_stats['Median'],
            'Low Median': low_stats['Median'],
            'High Std Dev': high_stats['Std Dev'],
            'Low Std Dev': low_stats['Std Dev'],
            'High Min': high_stats['Min'],
            'Low Min': low_stats['Min'],
            'High Max': high_stats['Max'],
            'Low Max': low_stats['Max'],
            'High Total Mobile Duration': round(high_mobile_duration, 2),
            'Low Total Mobile Duration': round(low_mobile_duration, 2)
        })

    return pd.DataFrame(results)

def create_median_cutoff_chart(population_stats):
    frag_columns = population_stats['Fragmentation']
    median_cutoffs = population_stats['Median Cut-off']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(frag_columns, median_cutoffs)
    plt.title('Median Cut-off Values for Fragmentation Indices')
    plt.xlabel('Fragmentation Index')
    plt.ylabel('Median Cut-off Value')
    plt.xticks(rotation=45)

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(get_output_path('median_cutoff_chart.png'))
    print(f"Median cut-off chart saved: {get_output_path('median_cutoff_chart.png')}")
    plt.close()

def create_spider_web_chart(high_data, low_data, labels):
    num_vars = len(labels)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    max_value = max(max(high_data), max(low_data))

    high_data += high_data[:1]
    ax.plot(angles, high_data, 'o-', linewidth=2, label='High Fragmentation')
    ax.fill(angles, high_data, alpha=0.25)

    low_data += low_data[:1]
    ax.plot(angles, low_data, 'o-', linewidth=2, label='Low Fragmentation')
    ax.fill(angles, low_data, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max_value)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Emotional Scores: High vs Low Fragmentation")
    plt.tight_layout()
    plt.savefig(get_output_path('spider_web_chart.png'))
    print(f"Spider Web Chart saved: {get_output_path('spider_web_chart.png')}")
    plt.close()
    
    
    

def main():
    output_capture = None
    try:
        output_file = get_output_path('analysis_output.txt')
        output_capture = OutputCapture(output_file)
        sys.stdout = output_capture
        print(f"Console output is being saved to: {output_file}")

        # Load the fragmentation results
        frag_results = pd.read_csv('/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/fragmentation_daily_summary.csv')
        frag_results = round_and_abs_aid(frag_results)
        print("Fragmentation results shape:", frag_results.shape)
        print("Fragmentation results columns:", frag_results.columns)
        print("Fragmentation results sample:\n", frag_results.head())

        # Load the survey responses
        survey_responses = pd.read_excel('/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/Survey/End_of_the_day_questionnaire.xlsx')
        survey_responses = round_and_abs_aid(survey_responses)
        print("\nSurvey responses shape:", survey_responses.shape)
        print("Survey responses columns:", survey_responses.columns)
        print("Survey responses sample:\n", survey_responses.head())

        # Convert 'date' column in frag_results to datetime
        frag_results['date'] = pd.to_datetime(frag_results['date']).dt.date

        # Convert 'StartDate' column in survey_responses to datetime and extract date
        survey_responses['date'] = pd.to_datetime(survey_responses['StartDate']).dt.date

        # Ensure participant_id is of the same type in both datasets
        frag_results['participant_id'] = frag_results['participant_id'].astype(int)
        survey_responses['participant_id'] = survey_responses['Participant_ID'].astype(int)

        print("\nNumber of unique participants in fragmentation results:", frag_results['participant_id'].nunique())
        print("Number of unique participants in survey responses:", survey_responses['participant_id'].nunique())
        print("Total number of fragmentation records:", len(frag_results))
        print("Total number of survey responses:", len(survey_responses))

        # Merge the datasets
        merged_data = pd.merge(frag_results, survey_responses,
                               on=['participant_id', 'date'],
                               how='inner')
        merged_data = round_and_abs_aid(merged_data)

        print("\nMerged data shape:", merged_data.shape)
        print("Number of unique participants in merged data:", merged_data['participant_id'].nunique())
        print("Total number of matched records:", len(merged_data))
        print("Merged data columns:", merged_data.columns)
        print("Merged data sample:\n", merged_data.head())

        # Ensure emotional scores are numeric
        emotion_columns = ['PEACE', 'TENSE', 'IRRITATION', 'RELAXATION', 'SATISFACTION', 'WORRY', 'HAPPY']
        for col in emotion_columns:
            merged_data[col] = pd.to_numeric(merged_data[col], errors='coerce')

        # Print diagnostics for emotional scores and fragmentation indices
        frag_columns = ['Stationary_index', 'Mobile_index', 'Stationary_AID', 'Mobile_AID']
        for col in emotion_columns + frag_columns:
            print(f"\n{col}:")
            print("Number of values:", merged_data[col].count())
            print("Mean:", round(merged_data[col].mean(), 2))
            print("Median:", round(merged_data[col].median(), 2))
            print("Standard deviation:", round(merged_data[col].std(), 2))
            print("Min:", round(merged_data[col].min(), 2))
            print("Max:", round(merged_data[col].max(), 2))

        # Visualize distributions
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle('Distributions of Emotional Scores and Fragmentation Indices')
        for i, col in enumerate(emotion_columns + frag_columns):
            sns.histplot(merged_data[col], ax=axes[i//4, i%4], kde=True)
            axes[i//4, i%4].set_title(col)
        plt.tight_layout()
        plt.savefig(get_output_path('distributions.png'))
        print(f"Distributions plot saved: {get_output_path('distributions.png')}")
        plt.close()

        # Calculate correlation matrix and p-values
        columns_for_correlation = frag_columns + emotion_columns
        correlation_matrix = merged_data[columns_for_correlation].corr().round(2)

        def calculate_pvalues(df):
            df = df.dropna()._get_numeric_data()
            dfcols = pd.DataFrame(columns=df.columns)
            pvalues = dfcols.transpose().join(dfcols, how='outer')
            for r in df.columns:
                for c in df.columns:
                    pvalues.loc[r,c] = round(stats.pearsonr(df[r], df[c])[1], 4)
            return pvalues

        p_values = calculate_pvalues(merged_data[columns_for_correlation])

        # Create heatmap of correlation matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f')
        plt.title('Correlation Heatmap: Fragmentation Indices vs. Emotional Well-being')
        plt.tight_layout()
        plt.savefig(get_output_path('correlation_heatmap.png'))
        print(f"Correlation heatmap saved: {get_output_path('correlation_heatmap.png')}")
        plt.close()

        print("\nCorrelation matrix:\n", correlation_matrix)

        # Calculate median values for fragmentation indices
        median_stationary = merged_data['Stationary_index'].median()
        median_mobile = merged_data['Mobile_index'].median()

        # Split data into high and low fragmentation groups
        high_stationary = merged_data[merged_data['Stationary_index'] > median_stationary]
        low_stationary = merged_data[merged_data['Stationary_index'] <= median_stationary]
        high_mobile = merged_data[merged_data['Mobile_index'] > median_mobile]
        low_mobile = merged_data[merged_data['Mobile_index'] <= median_mobile]

        # Print group sizes and median values for debugging
        print("\nGroup sizes and median values:")
        print(f"Stationary median: {median_stationary:.4f}")
        print(f"High Stationary: {len(high_stationary)}")
        print(f"Low Stationary: {len(low_stationary)}")
        print(f"Mobile median: {median_mobile:.4f}")
        print(f"High Mobile: {len(high_mobile)}")
        print(f"Low Mobile: {len(low_mobile)}")

        # Create and save all t-test results
        all_ttest_results = create_all_ttest_results(high_stationary, low_stationary, high_mobile, low_mobile)
        all_ttest_results_path = get_output_path('all_ttest_results.csv')
        all_ttest_results.to_csv(all_ttest_results_path, index=False)
        print(f"All t-test results saved: {all_ttest_results_path}")

        # Print all t-test results
        print("\nAll T-test Results:")
        print(all_ttest_results.to_string(index=False))

        # Create comprehensive analysis table
        comprehensive_table = create_comprehensive_analysis_table(
            correlation_matrix, p_values, high_stationary, low_stationary, high_mobile, low_mobile
        )

        # Save comprehensive analysis table
        comprehensive_table_path = get_output_path('comprehensive_analysis.csv')
        comprehensive_table.to_csv(comprehensive_table_path, index=False)
        print(f"Comprehensive analysis table saved: {comprehensive_table_path}")

        # Print comprehensive analysis table
        print("\nComprehensive Analysis Results:")
        print(comprehensive_table.to_string(index=False))
        
        # Create population statistics
        population_stats = create_population_statistics(merged_data)
        population_stats_path = get_output_path('population_statistics.csv')
        population_stats.to_csv(population_stats_path, index=False)
        print(f"Population statistics saved: {population_stats_path}")

        # Print population statistics
        print("\nPopulation Statistics:")
        print(population_stats.to_string(index=False))

        # Create and save median cut-off chart
        create_median_cutoff_chart(population_stats)

        # Create Spider Web Chart
        print("\nCreating Spider Web Chart...")
        high_frag_means = high_stationary[emotion_columns].mean().round(2).tolist()
        low_frag_means = low_stationary[emotion_columns].mean().round(2).tolist()
        create_spider_web_chart(high_frag_means, low_frag_means, emotion_columns)

        print("Analysis complete. Check the output folder for results.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        # Restore the original stdout and close the output file
        if output_capture:
            sys.stdout = output_capture.terminal
            output_capture.close()

        # Make sure all plots are closed
        plt.close('all')

        print("Script execution completed.")

if __name__ == "__main__":
    main()
    
    