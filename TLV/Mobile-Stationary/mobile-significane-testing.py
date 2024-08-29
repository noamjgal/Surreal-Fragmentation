import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the fragmentation results
frag_results = pd.read_csv('/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/fragmentation_daily_summary.csv')
print("Fragmentation results shape:", frag_results.shape)
print("Fragmentation results columns:", frag_results.columns)
print("Fragmentation results sample:\n", frag_results.head())

# Load the survey responses
survey_responses = pd.read_excel('/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/Survey/End_of_the_day_questionnaire.xlsx')
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

print("\nMerged data shape:", merged_data.shape)
print("Number of unique participants in merged data:", merged_data['participant_id'].nunique())
print("Total number of matched records:", len(merged_data))
print("Merged data columns:", merged_data.columns)
print("Merged data sample:\n", merged_data.head())

# Ensure emotional scores are numeric
emotion_columns = ['PEACE', 'TENSE', 'IRRITATION', 'RELAXATION', 'SATISFACTION', 'WORRY', 'HAPPY']
for col in emotion_columns:
    merged_data[col] = pd.to_numeric(merged_data[col], errors='coerce')

# Print diagnostics for emotional scores
print("\nDiagnostics for emotional scores:")
for col in emotion_columns:
    print(f"\n{col}:")
    print("Number of responses:", merged_data[col].count())
    print("Mean:", merged_data[col].mean())
    print("Median:", merged_data[col].median())
    print("Standard deviation:", merged_data[col].std())
    print("Min:", merged_data[col].min())
    print("Max:", merged_data[col].max())

# Print diagnostics for fragmentation indices
frag_columns = ['Stationary_index', 'Mobile_index', 'Stationary_AID', 'Mobile_AID']
print("\nDiagnostics for fragmentation indices:")
for col in frag_columns:
    print(f"\n{col}:")
    print("Number of values:", merged_data[col].count())
    print("Mean:", merged_data[col].mean())
    print("Median:", merged_data[col].median())
    print("Standard deviation:", merged_data[col].std())
    print("Min:", merged_data[col].min())
    print("Max:", merged_data[col].max())

# Visualize distributions of emotional scores and fragmentation indices
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle('Distributions of Emotional Scores and Fragmentation Indices')
for i, col in enumerate(emotion_columns + frag_columns):
    sns.histplot(merged_data[col], ax=axes[i//4, i%4], kde=True)
    axes[i//4, i%4].set_title(col)
plt.tight_layout()
plt.savefig('distributions.png')
plt.close()

# Select relevant columns for correlation analysis
columns_for_correlation = frag_columns + emotion_columns

# Calculate correlation matrix
correlation_matrix = merged_data[columns_for_correlation].corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Heatmap: Fragmentation Indices vs. Emotional Well-being')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

print("\nCorrelation matrix:\n", correlation_matrix)

# Save the merged data to a new Excel file
merged_data.to_excel('fragmentation_survey_merged.xlsx', index=False)
print("\nMerged data saved to 'fragmentation_survey_merged.xlsx'")

# Calculate average emotional scores for high and low fragmentation
median_stationary = merged_data['Stationary_index'].median()
median_mobile = merged_data['Mobile_index'].median()

high_stationary = merged_data[merged_data['Stationary_index'] > median_stationary]
low_stationary = merged_data[merged_data['Stationary_index'] <= median_stationary]
high_mobile = merged_data[merged_data['Mobile_index'] > median_mobile]
low_mobile = merged_data[merged_data['Mobile_index'] <= median_mobile]

print("\nAverage emotional scores for high vs low Stationary fragmentation:")
print(high_stationary[emotion_columns].mean())
print(low_stationary[emotion_columns].mean())

print("\nAverage emotional scores for high vs low Mobile fragmentation:")
print(high_mobile[emotion_columns].mean())
print(low_mobile[emotion_columns].mean())

# Additional analysis: T-tests for emotional differences between high and low fragmentation
print("\nT-tests for emotional differences between high and low Stationary fragmentation:")
for emotion in emotion_columns:
    t_stat, p_value = stats.ttest_ind(high_stationary[emotion].dropna(), low_stationary[emotion].dropna())
    print(f"{emotion}: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")

print("\nT-tests for emotional differences between high and low Mobile fragmentation:")
for emotion in emotion_columns:
    t_stat, p_value = stats.ttest_ind(high_mobile[emotion].dropna(), low_mobile[emotion].dropna())
    print(f"{emotion}: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")