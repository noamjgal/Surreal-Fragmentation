import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Load the fragmentation results
frag_results = pd.read_csv('/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/fragmentation_daily_summary.csv')

# Load the survey responses
survey_responses = pd.read_excel('/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/Survey/End_of_the_day_questionnaire.xlsx')

# Data preprocessing
frag_results['date'] = pd.to_datetime(frag_results['date']).dt.date
survey_responses['date'] = pd.to_datetime(survey_responses['StartDate']).dt.date
frag_results['participant_id'] = frag_results['participant_id'].astype(int)
survey_responses['participant_id'] = survey_responses['Participant_ID'].astype(int)

# Merge the datasets
merged_data = pd.merge(frag_results, survey_responses,
                       on=['participant_id', 'date'],
                       how='inner')

# Ensure emotional scores are numeric
emotion_columns = ['PEACE', 'TENSE', 'IRRITATION', 'RELAXATION', 'SATISFACTION', 'WORRY', 'HAPPY']
for col in emotion_columns:
    merged_data[col] = pd.to_numeric(merged_data[col], errors='coerce')

# Define fragmentation indices
frag_columns = ['Stationary_index', 'Mobile_index', 'Indoor_index', 'Outdoor_index', 'Digital Usage_index', 'Zero-Digital Usage_index']

# Function to perform t-test and calculate effect size
def perform_ttest(high_group, low_group, emotion):
    t_stat, p_value = stats.ttest_ind(high_group[emotion].dropna(), low_group[emotion].dropna())
    effect_size = (high_group[emotion].mean() - low_group[emotion].mean()) / np.sqrt((high_group[emotion].std()**2 + low_group[emotion].std()**2) / 2)
    return t_stat, p_value, effect_size

# Perform t-tests for each fragmentation index
results = []
for frag_index in frag_columns:
    median = merged_data[frag_index].median()
    high_frag = merged_data[merged_data[frag_index] > median]
    low_frag = merged_data[merged_data[frag_index] <= median]
    
    for emotion in emotion_columns:
        t_stat, p_value, effect_size = perform_ttest(high_frag, low_frag, emotion)
        results.append({
            'Fragmentation Index': frag_index,
            'Emotion': emotion,
            't-statistic': t_stat,
            'p-value': p_value,
            'Effect Size': effect_size,
            'High Group Mean': high_frag[emotion].mean(),
            'Low Group Mean': low_frag[emotion].mean()
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Filter significant results (p < 0.05)
significant_results = results_df[results_df['p-value'] < 0.05].sort_values('p-value')

# Create HTML table for significant results
html_table = """
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: center;">
      <th>Fragmentation Index</th>
      <th>Emotion</th>
      <th>t-statistic</th>
      <th>p-value</th>
      <th>Effect Size</th>
      <th>High Group Mean</th>
      <th>Low Group Mean</th>
    </tr>
  </thead>
  <tbody>
"""

for _, row in significant_results.iterrows():
    html_table += f"""
    <tr>
      <td>{row['Fragmentation Index']}</td>
      <td>{row['Emotion']}</td>
      <td>{row['t-statistic']:.4f}</td>
      <td>{row['p-value']:.4f}</td>
      <td>{row['Effect Size']:.4f}</td>
      <td>{row['High Group Mean']:.4f}</td>
      <td>{row['Low Group Mean']:.4f}</td>
    </tr>
    """

html_table += """
  </tbody>
</table>
"""

# Define the output directory
output_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/analysis_results'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Save HTML table to file
html_file_path = os.path.join(output_dir, 'significant_results_table.html')
with open(html_file_path, 'w') as f:
    f.write(html_table)

print(f"Significant results table saved as '{html_file_path}'")

# Visualize distributions of emotional scores
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle('Distributions of Emotional Scores (Likert Scale 1-4)')
for i, emotion in enumerate(emotion_columns):
    ax = axes[i//3, i%3]
    sns.histplot(merged_data[emotion], ax=ax, kde=False, bins=[0.5, 1.5, 2.5, 3.5, 4.5])
    ax.set_title(emotion)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlim(0.5, 4.5)
    if i >= 6:  # Remove the last two empty subplots
        fig.delaxes(axes[2, 1])
        fig.delaxes(axes[2, 2])
plt.tight_layout()
emotion_dist_path = os.path.join(output_dir, 'emotion_distributions.png')
plt.savefig(emotion_dist_path)
plt.close()

print(f"Emotion distributions saved as '{emotion_dist_path}'")

# Visualize distributions of fragmentation indices
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distributions of Fragmentation Indices')
for i, frag_index in enumerate(frag_columns):
    sns.histplot(merged_data[frag_index], ax=axes[i//3, i%3], kde=True)
    axes[i//3, i%3].set_title(frag_index)
plt.tight_layout()
frag_dist_path = os.path.join(output_dir, 'fragmentation_distributions.png')
plt.savefig(frag_dist_path)
plt.close()

print(f"Fragmentation distributions saved as '{frag_dist_path}'")

# Calculate correlation matrix
correlation_matrix = merged_data[frag_columns + emotion_columns].corr()

# Function to calculate correlation p-value
def corr_pvalue(x, y):
    return stats.pearsonr(x, y)[1]

# Calculate p-values for correlations
p_values = merged_data[frag_columns + emotion_columns].corr(method=lambda x, y: corr_pvalue(x, y))

# Create a heatmap of the correlation matrix with significance indicators
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f')
for i in range(len(correlation_matrix)):
    for j in range(len(correlation_matrix)):
        if i != j:
            plt.text(j+0.5, i+0.5, 
                     '**' if p_values.iloc[i, j] < 0.01 else '*' if p_values.iloc[i, j] < 0.05 else '',
                     ha='center', va='center')
plt.title('Correlation Heatmap: Fragmentation Indices vs. Emotional Well-being\n* p<0.05, ** p<0.01')
plt.tight_layout()
heatmap_path = os.path.join(output_dir, 'correlation_heatmap.png')
plt.savefig(heatmap_path)
plt.close()

print(f"Correlation heatmap saved as '{heatmap_path}'")

# Print correlation between Mobile_index and WORRY
mobile_worry_corr = correlation_matrix.loc['Mobile_index', 'WORRY']
mobile_worry_pvalue = p_values.loc['Mobile_index', 'WORRY']
print(f"\nCorrelation between Mobile_index and WORRY:")
print(f"Correlation coefficient: {mobile_worry_corr:.4f}")
print(f"P-value: {mobile_worry_pvalue:.4f}")

print(f"\nAll analysis results have been saved in the directory: {output_dir}")