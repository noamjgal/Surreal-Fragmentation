import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from scipy import stats
import logging
from datetime import datetime

class FragmentationGroupAnalysis:
    def __init__(self, input_path: str, output_dir: str):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        self.comparison_results = []
        
        # Define main metrics as class attribute
        self.fragmentation_vars = [
            'digital_fragmentation_index',
            'moving_fragmentation_index',
            'digital_frag_during_mobility',
            'digital_total_duration',
            'moving_total_duration'
        ]
        
        # Define grouping variables as class attribute
        self.group_vars = ['Gender', 'School', 'is_weekend', 'Class']
        
    def _setup_logging(self):
        log_path = self.output_dir / 'group_analysis.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        """Load and preprocess the data, keeping only good quality records"""
        self.logger.info(f"Loading data from {self.input_path}")
        df = pd.read_csv(self.input_path)
        #df = df[df['data_quality'] == 'good'].copy()
        self.data = df
        return df

    def compare_groups(self):
        """Perform group comparisons for fragmentation metrics"""
        for frag_var in self.fragmentation_vars:
            self.logger.info(f"\nAnalyzing {frag_var}")
            
            for group_var in self.group_vars:
                try:
                    # Skip if too many missing values
                    if self.data[frag_var].isna().sum() / len(self.data) > 0.5:
                        self.logger.warning(f"Skipping {frag_var} due to too many missing values")
                        continue
                    
                    groups = self.data[group_var].unique()
                    
                    if len(groups) == 2:  # Binary comparison
                        # Get data for each group
                        group1_data = self.data[self.data[group_var] == groups[0]][frag_var].dropna()
                        group2_data = self.data[self.data[group_var] == groups[1]][frag_var].dropna()
                        
                        # Perform t-test
                        t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
                        
                        # Calculate Cohen's d
                        pooled_std = np.sqrt((np.var(group1_data) + np.var(group2_data)) / 2)
                        cohens_d = (np.mean(group1_data) - np.mean(group2_data)) / pooled_std if pooled_std != 0 else np.nan
                        
                        result = {
                            'metric': frag_var,
                            'group_variable': group_var,
                            'test': 't-test',
                            'group1': groups[0],
                            'group2': groups[1],
                            'group1_mean': np.mean(group1_data),
                            'group2_mean': np.mean(group2_data),
                            'group1_std': np.std(group1_data),
                            'group2_std': np.std(group2_data),
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'cohens_d': cohens_d,
                            'n1': len(group1_data),
                            'n2': len(group2_data)
                        }
                        
                    else:  # Multi-group comparison (e.g., Class)
                        # Prepare data for ANOVA
                        group_data = [
                            self.data[self.data[group_var] == g][frag_var].dropna() 
                            for g in groups
                        ]
                        
                        # Perform one-way ANOVA
                        f_stat, p_value = stats.f_oneway(*group_data)
                        
                        # Calculate eta-squared
                        groups_concat = np.concatenate(group_data)
                        grand_mean = np.mean(groups_concat)
                        ss_total = np.sum((groups_concat - grand_mean) ** 2)
                        ss_between = np.sum([
                            len(g) * (np.mean(g) - grand_mean) ** 2 
                            for g in group_data
                        ])
                        eta_squared = ss_between / ss_total
                        
                        result = {
                            'metric': frag_var,
                            'group_variable': group_var,
                            'test': 'ANOVA',
                            'f_statistic': f_stat,
                            'p_value': p_value,
                            'eta_squared': eta_squared
                        }
                        
                        # Add group-specific statistics
                        for i, (g, data) in enumerate(zip(groups, group_data)):
                            result.update({
                                f'group{i+1}': g,
                                f'group{i+1}_mean': np.mean(data),
                                f'group{i+1}_std': np.std(data),
                                f'group{i+1}_n': len(data)
                            })
                    
                    self.comparison_results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Error analyzing {frag_var} by {group_var}: {str(e)}")

    def save_results(self):
        """Save results in a clean, simple format"""
        if not self.comparison_results:
            self.logger.warning("No results to save")
            return
            
        df_results = pd.DataFrame(self.comparison_results)
        
        # Sort by p-value for easier interpretation
        df_results = df_results.sort_values('p_value')
        
        # Round numeric columns for cleaner display
        numeric_cols = df_results.select_dtypes(include=[np.number]).columns
        df_results[numeric_cols] = df_results[numeric_cols].round(3)
        
        # Save to Excel with simple formatting
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = self.output_dir / f'group_comparisons_{timestamp}.xlsx'
        
        with pd.ExcelWriter(output_path) as writer:
            # Write main results
            df_results.to_excel(writer, sheet_name='Results', index=False)
            
            # Add basic summary statistics
            summary = self.data[self.fragmentation_vars].describe()
            summary.round(3).to_excel(writer, sheet_name='Summary Stats')
            
        self.logger.info(f"Saved results to {output_path}")

def main():
    
    input_path = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/metrics/combined_metrics.csv'
    output_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/analysis_results'
    
    
    analyzer = FragmentationGroupAnalysis(input_path, output_dir)
    analyzer.load_data()
    analyzer.compare_groups()
    analyzer.save_results()
    
    print("Group comparison analysis completed successfully")

if __name__ == "__main__":
    main()