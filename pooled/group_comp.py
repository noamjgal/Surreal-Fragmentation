#!/usr/bin/env python3
"""
Simplified Group Comparison T-Test Script with Confidence Intervals

This script performs t-tests using standardized data from the pooled dataset (SURREAL and TLV),
properly accounting for unique participants across datasets and includes 95% confidence intervals.
"""
import sys
print(sys.executable)  # This will show which Python is actually running
print(sys.path)        # This will show the module search paths

import pandas as pd
import numpy as np
from scipy import stats
import logging
from datetime import datetime
import warnings
import os

class SimplifiedGroupAnalysis:
    def __init__(self, output_dir=None, debug=False):
        """Initialize the group comparison analysis class."""
        # Set paths for pooled data
        self.population_file = "processed/pooled_stai_data_population.csv"
        
        # Set output directory
        if output_dir:
            self.output_dir = os.path.join(output_dir)
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.output_dir = os.path.join(script_dir, "results", "group_comparison")
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.debug = debug
        self._setup_logging()
        
        # Define metrics for the pooled dataset - only the ones specified
        self.fragmentation_metrics = [
            'digital_fragmentation', 
            'mobility_fragmentation', 
            'overlap_fragmentation',
            'digital_home_fragmentation',
            'digital_home_mobility_delta'
        ]
        
        # Define demographic variables
        self.demographic_vars = [
            'gender_standardized',  # female/male
            'location_type',        # city_center/suburb
            'age_group'             # adult/adolescent
        ]
        
        # Results container
        self.ttest_results = []
        
    def _setup_logging(self):
        """Set up logging configuration"""
        log_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'group_analysis_{timestamp}.log')
        
        log_level = logging.DEBUG if self.debug else logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing simplified group comparison analysis")
        self.logger.info(f"Population-normalized data: {self.population_file}")
        self.logger.info(f"Output directory: {self.output_dir}")

    def load_data(self):
        """Load population-normalized pooled data."""
        # Load population-normalized data
        self.logger.info(f"Loading population-normalized data from {self.population_file}")
        try:
            population_df = pd.read_csv(self.population_file)
            self.logger.info(f"Population data loaded with shape: {population_df.shape}")
            
            # Create unique participant identifiers by combining dataset and participant ID
            population_df['unique_participant_id'] = population_df['dataset_source'] + '_' + population_df['participant_id'].astype(str)
            self.logger.info(f"Created unique participant IDs: {population_df['unique_participant_id'].nunique()} unique participants")
            
        except Exception as e:
            self.logger.error(f"Error loading population data: {str(e)}")
            population_df = None
        
        return population_df
    
    def run_analyses(self):
        """Run t-tests accounting for unique participants."""
        # Load dataset
        population_df = self.load_data()
        
        # Validate dataset
        if population_df is None:
            self.logger.error("Population dataset failed to load, cannot continue with t-tests")
            return False
        
        # Run analyses on the full pooled dataset
        self.logger.info("Running analyses on complete pooled dataset")
        
        # Process population-normalized data for t-tests
        if population_df is not None:
            self.logger.info("Running t-tests on population-normalized data")
            for outcome_var in self.fragmentation_metrics:
                for demo_var in self.demographic_vars:
                    self.logger.info(f"Testing {outcome_var} by {demo_var}")
                    result = self._run_t_test_comparison_with_unique_ids(population_df, outcome_var, demo_var)
                    if result:
                        self.ttest_results.append({
                            'dv': outcome_var,
                            'predictor': demo_var,
                            'n_obs': result.get('total_n', 0),
                            **self._extract_t_test_stats(result)
                        })
        
        self.logger.info(f"Completed all analyses. Generated {len(self.ttest_results)} t-test results.")
        
        return True
    
    def _extract_t_test_stats(self, result):
        """Extract statistics from t-test result into a flattened format."""
        stats = {}
        
        # Add group-specific statistics
        for group_num, group_label in [(1, 'group1'), (2, 'group2')]:
            group_name = result.get(f'{group_label}', '')
            stats[f'group{group_num}_name'] = group_name
            stats[f'group{group_num}_n'] = result.get(f'{group_label}_n', 0)
            stats[f'group{group_num}_mean'] = result.get(f'{group_label}_mean', np.nan)
            stats[f'group{group_num}_std'] = result.get(f'{group_label}_std', np.nan)
        
        # Add test statistics
        stats['t_statistic'] = result.get('statistic', np.nan)
        stats['p_value'] = result.get('p_value', np.nan)
        stats['effect_size'] = result.get('effect_size', np.nan)
        
        # Add confidence interval
        stats['ci_lower'] = result.get('ci_lower', np.nan)
        stats['ci_upper'] = result.get('ci_upper', np.nan)
        
        # Add significance indicator
        p_val = result.get('p_value', 1.0)
        if p_val < 0.001:
            stats['sig_level'] = '***'
        elif p_val < 0.01:
            stats['sig_level'] = '**'
        elif p_val < 0.05:
            stats['sig_level'] = '*'
        elif p_val < 0.1:
            stats['sig_level'] = '†'
        else:
            stats['sig_level'] = ''
        
        return stats
    
    def _run_t_test_comparison_with_unique_ids(self, df, outcome_var, group_var):
        """Run a t-test comparison between groups on an outcome variable, using unique participant IDs."""
        try:
            # Validate columns
            if outcome_var not in df.columns or group_var not in df.columns:
                self.logger.error(f"Columns {outcome_var} or {group_var} not found in dataframe")
                return None
            
            # Check for sufficient non-missing values
            if df[outcome_var].isna().sum() > 0.5 * len(df) or df[group_var].isna().sum() > 0.5 * len(df):
                self.logger.error(f"Too many missing values in {outcome_var} or {group_var}")
                return None
            
            # Get groups
            groups = df[group_var].dropna().unique()
            if len(groups) < 2:
                self.logger.error(f"Not enough groups in {group_var}: {groups}")
                return None
            
            # For more than 2 groups, use the first two
            if len(groups) > 2:
                groups = groups[:2]
                
            # Aggregate data by unique participant to account for repeated measurements
            self.logger.info(f"Aggregating data by unique participant for {outcome_var} by {group_var}")
            
            # Ensure unique_participant_id is available
            if 'unique_participant_id' not in df.columns:
                self.logger.error("unique_participant_id column not found, cannot aggregate data")
                return None
                
            # Aggregate data by unique participant ID and group
            # Using mean of each measure for each participant
            participant_means = df.groupby(['unique_participant_id', group_var])[outcome_var].mean().reset_index()
            
            # Get data for each group at the participant level
            g1_data = participant_means[participant_means[group_var] == groups[0]][outcome_var].dropna()
            g2_data = participant_means[participant_means[group_var] == groups[1]][outcome_var].dropna()
            
            # Minimum observations check
            if len(g1_data) < 3 or len(g2_data) < 3:
                self.logger.info(f"Insufficient participants for comparison between {groups[0]} and {groups[1]} groups")
                return None
            
            # Run t-test on participant-level means
            t_stat, p_val = stats.ttest_ind(g1_data, g2_data, equal_var=False)
            
            # Calculate effect size (Cohen's d)
            g1_mean, g1_std = g1_data.mean(), g1_data.std()
            g2_mean, g2_std = g2_data.mean(), g2_data.std()
            
            # Pooled standard deviation
            pooled_std = np.sqrt(((len(g1_data) - 1) * g1_std**2 + 
                                 (len(g2_data) - 1) * g2_std**2) / 
                                (len(g1_data) + len(g2_data) - 2))
            
            # Cohen's d
            cohen_d = abs(g1_mean - g2_mean) / pooled_std if pooled_std != 0 else np.nan
            
            # Calculate 95% confidence interval for mean difference
            # Standard error of the difference between means
            se_diff = np.sqrt((g1_std**2 / len(g1_data)) + (g2_std**2 / len(g2_data)))
            
            # Degrees of freedom using Welch-Satterthwaite equation for unequal variances
            df_welch = ((g1_std**2 / len(g1_data) + g2_std**2 / len(g2_data))**2) / \
                      ((g1_std**2 / len(g1_data))**2 / (len(g1_data) - 1) + 
                       (g2_std**2 / len(g2_data))**2 / (len(g2_data) - 1))
            
            # Critical t-value for 95% confidence interval
            t_crit = stats.t.ppf(0.975, df_welch)  # 0.975 for 95% CI (two-tailed)
            
            # Mean difference
            mean_diff = g1_mean - g2_mean
            
            # Calculate confidence interval
            margin = t_crit * se_diff
            ci_lower = mean_diff - margin
            ci_upper = mean_diff + margin
            
            result = {
                'outcome': outcome_var,
                'group_variable': group_var,
                'statistic': float(t_stat),
                'p_value': float(p_val),
                'effect_size': float(cohen_d),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'group1': str(groups[0]),
                'group1_mean': float(g1_mean),
                'group1_std': float(g1_std),
                'group1_n': int(len(g1_data)),
                'group2': str(groups[1]),
                'group2_mean': float(g2_mean),
                'group2_std': float(g2_std),
                'group2_n': int(len(g2_data)),
                'total_n': int(len(g1_data) + len(g2_data))
            }
            
            self.logger.info(
                f"Participant-level comparison of {outcome_var} between {group_var} groups: "
                f"{groups[0]} (n={len(g1_data)}, mean={g1_mean:.4f}) vs "
                f"{groups[1]} (n={len(g2_data)}, mean={g2_mean:.4f}), "
                f"t={t_stat:.4f}, p={p_val:.4f}, d={cohen_d:.4f}, "
                f"95% CI [{ci_lower:.4f}, {ci_upper:.4f}]"
            )
            
            return result
                
        except Exception as e:
            self.logger.error(f"Error comparing {outcome_var} by {group_var} groups: {str(e)}")
            if self.debug:
                import traceback
                self.logger.error(traceback.format_exc())
            
            return None
    
    def _create_beautified_table(self, df):
        """Create a beautified version of the results for presentation."""
        # Map column names to more readable versions
        column_mapping = {
            'digital_fragmentation': 'Digital Fragmentation',
            'mobility_fragmentation': 'Mobility Fragmentation',
            'overlap_fragmentation': 'Digital Mobile Fragmentation',
            'digital_home_fragmentation': 'Digital Home Fragmentation',
            'digital_home_mobility_delta': 'Digital Home Mobility Delta',
            'gender_standardized': 'Gender',
            'location_type': 'Location',
            'age_group': 'Age'
        }
        
        # Create a new dataframe for the beautified table
        rows = []
        
        # Process each t-test result
        for _, row in df.iterrows():
            dv = column_mapping.get(row['dv'], row['dv'])
            predictor = column_mapping.get(row['predictor'], row['predictor'])
            
            # Format means and SDs with 2 decimal places
            group1_mean_sd = f"{row['group1_mean']:.2f} ({row['group1_std']:.2f})"
            group2_mean_sd = f"{row['group2_mean']:.2f} ({row['group2_std']:.2f})"
            
            # Format t-statistic with 2 decimal places
            t_statistic = f"t({row['group1_n'] + row['group2_n'] - 2}) = {row['t_statistic']:.2f}"
            
            # Format p-value with 3 decimal places and add significance markers
            if row['p_value'] < 0.001:
                p_value = f"<.001{row['sig_level']}"
            else:
                p_value = f"{row['p_value']:.3f}{row['sig_level']}"
            
            # Format effect size with 2 decimal places
            effect_size = f"{row['effect_size']:.2f}"
            
            # Format confidence interval with 2 decimal places
            confidence_interval = f"[{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]"
            
            # Add row to results
            rows.append({
                'Measure': dv,
                'Groups': predictor,
                f"{row['group1_name']}": f"{row['group1_n']}",
                f"M (SD)": group1_mean_sd,
                f"{row['group2_name']}": f"{row['group2_n']}",
                f"M (SD)": group2_mean_sd,
                't(df)': t_statistic,
                'p': p_value,
                'd': effect_size,
                '95% CI': confidence_interval
            })
        
        # Create the dataframe
        beautified_df = pd.DataFrame(rows)
        
        return beautified_df

    def _create_descriptive_stats_table(self, df):
        """Create a descriptive statistics table for fragmentation metrics by age group at the observation level."""
        # Define the metrics to include (excluding digital_home_mobility_delta)
        metrics = ['digital_fragmentation', 'mobility_fragmentation', 
                  'overlap_fragmentation', 'digital_home_fragmentation']
        
        # Create a new dataframe for descriptive statistics
        rows = []
        
        # Process each metric
        for metric in metrics:
            # Get observation-level data for each age group
            adolescent_obs = df[df['age_group'] == 'adolescent'][metric].dropna()
            adult_obs = df[df['age_group'] == 'adult'][metric].dropna()
            all_obs = df[metric].dropna()
            
            # Calculate detailed statistics
            adolescent_mean = adolescent_obs.mean()
            adolescent_sd = adolescent_obs.std()
            adult_mean = adult_obs.mean()
            adult_sd = adult_obs.std()
            sd_ratio = adult_sd / adolescent_sd if adolescent_sd > 0 else float('inf')
            
            # Calculate statistics for each group at observation level
            stats = {
                'Measure': metric.replace('_', ' ').title(),
                'Adolescents N': f"{len(adolescent_obs)}",
                'Adolescents M (SD)': f"{adolescent_mean:.2f} ({adolescent_sd:.2f})",
                'Adults N': f"{len(adult_obs)}",
                'Adults M (SD)': f"{adult_mean:.2f} ({adult_sd:.2f})",
                'Total N': f"{len(all_obs)}",
                'Total M (SD)': f"{all_obs.mean():.2f} ({all_obs.std():.2f})",
                'Adult/Adol SD Ratio': f"{sd_ratio:.2f}",
                'Mean Diff': f"{adult_mean - adolescent_mean:.2f}"
            }
            rows.append(stats)
        
        # Create the dataframe
        desc_stats_df = pd.DataFrame(rows)
        
        return desc_stats_df

    def save_results(self, beautify=True):
        """Save results to CSV file and optionally create a beautified table version."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Load the data for descriptive statistics
        population_df = self.load_data()
        
        # Save t-test results to CSV
        if self.ttest_results:
            ttest_df = pd.DataFrame(self.ttest_results)
            
            # Round numeric columns
            numeric_cols = ttest_df.select_dtypes(include=[np.number]).columns
            ttest_df[numeric_cols] = ttest_df[numeric_cols].round(4)
            
            # Save to CSV
            csv_path = os.path.join(self.output_dir, f'group_ttest_results_{timestamp}.csv')
            ttest_df.to_csv(csv_path, index=False)
            self.logger.info(f"Saved {len(ttest_df)} t-test results to {csv_path}")
            
            # Create and save descriptive statistics table
            if population_df is not None:
                desc_stats_df = self._create_descriptive_stats_table(population_df)
                desc_stats_path = os.path.join(self.output_dir, f'descriptive_statistics_{timestamp}.csv')
                desc_stats_df.to_csv(desc_stats_path, index=False)
                self.logger.info(f"Saved descriptive statistics to {desc_stats_path}")
            
            # Create beautified table if requested
            if beautify:
                beautified_table = self._create_beautified_table(ttest_df)
                table_path = os.path.join(self.output_dir, f'group_ttest_beautiful_table_{timestamp}.csv')
                beautified_table.to_csv(table_path, index=False)
                self.logger.info(f"Saved beautified table to {table_path}")
            
            return csv_path
        else:
            self.logger.warning("No results to save")
            return None

def main():
    """Main function to run the group comparison analysis."""
    try:
        # Create analyzer
        analyzer = SimplifiedGroupAnalysis(debug=True)
        
        # Run analyses
        if analyzer.run_analyses():
            # Save results
            results_path = analyzer.save_results(beautify=True)
            
            print(f"Group comparison analysis completed successfully!")
            print(f"Results saved to: {results_path}")
            return 0
        else:
            print("Error: Failed to run analyses")
            return 1
        
    except Exception as e:
        print(f"Error running analysis: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return 1

if __name__ == "__main__":
    # Ignore certain warnings
    warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
    
    exit(main())