#!/usr/bin/env python3
"""
Pooled Group Comparison Analysis

This script performs t-tests using population-standardized data from the pooled STAI dataset (SURREAL and TLV),
properly accounting for repeated measurements by aggregating data at the participant level.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import logging
from datetime import datetime
import warnings

class PooledGroupAnalysis:
    def __init__(self, output_dir=None, debug=False):
        """Initialize the pooled group comparison analysis class."""
        # Set paths for pooled data
        self.population_file = Path("pooled/processed/pooled_stai_data_population.csv")
        self.participant_file = Path("pooled/processed/pooled_stai_data_participant.csv")
        
        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            script_dir = Path(__file__).parent
            self.output_dir = script_dir / "results" / "group_comparison"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug = debug
        self._setup_logging()
        
        # Define metrics for the pooled dataset
        self.fragmentation_metrics = [
            'digital_fragmentation', 
            'mobility_fragmentation', 
            'overlap_fragmentation',
            'digital_home_fragmentation',  # Added digital home fragmentation
            'digital_home_mobility_delta'  # Added digital home mobility delta
        ]
        
        self.episode_metrics = [
            'digital_episodes',
            'mobility_episodes',
            'overlap_episodes'
        ]
        
        self.duration_metrics = [
            'digital_duration',
            'mobility_duration',
            'overlap_duration',
            'digital_home_total_duration',  # Added digital home total duration
            'home_duration',               # Added home duration
            'active_transport_duration',    # Added active transport duration
            'mechanized_transport_duration',# Added mechanized transport duration
            'out_of_home_duration'          # Added out of home duration
        ]
        
        # Define emotion metrics - updated to only use standardized scores
        self.anxiety_metrics = ['anxiety_score_std']  # Only standardized anxiety score
        self.mood_metrics = ['mood_score_std']        # Only standardized mood/depression score
        
        # Define demographic variables
        self.demographic_vars = [
            'gender_standardized',  # female/male
            'location_type',        # city_center/suburb
            'age_group',            # adult/adolescent
            'dataset_source'        # surreal/tlv
        ]
        
        # Define subset analyses to run
        self.subsets = [
            {'name': 'tlv', 'filter_column': 'dataset_source', 'filter_value': 'tlv'},
            {'name': 'surreal', 'filter_column': 'dataset_source', 'filter_value': 'surreal'},
            {'name': 'adults', 'filter_column': 'age_group', 'filter_value': 'adult'},
            {'name': 'adolescents', 'filter_column': 'age_group', 'filter_value': 'adolescent'}
        ]
        
        # Results containers - now with subset tracking
        self.ttest_results = []
        
    def _setup_logging(self):
        """Set up logging configuration"""
        log_dir = self.output_dir / 'logs'
        log_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'pooled_group_analysis_{timestamp}.log'
        
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
        self.logger.info(f"Initializing pooled group comparison analysis")
        self.logger.info(f"Population-normalized data: {self.population_file}")
        self.logger.info(f"Participant-normalized data: {self.participant_file}")
        self.logger.info(f"Output directory: {self.output_dir}")

    def load_data(self):
        """Load population-normalized pooled data."""
        # Load population-normalized data
        self.logger.info(f"Loading population-normalized data from {self.population_file}")
        try:
            population_df = pd.read_csv(self.population_file)
            self.logger.info(f"Population data loaded with shape: {population_df.shape}")
        except Exception as e:
            self.logger.error(f"Error loading population data: {str(e)}")
            population_df = None
        
        return population_df
    
    def run_analyses(self):
        """Run t-tests accounting for repeated measurements."""
        # Load dataset
        population_df = self.load_data()
        
        # Validate dataset
        if population_df is None:
            self.logger.error("Population dataset failed to load, cannot continue with t-tests")
            return False
        
        # First run analyses on the full pooled dataset
        self.logger.info("Running analyses on complete pooled dataset")
        
        # Process population-normalized data for t-tests
        if population_df is not None:
            self.logger.info("Running t-tests on population-normalized data")
            pop_ttests = self._run_ttests(population_df, subset_name="pooled")
            self.ttest_results.extend(pop_ttests)
        
        # Now run analyses on each subset
        for subset in self.subsets:
            subset_name = subset['name']
            filter_column = subset['filter_column']
            filter_value = subset['filter_value']
            
            self.logger.info(f"Running analyses on {subset_name} subset")
            
            # Filter datasets for the current subset
            if population_df is not None:
                if filter_column in population_df.columns:
                    subset_pop_df = population_df[population_df[filter_column] == filter_value].copy()
                    self.logger.info(f"Filtered population dataset for {subset_name}: {len(subset_pop_df)} rows")
                    
                    if len(subset_pop_df) >= 10:  # Minimum sample size for meaningful analysis
                        self.logger.info(f"Running t-tests on {subset_name} population-normalized data")
                        subset_ttests = self._run_ttests(subset_pop_df, subset_name=subset_name)
                        self.ttest_results.extend(subset_ttests)
                    else:
                        self.logger.warning(f"Insufficient data in {subset_name} population subset ({len(subset_pop_df)} rows)")
                else:
                    self.logger.warning(f"Filter column {filter_column} not found in population dataset")
        
        self.logger.info(f"Completed all analyses. Generated {len(self.ttest_results)} t-test results.")
        
        return True
    
    def _run_ttests(self, df, subset_name="pooled"):
        """Run t-test comparisons on population-normalized data, accounting for repeated measurements."""
        results = []
        
        # Define metrics to analyze
        all_metrics = (
            self.fragmentation_metrics + 
            self.episode_metrics + 
            self.duration_metrics + 
            self.anxiety_metrics + 
            self.mood_metrics
        )
        
        # Focus on key demographic comparisons
        key_demographics = ['gender_standardized', 'location_type']
        
        # For subset-specific analyses, adjust the demographics list
        if subset_name == "tlv" or subset_name == "surreal":
            # Don't include dataset_source when already filtering by it
            if 'age_group' not in key_demographics:
                key_demographics.append('age_group')
        elif subset_name == "adults" or subset_name == "adolescents":
            # Don't include age_group when already filtering by it
            if 'dataset_source' not in key_demographics:
                key_demographics.append('dataset_source')
        else:
            # For pooled analysis, include both age_group and dataset_source
            if 'age_group' not in key_demographics:
                key_demographics.append('age_group')
            if 'dataset_source' not in key_demographics:
                key_demographics.append('dataset_source')
        
        # 1. Compare emotional and fragmentation metrics across demographic groups
        for outcome_var in all_metrics:
            if outcome_var not in df.columns:
                continue
                
            for demo_var in key_demographics:
                if demo_var not in df.columns:
                    continue
                    
                result = self._run_t_test_comparison_aggregated(df, outcome_var, demo_var)
                if result:
                    # Determine outcome category
                    outcome_category = self._determine_variable_category(outcome_var)
                    
                    # Add to results
                    results.append({
                        'model_name': f"{subset_name.capitalize()}: {outcome_var} ~ {demo_var}",
                        'dv': outcome_var,
                        'dv_category': outcome_category,
                        'predictor': demo_var,
                        'predictor_category': 'demographic',
                        'normalization': 'population',
                        'test_type': 't-test',
                        'subset': subset_name,
                        'n_obs': result.get('total_n', 0),
                        **self._extract_t_test_stats(result)
                    })
        
        # 2. Median-split analyses - focus on relationship between fragmentation and emotions
        behavioral_metrics = self.fragmentation_metrics
        emotional_metrics = self.anxiety_metrics + self.mood_metrics
        
        for predictor in behavioral_metrics:
            if predictor not in df.columns:
                continue
                
            for outcome in emotional_metrics:
                if outcome not in df.columns or predictor == outcome:
                    continue
                
                # Create median-split group at participant level
                # First, aggregate predictor at participant level
                participant_means = df.groupby('participant_id')[predictor].mean().reset_index()
                median_val = participant_means[predictor].median()
                
                # Create group labels
                participant_means[f'{predictor}_group'] = participant_means[predictor].apply(
                    lambda x: 'high' if x > median_val else 'low' if pd.notna(x) else np.nan
                )
                
                # Merge back to original data
                temp_df = df.merge(
                    participant_means[['participant_id', f'{predictor}_group']], 
                    on='participant_id', 
                    how='left'
                )
                
                # Run t-test with median split
                result = self._run_t_test_comparison_aggregated(temp_df, outcome, f'{predictor}_group')
                
                if result:
                    dv_category = 'anxiety' if 'anxiety' in outcome else 'mood'
                    predictor_category = 'fragmentation'
                    
                    results.append({
                        'model_name': f"{subset_name.capitalize()}: {outcome} ~ {predictor} (median split)",
                        'dv': outcome,
                        'dv_category': dv_category,
                        'predictor': predictor,
                        'predictor_category': predictor_category,
                        'normalization': 'population',
                        'test_type': 'median-split t-test',
                        'subset': subset_name,
                        'n_obs': result.get('total_n', 0),
                        **self._extract_t_test_stats(result)
                    })
        
        return results
    
    def _determine_variable_category(self, var_name):
        """Determine the category of a variable based on its name."""
        if var_name in self.anxiety_metrics:
            return "anxiety"
        elif var_name in self.mood_metrics:
            return "mood"
        elif var_name in self.fragmentation_metrics:
            return "fragmentation"
        elif var_name in self.episode_metrics:
            return "episode"
        elif var_name in self.duration_metrics:
            return "duration"
        else:
            return "other"
    
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
        stats['effect_size_type'] = "Cohen's d"
        
        # Add significance indicator
        p_val = result.get('p_value', 1.0)
        stats['sig_level'] = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        
        return stats
    
    def _run_t_test_comparison_aggregated(self, df, outcome_var, group_var):
        """Run a t-test comparison between groups on an outcome variable, aggregating by participant."""
        try:
            # Validate columns
            if outcome_var not in df.columns or group_var not in df.columns:
                return None
            
            # Check for sufficient non-missing values
            if df[outcome_var].isna().sum() > 0.5 * len(df) or df[group_var].isna().sum() > 0.5 * len(df):
                return None
            
            # Get groups
            groups = df[group_var].dropna().unique()
            if len(groups) < 2:
                return None
            
            # For more than 2 groups, use the first two
            if len(groups) > 2:
                groups = groups[:2]
                
            # First aggregate the data by participant to account for repeated measurements
            self.logger.info(f"Aggregating data by participant for {outcome_var} by {group_var}")
            
            # Ensure participant_id is available
            if 'participant_id' not in df.columns:
                self.logger.error("participant_id column not found, cannot aggregate data")
                return None
                
            # Aggregate data by participant and group
            participant_means = df.groupby(['participant_id', group_var])[outcome_var].mean().reset_index()
            
            # Get data for each group at the participant level
            g1_data = participant_means[participant_means[group_var] == groups[0]][outcome_var].dropna()
            g2_data = participant_means[participant_means[group_var] == groups[1]][outcome_var].dropna()
            
            # Minimum observations check
            if len(g1_data) < 5 or len(g2_data) < 5:
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
            
            result = {
                'outcome': outcome_var,
                'group_variable': group_var,
                'statistic': float(t_stat),
                'p_value': float(p_val),
                'effect_size': float(cohen_d),
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
                f"{groups[0]} (n={len(g1_data)}, mean={g1_mean:.2f}) vs "
                f"{groups[1]} (n={len(g2_data)}, mean={g2_mean:.2f}), "
                f"t={t_stat:.2f}, p={p_val:.4f}, d={cohen_d:.2f}"
            )
            
            return result
                
        except Exception as e:
            self.logger.error(f"Error comparing {outcome_var} by {group_var} groups: {str(e)}")
            if self.debug:
                import traceback
                self.logger.error(traceback.format_exc())
            
            return None
    
    def save_results(self):
        """Save results to Excel file for t-tests"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save t-test results
        if self.ttest_results:
            ttest_df = pd.DataFrame(self.ttest_results)
            
            # Round numeric columns
            numeric_cols = ttest_df.select_dtypes(include=[np.number]).columns
            ttest_df[numeric_cols] = ttest_df[numeric_cols].round(4)
            
            # Save to Excel
            ttest_path = self.output_dir / f'pooled_ttest_results_{timestamp}.xlsx'
            
            with pd.ExcelWriter(ttest_path) as writer:
                # All results
                ttest_df.to_excel(writer, sheet_name='All T-Tests', index=False)
                
                # Filter by subsets
                for subset in ['pooled'] + [s['name'] for s in self.subsets]:
                    subset_data = ttest_df[ttest_df['subset'] == subset]
                    if not subset_data.empty:
                        sheet_name = f'{subset.capitalize()} Tests'
                        subset_data.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # By outcome category
                for category in sorted(ttest_df['dv_category'].unique()):
                    cat_subset = ttest_df[ttest_df['dv_category'] == category]
                    if not cat_subset.empty:
                        cat_subset.to_excel(writer, sheet_name=f'{category.capitalize()}', index=False)
                
                # Special adult vs adolescent comparison summary
                age_comparisons = ttest_df[ttest_df['predictor'] == 'age_group']
                if not age_comparisons.empty:
                    age_comparisons.to_excel(writer, sheet_name='Age Group Comparisons', index=False)
                
                # Significant results
                sig_results = ttest_df[ttest_df['p_value'] < 0.05]
                if not sig_results.empty:
                    sig_results.to_excel(writer, sheet_name='Significant', index=False)
                    
                # Top effect sizes
                top_effects = ttest_df.sort_values(by='effect_size', ascending=False).head(20)
                if not top_effects.empty:
                    top_effects.to_excel(writer, sheet_name='Top Effects', index=False)
            
            self.logger.info(f"Saved {len(ttest_df)} t-test results to {ttest_path}")
        
        # Create a summary file
        self._create_summary_file(timestamp)
        
        return True
    
    def _create_summary_file(self, timestamp):
        """Create a summary file with key findings"""
        summary_data = []
        
        # For each subset, create a summary section
        for subset in ['pooled'] + [s['name'] for s in self.subsets]:
            # Add a separator row
            summary_data.append({
                'Analysis Type': f"--- {subset.upper()} SUBSET SUMMARY ---",
                'Total Count': "",
                'Significant Count': "",
                'Significant %': "",
                'Notes': ""
            })
            
            # T-test summary for this subset
            if self.ttest_results:
                ttest_df = pd.DataFrame(self.ttest_results)
                subset_ttests = ttest_df[ttest_df['subset'] == subset]
                
                if not subset_ttests.empty:
                    total_ttests = len(subset_ttests)
                    sig_ttests = len(subset_ttests[subset_ttests['p_value'] < 0.05])
                    sig_percent = sig_ttests / total_ttests * 100 if total_ttests > 0 else 0
                    
                    summary_data.append({
                        'Analysis Type': f'T-tests ({subset})',
                        'Total Count': total_ttests,
                        'Significant Count': sig_ttests,
                        'Significant %': f"{sig_percent:.1f}%",
                        'Notes': 'Group comparisons'
                    })
                    
                    # Breakdown by outcome category
                    for category in sorted(subset_ttests['dv_category'].unique()):
                        cat_subset = subset_ttests[subset_ttests['dv_category'] == category]
                        cat_total = len(cat_subset)
                        cat_sig = len(cat_subset[cat_subset['p_value'] < 0.05])
                        cat_percent = cat_sig / cat_total * 100 if cat_total > 0 else 0
                        
                        summary_data.append({
                            'Analysis Type': f'T-tests ({subset}: {category})',
                            'Total Count': cat_total,
                            'Significant Count': cat_sig,
                            'Significant %': f"{cat_percent:.1f}%",
                            'Notes': ''
                        })
                    
                    # Top t-test effects for this subset
                    top_ttests = subset_ttests.sort_values(by='effect_size', ascending=False).head(3)
                    for i, row in top_ttests.iterrows():
                        summary_data.append({
                            'Analysis Type': f'Top {subset} t-test effect',
                            'Total Count': '',
                            'Significant Count': '',
                            'Significant %': f"d={row['effect_size']:.2f}",
                            'Notes': f"{row['dv']} by {row['predictor']}, p={row['p_value']:.4f}"
                        })
        
        # Create summary dataframe
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary to Excel
        summary_path = self.output_dir / f'pooled_analysis_summary_{timestamp}.xlsx'
        summary_df.to_excel(summary_path, index=False)
        
        self.logger.info(f"Saved analysis summary to {summary_path}")
        return summary_path

def main():
    """Main function to run the pooled group comparison analysis."""
    try:
        # Create analyzer
        analyzer = PooledGroupAnalysis(debug=True)
        
        # Run analyses
        if analyzer.run_analyses():
            # Save results
            analyzer.save_results()
            
            print(f"Pooled group comparison analysis completed successfully!")
            print(f"Results saved to: {analyzer.output_dir}")
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