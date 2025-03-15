#!/usr/bin/env python3
"""
Pooled Group Comparison Analysis

This script performs t-tests using population-standardized data and correlations using 
participant-standardized data from the pooled STAI dataset (SURREAL and TLV).
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
            'overlap_fragmentation'
        ]
        
        self.episode_metrics = [
            'digital_episodes',
            'mobility_episodes',
            'overlap_episodes'
        ]
        
        self.duration_metrics = [
            'digital_duration',
            'mobility_duration',
            'overlap_duration'
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
        self.correlation_results = []
        
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
        """Load both population-normalized and participant-normalized pooled data."""
        # Load population-normalized data
        self.logger.info(f"Loading population-normalized data from {self.population_file}")
        try:
            population_df = pd.read_csv(self.population_file)
            self.logger.info(f"Population data loaded with shape: {population_df.shape}")
        except Exception as e:
            self.logger.error(f"Error loading population data: {str(e)}")
            population_df = None
        
        # Load participant-normalized data
        self.logger.info(f"Loading participant-normalized data from {self.participant_file}")
        try:
            participant_df = pd.read_csv(self.participant_file)
            self.logger.info(f"Participant data loaded with shape: {participant_df.shape}")
        except Exception as e:
            self.logger.error(f"Error loading participant data: {str(e)}")
            participant_df = None
        
        return population_df, participant_df
    
    def run_analyses(self):
        """Run t-tests and correlations using appropriate datasets for each."""
        # Load both datasets
        population_df, participant_df = self.load_data()
        
        # Validate datasets
        if population_df is None:
            self.logger.error("Population dataset failed to load, cannot continue with t-tests")
            return False
        
        if participant_df is None:
            self.logger.error("Participant dataset failed to load, cannot continue with correlations")
            return False
        
        # First run analyses on the full pooled dataset
        self.logger.info("Running analyses on complete pooled dataset")
        
        # Process population-normalized data for t-tests
        if population_df is not None:
            self.logger.info("Running t-tests on population-normalized data")
            pop_ttests = self._run_ttests(population_df, subset_name="pooled")
            self.ttest_results.extend(pop_ttests)
        
        # Process participant-normalized data for correlations
        if participant_df is not None:
            self.logger.info("Running correlations on participant-normalized data")
            part_correlations = self._run_correlations(participant_df, subset_name="pooled")
            self.correlation_results.extend(part_correlations)
        
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
            
            if participant_df is not None:
                if filter_column in participant_df.columns:
                    subset_part_df = participant_df[participant_df[filter_column] == filter_value].copy()
                    self.logger.info(f"Filtered participant dataset for {subset_name}: {len(subset_part_df)} rows")
                    
                    if len(subset_part_df) >= 10:  # Minimum sample size for meaningful analysis
                        self.logger.info(f"Running correlations on {subset_name} participant-normalized data")
                        subset_correlations = self._run_correlations(subset_part_df, subset_name=subset_name)
                        self.correlation_results.extend(subset_correlations)
                    else:
                        self.logger.warning(f"Insufficient data in {subset_name} participant subset ({len(subset_part_df)} rows)")
                else:
                    self.logger.warning(f"Filter column {filter_column} not found in participant dataset")
        
        self.logger.info(f"Completed all analyses. Generated {len(self.ttest_results)} t-test results and {len(self.correlation_results)} correlation results.")
        
        return True
    
    def _run_ttests(self, df, subset_name="pooled"):
        """Run t-test comparisons on population-normalized data."""
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
                    
                result = self._run_t_test_comparison(df, outcome_var, demo_var)
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
                
                # Create median-split group
                temp_df = df.copy()
                median_val = temp_df[predictor].median()
                temp_df[f'{predictor}_group'] = temp_df[predictor].apply(
                    lambda x: 'high' if x > median_val else 'low' if pd.notna(x) else np.nan
                )
                
                # Run t-test with median split
                result = self._run_t_test_comparison(temp_df, outcome, f'{predictor}_group')
                
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
    
    def _run_correlations(self, df, subset_name="pooled"):
        """Run correlation analyses on participant-normalized data."""
        results = []
        tested_pairs = set()  # Track which pairs we've already tested
        
        # 1. Correlations between fragmentation metrics and emotion metrics
        for frag_var in self.fragmentation_metrics:
            if frag_var not in df.columns:
                continue
            
            for emotion_var in self.anxiety_metrics + self.mood_metrics:
                if emotion_var not in df.columns:
                    continue
                
                # Create a unique pair identifier
                pair_key = tuple(sorted([frag_var, emotion_var]))
                if pair_key in tested_pairs:
                    continue
                    
                tested_pairs.add(pair_key)
                
                # Run correlation
                result = self._run_correlation_test(df, frag_var, emotion_var)
                
                if result:
                    # Determine categories
                    frag_category = 'fragmentation'
                    emotion_category = 'anxiety' if 'anxiety' in emotion_var else 'mood'
                    
                    results.append({
                        'model_name': f"{subset_name.capitalize()}: {frag_var} ~ {emotion_var}",
                        'var1': frag_var,
                        'var1_category': frag_category,
                        'var2': emotion_var,
                        'var2_category': emotion_category,
                        'normalization': 'participant',
                        'test_type': 'correlation',
                        'subset': subset_name,
                        'n': result.get('n', 0),
                        **self._extract_correlation_stats(result)
                    })
        
        # 2. Correlations between different types of fragmentation metrics
        fragmentation_pairs = [
            ('digital_fragmentation', 'mobility_fragmentation'),
            ('digital_fragmentation', 'overlap_fragmentation'),
            ('mobility_fragmentation', 'overlap_fragmentation')
        ]
        
        for var1, var2 in fragmentation_pairs:
            if var1 not in df.columns or var2 not in df.columns:
                continue
                
            pair_key = tuple(sorted([var1, var2]))
            if pair_key in tested_pairs:
                continue
                
            tested_pairs.add(pair_key)
            
            result = self._run_correlation_test(df, var1, var2)
            
            if result:
                results.append({
                    'model_name': f"{subset_name.capitalize()}: {var1} ~ {var2}",
                    'var1': var1,
                    'var1_category': 'fragmentation',
                    'var2': var2,
                    'var2_category': 'fragmentation',
                    'normalization': 'participant',
                    'test_type': 'correlation',
                    'subset': subset_name,
                    'n': result.get('n', 0),
                    **self._extract_correlation_stats(result)
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
    
    def _extract_correlation_stats(self, result):
        """Extract statistics from correlation result."""
        stats = {}
        
        # Add correlation statistics
        stats['correlation'] = result.get('correlation', np.nan)
        stats['p_value'] = result.get('p_value', np.nan)
        
        # Add significance indicator
        p_val = result.get('p_value', 1.0)
        stats['sig_level'] = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        
        # Add effect size interpretation
        corr = abs(result.get('correlation', 0))
        if corr < 0.1:
            effect_size = "Negligible"
        elif corr < 0.3:
            effect_size = "Small"
        elif corr < 0.5:
            effect_size = "Medium"
        else:
            effect_size = "Large"
        
        stats['effect_size'] = corr
        stats['effect_size_type'] = "Pearson's r"
        stats['effect_interpretation'] = effect_size
        
        return stats
    
    def _run_correlation_test(self, df, var1, var2):
        """Run a correlation test between two continuous variables."""
        try:
            # Validate columns
            if var1 not in df.columns or var2 not in df.columns:
                return None
            
            # Get data with common indices
            data1 = df[var1].dropna()
            data2 = df[var2].dropna()
            common_indices = data1.index.intersection(data2.index)
            data1 = data1.loc[common_indices]
            data2 = data2.loc[common_indices]
            
            # Minimum observations check
            if len(data1) < 5:
                self.logger.info(f"Insufficient observations for correlation between {var1} and {var2}")
                return None
            
            # Run correlation test
            corr, p_val = stats.pearsonr(data1, data2)
            
            result = {
                'correlation': float(corr),
                'p_value': float(p_val),
                'n': int(len(data1))
            }
            
            self.logger.info(
                f"Correlation between {var1} and {var2}: "
                f"r={corr:.2f}, p={p_val:.4f}, n={len(data1)}"
            )
            
            return result
                
        except Exception as e:
            self.logger.error(f"Error computing correlation between {var1} and {var2}: {str(e)}")
            if self.debug:
                import traceback
                self.logger.error(traceback.format_exc())
            
            return None
    
    def _run_t_test_comparison(self, df, outcome_var, group_var):
        """Run a t-test comparison between groups on an outcome variable."""
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
            
            # Get data for each group
            g1_data = df[df[group_var] == groups[0]][outcome_var].dropna()
            g2_data = df[df[group_var] == groups[1]][outcome_var].dropna()
            
            # Minimum observations check
            if len(g1_data) < 5 or len(g2_data) < 5:
                return None
            
            # Run t-test
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
                f"Compared {outcome_var} between {group_var} groups: "
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
        """Save results to separate Excel files for t-tests and correlations"""
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
        
        # Save correlation results
        if self.correlation_results:
            corr_df = pd.DataFrame(self.correlation_results)
            
            # Round numeric columns
            numeric_cols = corr_df.select_dtypes(include=[np.number]).columns
            corr_df[numeric_cols] = corr_df[numeric_cols].round(4)
            
            # Save to Excel
            corr_path = self.output_dir / f'pooled_correlation_results_{timestamp}.xlsx'
            
            with pd.ExcelWriter(corr_path) as writer:
                # All results
                corr_df.to_excel(writer, sheet_name='All Correlations', index=False)
                
                # Filter by subsets
                for subset in ['pooled'] + [s['name'] for s in self.subsets]:
                    subset_data = corr_df[corr_df['subset'] == subset]
                    if not subset_data.empty:
                        sheet_name = f'{subset.capitalize()} Correlations'
                        subset_data.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # By variable category combinations
                var_combinations = []
                for var1_cat in sorted(corr_df['var1_category'].unique()):
                    for var2_cat in sorted(corr_df['var2_category'].unique()):
                        var_combinations.append((var1_cat, var2_cat))
                
                for var1_cat, var2_cat in var_combinations:
                    cat_subset = corr_df[
                        (corr_df['var1_category'] == var1_cat) & 
                        (corr_df['var2_category'] == var2_cat)
                    ]
                    if not cat_subset.empty and len(cat_subset) > 1:
                        sheet_name = f'{var1_cat[:4]}_{var2_cat[:4]}'
                        cat_subset.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Significant results
                sig_results = corr_df[corr_df['p_value'] < 0.05]
                if not sig_results.empty:
                    sig_results.to_excel(writer, sheet_name='Significant', index=False)
                    
                # Top correlations
                top_corrs = corr_df.sort_values(by='correlation', key=abs, ascending=False).head(20)
                if not top_corrs.empty:
                    top_corrs.to_excel(writer, sheet_name='Top Correlations', index=False)
            
            self.logger.info(f"Saved {len(corr_df)} correlation results to {corr_path}")
        
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
            
            # Correlation summary for this subset
            if self.correlation_results:
                corr_df = pd.DataFrame(self.correlation_results)
                subset_corrs = corr_df[corr_df['subset'] == subset]
                
                if not subset_corrs.empty:
                    total_corrs = len(subset_corrs)
                    sig_corrs = len(subset_corrs[subset_corrs['p_value'] < 0.05])
                    sig_percent = sig_corrs / total_corrs * 100 if total_corrs > 0 else 0
                    
                    summary_data.append({
                        'Analysis Type': f'Correlations ({subset})',
                        'Total Count': total_corrs,
                        'Significant Count': sig_corrs,
                        'Significant %': f"{sig_percent:.1f}%",
                        'Notes': 'Continuous relationships'
                    })
                    
                    # Top correlations for this subset
                    top_corrs = subset_corrs.sort_values(by='correlation', key=abs, ascending=False).head(3)
                    for i, row in top_corrs.iterrows():
                        summary_data.append({
                            'Analysis Type': f'Top {subset} correlation',
                            'Total Count': '',
                            'Significant Count': '',
                            'Significant %': f"r={row['correlation']:.2f}",
                            'Notes': f"{row['var1']} ~ {row['var2']}, p={row['p_value']:.4f}"
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