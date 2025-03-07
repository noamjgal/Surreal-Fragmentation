#!/usr/bin/env python3
"""
Split Population Comparison Analysis

This script performs statistical comparisons between demographic groups and emotion metrics,
saving t-tests and correlations to separate files for easier interpretation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import logging
from datetime import datetime

class SplitPopulationAnalysis:
    def __init__(self, debug=False):
        """Initialize the population comparison analysis class with hardcoded paths."""
        # Hardcoded paths
        self.population_file = "/Users/noamgal/DSProjects/Fragmentation/SURREAL/processed/merged_data/ema_fragmentation_demographics_population_norm.csv"
        self.participant_file = "/Users/noamgal/DSProjects/Fragmentation/SURREAL/processed/merged_data/ema_fragmentation_demographics_participant_norm.csv"
        self.output_dir = Path("SURREAL/results/population_comparison")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug = debug
        self._setup_logging()
        
        # The metrics we're interested in
        self.fragmentation_metrics = [
            'frag_digital_fragmentation_index', 
            'frag_mobility_fragmentation_index', 
            'frag_overlap_fragmentation_index'
        ]
        
        self.episode_metrics = [
            'frag_digital_episode_count',
            'frag_mobility_episode_count',
            'frag_overlap_episode_count'
        ]
        
        self.duration_metrics = [
            'frag_digital_total_duration',
            'frag_mobility_total_duration',
            'frag_overlap_total_duration'
        ]
        
        # Updated emotion metrics based on actual column names
        self.anxiety_metrics = ['ema_STAI_Y_A_6_zstd', 'ema_STAI_Y_A_6_raw']
        self.mood_metrics = ['ema_CES_D_8_zstd', 'ema_CES_D_8_raw']
        
        # Updated demographic variables based on actual column names
        self.demographic_vars = [
            'Gender', 
            'City.center',        # Location (Yes = city center, No = suburb)
            'gender_code'         # Gender as numeric code
        ]
        
        # Separate results containers
        self.ttest_results = []
        self.correlation_results = []
        
    def _setup_logging(self):
        """Set up logging configuration"""
        log_dir = self.output_dir / 'logs'
        log_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'split_analysis_{timestamp}.log'
        
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
        self.logger.info(f"Initializing split population analysis")
        self.logger.info(f"Population-normalized data: {self.population_file}")
        self.logger.info(f"Participant-normalized data: {self.participant_file}")
        self.logger.info(f"Output directory: {self.output_dir}")

    def load_data(self):
        """Load both population-normalized and participant-normalized data.
        
        Returns:
            tuple: (population_df, participant_df) - both datasets
        """
        # Load population-normalized data
        self.logger.info(f"Loading population-normalized data from {self.population_file}")
        try:
            population_df = pd.read_csv(self.population_file)
            self.logger.info(f"Population data loaded successfully with shape: {population_df.shape}")
            self.logger.info(f"Population data columns: {', '.join(population_df.columns[:10])}...")
        except Exception as e:
            self.logger.error(f"Error loading population data: {str(e)}")
            population_df = None
        
        # Load participant-normalized data
        self.logger.info(f"Loading participant-normalized data from {self.participant_file}")
        try:
            participant_df = pd.read_csv(self.participant_file)
            self.logger.info(f"Participant data loaded successfully with shape: {participant_df.shape}")
            self.logger.info(f"Participant data columns: {', '.join(participant_df.columns[:10])}...")
        except Exception as e:
            self.logger.error(f"Error loading participant data: {str(e)}")
            participant_df = None
        
        # Check for expected columns in population data
        if population_df is not None:
            all_expected_cols = (
                self.fragmentation_metrics + 
                self.anxiety_metrics + 
                self.mood_metrics + 
                self.demographic_vars
            )
            
            missing_pop_cols = [col for col in all_expected_cols if col not in population_df.columns]
            if missing_pop_cols:
                self.logger.warning(f"Missing expected columns in population data: {missing_pop_cols}")
        
        # Check for expected columns in participant data
        if participant_df is not None:
            all_expected_cols = (
                self.fragmentation_metrics + 
                self.anxiety_metrics + 
                self.mood_metrics + 
                self.demographic_vars
            )
            
            missing_part_cols = [col for col in all_expected_cols if col not in participant_df.columns]
            if missing_part_cols:
                self.logger.warning(f"Missing expected columns in participant data: {missing_part_cols}")
        
        return population_df, participant_df
    
    def _preprocess_data(self, df):
        """Preprocess data for analysis
        
        Args:
            df (DataFrame): Dataset to preprocess
            
        Returns:
            DataFrame: Preprocessed dataset
        """
        if df is None:
            return None
            
        # Create a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Calculate age groups if age is available
        if 'age' in processed_df.columns:
            processed_df['age_group'] = processed_df['age'].apply(
                lambda x: 'adolescent' if x < 18 else 'adult' if pd.notna(x) else np.nan
            )
        
        # Create location type from City.center
        if 'City.center' in processed_df.columns:
            processed_df['location_type'] = processed_df['City.center'].apply(
                lambda x: 'city_center' if x == 'Yes' else 'suburb' if x == 'No' else np.nan
            )
        
        # Standardize gender naming
        if 'Gender' in processed_df.columns:
            processed_df['gender_standardized'] = processed_df['Gender'].apply(
                lambda x: 'female' if x == 'F' else 'male' if x == 'M' else np.nan
            )
        
        # Create dataset_source based on participant ID if available
        if 'Participant_ID_x' in processed_df.columns:
            processed_df['dataset_source'] = processed_df['Participant_ID_x'].apply(
                lambda x: str(x).split('_')[0].lower() if pd.notna(x) else np.nan
            )
        
        # Replace hyphens in column names with underscores for easier referencing
        processed_df.columns = [col.replace('-', '_') for col in processed_df.columns]
        
        # Update anxiety and mood metrics in the object to match the new column names
        self.anxiety_metrics = [metric.replace('-', '_') for metric in self.anxiety_metrics]
        self.mood_metrics = [metric.replace('-', '_') for metric in self.mood_metrics]
        
        return processed_df
    
    def run_analyses(self):
        """Run t-tests and correlations separately, using appropriate datasets for each."""
        # Load both datasets
        population_df, participant_df = self.load_data()
        
        if population_df is None:
            self.logger.error("Population dataset failed to load, cannot continue with t-tests")
            return False
        
        if participant_df is None:
            self.logger.error("Participant dataset failed to load, cannot continue with correlations")
            return False
        
        # Preprocess both datasets
        population_df = self._preprocess_data(population_df)
        participant_df = self._preprocess_data(participant_df)
        
        # Process population-normalized data for t-tests
        if population_df is not None:
            self.logger.info("Running t-tests on population-normalized data")
            pop_ttests = self._run_ttests(population_df, "population")
            self.ttest_results.extend(pop_ttests)
        
        # Process participant-normalized data for correlations
        if participant_df is not None:
            self.logger.info("Running correlations on participant-normalized data")
            part_correlations = self._run_correlations(participant_df, "participant")
            self.correlation_results.extend(part_correlations)
        
        self.logger.info(f"Completed all analyses. Generated {len(self.ttest_results)} t-test results and {len(self.correlation_results)} correlation results.")
        
        return True
    
    def _run_ttests(self, df, normalization_type):
        """Run all t-test comparisons on a dataset.
        
        Args:
            df (DataFrame): Dataset to analyze
            normalization_type (str): Type of normalization ("population" or "participant")
            
        Returns:
            list: T-test results in tabular format
        """
        results = []
        
        # 1. Compare all metrics across demographic variables
        all_metrics = (
            self.fragmentation_metrics + 
            self.episode_metrics + 
            self.duration_metrics + 
            self.anxiety_metrics + 
            self.mood_metrics
        )
        
        # Use a prioritized and reduced set of demographic variables to avoid redundancy
        # For each demographic concept, pick the best representation
        prioritized_demographics = [
            'gender_standardized',  # Use standardized version instead of 'Gender' and 'gender_code'
            'location_type',        # Use clearer version instead of 'City.center'
        ]
        
        for outcome_var in all_metrics:
            if outcome_var not in df.columns:
                self.logger.warning(f"Outcome variable {outcome_var} not found in dataset")
                continue
            
            for demo_var in prioritized_demographics:
                if demo_var not in df.columns:
                    self.logger.warning(f"Demographic variable {demo_var} not found in dataset")
                    continue
                    
                result = self._run_t_test_comparison(df, outcome_var, demo_var)
                if result:
                    # Determine outcome category
                    if outcome_var in self.anxiety_metrics:
                        outcome_category = "anxiety"
                    elif outcome_var in self.mood_metrics:
                        outcome_category = "mood"
                    elif outcome_var in self.fragmentation_metrics:
                        outcome_category = "fragmentation"
                    elif outcome_var in self.episode_metrics:
                        outcome_category = "episode"
                    elif outcome_var in self.duration_metrics:
                        outcome_category = "duration"
                    else:
                        outcome_category = "other"
                    
                    results.append({
                        'model_name': f"{normalization_type.capitalize()}: {outcome_var} ~ {demo_var}",
                        'dv': outcome_var,
                        'dv_category': outcome_category,
                        'predictor': demo_var,
                        'predictor_category': 'demographic',
                        'normalization': normalization_type,
                        'test_type': 't-test',
                        'n_obs': result.get('total_n', 0),
                        **self._extract_t_test_stats(result)
                    })
        
        # 2. Median-split analyses for targeted combinations only
        # Only compare fragmentation/episode/duration metrics with anxiety/mood metrics
        behavioral_metrics = self.fragmentation_metrics + self.episode_metrics + self.duration_metrics
        emotional_metrics = self.anxiety_metrics + self.mood_metrics
        
        # Only do splits in one direction to avoid duplication
        # Use behavioral metrics as predictors, emotional metrics as outcomes
        for predictor in behavioral_metrics:
            for outcome in emotional_metrics:
                if predictor == outcome or predictor not in df.columns or outcome not in df.columns:
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
                    # Determine categories
                    if outcome in self.anxiety_metrics:
                        dv_category = "anxiety"
                    elif outcome in self.mood_metrics:
                        dv_category = "mood"
                    else:
                        dv_category = "other"
                        
                    if predictor in self.fragmentation_metrics:
                        predictor_category = "fragmentation"
                    elif predictor in self.episode_metrics:
                        predictor_category = "episode"
                    elif predictor in self.duration_metrics:
                        predictor_category = "duration"
                    else:
                        predictor_category = "other"
                    
                    results.append({
                        'model_name': f"{normalization_type.capitalize()}: {outcome} ~ {predictor} (median split)",
                        'dv': outcome,
                        'dv_category': dv_category,
                        'predictor': predictor,
                        'predictor_category': predictor_category,
                        'normalization': normalization_type,
                        'test_type': 'median-split t-test',
                        'n_obs': result.get('total_n', 0),
                        **self._extract_t_test_stats(result)
                    })
        
        return results
    
    def _run_correlations(self, df, normalization_type):
        """Run all correlation analyses on a dataset.
        
        Args:
            df (DataFrame): Dataset to analyze
            normalization_type (str): Type of normalization ("population" or "participant")
            
        Returns:
            list: Correlation results in tabular format
        """
        results = []
        tested_pairs = set()  # Track which pairs we've already tested
        
        # 1. Correlations between fragmentation metrics and emotion metrics
        for frag_var in self.fragmentation_metrics + self.episode_metrics + self.duration_metrics:
            if frag_var not in df.columns:
                continue
            
            for emotion_var in self.anxiety_metrics + self.mood_metrics:
                if emotion_var not in df.columns:
                    continue
                
                # Create a unique pair identifier (alphabetically sorted)
                pair_key = tuple(sorted([frag_var, emotion_var]))
                
                # Skip if we've already tested this pair
                if pair_key in tested_pairs:
                    continue
                
                # Mark pair as tested
                tested_pairs.add(pair_key)
                
                # Run the correlation
                result = self._run_correlation_test(df, frag_var, emotion_var)
                
                if result:
                    # Determine categories
                    if frag_var in self.fragmentation_metrics:
                        frag_category = "fragmentation"
                    elif frag_var in self.episode_metrics:
                        frag_category = "episode"
                    elif frag_var in self.duration_metrics:
                        frag_category = "duration"
                    else:
                        frag_category = "other"
                        
                    if emotion_var in self.anxiety_metrics:
                        emotion_category = "anxiety"
                    elif emotion_var in self.mood_metrics:
                        emotion_category = "mood"
                    else:
                        emotion_category = "other"
                    
                    results.append({
                        'model_name': f"{normalization_type.capitalize()}: {frag_var} ~ {emotion_var}",
                        'var1': frag_var,
                        'var1_category': frag_category,
                        'var2': emotion_var,
                        'var2_category': emotion_category,
                        'normalization': normalization_type,
                        'test_type': 'correlation',
                        'n': result.get('n', 0),
                        **self._extract_correlation_stats(result)
                    })
        
        # 2. Correlations between different types of fragmentation metrics (but not with their own components)
        fragmentation_by_type = {
            'digital': [var for var in self.fragmentation_metrics if 'digital' in var],
            'mobility': [var for var in self.fragmentation_metrics if 'mobility' in var],
            'overlap': [var for var in self.fragmentation_metrics if 'overlap' in var]
        }
        
        # Only compare types in one direction to avoid duplication
        # Sort the type keys to ensure consistent ordering
        type_pairs = []
        sorted_types = sorted(fragmentation_by_type.keys())
        for i, type1 in enumerate(sorted_types):
            for type2 in sorted_types[i+1:]:  # only compare with types that come later
                type_pairs.append((type1, type2))
        
        # Process each type pair in only one direction
        for type1, type2 in type_pairs:
            vars1 = fragmentation_by_type[type1]
            vars2 = fragmentation_by_type[type2]
            
            for var1 in vars1:
                for var2 in vars2:
                    if var1 not in df.columns or var2 not in df.columns:
                        continue
                    
                    # Create a unique pair identifier (already sorted by the type ordering)
                    pair_key = tuple(sorted([var1, var2]))
                    
                    # Skip if we've already tested this pair
                    if pair_key in tested_pairs:
                        continue
                    
                    # Mark pair as tested
                    tested_pairs.add(pair_key)
                    
                    result = self._run_correlation_test(df, var1, var2)
                    
                    if result:
                        results.append({
                            'model_name': f"{normalization_type.capitalize()}: {var1} ~ {var2}",
                            'var1': var1,
                            'var1_category': 'fragmentation',
                            'var2': var2,
                            'var2_category': 'fragmentation',
                            'normalization': normalization_type,
                            'test_type': 'correlation',
                            'n': result.get('n', 0),
                            **self._extract_correlation_stats(result)
                        })
        
        return results
    
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
        
        # Add effect size interpretation for correlation
        corr = abs(result.get('correlation', 0))
        if corr < 0.1:
            effect_size = "Negligible"
        elif corr < 0.3:
            effect_size = "Small"
        elif corr < 0.5:
            effect_size = "Medium"
        else:
            effect_size = "Large"
        
        stats['effect_size'] = corr  # Use the correlation coefficient as effect size
        stats['effect_size_type'] = "Pearson's r"
        stats['effect_interpretation'] = effect_size
        
        return stats
    
    def _run_correlation_test(self, df, var1, var2):
        """Run a correlation test between two continuous variables.
        
        Args:
            df (DataFrame): Dataset
            var1 (str): First variable
            var2 (str): Second variable
            
        Returns:
            dict: Correlation results or None if correlation couldn't be performed
        """
        try:
            # Skip if columns don't exist
            if var1 not in df.columns or var2 not in df.columns:
                self.logger.warning(f"Skipping correlation between {var1} and {var2}, column(s) not found")
                return None
            
            # Get data for each variable
            data1 = df[var1].dropna()
            data2 = df[var2].dropna()
            
            # Get common indices (participants that have values for both variables)
            common_indices = data1.index.intersection(data2.index)
            
            # Filter data to common indices
            data1 = data1.loc[common_indices]
            data2 = data2.loc[common_indices]
            
            # Minimum number of observations required
            if len(data1) < 5:
                self.logger.info(
                    f"Skipping correlation between {var1} and {var2}: "
                    "insufficient observations"
                )
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
        """Run a t-test comparison between groups on an outcome variable.
        
        Args:
            df (DataFrame): Dataset
            outcome_var (str): Variable to analyze
            group_var (str): Grouping variable
            
        Returns:
            dict: Comparison results or None if comparison couldn't be performed
        """
        try:
            # Skip if columns don't exist
            if outcome_var not in df.columns or group_var not in df.columns:
                self.logger.warning(f"Skipping {outcome_var} by {group_var}, column(s) not found")
                return None
            
            # Skip if too many missing values
            if df[outcome_var].isna().sum() > 0.5 * len(df) or df[group_var].isna().sum() > 0.5 * len(df):
                self.logger.warning(f"Skipping {outcome_var} by {group_var}, too many missing values")
                return None
            
            # Get unique groups
            groups = df[group_var].dropna().unique()
            
            # Skip if less than 2 groups
            if len(groups) < 2:
                self.logger.info(f"Skipping {group_var}, only one group found: {groups}")
                return None
            
            # For more than 2 groups, use the first two for t-test
            if len(groups) > 2:
                self.logger.info(f"{group_var} has {len(groups)} groups: {groups}, using first two for t-test")
                groups = groups[:2]
            
            # Get data for each group
            g1_data = df[df[group_var] == groups[0]][outcome_var].dropna()
            g2_data = df[df[group_var] == groups[1]][outcome_var].dropna()
            
            # Minimum number of observations required
            if len(g1_data) < 5 or len(g2_data) < 5:
                self.logger.info(
                    f"Skipping {outcome_var} comparison by {group_var} groups: "
                    f"insufficient observations (group1: {len(g1_data)}, group2: {len(g2_data)})"
                )
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
            ttest_path = self.output_dir / f'ttest_results_{timestamp}.xlsx'
            
            with pd.ExcelWriter(ttest_path) as writer:
                # All results
                ttest_df.to_excel(writer, sheet_name='All T-Tests', index=False)
                
                # By normalization type
                for norm_type in ttest_df['normalization'].unique():
                    norm_subset = ttest_df[ttest_df['normalization'] == norm_type]
                    if not norm_subset.empty:
                        norm_subset.to_excel(writer, sheet_name=f'{norm_type.capitalize()}', index=False)
                
                # By outcome category
                for category in sorted(ttest_df['dv_category'].unique()):
                    cat_subset = ttest_df[ttest_df['dv_category'] == category]
                    if not cat_subset.empty:
                        cat_subset.to_excel(writer, sheet_name=f'{category.capitalize()}', index=False)
                
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
            corr_path = self.output_dir / f'correlation_results_{timestamp}.xlsx'
            
            with pd.ExcelWriter(corr_path) as writer:
                # All results
                corr_df.to_excel(writer, sheet_name='All Correlations', index=False)
                
                # By normalization type
                for norm_type in corr_df['normalization'].unique():
                    norm_subset = corr_df[corr_df['normalization'] == norm_type]
                    if not norm_subset.empty:
                        norm_subset.to_excel(writer, sheet_name=f'{norm_type.capitalize()}', index=False)
                
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
        
        # Also create a summary file
        self._create_summary_file(timestamp)
        
        return True
    
    def _create_summary_file(self, timestamp):
        """Create a summary file with key findings"""
        summary_data = []
        
        # Count significant results
        if self.ttest_results:
            ttest_df = pd.DataFrame(self.ttest_results)
            total_ttests = len(ttest_df)
            sig_ttests = len(ttest_df[ttest_df['p_value'] < 0.05])
            sig_percent = sig_ttests / total_ttests * 100 if total_ttests > 0 else 0
            
            summary_data.append({
                'Analysis Type': 'T-tests',
                'Total Count': total_ttests,
                'Significant Count': sig_ttests,
                'Significant %': f"{sig_percent:.1f}%",
                'Notes': 'Group comparisons'
            })
            
            # Breakdown by normalization
            for norm_type in ttest_df['normalization'].unique():
                norm_subset = ttest_df[ttest_df['normalization'] == norm_type]
                norm_total = len(norm_subset)
                norm_sig = len(norm_subset[norm_subset['p_value'] < 0.05])
                norm_percent = norm_sig / norm_total * 100 if norm_total > 0 else 0
                
                summary_data.append({
                    'Analysis Type': f'T-tests ({norm_type})',
                    'Total Count': norm_total,
                    'Significant Count': norm_sig,
                    'Significant %': f"{norm_percent:.1f}%",
                    'Notes': ''
                })
            
            # Breakdown by DV category
            for category in sorted(ttest_df['dv_category'].unique()):
                cat_subset = ttest_df[ttest_df['dv_category'] == category]
                cat_total = len(cat_subset)
                cat_sig = len(cat_subset[cat_subset['p_value'] < 0.05])
                cat_percent = cat_sig / cat_total * 100 if cat_total > 0 else 0
                
                summary_data.append({
                    'Analysis Type': f'T-tests ({category})',
                    'Total Count': cat_total,
                    'Significant Count': cat_sig,
                    'Significant %': f"{cat_percent:.1f}%",
                    'Notes': ''
                })
            
            # Top 5 t-test effects
            top_ttests = ttest_df.sort_values(by='effect_size', ascending=False).head(5)
            for i, row in top_ttests.iterrows():
                summary_data.append({
                    'Analysis Type': 'Top t-test effect',
                    'Total Count': '',
                    'Significant Count': '',
                    'Significant %': f"d={row['effect_size']:.2f}",
                    'Notes': f"{row['dv']} by {row['predictor']}, p={row['p_value']:.4f}"
                })
        
        # Correlation summary
        if self.correlation_results:
            corr_df = pd.DataFrame(self.correlation_results)
            total_corrs = len(corr_df) // 2  # Divide by 2 because we store each correlation twice (both directions)
            sig_corrs = len(corr_df[corr_df['p_value'] < 0.05]) // 2
            sig_percent = sig_corrs / total_corrs * 100 if total_corrs > 0 else 0
            
            summary_data.append({
                'Analysis Type': 'Correlations',
                'Total Count': total_corrs,
                'Significant Count': sig_corrs,
                'Significant %': f"{sig_percent:.1f}%",
                'Notes': 'Continuous relationships'
            })
            
            # Breakdown by normalization
            for norm_type in corr_df['normalization'].unique():
                norm_subset = corr_df[corr_df['normalization'] == norm_type]
                norm_total = len(norm_subset) // 2
                norm_sig = len(norm_subset[norm_subset['p_value'] < 0.05]) // 2
                norm_percent = norm_sig / norm_total * 100 if norm_total > 0 else 0
                
                summary_data.append({
                    'Analysis Type': f'Correlations ({norm_type})',
                    'Total Count': norm_total,
                    'Significant Count': norm_sig,
                    'Significant %': f"{norm_percent:.1f}%",
                    'Notes': ''
                })
            
            # Top 5 correlations
            unique_corrs = corr_df.drop_duplicates(subset=['var1', 'var2', 'normalization'])
            top_corrs = unique_corrs.sort_values(by='correlation', key=abs, ascending=False).head(5)
            for i, row in top_corrs.iterrows():
                summary_data.append({
                    'Analysis Type': 'Top correlation',
                    'Total Count': '',
                    'Significant Count': '',
                    'Significant %': f"r={row['correlation']:.2f}",
                    'Notes': f"{row['var1']} ~ {row['var2']}, p={row['p_value']:.4f}"
                })
        
        # Create summary dataframe
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary to Excel
        summary_path = self.output_dir / f'analysis_summary_{timestamp}.xlsx'
        summary_df.to_excel(summary_path, index=False)
        
        self.logger.info(f"Saved analysis summary to {summary_path}")
        return summary_path

def main():
    """Main function to run the split population analysis."""
    try:
        # Create analyzer with hardcoded paths
        analyzer = SplitPopulationAnalysis(debug=True)
        
        # Run analyses
        if analyzer.run_analyses():
            # Save results
            analyzer.save_results()
            
            print(f"Population comparison analysis completed successfully!")
            print(f"Results saved to separate t-test and correlation files in {analyzer.output_dir}")
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
    exit(main())

