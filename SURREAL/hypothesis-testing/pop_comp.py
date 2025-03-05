#!/usr/bin/env python3
"""
Population Comparison Analysis for SURREAL Fragmentation Metrics

This script performs statistical comparisons (t-tests, ANOVAs) between different 
demographic groups on various fragmentation metrics, producing comprehensive
reports of the differences.

"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from scipy import stats
import logging
import argparse
from datetime import datetime
import re

class PopulationComparisonAnalysis:
    def __init__(self, input_path, output_dir, debug=False):
        """Initialize the population comparison analysis class.
        
        Args:
            input_path (str): Path to merged data file
            output_dir (str): Directory to save analysis results
            debug (bool): Enable debug logging
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug = debug
        self._setup_logging()
        self.comparison_results = []
        
        # These will be identified based on available columns
        self.fragmentation_metrics = []
        self.demographic_vars = []
        self.emotion_vars = {'stai': [], 'cesd': []}
        
    def _setup_logging(self):
        """Set up logging configuration"""
        log_dir = self.output_dir / 'logs'
        log_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'population_comparison_{timestamp}.log'
        
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
        self.logger.info(f"Initializing population comparison analysis with input: {self.input_path}")

    def load_and_preprocess_data(self):
        """Load and preprocess data for analysis.
        
        Returns:
            tuple: (daily_df, participant_df) containing daily observations and participant summaries
        """
        self.logger.info(f"Loading data from {self.input_path}")
        
        try:
            # Load data
            df = pd.read_csv(self.input_path)
            self.logger.info(f"Data loaded successfully with shape: {df.shape}")
            
            # Create a safe column name mapper to avoid formula issues
            self.col_name_map = {}
            safe_cols = []
            
            for col in df.columns:
                # Check if column has special characters
                if re.search(r'[-]', col):
                    # Create safe name by replacing special chars with underscore
                    safe_name = re.sub(r'[-]', '_', col)
                    self.col_name_map[col] = safe_name
                    safe_cols.append(safe_name)
                else:
                    safe_cols.append(col)
            
            # Apply the mapping to rename columns
            if self.col_name_map:
                df = df.rename(columns=self.col_name_map)
                self.logger.info(f"Renamed {len(self.col_name_map)} columns to avoid formula issues")
                self.logger.debug(f"Column mapping: {self.col_name_map}")
            
            # Identify key column types
            self._identify_column_types(df)
            
            # Create weekend indicator
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df['is_weekend'] = df['date'].dt.dayofweek >= 5
                df['is_weekend'] = df['is_weekend'].astype(int)
                self.logger.info("Created weekend indicator variable")
            
            # Ensure proper participant ID formatting
            id_col = self._get_participant_id_column(df)
            if id_col:
                # Create clean participant ID
                df['participant_id_clean'] = df[id_col].astype(str).str.replace(r'\D', '', regex=True)
                self.logger.info(f"Created clean participant ID from {id_col}")
            else:
                self.logger.warning("Could not identify participant ID column")
                return None, None
            
            # Create two dataframes: one for daily measures and one for participant-level summaries
            daily_df = df.copy()
            
            # Create participant summary dataframe with mean values
            participant_df = df.groupby('participant_id_clean').agg({
                col: 'mean' for col in df.columns if df[col].dtype.kind in 'fc' and col != 'date'
            }).reset_index()
            
            for col in df.columns:
                if df[col].dtype == 'object' and col != 'date' and col != id_col:
                    # Take the most common value for categorical variables
                    most_common = df.groupby('participant_id_clean')[col].agg(
                        lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
                    )
                    participant_df[col] = most_common.values
            
            self.logger.info(f"Created participant-level summary with {len(participant_df)} participants")
            self.logger.info(f"Daily-level data has {len(daily_df)} observations")
            
            # Log the identified metrics
            self.logger.info(f"Identified {len(self.fragmentation_metrics)} fragmentation metrics: {self.fragmentation_metrics}")
            self.logger.info(f"Identified {len(self.demographic_vars)} demographic variables: {self.demographic_vars}")
            self.logger.info(f"Identified STAI variables: {self.emotion_vars['stai']}")
            self.logger.info(f"Identified CESD variables: {self.emotion_vars['cesd']}")
            
            return daily_df, participant_df
            
        except Exception as e:
            self.logger.error(f"Error loading or preprocessing data: {str(e)}")
            if self.debug:
                import traceback
                self.logger.error(traceback.format_exc())
            return None, None
    
    def _identify_column_types(self, df):
        """Identify different types of columns in the dataframe"""
        # Identify fragmentation metrics
        frag_patterns = [
            'digital_fragmentation', 'mobility_fragmentation', 'overlap_fragmentation',
            'digital_frag', 'moving_frag', 'fragmentation_index'
        ]
        
        self.fragmentation_metrics = [
            col for col in df.columns 
            if any(pattern in col.lower() for pattern in frag_patterns)
        ]
        
        # Add episode counts and durations
        episode_patterns = ['episode_count', 'total_duration']
        for pattern in episode_patterns:
            for prefix in ['digital_', 'mobility_', 'moving_', 'overlap_']:
                col_name = f"{prefix}{pattern}"
                if col_name in df.columns:
                    self.fragmentation_metrics.append(col_name)
        
        # Identify demographic variables
        demo_patterns = [
            'gender', 'age', 'sex', 'education', 'income', 'ethnicity', 
            'race', 'marital', 'employment', 'city', 'urban'
        ]
        
        self.demographic_vars = [
            col for col in df.columns 
            if any(pattern in col.lower() for pattern in demo_patterns) or
               col in ['is_weekend']  # Add weekend as demographic
        ]
        
        # Identify STAI variables
        self.emotion_vars['stai'] = [
            col for col in df.columns
            if 'stai' in col.lower() and 'zstd' in col.lower()
        ]
        
        # Identify CESD variables
        self.emotion_vars['cesd'] = [
            col for col in df.columns
            if 'ces' in col.lower() and 'zstd' in col.lower()
        ]
        
        # If we don't find the z-standardized versions, look for raw versions
        if not self.emotion_vars['stai']:
            self.emotion_vars['stai'] = [
                col for col in df.columns
                if 'stai' in col.lower()
            ]
            
        if not self.emotion_vars['cesd']:
            self.emotion_vars['cesd'] = [
                col for col in df.columns
                if 'ces' in col.lower()
            ]
    
    def _get_participant_id_column(self, df):
        """Identify the participant ID column"""
        id_patterns = ['participant_id', 'subject_id', 'user_id', 'id', 'pid']
        
        for pattern in id_patterns:
            matching_cols = [col for col in df.columns if pattern.lower() in col.lower()]
            if matching_cols:
                return matching_cols[0]
        
        return None
    
    def run_population_comparisons(self):
        """Run a series of population comparisons between demographic groups.
        
        Returns:
            dict: Dictionary of results for different comparison types
        """
        # Run the analysis twice - once for STAI and once for CESD
        all_results = {
            'stai': {'daily': [], 'participant': []},
            'cesd': {'daily': [], 'participant': []}
        }
        
        for emotion_type in ['stai', 'cesd']:
            emotion_vars = self.emotion_vars[emotion_type]
            
            if not emotion_vars:
                self.logger.warning(f"No {emotion_type.upper()} variables found, skipping analysis")
                continue
                
            self.logger.info(f"\n--- Running {emotion_type.upper()} Analysis ---")
            
            # Extract the first emotion variable for analysis
            emotion_var = emotion_vars[0]
            self.logger.info(f"Using {emotion_var} as primary outcome")
            
            # Fragmentation metrics as predictors
            for level, df in [('daily', self.daily_df), ('participant', self.participant_df)]:
                self.logger.info(f"\nRunning {level}-level {emotion_type.upper()} analysis")
                
                # Skip if dataframe is empty
                if df is None or df.empty:
                    self.logger.warning(f"No data available for {level}-level analysis")
                    continue
                
                # Run demographic group comparisons for all fragmentation metrics
                results = self._run_demographic_comparisons(df, level, emotion_type, emotion_var)
                all_results[emotion_type][level].extend(results)
                
                # Run fragmentation metric comparisons for STAI/CESD high vs low groups
                more_results = self._run_emotion_group_comparisons(df, level, emotion_type, emotion_var)
                all_results[emotion_type][level].extend(more_results)
        
        # Store all results together
        self.comparison_results = []
        for emotion_type in all_results:
            for level in all_results[emotion_type]:
                self.comparison_results.extend(all_results[emotion_type][level])
                
        return self.comparison_results
    
    def _run_demographic_comparisons(self, df, level, emotion_type, emotion_var):
        """Run comparisons between demographic groups on emotion outcomes"""
        results = []
        
        for demo_var in self.demographic_vars:
            try:
                # Skip if too many missing values
                if df[demo_var].isna().sum() > 0.5 * len(df):
                    self.logger.warning(f"Skipping {demo_var}, too many missing values")
                    continue
                
                # Get unique groups
                groups = df[demo_var].dropna().unique()
                
                # Skip if only one group
                if len(groups) < 2:
                    self.logger.info(f"Skipping {demo_var}, only one group found")
                    continue
                    
                # Skip if emotion variable has too many missing values
                if df[emotion_var].isna().sum() > 0.5 * len(df):
                    self.logger.warning(f"Skipping {emotion_var}, too many missing values")
                    continue
                
                if len(groups) == 2:  # Binary comparison
                    # Run t-test
                    g1_data = df[df[demo_var] == groups[0]][emotion_var].dropna()
                    g2_data = df[df[demo_var] == groups[1]][emotion_var].dropna()
                    
                    # Minimum number of observations required
                    if len(g1_data) < 5 or len(g2_data) < 5:
                        self.logger.info(f"Skipping {demo_var} for {emotion_var}: insufficient observations")
                        continue
                    
                    t_stat, p_val = stats.ttest_ind(g1_data, g2_data, equal_var=False)
                    
                    # Calculate effect size (Cohen's d)
                    g1_mean, g1_std = g1_data.mean(), g1_data.std()
                    g2_mean, g2_std = g2_data.mean(), g2_data.std()
                    
                    # Pooled standard deviation
                    pooled_std = np.sqrt(((len(g1_data) - 1) * g1_std**2 + 
                                         (len(g2_data) - 1) * g2_std**2) / 
                                        (len(g1_data) + len(g2_data) - 2))
                    
                    # Cohen's d
                    cohen_d = abs(g1_mean - g2_mean) / pooled_std
                    
                    result = {
                        'analysis_type': f"{emotion_type}_by_demographic",
                        'level': level,
                        'outcome': emotion_var, 
                        'group_variable': demo_var,
                        'test': 't-test',
                        'statistic': float(t_stat),
                        'p_value': float(p_val),
                        'effect_size': float(cohen_d),
                        'effect_size_type': "Cohen's d",
                        'sig_level': '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else '',
                        'group1': str(groups[0]),
                        'group1_mean': float(g1_mean),
                        'group1_std': float(g1_std),
                        'group1_n': int(len(g1_data)),
                        'group2': str(groups[1]),
                        'group2_mean': float(g2_mean),
                        'group2_std': float(g2_std),
                        'group2_n': int(len(g2_data)),
                    }
                    
                    results.append(result)
                    self.logger.info(f"Compared {demo_var} groups on {emotion_var}: t={t_stat:.2f}, p={p_val:.4f}")
                    
                else:  # Multi-group comparison (ANOVA)
                    # Prepare data for ANOVA
                    anova_groups = []
                    group_stats = {}
                    
                    for i, group in enumerate(groups):
                        group_data = df[df[demo_var] == group][emotion_var].dropna()
                        
                        # Skip if too few observations
                        if len(group_data) < 5:
                            continue
                            
                        anova_groups.append(group_data)
                        group_stats[f'group{i+1}'] = {
                            'name': str(group),
                            'mean': float(group_data.mean()),
                            'std': float(group_data.std()),
                            'n': int(len(group_data))
                        }
                    
                    # Run ANOVA if we have at least 2 groups
                    if len(anova_groups) >= 2:
                        f_stat, p_val = stats.f_oneway(*anova_groups)
                        
                        # Calculate effect size (eta-squared)
                        # Use statsmodels for ANOVA to get SS values
                        formula = f"{emotion_var} ~ C({demo_var})"
                        try:
                            model = sm.formula.ols(formula, data=df).fit()
                            anova_table = sm.stats.anova_lm(model, typ=2)
                            
                            # Calculate eta-squared
                            ss_effect = anova_table.iloc[0, 0]  # Sum of squares for the effect
                            ss_total = ss_effect + anova_table.iloc[1, 0]  # Total sum of squares
                            eta_squared = ss_effect / ss_total
                            
                            result = {
                                'analysis_type': f"{emotion_type}_by_demographic",
                                'level': level,
                                'outcome': emotion_var,
                                'group_variable': demo_var,
                                'test': 'ANOVA',
                                'statistic': float(f_stat),
                                'p_value': float(p_val),
                                'effect_size': float(eta_squared),
                                'effect_size_type': 'Eta-squared',
                                'sig_level': '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else '',
                                'num_groups': len(anova_groups)
                            }
                            
                            # Add group-specific stats
                            for i, stats_dict in group_stats.items():
                                for key, value in stats_dict.items():
                                    result[f'{i}_{key}'] = value
                            
                            results.append(result)
                            self.logger.info(f"Compared {demo_var} groups on {emotion_var}: F={f_stat:.2f}, p={p_val:.4f}")
                        
                        except Exception as e:
                            self.logger.error(f"Error in ANOVA for {demo_var} on {emotion_var}: {str(e)}")
            
            except Exception as e:
                self.logger.error(f"Error analyzing {demo_var} for {emotion_var}: {str(e)}")
                if self.debug:
                    import traceback
                    self.logger.error(traceback.format_exc())
        
        return results
    
    def _run_emotion_group_comparisons(self, df, level, emotion_type, emotion_var):
        """Compare fragmentation metrics between high and low emotion groups"""
        results = []
        
        try:
            # Create high/low emotion groups based on median split
            median_val = df[emotion_var].median()
            df['emotion_group'] = df[emotion_var].apply(lambda x: 'high' if x > median_val else 'low')
            
            self.logger.info(f"Created {emotion_type} groups based on median split ({median_val:.2f})")
            
            # Compare each fragmentation metric between high/low groups
            for frag_metric in self.fragmentation_metrics:
                # Skip if too many missing values
                if df[frag_metric].isna().sum() > 0.5 * len(df):
                    self.logger.warning(f"Skipping {frag_metric}, too many missing values")
                    continue
                
                # Get data for each group
                high_group = df[df['emotion_group'] == 'high'][frag_metric].dropna()
                low_group = df[df['emotion_group'] == 'low'][frag_metric].dropna()
                
                # Minimum number of observations required
                if len(high_group) < 5 or len(low_group) < 5:
                    self.logger.info(f"Skipping {frag_metric} comparison: insufficient observations")
                    continue
                
                # Run t-test
                t_stat, p_val = stats.ttest_ind(high_group, low_group, equal_var=False)
                
                # Calculate effect size (Cohen's d)
                high_mean, high_std = high_group.mean(), high_group.std()
                low_mean, low_std = low_group.mean(), low_group.std()
                
                # Pooled standard deviation
                pooled_std = np.sqrt(((len(high_group) - 1) * high_std**2 + 
                                     (len(low_group) - 1) * low_std**2) / 
                                    (len(high_group) + len(low_group) - 2))
                
                # Cohen's d
                cohen_d = abs(high_mean - low_mean) / pooled_std
                
                result = {
                    'analysis_type': f"fragmentation_by_{emotion_type}_group",
                    'level': level,
                    'outcome': frag_metric, 
                    'group_variable': f"{emotion_type}_group",
                    'test': 't-test',
                    'statistic': float(t_stat),
                    'p_value': float(p_val),
                    'effect_size': float(cohen_d),
                    'effect_size_type': "Cohen's d",
                    'sig_level': '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else '',
                    'group1': 'high',
                    'group1_mean': float(high_mean),
                    'group1_std': float(high_std),
                    'group1_n': int(len(high_group)),
                    'group2': 'low',
                    'group2_mean': float(low_mean),
                    'group2_std': float(low_std),
                    'group2_n': int(len(low_group)),
                }
                
                results.append(result)
                self.logger.info(f"Compared {frag_metric} between {emotion_type} groups: t={t_stat:.2f}, p={p_val:.4f}")
                
        except Exception as e:
            self.logger.error(f"Error running emotion group comparisons: {str(e)}")
            if self.debug:
                import traceback
                self.logger.error(traceback.format_exc())
        
        return results
            
    def save_results(self):
        """Save comparison results to Excel files"""
        if not self.comparison_results:
            self.logger.warning("No results to save")
            return
        
        try:
            # Create results directory
            results_dir = self.output_dir
            results_dir.mkdir(exist_ok=True, parents=True)
            
            # Generate timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Convert results to DataFrame
            results_df = pd.DataFrame(self.comparison_results)
            
            # Round numeric columns for cleaner output
            numeric_cols = results_df.select_dtypes(include=[np.number]).columns
            results_df[numeric_cols] = results_df[numeric_cols].round(4)
            
            # Save all results to a single Excel file
            all_results_path = results_dir / f'all_population_comparisons_{timestamp}.xlsx'
            
            with pd.ExcelWriter(all_results_path) as writer:
                # Save all results to first sheet
                results_df.to_excel(writer, sheet_name='All Results', index=False)
                
                # Create summary sheets by analysis type
                for analysis_type in results_df['analysis_type'].unique():
                    type_results = results_df[results_df['analysis_type'] == analysis_type]
                    type_results.to_excel(writer, sheet_name=analysis_type[:31], index=False)  # Excel limits sheet names to 31 chars
                
                # Create summary sheets by level
                for level in results_df['level'].unique():
                    level_results = results_df[results_df['level'] == level]
                    level_results.to_excel(writer, sheet_name=f'{level}_level', index=False)
                
                # Create significant results summary
                sig_results = results_df[results_df['p_value'] < 0.05]
                if not sig_results.empty:
                    sig_results.to_excel(writer, sheet_name='Significant Results', index=False)
            
            self.logger.info(f"Saved all results to {all_results_path}")
            
            # Generate individual results files by emotion type
            for emotion_type in ['stai', 'cesd']:
                emotion_results = [r for r in self.comparison_results if emotion_type in r.get('analysis_type', '')]
                
                if emotion_results:
                    emotion_df = pd.DataFrame(emotion_results)
                    numeric_cols = emotion_df.select_dtypes(include=[np.number]).columns
                    emotion_df[numeric_cols] = emotion_df[numeric_cols].round(4)
                    
                    emotion_path = results_dir / f'{emotion_type}_comparisons_{timestamp}.xlsx'
                    
                    with pd.ExcelWriter(emotion_path) as writer:
                        # All results for this emotion
                        emotion_df.to_excel(writer, sheet_name=f'All {emotion_type.upper()}', index=False)
                        
                        # Results by level
                        for level in emotion_df['level'].unique():
                            level_results = emotion_df[emotion_df['level'] == level]
                            level_results.to_excel(writer, sheet_name=f'{level}_level', index=False)
                        
                        # Significant results only
                        sig_results = emotion_df[emotion_df['p_value'] < 0.05]
                        if not sig_results.empty:
                            sig_results.to_excel(writer, sheet_name='Significant Results', index=False)
                    
                    self.logger.info(f"Saved {emotion_type.upper()} results to {emotion_path}")
                else:
                    self.logger.warning(f"No results for {emotion_type.upper()}")
            
            return all_results_path
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            if self.debug:
                import traceback
                self.logger.error(traceback.format_exc())
            return None

def main():
    """Main function to run the population comparison analysis."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run population comparison analysis on SURREAL data')
    
    parser.add_argument('--input_file', type=str, 
                        default='/Users/noamgal/DSProjects/Fragmentation/SURREAL/processed/merged_data/ema_fragmentation_daily_demographics.csv',
                        help='Path to merged data file')
    
    parser.add_argument('--output_dir', type=str,
                        default='/Users/noamgal/DSProjects/Fragmentation/SURREAL/results/population_comparison',
                        help='Directory to save analysis results')
    
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Verify input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file not found: {args.input_file}")
        return 1
    
    # Run analysis
    try:
        analyzer = PopulationComparisonAnalysis(
            input_path=args.input_file,
            output_dir=args.output_dir,
            debug=args.debug
        )
        
        # Load and preprocess data
        daily_df, participant_df = analyzer.load_and_preprocess_data()
        
        if daily_df is None or daily_df.empty:
            print("Error: Failed to load or preprocess data")
            return 1
        
        # Store dataframes for use in analysis
        analyzer.daily_df = daily_df
        analyzer.participant_df = participant_df
        
        # Run population comparisons
        results = analyzer.run_population_comparisons()
        
        # Save results
        analyzer.save_results()
        
        print("Population comparison analysis completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error running analysis: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main())
