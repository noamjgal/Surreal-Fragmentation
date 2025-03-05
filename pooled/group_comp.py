#!/usr/bin/env python3
"""
Population Comparison Analysis Script

This script performs statistical comparisons (t-tests) between different 
demographic groups (gender, age group, location) on anxiety scores, mood scores, 
and fragmentation metrics, producing comprehensive reports of the differences.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import logging
import argparse
from datetime import datetime

class PopulationComparisonAnalysis:
    def __init__(self, input_path, output_dir, debug=False):
        """Initialize the population comparison analysis class.
        
        Args:
            input_path (str): Path to data file
            output_dir (str): Directory to save analysis results
            debug (bool): Enable debug logging
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug = debug
        self._setup_logging()
        self.comparison_results = []
        
        # The metrics we're interested in
        self.fragmentation_metrics = [
            'digital_fragmentation', 'mobility_fragmentation', 'overlap_fragmentation'
        ]
        self.anxiety_metrics = ['anxiety_score_std', 'anxiety_score_raw']
        self.mood_metrics = ['mood_score_std', 'mood_score_raw']
        
        # The demographic variables we're interested in
        self.demographic_vars = [
            'gender_standardized', 'age_group', 'location_type'
        ]
        
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

    def load_data(self):
        """Load data for analysis.
        
        Returns:
            DataFrame: The loaded dataset
        """
        self.logger.info(f"Loading data from {self.input_path}")
        
        try:
            # Load data
            df = pd.read_csv(self.input_path)
            self.logger.info(f"Data loaded successfully with shape: {df.shape}")
            
            # Verify that our expected columns exist
            all_expected_cols = (
                self.fragmentation_metrics + 
                self.anxiety_metrics + 
                self.mood_metrics + 
                self.demographic_vars
            )
            
            missing_cols = [col for col in all_expected_cols if col not in df.columns]
            if missing_cols:
                self.logger.warning(f"Missing expected columns: {missing_cols}")
            
            # Log basic dataset info
            self.logger.info(f"Dataset source values: {df['dataset_source'].unique()}")
            self.logger.info(f"Gender values: {df['gender_standardized'].unique()}")
            self.logger.info(f"Age group values: {df['age_group'].unique()}")
            self.logger.info(f"Location type values: {df['location_type'].unique()}")
            
            # Check for missing values
            for col in all_expected_cols:
                if col in df.columns:
                    missing_count = df[col].isna().sum()
                    missing_pct = round(missing_count / len(df) * 100, 2)
                    self.logger.info(f"Column '{col}' has {missing_count} missing values ({missing_pct}%)")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            if self.debug:
                import traceback
                self.logger.error(traceback.format_exc())
            return None
    
    def run_comparisons(self, df):
        """Run all demographic comparisons on anxiety, mood, and fragmentation metrics.
        
        Args:
            df (DataFrame): The dataset to analyze
            
        Returns:
            list: A list of dictionaries containing comparison results
        """
        self.comparison_results = []
        
        # 1. Compare anxiety metrics across demographic groups
        for anxiety_metric in self.anxiety_metrics:
            self.logger.info(f"\nAnalyzing {anxiety_metric} across demographic groups")
            for demo_var in self.demographic_vars:
                results = self._run_comparison(df, anxiety_metric, demo_var, "anxiety")
                self.comparison_results.extend(results)
        
        # 2. Compare mood metrics across demographic groups
        for mood_metric in self.mood_metrics:
            self.logger.info(f"\nAnalyzing {mood_metric} across demographic groups")
            for demo_var in self.demographic_vars:
                results = self._run_comparison(df, mood_metric, demo_var, "mood")
                self.comparison_results.extend(results)
        
        # 3. Compare fragmentation metrics across demographic groups
        for frag_metric in self.fragmentation_metrics:
            self.logger.info(f"\nAnalyzing {frag_metric} across demographic groups")
            for demo_var in self.demographic_vars:
                results = self._run_comparison(df, frag_metric, demo_var, "fragmentation")
                self.comparison_results.extend(results)
        
        # 4. Compare fragmentation by high/low anxiety
        for anxiety_metric in self.anxiety_metrics:
            self.logger.info(f"\nAnalyzing fragmentation metrics by {anxiety_metric} groups")
            for frag_metric in self.fragmentation_metrics:
                results = self._run_metric_by_emotion_group(df, frag_metric, anxiety_metric, "anxiety")
                self.comparison_results.extend(results)
        
        # 5. Compare fragmentation by high/low mood
        for mood_metric in self.mood_metrics:
            self.logger.info(f"\nAnalyzing fragmentation metrics by {mood_metric} groups")
            for frag_metric in self.fragmentation_metrics:
                results = self._run_metric_by_emotion_group(df, frag_metric, mood_metric, "mood")
                self.comparison_results.extend(results)
        
        self.logger.info(f"Completed all comparisons. Generated {len(self.comparison_results)} results.")
        return self.comparison_results
    
    def _run_comparison(self, df, outcome_var, group_var, analysis_type):
        """Run a comparison between demographic groups on an outcome variable.
        
        Args:
            df (DataFrame): Dataset
            outcome_var (str): Variable to analyze
            group_var (str): Grouping variable
            analysis_type (str): Type of analysis (anxiety, mood, fragmentation)
            
        Returns:
            list: Comparison results
        """
        results = []
        
        try:
            # Skip if too many missing values
            if outcome_var not in df.columns:
                self.logger.warning(f"Skipping {outcome_var}, column not found")
                return results
                
            if group_var not in df.columns:
                self.logger.warning(f"Skipping {group_var}, column not found")
                return results
                
            if df[outcome_var].isna().sum() > 0.5 * len(df):
                self.logger.warning(f"Skipping {outcome_var}, too many missing values")
                return results
                
            if df[group_var].isna().sum() > 0.5 * len(df):
                self.logger.warning(f"Skipping {group_var}, too many missing values")
                return results
            
            # Get unique groups
            groups = df[group_var].dropna().unique()
            
            # Skip if only one group
            if len(groups) < 2:
                self.logger.info(f"Skipping {group_var}, only one group found")
                return results
            
            if len(groups) == 2:  # Binary comparison (t-test)
                g1_data = df[df[group_var] == groups[0]][outcome_var].dropna()
                g2_data = df[df[group_var] == groups[1]][outcome_var].dropna()
                
                # Minimum number of observations required
                if len(g1_data) < 5 or len(g2_data) < 5:
                    self.logger.info(f"Skipping {group_var} for {outcome_var}: insufficient observations")
                    return results
                
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
                    'analysis_type': f"{analysis_type}_by_demographic",
                    'outcome': outcome_var, 
                    'group_variable': group_var,
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
                self.logger.info(
                    f"Compared {outcome_var} between {group_var} groups: "
                    f"{groups[0]} (n={len(g1_data)}, mean={g1_mean:.2f}) vs "
                    f"{groups[1]} (n={len(g2_data)}, mean={g2_mean:.2f}), "
                    f"t={t_stat:.2f}, p={p_val:.4f}, d={cohen_d:.2f}"
                )
            
            else:  # More than two groups - compare each pair
                # For each pair of groups, run a t-test
                for i in range(len(groups)):
                    for j in range(i+1, len(groups)):
                        g1_data = df[df[group_var] == groups[i]][outcome_var].dropna()
                        g2_data = df[df[group_var] == groups[j]][outcome_var].dropna()
                        
                        # Minimum number of observations required
                        if len(g1_data) < 5 or len(g2_data) < 5:
                            self.logger.info(
                                f"Skipping comparison of {groups[i]} vs {groups[j]} "
                                f"for {outcome_var}: insufficient observations"
                            )
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
                        cohen_d = abs(g1_mean - g2_mean) / pooled_std if pooled_std != 0 else np.nan
                        
                        result = {
                            'analysis_type': f"{analysis_type}_by_demographic",
                            'outcome': outcome_var, 
                            'group_variable': group_var,
                            'test': 't-test',
                            'statistic': float(t_stat),
                            'p_value': float(p_val),
                            'effect_size': float(cohen_d),
                            'effect_size_type': "Cohen's d",
                            'sig_level': '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else '',
                            'group1': str(groups[i]),
                            'group1_mean': float(g1_mean),
                            'group1_std': float(g1_std),
                            'group1_n': int(len(g1_data)),
                            'group2': str(groups[j]),
                            'group2_mean': float(g2_mean),
                            'group2_std': float(g2_std),
                            'group2_n': int(len(g2_data)),
                        }
                        
                        results.append(result)
                        self.logger.info(
                            f"Compared {outcome_var} between {group_var} groups: "
                            f"{groups[i]} (n={len(g1_data)}, mean={g1_mean:.2f}) vs "
                            f"{groups[j]} (n={len(g2_data)}, mean={g2_mean:.2f}), "
                            f"t={t_stat:.2f}, p={p_val:.4f}, d={cohen_d:.2f}"
                        )
                        
        except Exception as e:
            self.logger.error(f"Error comparing {outcome_var} across {group_var} groups: {str(e)}")
            if self.debug:
                import traceback
                self.logger.error(traceback.format_exc())
        
        return results
    
    def _run_metric_by_emotion_group(self, df, outcome_var, emotion_var, emotion_type):
        """Compare a metric between high and low emotion groups.
        
        Args:
            df (DataFrame): Dataset
            outcome_var (str): Outcome variable (fragmentation metric)
            emotion_var (str): Emotion variable (anxiety or mood)
            emotion_type (str): Type of emotion (anxiety or mood)
            
        Returns:
            list: Comparison results
        """
        results = []
        
        try:
            # Skip if columns don't exist or have too many missing values
            if outcome_var not in df.columns or emotion_var not in df.columns:
                self.logger.warning(f"Skipping {outcome_var} by {emotion_var}, column(s) not found")
                return results
                
            if df[outcome_var].isna().sum() > 0.5 * len(df) or df[emotion_var].isna().sum() > 0.5 * len(df):
                self.logger.warning(f"Skipping {outcome_var} by {emotion_var}, too many missing values")
                return results
            
            # Create high/low emotion groups based on median split
            median_val = df[emotion_var].median()
            temp_df = df.copy()
            temp_df['emotion_group'] = temp_df[emotion_var].apply(lambda x: 'high' if x > median_val else 'low')
            
            self.logger.info(f"Created {emotion_type} groups based on median split of {emotion_var} ({median_val:.2f})")
            
            # Get data for each group
            high_group = temp_df[temp_df['emotion_group'] == 'high'][outcome_var].dropna()
            low_group = temp_df[temp_df['emotion_group'] == 'low'][outcome_var].dropna()
            
            # Minimum number of observations required
            if len(high_group) < 5 or len(low_group) < 5:
                self.logger.info(
                    f"Skipping {outcome_var} comparison by {emotion_var} groups: "
                    "insufficient observations"
                )
                return results
            
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
            cohen_d = abs(high_mean - low_mean) / pooled_std if pooled_std != 0 else np.nan
            
            result = {
                'analysis_type': f"fragmentation_by_{emotion_type}",
                'outcome': outcome_var, 
                'group_variable': f"{emotion_var}_group",
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
            self.logger.info(
                f"Compared {outcome_var} between {emotion_var} groups: "
                f"high (n={len(high_group)}, mean={high_mean:.2f}) vs "
                f"low (n={len(low_group)}, mean={low_mean:.2f}), "
                f"t={t_stat:.2f}, p={p_val:.4f}, d={cohen_d:.2f}"
            )
                
        except Exception as e:
            self.logger.error(f"Error comparing {outcome_var} by {emotion_var} groups: {str(e)}")
            if self.debug:
                import traceback
                self.logger.error(traceback.format_exc())
        
        return results
    
    def save_results(self):
        """Save comparison results to Excel files"""
        if not self.comparison_results:
            self.logger.warning("No results to save")
            return None
        
        try:
            # Create results directory if it doesn't exist
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Convert results to DataFrame
            results_df = pd.DataFrame(self.comparison_results)
            
            # Round numeric columns for cleaner output
            numeric_cols = results_df.select_dtypes(include=[np.number]).columns
            results_df[numeric_cols] = results_df[numeric_cols].round(4)
            
            # Save all results to a single Excel file
            all_results_path = self.output_dir / f'population_comparisons_{timestamp}.xlsx'
            
            with pd.ExcelWriter(all_results_path) as writer:
                # Save all results to first sheet
                results_df.to_excel(writer, sheet_name='All Results', index=False)
                
                # Create summary sheets by analysis type
                for analysis_type in results_df['analysis_type'].unique():
                    type_results = results_df[results_df['analysis_type'] == analysis_type]
                    sheet_name = analysis_type[:31]  # Excel limits sheet names to 31 chars
                    type_results.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Create sheets for each outcome variable
                for outcome in sorted(results_df['outcome'].unique()):
                    outcome_results = results_df[results_df['outcome'] == outcome]
                    sheet_name = f"{outcome[:28]}"  # Keep within Excel's 31 char limit
                    outcome_results.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Create significant results summary
                sig_results = results_df[results_df['p_value'] < 0.05]
                if not sig_results.empty:
                    sig_results.to_excel(writer, sheet_name='Significant Results', index=False)
            
            self.logger.info(f"Saved all results to {all_results_path}")
            return all_results_path
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            if self.debug:
                import traceback
                self.logger.error(traceback.format_exc())
            return None

def main():
    """Main function to run the population comparison analysis."""
    
    # Hardcoded input and output paths
    input_file = "pooled/processed/pooled_stai_data.csv"
    output_dir = "pooled/results/groups"
    debug = False  # Set to True if you want debug logging
    
    # Verify input file exists
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_file}")
        return 1
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run analysis
    try:
        analyzer = PopulationComparisonAnalysis(
            input_path=input_file,
            output_dir=output_dir,
            debug=debug
        )
        
        # Load data
        df = analyzer.load_data()
        
        if df is None or df.empty:
            print("Error: Failed to load data")
            return 1
        
        # Run comparisons
        analyzer.run_comparisons(df)
        
        # Save results
        results_path = analyzer.save_results()
        
        if results_path:
            print(f"Population comparison analysis completed successfully!")
            print(f"Results saved to: {results_path}")
            return 0
        else:
            print("Error: Failed to save results")
            return 1
        
    except Exception as e:
        print(f"Error running analysis: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main())