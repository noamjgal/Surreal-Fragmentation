#!/usr/bin/env python3
"""
Stepwise Regression Analysis for Fragmentation and Stress

This script performs stepwise regression to explore the relationship between
fragmentation metrics and stress/anxiety measures in the participant-normalized data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from statsmodels.formula.api import ols
import logging
from datetime import datetime

class StepwiseRegressionAnalysis:
    def __init__(self, debug=False):
        """Initialize the stepwise regression analysis class with hardcoded paths."""
        # Hardcoded paths
        self.participant_file = "/Users/noamgal/DSProjects/Fragmentation/SURREAL/processed/merged_data/ema_fragmentation_demographics_participant_norm.csv"
        self.output_dir = Path("SURREAL/results/regression_analysis")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug = debug
        self._setup_logging()
        self.regression_results = []
        
        # The metrics we're interested in
        self.fragmentation_predictors = [
            'frag_digital_fragmentation_index', 
            'frag_mobility_fragmentation_index', 
            'frag_overlap_fragmentation_index'
        ]
        
        # Additional predictors
        self.episode_predictors = [
            'frag_digital_episode_count',
            'frag_mobility_episode_count',
            'frag_overlap_episode_count'
        ]
        
        self.duration_predictors = [
            'frag_digital_total_duration',
            'frag_mobility_total_duration',
            'frag_overlap_total_duration'
        ]
        
        # Outcome variables (stress/anxiety measures)
        self.outcome_variables = [
            'ema_STAI_Y_A_6_zstd',  # Standardized anxiety score
            'ema_STAI_Y_A_6_raw',   # Raw anxiety score
            'ema_CES_D_8_zstd',     # Standardized mood/depression score
            'ema_CES_D_8_raw'       # Raw mood/depression score
        ]
        
        # Control variables - note we'll handle 'City.center' differently
        self.control_variables = [
            'Gender'
        ]
        
    def _setup_logging(self):
        """Set up logging configuration"""
        log_dir = self.output_dir / 'logs'
        log_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'stepwise_regression_{timestamp}.log'
        
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
        self.logger.info(f"Initializing stepwise regression analysis")
        self.logger.info(f"Participant-normalized data: {self.participant_file}")
        self.logger.info(f"Output directory: {self.output_dir}")

    def load_data(self):
        """Load participant-normalized data.
        
        Returns:
            DataFrame: Participant-normalized dataset
        """
        # Load participant-normalized data
        self.logger.info(f"Loading participant-normalized data from {self.participant_file}")
        try:
            df = pd.read_csv(self.participant_file)
            self.logger.info(f"Data loaded successfully with shape: {df.shape}")
            self.logger.info(f"Data columns: {', '.join(df.columns[:10])}...")
            
            # Replace hyphens in column names with underscores for easier referencing
            df.columns = [col.replace('-', '_') for col in df.columns]
            
            # Update outcome variables to match the new column names
            self.outcome_variables = [var.replace('-', '_') for var in self.outcome_variables]
            
            # Handle problematic column names
            if 'City.center' in df.columns:
                # Rename City.center to avoid patsy formula issues
                df['city_center'] = df['City.center']
                self.logger.info("Renamed 'City.center' to 'city_center' to avoid patsy formula issues")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            if self.debug:
                import traceback
                self.logger.error(traceback.format_exc())
            return None
        
    def run_stepwise_regressions(self, df):
        """Run stepwise regression for each outcome variable.
        
        Args:
            df (DataFrame): The participant-normalized dataset
            
        Returns:
            list: A list of regression results
        """
        if df is None or df.empty:
            self.logger.error("No data available for analysis")
            return False
        
        # Check if required columns exist
        all_required_cols = (
            self.fragmentation_predictors + 
            self.episode_predictors + 
            self.duration_predictors + 
            self.outcome_variables + 
            self.control_variables
        )
        
        if 'city_center' in df.columns:
            all_required_cols = list(all_required_cols) + ['city_center']
            self.control_variables = list(self.control_variables) + ['city_center']
        
        missing_cols = [col for col in all_required_cols if col not in df.columns]
        if missing_cols:
            self.logger.warning(f"Missing columns: {missing_cols}")
            
            # Update lists to only include available columns
            self.fragmentation_predictors = [col for col in self.fragmentation_predictors if col in df.columns]
            self.episode_predictors = [col for col in self.episode_predictors if col in df.columns]
            self.duration_predictors = [col for col in self.duration_predictors if col in df.columns]
            self.outcome_variables = [col for col in self.outcome_variables if col in df.columns]
            self.control_variables = [col for col in self.control_variables if col in df.columns]
        
        results = []
        
        # Run stepwise regression for each outcome variable
        for outcome_var in self.outcome_variables:
            self.logger.info(f"Running stepwise regression for {outcome_var}")
            
            # Step 1: Base model with controls only
            if self.control_variables:
                step1_formula = f"{outcome_var} ~ " + " + ".join(self.control_variables)
                step1_results = self._run_regression(df, step1_formula, outcome_var, "Step 1: Controls Only")
                results.append(step1_results)
            else:
                self.logger.warning("No control variables available, skipping step 1")
                results.append({
                    'outcome': outcome_var,
                    'step': "Step 1: Controls Only",
                    'formula': f"{outcome_var} ~ 1",
                    'n_obs': 0,
                    'converged': False,
                    'error': "No control variables available"
                })
            
            # Step 2: Add fragmentation predictors
            if self.fragmentation_predictors:
                all_predictors = self.control_variables + self.fragmentation_predictors
                step2_formula = f"{outcome_var} ~ " + " + ".join(all_predictors)
                step2_results = self._run_regression(df, step2_formula, outcome_var, "Step 2: Controls + Fragmentation")
                results.append(step2_results)
            else:
                self.logger.warning("No fragmentation predictors available, skipping step 2")
                results.append({
                    'outcome': outcome_var,
                    'step': "Step 2: Controls + Fragmentation",
                    'formula': f"{outcome_var} ~ " + " + ".join(self.control_variables) if self.control_variables else "1",
                    'n_obs': 0,
                    'converged': False,
                    'error': "No fragmentation predictors available"
                })
            
            # Step 3: Add episode counts
            if self.episode_predictors:
                all_predictors = self.control_variables + self.fragmentation_predictors + self.episode_predictors
                step3_formula = f"{outcome_var} ~ " + " + ".join(all_predictors)
                step3_results = self._run_regression(df, step3_formula, outcome_var, "Step 3: Controls + Fragmentation + Episodes")
                results.append(step3_results)
            else:
                self.logger.warning("No episode predictors available, skipping step 3")
                results.append({
                    'outcome': outcome_var,
                    'step': "Step 3: Controls + Fragmentation + Episodes",
                    'formula': f"{outcome_var} ~ " + " + ".join(self.control_variables + self.fragmentation_predictors),
                    'n_obs': 0,
                    'converged': False,
                    'error': "No episode predictors available"
                })
            
            # Step 4: Full model (add durations)
            if self.duration_predictors:
                all_predictors = self.control_variables + self.fragmentation_predictors + self.episode_predictors + self.duration_predictors
                step4_formula = f"{outcome_var} ~ " + " + ".join(all_predictors)
                step4_results = self._run_regression(df, step4_formula, outcome_var, "Step 4: Full Model")
                results.append(step4_results)
            else:
                self.logger.warning("No duration predictors available, skipping step 4")
                results.append({
                    'outcome': outcome_var,
                    'step': "Step 4: Full Model",
                    'formula': f"{outcome_var} ~ " + " + ".join(self.control_variables + self.fragmentation_predictors + self.episode_predictors),
                    'n_obs': 0,
                    'converged': False,
                    'error': "No duration predictors available"
                })
            
            # Also run individual regressions for each fragmentation metric
            for frag_metric in self.fragmentation_predictors:
                predictors = self.control_variables + [frag_metric]
                formula = f"{outcome_var} ~ " + " + ".join(predictors)
                individual_results = self._run_regression(df, formula, outcome_var, f"Individual: {frag_metric}")
                results.append(individual_results)
        
        self.regression_results = results
        self.logger.info(f"Completed all stepwise regressions. Generated {len(results)} models.")
        
        return True
    
    def _run_regression(self, df, formula, outcome_var, step_name):
        """Run a single regression model.
        
        Args:
            df (DataFrame): Dataset
            formula (str): Regression formula
            outcome_var (str): Outcome variable name
            step_name (str): Name of the regression step
            
        Returns:
            dict: Regression results
        """
        self.logger.info(f"Running regression: {formula}")
        
        try:
            # Clean data for regression (remove missing values)
            model_vars = [var.strip() for var in formula.replace(f"{outcome_var} ~", "").split("+")]
            model_vars = [var.strip() for var in model_vars] + [outcome_var]
            
            # Get data for regression
            regression_data = df[model_vars].dropna()
            n_obs = len(regression_data)
            
            if n_obs < 10:
                self.logger.warning(f"Too few observations ({n_obs}) for reliable regression")
                return {
                    'outcome': outcome_var,
                    'step': step_name,
                    'formula': formula,
                    'n_obs': n_obs,
                    'converged': False,
                    'error': "Too few observations"
                }
            
            # Fit the model
            model = ols(formula, data=regression_data)
            results = model.fit()
            
            # Extract key statistics
            r_squared = results.rsquared
            adj_r_squared = results.rsquared_adj
            aic = results.aic
            bic = results.bic
            f_stat = results.fvalue
            f_pvalue = results.f_pvalue
            
            # Extract coefficients, standard errors, t-stats, and p-values
            coefficients = {}
            for name in results.params.index:
                coefficients[f'coef_{name}'] = results.params[name]
                coefficients[f'se_{name}'] = results.bse[name]
                coefficients[f't_{name}'] = results.tvalues[name]
                coefficients[f'p_{name}'] = results.pvalues[name]
                # Add significance indicators
                p_val = results.pvalues[name]
                coefficients[f'sig_{name}'] = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            
            # Prepare results dictionary
            result_dict = {
                'outcome': outcome_var,
                'step': step_name,
                'formula': formula,
                'n_obs': n_obs,
                'converged': True,
                'r_squared': r_squared,
                'adj_r_squared': adj_r_squared,
                'aic': aic,
                'bic': bic,
                'f_statistic': f_stat,
                'f_pvalue': f_pvalue,
                **coefficients
            }
            
            # Log key results
            self.logger.info(
                f"Regression results for {outcome_var} ({step_name}): "
                f"n={n_obs}, R²={r_squared:.4f}, Adj. R²={adj_r_squared:.4f}, "
                f"F={f_stat:.2f}, p={f_pvalue:.4f}"
            )
            
            return result_dict
            
        except Exception as e:
            self.logger.error(f"Error in regression: {str(e)}")
            if self.debug:
                import traceback
                self.logger.error(traceback.format_exc())
                
            return {
                'outcome': outcome_var,
                'step': step_name,
                'formula': formula,
                'n_obs': 0,  # Add default n_obs to avoid KeyError
                'converged': False,
                'error': str(e)
            }
    
    def save_results(self):
        """Save regression results to Excel files"""
        if not self.regression_results:
            self.logger.warning("No results to save")
            return None
        
        try:
            # Generate timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Convert results to DataFrame
            results_df = pd.DataFrame(self.regression_results)
            
            # Round numeric columns for cleaner output
            numeric_cols = results_df.select_dtypes(include=[np.number]).columns
            results_df[numeric_cols] = results_df[numeric_cols].round(4)
            
            # Save to Excel
            output_path = self.output_dir / f'stepwise_regression_results_{timestamp}.xlsx'
            
            with pd.ExcelWriter(output_path) as writer:
                # Save all results to first sheet
                results_df.to_excel(writer, sheet_name='All Results', index=False)
                
                # Create separate sheets for each outcome variable
                for outcome_var in results_df['outcome'].unique():
                    outcome_results = results_df[results_df['outcome'] == outcome_var]
                    if not outcome_results.empty:
                        # Shorten sheet name if needed (Excel limit is 31 chars)
                        sheet_name = outcome_var
                        if len(sheet_name) > 30:
                            sheet_name = outcome_var.split('_')[-2] + '_' + outcome_var.split('_')[-1]
                        outcome_results.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Create summary sheet with model comparisons
                summary_data = []
                for outcome_var in results_df['outcome'].unique():
                    outcome_steps = results_df[
                        (results_df['outcome'] == outcome_var) & 
                        (results_df['step'].str.startswith('Step'))
                    ]
                    
                    if not outcome_steps.empty:
                        for _, row in outcome_steps.iterrows():
                            # Check if model converged
                            if not row['converged']:
                                summary_data.append({
                                    'Outcome': outcome_var,
                                    'Step': row['step'],
                                    'N': row['n_obs'],
                                    'R²': np.nan,
                                    'Adj. R²': np.nan,
                                    'AIC': np.nan,
                                    'BIC': np.nan,
                                    'F': np.nan,
                                    'p': np.nan,
                                    'Key Fragmentation Effects': f"Model Error: {row.get('error', 'Unknown')}"
                                })
                                continue
                        
                            # Extract key fragmentation coefficients if they exist
                            frag_coeffs = []
                            for frag_metric in self.fragmentation_predictors:
                                coef_col = f'coef_{frag_metric}'
                                p_col = f'p_{frag_metric}'
                                if coef_col in row and p_col in row:
                                    coef = row[coef_col]
                                    p_val = row[p_col]
                                    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                                    frag_coeffs.append(f"{frag_metric}: β={coef:.4f}{sig}")
                            
                            summary_data.append({
                                'Outcome': outcome_var,
                                'Step': row['step'],
                                'N': row['n_obs'],
                                'R²': row['r_squared'],
                                'Adj. R²': row['adj_r_squared'],
                                'AIC': row['aic'],
                                'BIC': row['bic'],
                                'F': row['f_statistic'],
                                'p': row['f_pvalue'],
                                'Key Fragmentation Effects': '; '.join(frag_coeffs) if frag_coeffs else 'None'
                            })
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Model Comparison', index=False)
                    
                    # Also create a coefficients-focused summary
                    coef_summary = []
                    for outcome_var in results_df['outcome'].unique():
                        # Get the full model results
                        full_model = results_df[
                            (results_df['outcome'] == outcome_var) & 
                            (results_df['step'] == 'Step 4: Full Model') &
                            (results_df['converged'] == True)
                        ]
                        
                        if not full_model.empty:
                            row = full_model.iloc[0]
                            
                            # Get all coefficients
                            for name in self.fragmentation_predictors + self.episode_predictors + self.duration_predictors:
                                coef_col = f'coef_{name}'
                                se_col = f'se_{name}'
                                p_col = f'p_{name}'
                                
                                if coef_col in row and p_col in row:
                                    coef_summary.append({
                                        'Outcome': outcome_var,
                                        'Predictor': name,
                                        'Coefficient': row[coef_col],
                                        'Std. Error': row[se_col] if se_col in row else np.nan,
                                        'p-value': row[p_col],
                                        'Significance': '***' if row[p_col] < 0.001 else '**' if row[p_col] < 0.01 else '*' if row[p_col] < 0.05 else ''
                                    })
                    
                    if coef_summary:
                        coef_df = pd.DataFrame(coef_summary)
                        coef_df.to_excel(writer, sheet_name='Coefficients', index=False)
            
            self.logger.info(f"Saved regression results to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            if self.debug:
                import traceback
                self.logger.error(traceback.format_exc())
            return None

def main():
    """Main function to run the stepwise regression analysis."""
    try:
        # Create analyzer
        analyzer = StepwiseRegressionAnalysis(debug=True)
        
        # Load data
        df = analyzer.load_data()
        
        if df is None or df.empty:
            print("Error: Failed to load data")
            return 1
        
        # Run regressions
        if analyzer.run_stepwise_regressions(df):
            # Save results
            results_path = analyzer.save_results()
            
            if results_path:
                print(f"Stepwise regression analysis completed successfully!")
                print(f"Results saved to: {results_path}")
                return 0
            else:
                print("Error: Failed to save results")
                return 1
        else:
            print("Error: Failed to run regressions")
            return 1
        
    except Exception as e:
        print(f"Error running analysis: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main())